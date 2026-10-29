# End-to-end decode through TinyChat's own llama module (tinychat/models/llama.py) with 4-bit AWQ
# WQLinear projections: prefill, then greedy decode where every step runs the fused
# single_query_attention (rotary + KV-cache update + attention). Reference: the non-fused path,
# re-running the whole sequence through the same model's prefill branch (plain softmax attention)
# at every step, teacher-forced on the fused path's tokens.
# transformers, flash_attn and a checkpoint/tokenizer are not available on this host, so the two
# imports llama.py needs are stubbed (LlamaRotaryEmbedding is unused there; flash_attn_func is
# replaced by exact causal softmax attention) and the weights are random.
import sys, types, importlib.util, os
import torch, torch.nn.functional as F

SRC = sys.argv[1]
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(SRC, "awq", "kernels"))

tf = types.ModuleType("transformers"); tfm = types.ModuleType("transformers.models")
tfl = types.ModuleType("transformers.models.llama"); tflm = types.ModuleType("transformers.models.llama.modeling_llama")
class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, *a, **k): super().__init__()
tflm.LlamaRotaryEmbedding = LlamaRotaryEmbedding
sys.modules.update({"transformers": tf, "transformers.models": tfm, "transformers.models.llama": tfl,
                    "transformers.models.llama.modeling_llama": tflm})
fa = types.ModuleType("flash_attn")
def flash_attn_func(q, k, v, causal=True):
    # q [B, S, H, D], k/v [B, S, Hkv, D]
    rep = q.shape[2] // k.shape[2]
    k = k.repeat_interleave(rep, dim=2); v = v.repeat_interleave(rep, dim=2)
    o = F.scaled_dot_product_attention(q.transpose(1, 2).float(), k.transpose(1, 2).float(),
                                       v.transpose(1, 2).float(), is_causal=causal)
    return o.transpose(1, 2).to(q.dtype)
fa.flash_attn_func = flash_attn_func
sys.modules["flash_attn"] = fa

import tinychat.utils.constants  # noqa: E402
tinychat.utils.constants.init()
spec = importlib.util.spec_from_file_location("tc_llama", os.path.join(SRC, "tinychat/models/llama.py"))
llama = importlib.util.module_from_spec(spec); spec.loader.exec_module(llama)
from awq.quantize.qmodule import WQLinear  # noqa: E402


def quantize_linears(module, group=128):
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear) and name != "lm_head":
            w = child.weight.data.float()
            o, i = w.shape
            wg = w.reshape(o, i // group, group)
            mx, mn = wg.amax(-1), wg.amin(-1)
            scales = ((mx - mn).clamp(min=1e-5) / 15)
            zeros = torch.round(-mn / scales).clamp(0, 15)
            # snap to the 4-bit grid first: from_linear does not clamp q to [0, 15]
            q = torch.round(wg / scales[..., None] + zeros[..., None]).clamp(0, 15)
            child.weight.data = ((q - zeros[..., None]) * scales[..., None]).reshape(o, i).half()
            setattr(module, name, WQLinear.from_linear(child, 4, group, scales=scales.half(), zeros=zeros.half()))
        else:
            quantize_linears(child, group)


class Args:
    pass

def make_args(H, Hkv, D, layers):
    a = Args()
    a.vocab_size = 512; a.hidden_size = H * D; a.num_attention_heads = H; a.num_key_value_heads = Hkv
    a.intermediate_size = 2 * H * D; a.num_hidden_layers = layers; a.rms_norm_eps = 1e-5
    a.max_position_embeddings = 512; a.rope_theta = 10000.0; a.rope_scaling = None
    return a


@torch.inference_mode()
def run(H, Hkv, D, layers=4, prompt=24, steps=40, seed=0):
    torch.manual_seed(seed)
    args = make_args(H, Hkv, D, layers)
    model = llama.LlamaForCausalLM(args)
    for p in model.parameters():
        p.data.normal_(0, 0.02 if p.dim() > 1 else 0.0).add_(1.0 if p.dim() == 1 else 0.0)
    model.model.embed_tokens.weight.data.normal_(0, 1.0)
    model = model.half().cuda()
    quantize_linears(model)
    model = model.cuda()
    for layer in model.model.layers:
        a = layer.self_attn
        a.cache_k = a.cache_k[:1].contiguous(); a.cache_v = a.cache_v[:1].contiguous()

    tokens = torch.randint(0, args.vocab_size, (1, prompt), device="cuda")
    logits = model(tokens, 0, quant=False)[:, -1]
    seq = tokens
    fused_logits = []
    for s in range(steps):
        nxt = logits.argmax(-1, keepdim=True)
        seq = torch.cat([seq, nxt], 1)
        logits = model(nxt, seq.shape[1] - 1, quant=False)[:, -1]   # decode: single_query_attention
        fused_logits.append(logits.float())

    max_diff = 0.0; agree = 0; cos_min = 1.0
    for s in range(steps):
        n = prompt + s + 1
        ref = model(seq[:, :n], 0, quant=False)[:, -1].float()        # non-fused attention over n tokens
        d = (fused_logits[s] - ref).abs().max().item()
        max_diff = max(max_diff, d)
        cos_min = min(cos_min, F.cosine_similarity(fused_logits[s], ref).item())
        agree += int(fused_logits[s].argmax() == ref.argmax())
    scale = max(l.abs().max().item() for l in fused_logits)
    ok = cos_min > 0.999 and agree >= steps - 1 and all(torch.isfinite(l).all() for l in fused_logits)
    print(f"{'PASS' if ok else 'FAIL'} H={H} Hkv={Hkv} D={D} layers={layers}: steps={steps} "
          f"argmax agree {agree}/{steps}, max|dlogit|={max_diff:.3e} (|logit|max {scale:.2f}), min cos={cos_min:.6f}",
          flush=True)
    return ok


oks = [run(8, 8, 128), run(16, 4, 64), run(8, 2, 128, steps=64), run(8, 1, 80)]
print(f"{sum(oks)}/{len(oks)} passed")
sys.exit(0 if all(oks) else 1)
