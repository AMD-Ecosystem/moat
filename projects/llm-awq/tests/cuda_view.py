# Print a source file as the CUDA build sees it: resolve only #if/#elif/#else blocks whose condition
# is exactly defined(__HIP_PLATFORM_AMD__) or !defined(__HIP_PLATFORM_AMD__) (macro undefined).
import re, sys
HIP = re.compile(r'^\s*#\s*if\s+(!?)\s*defined\s*\(\s*__HIP_PLATFORM_AMD__\s*\)\s*(//.*)?$')
out, stack = [], []  # stack entries: None for foreign #if, or dict(keep=bool, taken=bool)
def emitting():
    return all(e is None or e["keep"] for e in stack)
for line in open(sys.argv[1]).read().split("\n"):
    m = HIP.match(line)
    s = line.strip()
    if m:
        keep = m.group(1) == "!"
        stack.append({"keep": keep, "taken": keep, "elif_open": False})
        continue
    if re.match(r'#\s*if', s):
        stack.append(None)
        if emitting(): out.append(line)
        continue
    top = stack[-1] if stack else None
    if re.match(r'#\s*elif', s) and isinstance(top, dict):
        if top["taken"]:
            top["keep"] = False
        else:
            # first surviving branch becomes a plain #if
            top["keep"] = True; top["taken"] = True; top["elif_open"] = True
            stack[-1] = top
            if all(e is None or e["keep"] for e in stack[:-1]):
                out.append(re.sub(r'#\s*elif', '#if', line, count=1))
        continue
    if re.match(r'#\s*else', s) and isinstance(top, dict):
        if top["elif_open"]:
            top["keep"] = True
            if all(e is None or e["keep"] for e in stack[:-1]): out.append(line)
        else:
            top["keep"] = not top["taken"]; top["taken"] = True
        continue
    if re.match(r'#\s*endif', s):
        e = stack.pop()
        if isinstance(e, dict):
            if e["elif_open"] and all(x is None or x["keep"] for x in stack): out.append(line)
            continue
        if emitting(): out.append(line)
        continue
    if emitting(): out.append(line)
print("\n".join(out), end="")
