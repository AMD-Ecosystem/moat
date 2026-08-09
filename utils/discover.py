#!/usr/bin/env python3
"""GitHub discovery + ranking for MOAT.

Honest limit: this is a best-effort union of targeted searches, NOT a census of
every CUDA repo. GitHub repo search caps at 1000 results per query;
`language:Cuda` only catches repos whose dominant language is Cuda (it misses
C++/Python repos that merely contain .cu files); `topic:cuda` only catches
self-tagged repos. Coverage grows by adding queries in config/discover.toml.

The metadata pass has a hard blind spot: `gh search repos` indexes only
name+description+topics, so a CUDA library is invisible if its dominant
language is not Cuda AND none of those fields contains "cuda" (exemplar:
facebookresearch/pytorch3d -- Python-dominant, empty topics, no "cuda" in
description, yet 328 KB of .cu). The `--code-search` pass closes this by
querying the REST code-search API on CUDA SOURCE (config `[code_search]`) and
merging new repos into candidates.json. The metadata pass is non-clobbering:
it preserves those code:-sourced records on its next run instead of
overwriting them (historically it truncated the file and silently erased
every code-search find -- the actual cause of the blind spot).

Output: data/candidates.json (ranked, filtered) and data/candidates.raw.jsonl
(every unique hit before filtering, for audit). Adopt rows with
`python3 utils/moatlib.py scaffold <owner/repo>`."""

import argparse
import datetime as _dt
import json
import math
import subprocess
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "config" / "discover.toml"
DATA = REPO_ROOT / "data"
# The licence is fetched so candidates.json carries one from the start, since a
# licence discovered after the port is finished can block the contribution the port
# existed to make. The field is not sufficient on its own -- it is absent or unparsed
# for roughly a fifth of repos -- so intake re-checks with a real detector
# (utils/licenses.py).
FIELDS = ("fullName,url,defaultBranch,stargazersCount,forksCount,pushedAt,"
          "isFork,isArchived,isDisabled,language,description,license")


def load_cfg():
    return tomllib.loads(CONFIG.read_text())


def run_query(query, limit):
    """One `gh search repos` call. Returns a list of repo dicts (gh handles
    paging up to the 1000 cap). Empty on failure; backs off on rate-limit."""
    cmd = ["gh", "search", "repos", *query.split(),
           "--sort", "stars", "--order", "desc",
           "--limit", str(limit), "--json", FIELDS]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(f"discover: query failed: {query}\n{r.stderr.strip()}\n")
        if "rate limit" in r.stderr.lower() or "API rate" in r.stderr:
            time.sleep(30)
        return []
    try:
        return json.loads(r.stdout or "[]")
    except json.JSONDecodeError:
        return []


def days_since(iso):
    if not iso:
        return 1e9
    try:
        t = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return 1e9
    return (datetime.now(timezone.utc) - t).total_seconds() / 86400.0


def gather(cfg, limit):
    """Run every query, dedupe by lowercased full_name, merge matched_queries,
    write the raw audit trail."""
    seen = {}
    DATA.mkdir(parents=True, exist_ok=True)
    raw = DATA / "candidates.raw.jsonl"
    min_stars = cfg["search"]["min_stars"]
    since_date = _since_date(cfg["search"]["since_months"])
    with open(raw, "w") as rawf:
        for q in cfg["search"]["queries"]:
            full_q = f"{q} stars:>={min_stars} pushed:>={since_date}"
            hits = run_query(full_q, limit)
            for h in hits:
                key = h["fullName"].lower()
                rawf.write(json.dumps(h) + "\n")
                if key in seen:
                    seen[key]["matched_queries"].append(q)
                else:
                    h["matched_queries"] = [q]
                    seen[key] = h
            time.sleep(cfg["search"]["throttle_seconds"])
    return list(seen.values())


def _since_date(months):
    days = int(months) * 30
    t = datetime.now(timezone.utc).timestamp() - days * 86400
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d")


def passes(h, cfg):
    s = cfg["search"]
    fn = h["fullName"].lower()
    if fn in {e.lower() for e in s["exclude"]}:
        return False
    if h.get("isArchived") or h.get("isDisabled"):
        return False
    if h.get("isFork") and fn not in {e.lower() for e in s["fork_exceptions"]}:
        return False
    if (h.get("stargazersCount") or 0) < s["min_stars"]:
        return False
    if days_since(h.get("pushedAt")) > int(s["since_months"]) * 30:
        return False
    return True


def score(h, w):
    stars = h.get("stargazersCount") or 0
    forks = h.get("forksCount") or 0
    dsp = days_since(h.get("pushedAt"))
    stars_term = w["stars"] * math.log10(stars + 1)
    forks_term = w["forks"] * math.log10(forks + 1)
    recency_term = w["recency"] * (0.5 ** (dsp / w["half_life_days"]))
    parts = {"stars_term": round(stars_term, 3),
             "forks_term": round(forks_term, 3),
             "recency_term": round(recency_term, 3)}
    return round(stars_term + forks_term + recency_term, 3), parts


def to_record(h, cfg):
    prio, parts = score(h, cfg["weights"])
    fn = h["fullName"]
    return {
        "full_name": fn,
        "url": h.get("url") or f"https://github.com/{fn}",
        "default_branch": h.get("defaultBranch"),
        "stars": h.get("stargazersCount") or 0,
        "forks": h.get("forksCount") or 0,
        "pushed_at": h.get("pushedAt"),
        "days_since_push": round(days_since(h.get("pushedAt")), 1),
        "is_fork": bool(h.get("isFork")),
        "fork_exception": h["fullName"].lower() in {e.lower() for e in cfg["search"]["fork_exceptions"]},
        "language": h.get("language"),
        "description": (h.get("description") or "").strip(),
        "matched_queries": sorted(set(h.get("matched_queries", []))),
        # gh returns {key, name, url} or null. The key is the closest thing to an
        # identifier it offers -- not an SPDX id, and absent for roughly a fifth of
        # repos -- so intake still reads the licence file. This is a starting point,
        # never the determination.
        "license_key": ((h.get("license") or {}).get("key") or None),
        "license_name": ((h.get("license") or {}).get("name") or None),
        "cuda_bytes": h.get("_cuda_bytes"),
        "cuda_dominant": h.get("language") == "Cuda",
        "priority": prio,
        "score_parts": parts,
    }


def cuda_bytes(full_name, cache):
    """Bytes of CUDA code in the repo via the languages API; 0 if none or on
    error. This is the decisive signal that a repo actually contains .cu/.cuh
    kernels, versus merely mentioning CUDA in its README."""
    if full_name in cache:
        return cache[full_name]
    r = subprocess.run(["gh", "api", f"repos/{full_name}/languages"],
                       capture_output=True, text=True)
    cb = 0
    if r.returncode == 0:
        try:
            cb = int(json.loads(r.stdout).get("Cuda", 0))
        except (json.JSONDecodeError, ValueError):
            cb = 0
    cache[full_name] = cb
    time.sleep(0.08)
    return cb


def code_search(query, page, per_page=100):
    """One REST code-search call. Returns (items, total_count). The `gh search
    code` CLI returns [] for these queries, so we hit the API directly. On a
    rate-limit (HTTP 403 / secondary/abuse limit) it retries with exponential
    backoff rather than giving up, so a transient throttle does not silently
    truncate the query (which is what loses high-signal mid-page repos)."""
    backoffs = [30, 60, 120, 240, 300, 300]
    for attempt in range(len(backoffs) + 1):
        r = subprocess.run(
            ["gh", "api", "-X", "GET", "search/code",
             "-f", f"q={query}", "-f", f"per_page={per_page}", "-f", f"page={page}"],
            capture_output=True, text=True)
        if r.returncode == 0:
            try:
                d = json.loads(r.stdout or "{}")
                return d.get("items", []), d.get("total_count")
            except json.JSONDecodeError:
                return [], None
        err = r.stderr.lower()
        is_rl = "rate limit" in err or "403" in err or "abuse" in err
        sys.stderr.write(f"discover: code-search failed (p{page}): {query}\n{r.stderr.strip()}\n")
        if is_rl and attempt < len(backoffs):
            wait = backoffs[attempt]
            sys.stderr.write(f"discover: rate-limited, backoff {wait}s "
                             f"(attempt {attempt + 1}/{len(backoffs)})\n")
            time.sleep(wait)
            continue
        return [], None


def code_search_repos(cfg, disk=None, save=None):
    """Run every [code_search] query, page up to the cap, dedupe by
    repository.full_name (preserving first-seen order). Returns an ordered list
    of [full_name, matched_query] -- matched_query is the first query that hit
    the repo. Throttles between every API call.

    Resumable: completed queries are recorded in disk['cs_done'] and the
    accumulated order in disk['cs_order'], persisted via save() after each
    query, so a run killed by a wall-clock limit resumes without re-spending
    the code-search rate budget on finished queries."""
    cs = cfg["code_search"]
    page_cap = int(cs.get("page_cap", 10))
    throttle = float(cs.get("throttle_seconds", 7))
    if disk is None:
        disk = {}
    done = set(disk.get("cs_done", []))
    order = [tuple(x) for x in disk.get("cs_order", [])]
    seen = {fn.lower() for fn, _ in order}
    for q in cs["queries"]:
        if q in done:
            sys.stderr.write(f"discover: code-search '{q}': cached, skipping\n")
            continue
        pages_done = 0
        for page in range(1, page_cap + 1):
            items, total = code_search(q, page)
            pages_done += 1
            time.sleep(throttle)
            if page == 1:
                sys.stderr.write(f"discover: code-search '{q}': total_count={total}\n")
            for it in items:
                fn = it.get("repository", {}).get("full_name")
                if not fn:
                    continue
                key = fn.lower()
                if key not in seen:
                    seen.add(key)
                    order.append((fn, q))
            # Stop early once this query is exhausted (fewer than a full page).
            if len(items) < 100:
                break
        done.add(q)
        disk["cs_done"] = sorted(done)
        disk["cs_order"] = [list(x) for x in order]
        if save:
            save(disk)
        sys.stderr.write(f"discover: code-search '{q}': {pages_done} page(s), "
                         f"{len(order)} unique repos cumulative\n")
    return order


def fetch_repo_meta(full_name):
    """Repo metadata via the REST API, mapped to the same shape gh search repos
    returns (so passes()/to_record() reuse works unchanged). None on error."""
    r = subprocess.run(["gh", "api", f"repos/{full_name}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None
    try:
        d = json.loads(r.stdout)
    except json.JSONDecodeError:
        return None
    return {
        "fullName": d.get("full_name") or full_name,
        "url": d.get("html_url") or f"https://github.com/{full_name}",
        "defaultBranch": d.get("default_branch"),
        "stargazersCount": d.get("stargazers_count") or 0,
        "forksCount": d.get("forks_count") or 0,
        "pushedAt": d.get("pushed_at"),
        "isFork": bool(d.get("fork")),
        "isArchived": bool(d.get("archived")),
        "isDisabled": bool(d.get("disabled")),
        "language": d.get("language"),
        "description": d.get("description") or "",
    }


def already_decided(full_name):
    """True for a repo a person has already ruled on: a maintainer opt-out or a
    skip disposition. Discovery drops these on the way in, so the queue never
    re-offers a decided repo -- the opt-out binding is mechanical, not something
    the triage view alone remembers. `verify` dispositions pass through: they
    mean investigate, and `triage.py review` tags them rather than hiding them."""
    sys.path.insert(0, str(Path(__file__).parent))
    import moatlib
    d = moatlib.get_disposition(full_name)
    return bool(moatlib.optout_for(full_name)
                or (d and d.get("disposition") == "skip"))


def adopted_projects():
    """Lowercased repo-name set of MOAT-adopted projects (projects/<name>/).
    The code-search pass should not re-surface a repo already being ported.
    Match is on the trailing repo name since projects/<name> drops the owner."""
    proot = REPO_ROOT / "projects"
    if not proot.is_dir():
        return set()
    return {p.name.lower() for p in proot.iterdir() if p.is_dir()}


CODE_CACHE = DATA / ".code_search_cache.json"


def _load_cache():
    if CODE_CACHE.exists():
        try:
            return json.loads(CODE_CACHE.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _save_cache(cache):
    CODE_CACHE.write_text(json.dumps(cache))


def run_code_search_pass(cfg, out_path):
    """Surface candidates by CUDA SOURCE (the discovery blind spot), merge them
    into the existing candidates.json, re-sort, and report. Existing records
    are preserved; only repos not already in candidates.json and not already an
    adopted project are evaluated.

    A disk cache (data/.code_search_cache.json) memoizes the per-repo metadata
    and Cuda-bytes results so a long fetch loop interrupted by a wall-clock
    limit resumes cheaply -- re-running the pass skips already-fetched repos."""
    s = cfg["search"]
    excl_orgs = {o.lower() for o in s.get("exclude_orgs", [])}
    excl_substr = [x.lower() for x in s.get("exclude_org_substrings", [])]
    threshold = max(1, int(s.get("min_cuda_bytes", 0)))
    max_fetches = int(cfg["code_search"].get("max_new_repo_fetches", 250))

    existing = json.loads(Path(out_path).read_text()) if Path(out_path).exists() else []
    known = {r["full_name"].lower() for r in existing}
    adopted_names = adopted_projects()

    disk = _load_cache()
    all_done = set(disk.get("cs_done", [])) >= set(cfg["code_search"]["queries"])
    if all_done and disk.get("cs_order"):
        surfaced = [tuple(x) for x in disk["cs_order"]]
        sys.stderr.write(f"discover: reusing cached code-search surfacing "
                         f"({len(surfaced)} repos)\n")
    else:
        surfaced = code_search_repos(cfg, disk=disk, save=_save_cache)
    n_unique = len(surfaced)

    # Partition: skip repos already known (in candidates.json), already adopted,
    # or already ruled on (opt-out or skip disposition).
    new_repos = []
    n_known = n_adopted = n_decided = 0
    for fn, q in surfaced:
        low = fn.lower()
        if low in known:
            n_known += 1
            continue
        if low.split("/")[-1] in adopted_names:
            n_adopted += 1
            continue
        if already_decided(fn):
            n_decided += 1
            continue
        new_repos.append((fn, q))

    truncated = 0
    if len(new_repos) > max_fetches:
        truncated = len(new_repos) - max_fetches
        sys.stderr.write(
            f"discover: code-search TRUNCATED -- {len(new_repos)} new repos exceed "
            f"max_new_repo_fetches={max_fetches}; deferring {truncated} (NOT silent)\n")
        new_repos = new_repos[:max_fetches]

    meta_cache = disk.setdefault("meta", {})   # full_name -> meta dict or None
    cb_cache = disk.setdefault("cuda_bytes", {})  # full_name -> int
    new_recs = []
    n_fork = n_org = n_filter = n_nocuda = n_meta = 0
    for i, (fn, q) in enumerate(new_repos):
        if fn in meta_cache:
            meta = meta_cache[fn]
        else:
            meta = fetch_repo_meta(fn)
            meta_cache[fn] = meta
            # ~1 req/s: GitHub's secondary (abuse) rate limit trips on bursty
            # request rates well before the primary 5000/hr core budget is
            # spent. 0.1s here triggered 403s, especially when porter agents
            # hit the API concurrently. The disk cache means this throttle is
            # only paid once per repo.
            time.sleep(1.0)
        if (i + 1) % 100 == 0:
            _save_cache(disk)
            sys.stderr.write(f"discover: code-search progress {i + 1}/{len(new_repos)}\n")
        if meta is None:
            n_meta += 1
            continue
        owner = meta["fullName"].split("/")[0].lower()
        if owner in excl_orgs or any(sub in owner for sub in excl_substr):
            n_org += 1
            continue
        if meta.get("isFork"):
            n_fork += 1
            continue
        if not passes(meta, cfg):
            n_filter += 1
            continue
        cb = cuda_bytes(meta["fullName"], cb_cache)
        meta["_cuda_bytes"] = cb
        if cb < threshold:
            n_nocuda += 1
            continue
        meta["matched_queries"] = [f"code:{q}"]
        new_recs.append(to_record(meta, cfg))

    _save_cache(disk)
    merged = existing + new_recs
    merged.sort(key=lambda r: (-r["priority"], r["full_name"].lower()))
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
        f.write("\n")

    # Durable side-channel: the new records alone, so a concurrent rebase that
    # clobbers candidates.json on the shared working tree cannot lose this run's
    # findings -- they can be re-merged from here. agent_space/ is gitignored.
    snap = REPO_ROOT / "agent_space" / "new_records_postrun.json"
    if snap.parent.is_dir():
        snap.write_text(json.dumps(new_recs, indent=2) + "\n")

    stats = {
        "queries": len(cfg["code_search"]["queries"]),
        "unique_repos": n_unique,
        "already_known": n_known,
        "already_adopted": n_adopted,
        "already_decided": n_decided,
        "new_evaluated": len(new_repos),
        "truncated": truncated,
        "meta_errors": n_meta,
        "dropped_fork": n_fork,
        "dropped_excluded_org": n_org,
        "dropped_filters": n_filter,
        "dropped_no_cuda": n_nocuda,
        "kept_new": len(new_recs),
        "total_candidates": len(merged),
    }
    sys.stderr.write("discover: code-search summary " + json.dumps(stats) + "\n")
    return new_recs, stats


def main(argv=None):
    ap = argparse.ArgumentParser(prog="discover")
    ap.add_argument("--limit", type=int, help="per-query result cap (<=1000)")
    ap.add_argument("--out", default=str(DATA / "candidates.json"))
    ap.add_argument("--allow-empty", action="store_true",
                    help="write an empty candidates.json instead of refusing")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the languages-API check that a repo really has CUDA code")
    ap.add_argument("--code-search", action="store_true",
                    help="run the CUDA-source code-search pass and merge new repos "
                         "into --out (the metadata-search blind-spot closer); "
                         "does not re-run the metadata pass")
    args = ap.parse_args(argv)
    cfg = load_cfg()
    if args.code_search:
        run_code_search_pass(cfg, args.out)
        return 0
    s = cfg["search"]
    limit = args.limit or s["per_query_limit"]
    raw_hits = gather(cfg, limit)

    excl_orgs = {o.lower() for o in s.get("exclude_orgs", [])}
    excl_substr = [x.lower() for x in s.get("exclude_org_substrings", [])]
    threshold = max(1, int(s.get("min_cuda_bytes", 0)))
    cache = {}
    recs = []
    n_filter = n_org = n_nocuda = n_decided = 0
    for h in raw_hits:
        if not passes(h, cfg):
            n_filter += 1
            continue
        owner = h["fullName"].split("/")[0].lower()
        if owner in excl_orgs or any(sub in owner for sub in excl_substr):
            n_org += 1
            continue
        if already_decided(h["fullName"]):
            n_decided += 1
            continue
        if not args.no_verify and h.get("language") != "Cuda":
            cb = cuda_bytes(h["fullName"], cache)
            h["_cuda_bytes"] = cb
            if cb < threshold:
                n_nocuda += 1
                continue
        recs.append(to_record(h, cfg))
    # The queue ACCUMULATES. A sweep is a way of finding candidates, not a
    # statement about which ones still exist, so anything already known is carried
    # forward whether or not this run re-found it. Repos age out of the search
    # window the moment they stop being pushed -- `pushed:>=` is a recency filter,
    # not a relevance one -- and a code-search finding cannot be reproduced by a
    # metadata query at all. Replacing the file instead of merging silently drops
    # both classes, which is how one routine run discarded 82 candidates.
    #
    # A re-found repo takes the fresh record: stars, push date and priority should
    # track reality. `first_seen` survives so the queue can be read chronologically,
    # and `last_seen` makes a stale entry visible instead of merely absent.
    out_path = Path(args.out)
    existing = json.loads(out_path.read_text()) if out_path.exists() else []
    today = _dt.date.today().isoformat()
    prior = {r["full_name"].lower(): r for r in existing}
    for r in recs:
        was = prior.get(r["full_name"].lower())
        r["first_seen"] = (was or {}).get("first_seen") or today
        r["last_seen"] = today
    fresh = {r["full_name"].lower() for r in recs}
    preserved = [r for r in existing if r["full_name"].lower() not in fresh]
    recs = recs + preserved
    recs.sort(key=lambda r: (-r["priority"], r["full_name"].lower()))
    if not recs and out_path.exists() and not args.allow_empty:
        sys.stderr.write(
            "discover: 0 candidates from this sweep -- REFUSING to overwrite "
            f"{out_path}. Every query returning nothing means the search failed "
            "(a bad field, a rate limit, an outage), not that CUDA left GitHub. "
            "Re-run; pass --allow-empty if you truly mean to empty the queue.\n")
        return 1
    with open(out_path, "w") as f:
        json.dump(recs, f, indent=2)
        f.write("\n")
    sys.stderr.write(
        f"discover: {len(raw_hits)} unique hits -> {len(recs)} candidates "
        f"({len(preserved)} code-search records preserved; dropped {n_filter} filters, "
        f"{n_org} excluded-org, {n_decided} already-decided, "
        f"{n_nocuda} no-CUDA-code) -> {out_path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
