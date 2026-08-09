#!/usr/bin/env python3
"""MOAT deferred-work registry: the answer to "what did we defer, and where do
we resume it?"

Two kinds of deferred work accumulate during a port and are easy to lose track
of because they live in prose:

  - rocm-bug-report: a bug isolated against a ROCm component (rocPRIM, hipCUB,
    hipSOLVER, the HIP runtime, ...), with a reproducer prepared under findings/,
    that has NOT yet been filed upstream against ROCm.
  - feature-port: a sub-feature of a project deliberately scoped out of the port
    (e.g. Open3D's NPP image filters, the SlabHash wave64 backend) that a later
    pass could pick up.
  - other: anything else we chose to revisit later.

A deferral belongs to the project that deferred it, so it is stored in that
project's own folder -- `projects/<name>/deferred.json` -- on that project's
branch, alongside the notes and the plan it came out of. It is discovered while
porting, it is evidence about that port, and it gets reviewed with that port
rather than broadcast on its own.

`data/deferred.json` remains for the items that are genuinely NOT project-scoped:
a bug isolated against a ROCm component with no port attached, and the record of
work deferred by a project that has since been removed.

`list` reads both, and reads across refs, so a deferral on a port branch nobody
has checked out still answers "what did we defer". It also flags any
findings/<dir> bug report that is not yet registered, so a prepared report cannot
silently fall off the radar, and `pending` lists the ones no person has ruled on.

    python3 utils/deferred.py list                 # open items + unregistered findings
    python3 utils/deferred.py list --all           # include filed/done
    python3 utils/deferred.py list --kind rocm-bug-report
    python3 utils/deferred.py add rocm-bug-report \
        --id hipcub-rocprim-beginbit --component rocPRIM \
        --summary "DeviceRadixSort drops keys with begin_bit>0 and end_bit==width" \
        --ref findings/hipcub-rocprim-beginbit/BUG_REPORT.md
    python3 utils/deferred.py set-status hipcub-rocprim-beginbit filed \
        --upstream https://github.com/ROCm/rocPRIM/issues/NNN
    python3 utils/deferred.py pending                 # awaiting a person's ruling
    python3 utils/deferred.py decide <id> --choice now --by <who>
"""

import argparse
import functools
import json
import subprocess
import sys

import moatlib

REGISTRY = moatlib.REPO_ROOT / "data" / "deferred.json"
FINDINGS = moatlib.REPO_ROOT / "findings"
KINDS = ("rocm-bug-report", "feature-port", "other")
STATUSES = ("open", "filed", "done")


def _empty():
    return {"schema_version": 1, "items": []}


def _load_global():
    if not REGISTRY.exists():
        return _empty()
    return json.loads(REGISTRY.read_text())


def _project_path(name):
    return moatlib.PROJECTS / name / "deferred.json"


@functools.lru_cache(maxsize=1)
def _branches_by_lowercase():
    """{lowercased name: actual branch name}, read once per run.

    `port_branches` shells out, and every resolution asks for it, so leaving it
    uncached put a subprocess per project on a path orient runs every time."""
    return {b.lower(): b for b in moatlib.port_branches()}


def _port_ref(name):
    """`origin/port/<name>`, tolerating a branch created under a different case.

    project_record carries the same fallback because the mismatch happened: a branch
    cut as `port/hami-core` for the project `HAMi-core` resolved to nothing, and a
    finished screen went invisible to every sweep."""
    return f"origin/port/{_branches_by_lowercase().get(name.lower(), name)}"


def _resolve_project(name):
    """(obj, where, ref) for a project's deferrals, resolved as project_record does.

    The project's OWN branch wins over the working tree, and that order is not
    cosmetic: an in-flight project is worked on its branch while the trunk may still
    carry a copy that predates it, and reading the trunk first is what made colmap's
    intake record invisible on the 2026-08-07 rerun. Being ON that branch is not a
    special case -- the working tree IS the branch then, so the local read is both
    correct and cheaper."""
    path = f"projects/{name}/deferred.json"
    ref = _port_ref(name)
    p = _project_path(name)
    if moatlib.current_branch() == f"port/{name}":
        return ((json.loads(p.read_text()) if p.exists() else _empty()), "local", ref)
    raw = moatlib._ref_read(ref, path)
    if raw:
        return (json.loads(raw), "branch", ref)
    if p.exists():
        return (json.loads(p.read_text()), "local", ref)
    raw = moatlib._ref_read("origin/main", path)
    if raw:
        return (json.loads(raw), "trunk", ref)
    return (_empty(), "local" if p.parent.exists() else "trunk", ref)


def _load_project(name):
    return _resolve_project(name)[0]


def _load_all():
    """Every deferral, project-scoped and global, each tagged with its owner."""
    items = []
    for it in _load_global()["items"]:
        items.append({**it, "_where": "global"})
    for name in sorted(moatlib.all_projects()):
        for it in _load_project(name)["items"]:
            items.append({**it, "project": name, "_where": name})
    return items


def _save_project(name, obj, message):
    """Write a project's deferrals to its own folder, wherever that folder lives.

    A ruling is a RECORD, not a working file. Whoever makes it already has the entry
    from `pending`, and nothing here opens plan.md or notes.md -- so this is the
    `release-forks` case rather than the selector's, and a project whose folder is on
    its own branch gets written there through `commit_to_branch` instead of refused.
    Refusing bought nothing: the ruling still had to reach that branch, and the
    person was sent to check out one branch per deferral.

    Presence is not ownership. A `port/<name>` branch carries every folder the trunk
    had when it was cut, so `projects/<other>/` existing in this tree says nothing
    about whether this checkout may write it. Gating on the path instead of on
    `writable_here` wrote one project's ruling into another project's branch, exit 0
    and silent -- the same canary bug that reverted the first folder migration.

    Returns the sha when the write went to a branch, else None."""
    _, where, ref = _resolve_project(name)
    if moatlib.writable_here(name, where):
        p = _project_path(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, indent=2) + "\n")
        return None
    if where == "branch":
        branch = ref.replace("origin/", "", 1)
        try:
            return moatlib.commit_to_branch(
                branch,
                {f"projects/{name}/deferred.json": json.dumps(obj, indent=2) + "\n"},
                message)
        except subprocess.CalledProcessError as e:
            # commit_to_branch parents on origin/<branch> as it reads it and pushes
            # unforced, so a branch that moved under us is REJECTED rather than
            # clobbered. That is the behaviour we want; it just needs to say so,
            # because the raw git failure reads like a broken tool.
            raise SystemExit(
                f"deferred: could not write {name}'s record to {branch} -- the branch "
                f"moved since this checkout last fetched it, so the push was refused "
                f"rather than overwriting somebody else's commit. `git fetch origin` "
                f"and run this again.\n{(e.stderr or '').strip()}")
    raise SystemExit(
        f"deferred: {name}'s record lives on the trunk, and this checkout is on "
        f"{moatlib.current_branch()}, which may not write it. `main` is protected, so a "
        f"trunk project's ruling reaches it by pull request: check out `main` (or a "
        f"branch off it) and record it there.")


def _save_global(obj):
    REGISTRY.write_text(json.dumps(obj, indent=2) + "\n")


def _find(obj, item_id):
    for it in obj["items"]:
        if it["id"] == item_id:
            return it
    return None


def _registered_finding_dirs(obj):
    dirs = set()
    for it in obj["items"]:
        for ref in it.get("refs", []):
            if ref.startswith("findings/"):
                dirs.add(ref.split("/", 2)[1])
    return dirs


def _owner(item_id):
    """(where, obj) for the store holding this id. `where` is a project name, or
    "global"."""
    g = _load_global()
    if _find(g, item_id):
        return ("global", g)
    for name in sorted(moatlib.all_projects()):
        obj = _load_project(name)
        if _find(obj, item_id):
            return (name, obj)
    return (None, None)


def _store(where, obj, message):
    """Write a store and say where it landed, so the caller can report it.

    Returns the sha when the write went to a branch this checkout does not hold,
    else None -- a branch write is a commit by construction, since there is no
    working tree to leave dirty."""
    if where == "global":
        _save_global(obj)
        return None
    return _save_project(where, obj, message)


def _report(where, sha, what):
    print(f"{what} on {where}" + (f" ({sha[:9]}, pushed to port/{where})" if sha else ""))


def add(args):
    where = args.project or "global"
    obj = _load_global() if where == "global" else _load_project(where)
    if _owner(args.id)[0]:
        sys.exit(f"deferred: id '{args.id}' already exists")
    item = {
        "id": args.id,
        "kind": args.kind,
        "component": args.component,
        "summary": args.summary,
        "refs": args.ref or [],
        "status": "open",
        "upstream_issue": None,
        "created_at": moatlib.now_iso(),
        # No ruling yet. The point of recording a deferral is that a person decides
        # whether it stays deferred or joins the port now, and an unrecorded decision
        # is how "we will get to it" becomes "nobody ever looked".
        "decided": None,
    }
    if where == "global":
        item["project"] = None
    obj["items"].append(item)
    msg = f"deferred: register {args.id} ({args.kind})"
    sha = _store(where, obj, msg)
    if not sha:
        _maybe_commit(args, where, msg)
    _report(where, sha, f"registered {args.id}")


def set_status(args):
    where, obj = _owner(args.id)
    if not where:
        sys.exit(f"deferred: unknown id '{args.id}'")
    it = _find(obj, args.id)
    it["status"] = args.status
    if args.upstream:
        it["upstream_issue"] = args.upstream
    msg = f"deferred: {args.id} -> {args.status}"
    sha = _store(where, obj, msg)
    if not sha:
        _maybe_commit(args, where, msg)
    _report(where, sha, f"{args.id} -> {args.status}")


def decide(args):
    """Record a person's ruling: does this stay deferred, or join the port now?

    `--by` is required and never defaulted, the same rule as a licence clearance, a
    gate waiver and a not-portable verdict. An agent may surface a deferral and may
    not rule on it -- deciding that a scoped-out feature stays scoped out is a
    judgement about what MOAT is delivering."""
    if not (args.by or "").strip():
        sys.exit("deferred: --by is required; ruling on a deferral is a person's call")
    where, obj = _owner(args.id)
    if not where:
        sys.exit(f"deferred: unknown id '{args.id}'")
    it = _find(obj, args.id)
    it["decided"] = {"choice": args.choice, "by": args.by, "at": moatlib.now_iso(),
                     **({"note": args.note} if args.note else {})}
    msg = f"deferred: {args.id} -> {args.choice} ({args.by})"
    sha = _store(where, obj, msg)
    if not sha:
        _maybe_commit(args, where, msg)
    _report(where, sha, f"{args.id}: {args.choice}, by {args.by}")


def pending(args):
    """Deferrals nobody has ruled on. This is the list a person is owed: every one
    of them is work that was set aside without anyone deciding it should be."""
    rows = [it for it in _load_all()
            if it.get("status") == "open" and not (it.get("decided") or {}).get("choice")]
    for it in sorted(rows, key=lambda i: (i["_where"], i["id"])):
        print(f"{it['_where']}\t{it['kind']}\t{it['id']}\t{it['summary'][:110]}")
    if rows:
        print(f"-- {len(rows)} deferral(s) with no ruling. Each is work set aside "
              f"without anyone deciding it should be: "
              f"`deferred.py decide <id> --choice defer|now --by <who>`",
              file=sys.stderr)


def list_items(args):
    items = _load_all()
    if not args.all:
        items = [it for it in items if it["status"] == "open"]
    if args.kind:
        items = [it for it in items if it["kind"] == args.kind]
    if args.project:
        items = [it for it in items if it.get("project") == args.project]

    if not items:
        print("(no matching deferred items)")
    for it in sorted(items, key=lambda i: (i["kind"], i["id"])):
        proj = f" [{it['project']}]" if it.get("project") else ""
        rule = ((it.get("decided") or {}).get("choice"))
        rule = f"  ruling: {rule} ({it['decided']['by']})" if rule else "  UNRULED"
        comp = f" <{it['component']}>" if it.get("component") else ""
        up = f"  filed: {it['upstream_issue']}" if it.get("upstream_issue") else ""
        print(f"- ({it['status']}) {it['kind']}{proj}{comp} {it['id']}{up}{rule}")
        print(f"    {it['summary']}")
        for ref in it.get("refs", []):
            print(f"    ref: {ref}")

    # Safety net: a prepared bug report under findings/ with no registry entry.
    if FINDINGS.exists():
        registered = _registered_finding_dirs({"items": _load_all()})
        orphans = sorted(d.name for d in FINDINGS.iterdir()
                         if d.is_dir() and d.name not in registered)
        if orphans:
            print("\nfindings/ bug reports NOT in the registry "
                  "(run `deferred.py add` to track):")
            for name in orphans:
                print(f"  - findings/{name}/")


def _maybe_commit(args, where, message):
    if getattr(args, "commit", False):
        path = str(REGISTRY) if where == "global" else str(_project_path(where))
        moatlib.commit_and_push([path], message)


def main(argv=None):
    p = argparse.ArgumentParser(description="MOAT deferred-work registry")
    p.add_argument("--commit", action="store_true",
                   help="commit+push the registry change to the MOAT repo")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="register a deferred item")
    a.add_argument("kind", choices=KINDS)
    a.add_argument("--id", required=True, help="unique kebab-case slug")
    a.add_argument("--project", help="related MOAT project (omit for global)")
    a.add_argument("--component", help="ROCm component, for rocm-bug-report")
    a.add_argument("--summary", required=True)
    a.add_argument("--ref", action="append", help="findings/ path, notes ref, URL (repeatable)")

    s = sub.add_parser("set-status", help="update an item's status")
    s.add_argument("id")
    s.add_argument("status", choices=STATUSES)
    s.add_argument("--upstream", help="upstream issue URL once filed")

    l = sub.add_parser("list", help="list deferred items (open by default)")
    l.add_argument("--all", action="store_true", help="include filed/done")
    l.add_argument("--kind", choices=KINDS)
    l.add_argument("--project")

    sub.add_parser("pending", help="deferrals no person has ruled on")

    d = sub.add_parser("decide", help="a person's ruling: stays deferred, or do it now")
    d.add_argument("id")
    d.add_argument("--choice", required=True, choices=("defer", "now"))
    d.add_argument("--by", required=True, help="who decided; never an agent")
    d.add_argument("--note")

    args = p.parse_args(argv)
    {"add": add, "set-status": set_status, "list": list_items,
     "pending": pending, "decide": decide}[args.cmd](args)


if __name__ == "__main__":
    main()
