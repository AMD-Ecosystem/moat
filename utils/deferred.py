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

Items are stored in data/deferred.json. `list` prints the open ones and also
flags any findings/<dir> bug report that is not yet registered, so a prepared
report cannot silently fall off the radar.

    python3 utils/deferred.py list                 # open items + unregistered findings
    python3 utils/deferred.py list --all           # include filed/done
    python3 utils/deferred.py list --kind rocm-bug-report
    python3 utils/deferred.py add rocm-bug-report \
        --id hipcub-rocprim-beginbit --component rocPRIM \
        --summary "DeviceRadixSort drops keys with begin_bit>0 and end_bit==width" \
        --ref findings/hipcub-rocprim-beginbit/BUG_REPORT.md
    python3 utils/deferred.py set-status hipcub-rocprim-beginbit filed \
        --upstream https://github.com/ROCm/rocPRIM/issues/NNN
"""

import argparse
import json
import sys
from pathlib import Path

import moatlib

REGISTRY = moatlib.REPO_ROOT / "data" / "deferred.json"
FINDINGS = moatlib.REPO_ROOT / "findings"
KINDS = ("rocm-bug-report", "feature-port", "other")
STATUSES = ("open", "filed", "done")


def _load():
    if not REGISTRY.exists():
        return {"schema_version": 1, "items": []}
    return json.loads(REGISTRY.read_text())


def _save(obj):
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


def add(args):
    obj = _load()
    if _find(obj, args.id):
        sys.exit(f"deferred: id '{args.id}' already exists")
    item = {
        "id": args.id,
        "kind": args.kind,
        "project": args.project,
        "component": args.component,
        "summary": args.summary,
        "refs": args.ref or [],
        "status": "open",
        "upstream_issue": None,
        "created_at": moatlib.now_iso(),
    }
    obj["items"].append(item)
    _save(obj)
    _maybe_commit(args, f"deferred: register {args.id} ({args.kind})")
    print(f"registered {args.id}")


def set_status(args):
    obj = _load()
    it = _find(obj, args.id)
    if not it:
        sys.exit(f"deferred: unknown id '{args.id}'")
    it["status"] = args.status
    if args.upstream:
        it["upstream_issue"] = args.upstream
    _save(obj)
    _maybe_commit(args, f"deferred: {args.id} -> {args.status}")
    print(f"{args.id} -> {args.status}")


def list_items(args):
    obj = _load()
    items = obj["items"]
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
        comp = f" <{it['component']}>" if it.get("component") else ""
        up = f"  filed: {it['upstream_issue']}" if it.get("upstream_issue") else ""
        print(f"- ({it['status']}) {it['kind']}{proj}{comp} {it['id']}{up}")
        print(f"    {it['summary']}")
        for ref in it.get("refs", []):
            print(f"    ref: {ref}")

    # Safety net: a prepared bug report under findings/ with no registry entry.
    if FINDINGS.exists():
        registered = _registered_finding_dirs(obj)
        orphans = sorted(d.name for d in FINDINGS.iterdir()
                         if d.is_dir() and d.name not in registered)
        if orphans:
            print("\nfindings/ bug reports NOT in the registry "
                  "(run `deferred.py add` to track):")
            for name in orphans:
                print(f"  - findings/{name}/")


def _maybe_commit(args, message):
    if getattr(args, "commit", False):
        moatlib.commit_and_push([str(REGISTRY)], message)


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

    args = p.parse_args(argv)
    {"add": add, "set-status": set_status, "list": list_items}[args.cmd](args)


if __name__ == "__main__":
    main()
