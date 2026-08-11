#!/usr/bin/env python3
"""Refuse `gh` invocations that would adopt a project or write upstream.

The autonomy boundary says an agent may not take a GitHub-visible action against an
upstream repo without a person's yes. Until now that rule lived only in prose, and
prose is the weakest thing to put in front of a behaviour that has already failed:
three reviewers reached for `gh pr create` in a single session, each in a different
way (see moatlib.set_review_pr, which closed the route it could close). This closes
the route prose cannot: `gh` reached directly from a shell.

The usual question is deliberately NOT "is this a write?" but "does this write touch a
repo outside AMD-Ecosystem?" A write to our own fork is ordinary porting work and must
stay friction-free; a write to somebody else's repo is the thing that needs a person.
Project adoption is the exception: `gh repo fork` is always refused because creating
the organization fork is itself the person's adoption decision. Unresolvable targets
count as foreign, so a command whose repo cannot be determined is refused rather than
waved through.

There is no exemption, and that is deliberate. MOAT's one automated route upstream --
`upstream.py --publish`, which re-checks the approval and every required gate before it
opens anything -- calls real_gh() directly instead of asking this guard for permission.

The first version DID have an exemption, inferred from process ancestry: a foreign
write was allowed if any ancestor process had `upstream.py --publish` on its command
line. It failed the first time it was exercised. An ancestor is not only the caller, it
is every enclosing shell, and a shell's command line holds the whole script being run,
so a test script that merely MENTIONED the phrase on one line granted the exemption to
an unrelated command three lines later. Two comments reached a live upstream issue
before anything noticed. The lesson generalises past this file: an exemption must be
proved by the caller, never inferred from ambient context that unrelated text can
satisfy. Trusted code calls the binary; it does not ask the guard to recognise it.

Scope: the guard binds only when the working directory is inside a MOAT checkout.
Outside it, `gh` behaves normally -- this machine is used for other work, and a guard
that broke `gh pr comment` on an unrelated repo would be uninstalled within the day.

    python3 utils/gh_guard.py --explain -- pr comment --repo foo/bar --body hi
    python3 utils/gh_guard.py --self-test

What this does NOT do: contain an agent that means to get around it. The real gh binary,
`curl`, and the API token are all still right there. It raises the cost of the failure
that actually happens -- reaching for the obvious command -- from one plausible line to
a deliberate circumvention of a named guard. Containment worth the name is a credential
with no write scope outside the org, which is a change to how the token is issued rather
than anything this file can do.

Everything here is OS-neutral: MOAT validates on Windows as well as Linux, so no path
layout, shell, or exec model belonging to either may be assumed. The platform-specific
part is confined to the one-line launcher install_hooks.py writes.
"""

import argparse
import os
import pathlib
import re
import shutil
import subprocess
import sys

OWNED_ORG = "AMD-Ecosystem"
REPO = pathlib.Path(__file__).resolve().parents[1]

# Stamped into the launcher so real_gh can recognise and skip it. MOAT runs on Windows
# as well as Linux, so nothing here may assume a path layout, a shell, or an exec model
# belonging to either.
SHIM_MARKER = "moat-gh-guard v1"


def is_shim(path):
    try:
        return SHIM_MARKER in pathlib.Path(path).read_text(errors="ignore")
    except OSError:
        return False


def real_gh():
    """The gh executable, skipping our launcher wherever PATH happens to put it.

    shutil.which rather than a hardcoded path: /usr/bin/gh is a Linux answer, and on
    Windows the thing to find is gh.exe via PATHEXT. When the first hit IS the launcher,
    drop that one directory and look again -- searching for "the next gh" by name would
    re-find it in the same directory and recurse.
    """
    first = shutil.which("gh")
    if first is None or not is_shim(first):
        return first
    home = pathlib.Path(first).parent.resolve()
    rest = [d for d in os.environ.get("PATH", "").split(os.pathsep)
            if d and pathlib.Path(d).resolve() != home]
    return shutil.which("gh", path=os.pathsep.join(rest))

# Subcommands that touch no repository at all.
LOCAL_NOUNS = {"auth", "config", "alias", "extension", "version", "help",
               "completion", "status", "search", "browse"}

# Everything not named here is treated as a write. A new `gh pr <verb>` should get the
# target check by default rather than a free pass; the cost of being wrong that way is
# a refusal message, and the cost of being wrong the other way is a post on somebody
# else's repo.
READ_VERBS = {"view", "list", "diff", "checks", "status", "download", "clone",
              "ls", "get", "describe", "cat", "shared-with-me"}

WRITE_METHODS = {"POST", "PATCH", "PUT", "DELETE"}

# `gh api` implicitly POSTs when any parameter flag is present; its own help uses
# posting an issue comment as the example. Reading -X alone would miss exactly that.
API_PARAM_FLAGS = {"-f", "--raw-field", "-F", "--field", "--input"}

SLUG_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
URL_RE = re.compile(r"https?://(?:www\.)?github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)")
API_PATH_RE = re.compile(r"(?:^|/)repos/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)")


class Decision:
    def __init__(self, allow, reason, targets=(), writes=False):
        self.allow = allow
        self.reason = reason
        self.targets = list(targets)
        self.writes = writes


def _strip_global_flags(argv):
    """Drop leading flags so argv[0] is the noun. `gh --repo x pr list` is legal."""
    out = list(argv)
    while out and out[0].startswith("-"):
        if out[0] in ("-R", "--repo") and len(out) > 1:
            out = out[2:]
        else:
            out = out[1:]
    return out


def explicit_repo(argv):
    for i, a in enumerate(argv):
        if a in ("-R", "--repo") and i + 1 < len(argv):
            return normalize(argv[i + 1])
        if a.startswith("--repo="):
            return normalize(a.split("=", 1)[1])
    return None


def normalize(ref):
    """A slug from anything gh accepts as a repo reference, or None."""
    if not ref:
        return None
    m = URL_RE.search(ref)
    if m:
        return m.group(1).removesuffix(".git")
    ref = ref.removesuffix(".git")
    if SLUG_RE.match(ref) and not ref.startswith("-"):
        return ref
    return None


def cwd_repo():
    """The slug of the clone we are standing in, from origin then upstream.

    A `gh pr comment 42` with no --repo acts on whichever clone is the working
    directory, and inside projects/<name>/src that is the fork. Resolving it wrong in
    the permissive direction would allow the exact command we are trying to stop.
    """
    for remote in ("origin", "upstream"):
        r = subprocess.run(["git", "remote", "get-url", remote],
                           capture_output=True, text=True)
        if r.returncode == 0:
            slug = normalize(r.stdout.strip())
            if slug:
                return slug
    return None


def api_targets(rest):
    """Repo slugs an `api` call addresses, and whether it is a graphql mutation."""
    targets, graphql_write = [], False
    for a in rest:
        if a.startswith("-"):
            continue
        if a == "graphql":
            continue
        for m in API_PATH_RE.finditer(a):
            targets.append(m.group(1))
    if any(a == "graphql" for a in rest):
        blob = " ".join(rest)
        if re.search(r"\bmutation\b", blob):
            graphql_write = True
        for m in re.finditer(r'owner:\s*\\?"([^"\\]+)\\?"', blob):
            targets.append(m.group(1) + "/?")
    return targets, graphql_write


def api_writes(rest):
    for i, a in enumerate(rest):
        if a in ("-X", "--method") and i + 1 < len(rest):
            return rest[i + 1].upper() in WRITE_METHODS
        if a.startswith("--method="):
            return a.split("=", 1)[1].upper() in WRITE_METHODS
    return any(a in API_PARAM_FLAGS or a.split("=", 1)[0] in API_PARAM_FLAGS
               for a in rest)


def owned(slug):
    return bool(slug) and slug.split("/", 1)[0].lower() == OWNED_ORG.lower()


def classify(argv):
    """Decide a single `gh` invocation from its arguments (gh itself excluded)."""
    global_target = explicit_repo(argv)
    argv = _strip_global_flags(argv)
    if not argv:
        return Decision(True, "no subcommand")

    noun = argv[0]
    rest = argv[1:]

    if noun in LOCAL_NOUNS:
        return Decision(True, f"`{noun}` does not write to a repository")

    if noun == "api":
        targets, graphql_write = api_targets(rest)
        writes = api_writes(rest) or graphql_write
        if not writes:
            return Decision(True, "read-only api call", targets)
        if not targets:
            return Decision(False, "an api write whose target repo could not be "
                                   "determined from the endpoint", targets, True)
        foreign = [t for t in targets if not owned(t)]
        if foreign:
            return Decision(False, f"api write to {', '.join(foreign)}", targets, True)
        return Decision(True, "api write inside " + OWNED_ORG, targets, True)

    verb = rest[0] if rest and not rest[0].startswith("-") else ""
    if verb in READ_VERBS:
        return Decision(True, f"`{noun} {verb}` is read-only")
    if noun == "repo" and verb == "fork":
        return Decision(False, "creating a fork is a human adoption decision", [], True)

    target = explicit_repo(argv) or global_target
    if not target:
        for a in rest[1:]:
            target = normalize(a)
            if target:
                break
    if not target:
        target = cwd_repo()

    label = f"`{noun} {verb}`".strip()
    if not target:
        return Decision(False, f"{label} writes, and no target repo could be resolved",
                        [], True)
    if owned(target):
        return Decision(True, f"{label} targets {target}", [target], True)
    return Decision(False, f"{label} targets {target}", [target], True)


REFUSAL = """moat: refused `gh {argv}`

{reason}.

The autonomy boundary reserves project adoption and every GitHub-visible action
against a repo we do not own for a person. That covers creating the organization fork,
PR and issue comments, reviews, edited bodies, and opening anything upstream.

The one automated route upstream is the approved one:
    python3 utils/upstream.py --publish --apply
It re-checks the recorded approval and every required gate, then opens the PR with the
approved title and body verbatim.

If a person has said yes and wants it done by hand, they can run {real} directly.
"""


def under_repo(path=None):
    """Is the working directory inside this MOAT checkout?

    The guard binds here and nowhere else. These machines are used for other work, and
    a guard that broke `gh pr comment` on an unrelated repo would be uninstalled within
    the day. Path.is_relative_to rather than string prefixes, so a sibling directory
    named moat-scratch is not mistaken for the checkout.
    """
    try:
        cwd = pathlib.Path(path or os.getcwd()).resolve()
    except OSError:
        return False
    return cwd == REPO or cwd.is_relative_to(REPO)


def delegate(argv):
    """Run the real gh and return its exit code.

    subprocess rather than os.exec: exec semantics differ on Windows, where the parent
    returns immediately and the caller sees a success it never waited for.
    """
    real = real_gh()
    if real is None:
        sys.stderr.write("moat: gh is not installed\n")
        return 127
    return subprocess.run([real] + list(argv)).returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--explain", action="store_true",
                    help="print the decision instead of running gh")
    ap.add_argument("--shim", action="store_true",
                    help="called as the PATH launcher: guard, then run the real gh")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("args", nargs=argparse.REMAINDER)
    ns = ap.parse_args()

    if ns.self_test:
        return self_test()

    argv = ns.args[1:] if ns.args[:1] == ["--"] else ns.args

    if ns.shim and not under_repo():
        return delegate(argv)

    d = classify(argv)

    if ns.explain:
        print(("ALLOW" if d.allow else "REFUSE") + ": " + d.reason)
        return 0 if d.allow else 1

    if not d.allow:
        sys.stderr.write(REFUSAL.format(argv=" ".join(argv), reason=d.reason,
                                        real=real_gh() or "gh"))
        return 13

    return delegate(argv)


CASES = [
    # (argv, expect_allow)
    (["pr", "view", "123", "--repo", "torvalds/linux"], True),
    (["pr", "diff", "--repo", "torvalds/linux"], True),
    (["search", "repos", "cuda"], True),
    (["api", "repos/torvalds/linux/languages"], True),
    (["api", "--paginate", "repos/torvalds/linux/pulls/1/commits"], True),
    (["api", "repos/torvalds/linux/issues/1/comments", "-f", "body=hi"], False),
    (["api", "-X", "POST", "repos/torvalds/linux/issues/1/comments"], False),
    (["api", "--method=DELETE", "repos/torvalds/linux/issues/1"], False),
    (["api", "-X", "PUT", f"repos/{OWNED_ORG}/fork/actions/permissions",
      "-F", "enabled=false"], True),
    (["api", "graphql", "-f", "query=mutation { addComment }"], False),
    (["pr", "comment", "--repo", "torvalds/linux", "--body", "hi"], False),
    (["pr", "create", "--repo", "torvalds/linux"], False),
    (["pr", "review", "https://github.com/torvalds/linux/pull/9", "--approve"], False),
    (["pr", "edit", "42", "--repo", "torvalds/linux", "--body", "x"], False),
    (["issue", "comment", "--repo", "torvalds/linux", "--body", "hi"], False),
    (["pr", "create", "--repo", f"{OWNED_ORG}/moat"], True),
    (["pr", "comment", "--repo", f"{OWNED_ORG}/somefork", "--body", "hi"], True),
    (["issue", "create", "--repo", f"{OWNED_ORG}/moat", "--title", "t"], True),
    (["repo", "delete", f"{OWNED_ORG}/somefork"], True),
    (["repo", "fork", "torvalds/linux", "--org", OWNED_ORG], False),
    (["repo", "fork", f"{OWNED_ORG}/somefork", "--org", OWNED_ORG], False),
    (["pr", "frobnicate", "--repo", f"{OWNED_ORG}/moat"], True),
    (["pr", "frobnicate", "--repo", "torvalds/linux"], False),
    (["--repo", "torvalds/linux", "pr", "list"], True),
    (["--repo", "torvalds/linux", "pr", "comment", "9", "--body", "hi"], False),
    (["--repo", f"{OWNED_ORG}/moat", "issue", "create", "--title", "t"], True),
]


def self_test():
    bad = 0
    for argv, want in CASES:
        got = classify(argv).allow
        if got != want:
            bad += 1
            print(f"FAIL want={'allow' if want else 'refuse'} "
                  f"got={'allow' if got else 'refuse'}: gh {' '.join(argv)}")
    print(f"{len(CASES) - bad}/{len(CASES)} cases pass")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
