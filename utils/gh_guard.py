#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
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

# Stamped into the launcher so real_gh can recognise and skip it. MOAT runs on Windows
# as well as Linux, so nothing here may assume a path layout, a shell, or an exec model
# belonging to either. v2 launchers run a host copy of the trunk's guard instead of a
# file inside one checkout; installers that predate v2 look for the v1 stamp, so they
# leave a v2 launcher alone instead of pointing it back at their own stale copy.
SHIM_MARKER = "moat-gh-guard v2"
SHIM_STAMP = "moat-gh-guard v"


def is_shim(path):
    try:
        return SHIM_STAMP in pathlib.Path(path).read_text(errors="ignore")
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

READ_METHODS = {"GET", "HEAD"}

# `clone` reads for a repository or a gist and writes for anything else: `gh label
# clone` copies labels INTO the target repository.
CLONE_READS = {"repo", "gist"}

# Flags whose value is free text. A URL inside a comment body names nothing the
# command acts on, so it must not be read as the command's target.
TEXT_FLAGS = {"-b", "--body", "-c", "--comment", "-t", "--title", "-n", "--notes",
              "-m", "--message", "--subject", "-d", "--description"}
# Flags naming a file whose contents become the body.
BODY_FILE_FLAGS = {"-F", "--body-file", "--notes-file", "--input"}

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


def api_method(rest):
    """The HTTP method an `api` call names, upper-cased, or None when it names none.
    gh accepts `-X POST`, `-XPOST`, `--method POST` and `--method=POST`."""
    for i, a in enumerate(rest):
        if a in ("-X", "--method"):
            return rest[i + 1].upper() if i + 1 < len(rest) else ""
        if a.startswith("--method="):
            return a.split("=", 1)[1].upper()
        if a.startswith("-X") and not a.startswith("--"):
            return a[2:].lstrip("=").upper()
    return None


def _is_param_flag(a):
    """`-f k=v`, `-fk=v`, `--field=k=v`, `--input file`: each makes gh POST."""
    if a in API_PARAM_FLAGS or a.split("=", 1)[0] in API_PARAM_FLAGS:
        return True
    return len(a) > 2 and a[:2] in ("-f", "-F") and not a.startswith("--")


def api_writes(rest):
    method = api_method(rest)
    if method is not None:
        # Anything that is not plainly a read is a write: an unrecognised or empty
        # method must not be waved through as one.
        return method not in READ_METHODS
    return any(_is_param_flag(a) for a in rest)


def owned(slug):
    return bool(slug) and slug.split("/", 1)[0].lower() == OWNED_ORG.lower()


def url_targets(rest):
    """Repos named by a positional GitHub URL. `gh pr comment <URL> --repo X` acts on
    the URL, so a foreign URL beside an owned --repo is still a foreign write."""
    out = []
    for i, a in enumerate(rest):
        if a.startswith("-") or (i and rest[i - 1] in TEXT_FLAGS | BODY_FILE_FLAGS):
            continue
        m = URL_RE.match(a)
        if m and not any(c.isspace() for c in a):
            out.append(m.group(1).removesuffix(".git"))
    return out


def _has_moat_command(text):
    """A stand-alone `/moat ...` line outside a code fence: the form the approval
    gate reads as a person's decision (moatlib._moat_command_lines)."""
    fenced = False
    for ln in (text or "").splitlines():
        s = ln.strip()
        if s.startswith("```"):
            fenced = not fenced
        elif not fenced and s.startswith("/moat"):
            return True
    return False


def carries_moat_command(argv):
    """Does this invocation post a `/moat` decision, inline or from a file?

    Review PRs are self-authored on the maintainer's credentials, so a comment an
    agent posts there is indistinguishable from the maintainer's own -- and
    `/moat approve` in one is what opens the upstream PR. Granting or repairing
    that decision is a person's, so no guarded `gh` call may carry it. A body read
    from stdin or an unreadable file cannot be inspected and is refused with it."""
    files = []
    for i, a in enumerate(argv):
        if _has_moat_command(a):
            return "a `/moat` decision line in its arguments"
        # -f body=..., -fbody=..., --field=body=...: the value after the first `=`.
        if "=" in a and _has_moat_command(a.split("=", 1)[1]):
            return "a `/moat` decision line in a field value"
        if a in BODY_FILE_FLAGS and i + 1 < len(argv):
            files.append(argv[i + 1])           # --body-file f, or api's -F k=@f
        elif "=" in a and a.split("=", 1)[0] in BODY_FILE_FLAGS:
            files.append(a.split("=", 1)[1])    # --body-file=f
        elif "=@" in a:
            files.append(a)                     # api's -Fk=@f
    for f in files:
        path = f.split("=@", 1)[1] if "=@" in f else f
        if path == "-":
            return "a body read from stdin, which cannot be checked for `/moat`"
        try:
            text = pathlib.Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue        # not a file: `-F k=v` is also api's plain field flag
        if _has_moat_command(text):
            return f"a `/moat` decision line in {path}"
    return None


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

    verb = rest[0] if rest and not rest[0].startswith("-") else ""
    reads = (verb in READ_VERBS and not (verb == "clone" and noun not in CLONE_READS))
    if noun == "api":
        targets, graphql_write = api_targets(rest)
        reads = not (api_writes(rest) or graphql_write)
    if not reads:
        moat = carries_moat_command(rest)
        if moat:
            return Decision(False, f"this call carries {moat}; granting or repairing "
                                   f"a `/moat` decision on a review PR is a person's",
                            [], True)

    if noun == "api":
        if reads:
            return Decision(True, "read-only api call", targets)
        if not targets:
            return Decision(False, "an api write whose target repo could not be "
                                   "determined from the endpoint", targets, True)
        foreign = [t for t in targets if not owned(t)]
        if foreign:
            return Decision(False, f"api write to {', '.join(foreign)}", targets, True)
        return Decision(True, "api write inside " + OWNED_ORG, targets, True)

    if reads:
        return Decision(True, f"`{noun} {verb}` is read-only")
    if noun == "repo" and verb == "fork":
        return Decision(False, "creating a fork is a human adoption decision", [], True)

    label = f"`{noun} {verb}`".strip()
    # A positional URL is what the command acts on, whatever --repo says.
    foreign = [t for t in url_targets(rest[1:]) if not owned(t)]
    if foreign:
        return Decision(False, f"{label} targets {', '.join(foreign)}", foreign, True)

    target = explicit_repo(argv) or global_target
    if not target:
        for a in rest[1:]:
            target = normalize(a)
            if target:
                break
    if not target:
        # gh's own order: --repo, then GH_REPO, then the clone it is standing in.
        target = normalize(os.environ.get("GH_REPO", "")) or cwd_repo()

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
    python3 utils/upstream.py --publish --apply --name <project>
It re-checks the recorded approval and every required gate, then opens the PR with the
approved title and body verbatim.

If a person has said yes and wants it done by hand, they can run {real} directly.
"""


def is_moat_checkout(d):
    """A MOAT checkout or worktree root: git metadata plus MOAT's own files."""
    return ((d / ".git").exists() and (d / "AGENTS.md").is_file()
            and (d / "utils" / "gh_guard.py").is_file())


def under_repo(path=None):
    """Is the working directory inside a MOAT checkout?

    The guard binds there and nowhere else. These machines are used for other work, and
    a guard that broke `gh pr comment` on an unrelated repo would be uninstalled within
    the day. Any MOAT checkout counts -- the main clone, each of its worktrees, every
    fleet job's clone: the launcher is one per host, and binding only to the checkout
    it was installed from left every other one unguarded.
    """
    try:
        cwd = pathlib.Path(path or os.getcwd()).resolve()
    except OSError:
        return False
    return any(is_moat_checkout(d) for d in (cwd, *cwd.parents))


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
    # Attached and unusual spellings of a write.
    (["api", "-XPOST", "repos/torvalds/linux/issues/1/comments"], False),
    (["api", "-XDELETE", "repos/torvalds/linux/issues/1"], False),
    (["api", "-X=PUT", "repos/torvalds/linux/pulls/1/merge"], False),
    (["api", "-X", "FROB", "repos/torvalds/linux/issues/1"], False),
    (["api", "repos/torvalds/linux/issues/1/comments", "-fbody=hi"], False),
    (["api", "repos/torvalds/linux/issues/1/comments", "--field=body=hi"], False),
    (["api", "-X", "GET", "repos/torvalds/linux/issues", "-f", "state=open"], True),
    (["api", "-XGET", "repos/torvalds/linux/issues", "-fstate=open"], True),
    (["label", "clone", f"{OWNED_ORG}/moat", "--repo", "torvalds/linux"], False),
    (["repo", "clone", "torvalds/linux"], True),
    # A positional URL is the target, whatever --repo says; a URL in a body is not.
    (["pr", "comment", "https://github.com/torvalds/linux/pull/9",
      "--repo", f"{OWNED_ORG}/moat", "--body", "hi"], False),
    (["pr", "close", f"https://github.com/{OWNED_ORG}/fork/pull/3", "--comment",
      "Submitted upstream as https://github.com/torvalds/linux/pull/9"], True),
    # A /moat decision is a person's, on any repository, in any spelling.
    (["pr", "comment", f"https://github.com/{OWNED_ORG}/fork/pull/3",
      "--body", "/moat approve"], False),
    (["pr", "review", "3", "--repo", f"{OWNED_ORG}/fork", "--comment",
      "--body", "looks right\n/moat approve\n"], False),
    (["api", f"repos/{OWNED_ORG}/fork/issues/3/comments",
      "-f", "body=/moat changes-requested"], False),
    (["api", f"repos/{OWNED_ORG}/fork/issues/3/comments", "-fbody=/moat approve"],
     False),
    (["pr", "comment", "3", "--repo", f"{OWNED_ORG}/fork", "--body-file", "-"], False),
    (["pr", "comment", "3", "--repo", f"{OWNED_ORG}/fork", "--body",
      "Approve with:\n```\n/moat approve\n```\n"], True),
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
