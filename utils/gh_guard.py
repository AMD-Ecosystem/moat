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

Scope: the full rule set binds when the working directory is inside a MOAT checkout.
Outside it, `gh` keeps working for this machine's other repositories -- a guard that
broke `gh pr comment` on an unrelated repo would be uninstalled within the day -- but
the launcher still judges every call, because three things do not depend on where the
shell is standing: a body that carries a `/moat` decision line, a review decision
(approve, request changes, dismiss, delete the comment that superseded one) on a
repository we own, and `gh repo fork --org` into the organization. Those are refused
from any directory. Before this, leaving the checkout was enough to post the approval
line unguarded, and that comment IS the approval.

The launcher is a PATH shim, so it guards nothing on a host where the real binary
sorts first, or for a command that names the binary by path. `--check-command` and
`--hook` judge the text of a shell command instead and are meant for a harness
PreToolUse hook, which sees the command before any PATH lookup happens.

    python3 utils/gh_guard.py --explain -- pr comment --repo foo/bar --body hi
    python3 utils/gh_guard.py --check-command 'gh pr comment 3 --repo foo/bar -b hi'
    python3 utils/gh_guard.py --hook < pretooluse-event.json
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
import json
import os
import pathlib
import re
import shlex
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

# Subcommands that touch no repository at all. `alias` and `extension` are NOT here:
# an alias or an extension can wrap any write, so each gets its own judgement below.
LOCAL_NOUNS = {"auth", "config", "version", "help", "completion", "status", "search",
               "browse"}

# Every noun gh ships. Inside a MOAT checkout anything else is an alias or an
# extension, and what it expands to cannot be seen from its arguments, so it is refused
# there instead of being judged as the command it is not. A noun gh adds later lands
# here by being named, the same default-deny READ_VERBS applies to verbs.
CORE_NOUNS = LOCAL_NOUNS | {
    "api", "pr", "issue", "repo", "release", "gist", "label", "run", "workflow",
    "cache", "secret", "variable", "ruleset", "project", "org", "codespace",
    "gpg-key", "ssh-key", "attestation", "alias", "extension", "preview",
    "agent-task", "co",
}

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

# Flags that take the next argument as their value, for walks that must not read a
# value as a flag of its own.
VALUE_FLAGS = TEXT_FLAGS | BODY_FILE_FLAGS | API_PARAM_FLAGS

SLUG_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
URL_RE = re.compile(r"https?://(?:www\.)?github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)")
API_PATH_RE = re.compile(r"(?:^|/)repos/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)")

# What explicit_repo returns for a --repo value that names no repository gh would
# accept. It is not "no flag": gh would fail on it, and falling through to the working
# directory would judge a different repository than the one the command names.
BAD_REPO = "(an unparseable --repo value)"


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


def repo_flag_values(argv):
    """Every value given to --repo/-R, in order, in each spelling gh accepts:
    `--repo x`, `-R x`, `--repo=x`, `-Rx` and `-R=x`. The value of a text or body flag
    is skipped, so a comment that reads `-R` is not a flag."""
    out, i = [], 0
    while i < len(argv):
        a = argv[i]
        if a in ("-R", "--repo"):
            if i + 1 < len(argv):
                out.append(argv[i + 1])
            i += 2
            continue
        if a.startswith("--repo="):
            out.append(a.split("=", 1)[1])
        elif a.startswith("-R") and not a.startswith("--"):
            out.append(a[2:].lstrip("="))
        elif a in VALUE_FLAGS:
            # `--comment` is a switch on `pr review` and takes a value on `pr close`;
            # a following flag settles which this is.
            if not (a in ("-c", "--comment") and i + 1 < len(argv)
                    and argv[i + 1].startswith("-")):
                i += 1
        i += 1
    return out


def explicit_repo(argv):
    """The repository --repo/-R names, None when the flag is absent. The LAST one
    wins, as it does in gh: taking the first let `--repo <ours> --repo <theirs>` be
    judged as a write to ours while gh sent it to theirs."""
    values = repo_flag_values(argv)
    if not values:
        return None
    return normalize(values[-1]) or BAD_REPO


def normalize(ref):
    """A slug from anything gh accepts as a repo reference, or None."""
    if not ref:
        return None
    m = URL_RE.search(ref)
    if m:
        return m.group(1).removesuffix(".git")
    ref = ref.removesuffix(".git")
    if ref.lower().startswith("github.com/"):       # gh's [HOST/]OWNER/REPO
        ref = ref.split("/", 1)[1]
    if SLUG_RE.match(ref) and not ref.startswith("-"):
        return ref
    return None


# gh's own order when no remote has been marked as the default: `upstream`, then
# `github`, then `origin`, then anything else.
_REMOTE_RANK = {"upstream": 0, "github": 1, "origin": 2}


def cwd_remotes(cwd=None):
    """(the slug gh would act on, every GitHub slug the clone's remotes name).

    A `gh pr comment 42` with no --repo acts on whichever clone is the working
    directory, and which of its remotes gh picks is gh's rule, not ours: a remote
    marked by `gh repo set-default` first, then `upstream` before `origin`. Reading
    `origin` first judged a fork clone that also has an `upstream` remote as a write
    to our fork while gh sent it upstream."""
    try:
        r = subprocess.run(["git", "config", "--get-regexp",
                            r"^remote\..*\.(url|gh-resolved)$"],
                           capture_output=True, text=True, cwd=cwd)
    except OSError:                  # the directory is not there, or git is not
        return None, []
    urls, resolved = {}, {}
    if r.returncode == 0:
        for line in r.stdout.splitlines():
            key, _, value = line.partition(" ")
            name, _, field = key[len("remote."):].rpartition(".")
            (urls if field == "url" else resolved)[name] = value.strip()
    slugs = {n: normalize(u) for n, u in urls.items()}
    every = sorted({s for s in slugs.values() if s})
    for name, value in resolved.items():
        return ((slugs.get(name) if value == "base" else normalize(value)) or BAD_REPO,
                every)
    for name in sorted(slugs, key=lambda n: (_REMOTE_RANK.get(n, 3), n)):
        if slugs[name]:
            return slugs[name], every
    return None, every


def cwd_repo(cwd=None):
    """The slug gh would act on in the clone we are standing in, or None."""
    return cwd_remotes(cwd)[0]


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


def _moat_in_payload(text):
    """Does this text carry a `/moat` line once GitHub has decoded it?

    A request body handed to `gh api --input` is JSON, and JSON spells a line break
    as the two characters `\\n`: scanned as raw text, `{"body": "ok\\n/moat approve"}`
    is one line that does not start with `/moat`, while GitHub stores a comment whose
    second line is the command. So every string inside a JSON value is scanned as
    the text it decodes to. Text that is not JSON gets the same escapes undone
    before a second look, which also covers a GraphQL string literal."""
    if _has_moat_command(text):
        return True
    try:
        todo = [json.loads(text)]
    except (ValueError, TypeError, RecursionError):
        todo = None
    if todo is None:
        loose = re.sub(r"\\r\\n|\\n|\\r", "\n", text or "")
        return loose != text and _has_moat_command(loose)
    while todo:
        x = todo.pop()
        if isinstance(x, str):
            if _has_moat_command(x):
                return True
        elif isinstance(x, dict):
            todo.extend(x.keys())
            todo.extend(x.values())
        elif isinstance(x, list):
            todo.extend(x)
    return False


def _read_body_file(path):
    return pathlib.Path(path).read_text(encoding="utf-8", errors="replace")


def body_files(noun, argv):
    """The files whose contents this invocation sends as a body or a field.

    `-F` means two things in gh. On `api` it is a typed field, `-F key=value`, and
    only a value spelled `@file` is read from disk (`--input file` is the whole
    request body). Everywhere else `-F file` is `--body-file`. Telling them apart is
    what lets an unreadable body file be refused instead of shrugged off as "probably
    a plain field"."""
    files = []
    for i, a in enumerate(argv):
        nxt = argv[i + 1] if i + 1 < len(argv) else None
        if noun == "api":
            val = None
            if a in ("-F", "--field"):
                val = nxt
            elif a.startswith("--field="):
                val = a.split("=", 1)[1]
            elif a.startswith("-F") and not a.startswith("--"):
                val = a[2:]
            if val and "=@" in val:
                files.append(val.split("=@", 1)[1])
            if a == "--input" and nxt is not None:
                files.append(nxt)
            elif a.startswith("--input="):
                files.append(a.split("=", 1)[1])
        elif a in BODY_FILE_FLAGS:
            if nxt is not None:
                files.append(nxt)
        elif a.startswith("--") and a.split("=", 1)[0] in BODY_FILE_FLAGS and "=" in a:
            files.append(a.split("=", 1)[1])
        elif a.startswith("-F") and not a.startswith("--"):
            files.append(a[2:].lstrip("="))
    return files


def reads_stdin(noun, argv):
    """Does this invocation take a body from standard input?"""
    return "-" in body_files(noun, argv)


def body_findings(noun, argv, stdin_text=None, read_file=None):
    """(moat, opaque): why this invocation carries a `/moat` decision, and why part of
    what it sends could not be inspected. Either may be None.

    Review PRs are self-authored on the maintainer's credentials, so a comment an
    agent posts there is indistinguishable from the maintainer's own -- and
    `/moat approve` in one is what opens the upstream PR. Granting or repairing that
    decision is a person's, so no guarded `gh` call may carry it, from any directory.
    A body that cannot be read (standard input nobody captured, a file that is not
    there) is reported separately: it may carry the line, and the caller refuses it
    wherever a `/moat` line could matter."""
    for a in argv:
        if _moat_in_payload(a):
            return ("a `/moat` decision line in its arguments", None)
        # -f body=..., -fbody=..., --field=body=...: the value after the first `=`.
        if "=" in a and _moat_in_payload(a.split("=", 1)[1]):
            return ("a `/moat` decision line in a field value", None)
    opaque = None
    for path in body_files(noun, argv):
        if path == "-":
            if stdin_text is None:
                opaque = "a body read from stdin, which cannot be checked for `/moat`"
                continue
            text, where = stdin_text, "standard input"
        else:
            try:
                text, where = (read_file or _read_body_file)(path), path
            except (OSError, ValueError):
                opaque = (f"a body file that could not be read ({path}), so it "
                          f"cannot be checked for `/moat`")
                continue
        if _moat_in_payload(text):
            return (f"a `/moat` decision line in {where}", None)
    return (None, opaque)


def carries_moat_command(argv, noun=None, stdin_text=None):
    """Why this invocation must not run on `/moat` grounds, or None: it carries a
    decision line, or it sends a body nobody could inspect for one."""
    moat, opaque = body_findings(noun, argv, stdin_text)
    return moat or opaque


# A review PR's decision is not only a `/moat` line. An APPROVED or CHANGES_REQUESTED
# review from an account with write access counts as one (moatlib._approving_review),
# dismissing a review withdraws one, and deleting a comment brings back the decision
# that comment superseded. GitHub refuses the first for a PR's author, but an agent
# running under a second account with write access is not the author.
_REVIEW_DECISION_FLAGS = {"-a", "--approve", "-r", "--request-changes"}
_DECISION_PATH_RE = re.compile(
    r"(?:^|/)repos/[^/]+/[^/]+/(?:pulls/\d+/reviews(?:/|$)"
    r"|(?:issues|pulls)/comments/\d+$)")


def decision_write(noun, verb, rest, method):
    """Why this write would grant, withdraw or repair a review decision, or None."""
    if noun == "pr" and verb == "review" and _REVIEW_DECISION_FLAGS & set(rest):
        return "`pr review` with --approve or --request-changes is a review decision"
    if noun in ("pr", "issue") and verb == "comment" and "--delete-last" in rest:
        return ("deleting a comment restores whichever decision that comment "
                "superseded")
    if noun == "api":
        path = next((a for a in rest if _DECISION_PATH_RE.search(a)), None)
        if path and "/reviews" in path:
            return "an api write to a pull request's reviews is a review decision"
        if path and method == "DELETE":
            return ("deleting a comment restores whichever decision that comment "
                    "superseded")
    return None


def _refuse_moat(reason):
    return Decision(False, f"this call carries {reason}; granting or repairing a "
                           f"`/moat` decision on a review PR is a person's", [], True)


def _classify_alias(rest, strict, stdin_text, read_file):
    verb = rest[0] if rest and not rest[0].startswith("-") else ""
    if verb in ("list", "delete", ""):
        return Decision(True, "listing or deleting an alias wraps nothing")
    # set / import: the expansion is a command somebody runs later under a name the
    # guard cannot read. One that carries a `/moat` line is refused everywhere.
    text = " ".join(rest[1:])
    opaque = False
    if verb == "import":
        for a in rest[1:]:
            if a.startswith("-") and a != "-":
                continue
            try:
                text += "\n" + (stdin_text if a == "-" and stdin_text is not None
                                else (read_file or _read_body_file)(a) if a != "-"
                                else "")
                opaque = opaque or (a == "-" and stdin_text is None)
            except (OSError, ValueError):
                opaque = True
    if "/moat" in text:
        return Decision(False, "an alias whose expansion carries a `/moat` command; "
                               "granting or repairing a `/moat` decision on a review "
                               "PR is a person's", [], True)
    if not strict:
        return Decision(True, f"`alias {verb}` outside a MOAT checkout")
    if verb == "set" and not opaque and "--shell" not in rest and "-s" not in rest:
        words = [a for a in rest[1:] if not a.startswith("-")]
        expansion = words[1] if len(words) > 1 else ""
        if expansion and not expansion.startswith("!"):
            try:
                inner = classify(shlex.split(expansion), strict=True)
            except ValueError:
                inner = None
            if inner is not None and inner.allow and not inner.writes:
                return Decision(True, "an alias for a read-only command")
    return Decision(False, f"`alias {verb}` could wrap a write the guard would never "
                           f"see; inside a MOAT checkout an alias may only stand "
                           f"for a read-only command", [], True)


def _classify_extension(rest, strict):
    verb = rest[0] if rest and not rest[0].startswith("-") else ""
    if verb in ("list", "search", "browse", "remove", ""):
        return Decision(True, "`extension` listing or removal runs nothing")
    if not strict:
        return Decision(True, f"`extension {verb}` outside a MOAT checkout")
    return Decision(False, f"`extension {verb}` runs or installs code that can write "
                           f"to any repository without the guard seeing it", [], True)


def classify(argv, strict=True, stdin_text=None, env_repo=None, cwd=None,
             read_file=None):
    """Decide a single `gh` invocation from its arguments (gh itself excluded).

    `strict` is the judgement inside a MOAT checkout: a write to a repository we do
    not own is refused. Outside one (`strict=False`) this machine's other work must
    keep working, so a write to somebody else's repository passes -- but three things
    are refused from anywhere, because the working directory is not what makes them
    dangerous: a body carrying a `/moat` line, a review decision on one of our own
    repositories, and creating the organization fork.

    `stdin_text` is what standard input holds when the caller captured it, `env_repo`
    a GH_REPO the command sets for itself, `cwd` the directory the command runs in
    (False when it cannot be known), `read_file` a replacement file reader."""
    global_target = explicit_repo(argv)
    argv = _strip_global_flags(argv)
    if not argv:
        return Decision(True, "no subcommand")

    noun = argv[0]
    rest = argv[1:]

    if noun in LOCAL_NOUNS:
        return Decision(True, f"`{noun}` does not write to a repository")
    if noun == "alias":
        return _classify_alias(rest, strict, stdin_text, read_file)
    if noun == "extension":
        return _classify_extension(rest, strict)

    verb = rest[0] if rest and not rest[0].startswith("-") else ""
    reads = (verb in READ_VERBS and not (verb == "clone" and noun not in CLONE_READS))
    targets, graphql_write = [], False
    if noun == "api":
        targets, graphql_write = api_targets(rest)
        reads = not (api_writes(rest) or graphql_write)
    opaque = None
    if not reads or noun not in CORE_NOUNS:
        moat, opaque = body_findings(noun, rest, stdin_text, read_file)
        if moat:
            return _refuse_moat(moat)
    if noun not in CORE_NOUNS:
        if strict:
            return Decision(False, f"`{noun}` is not a gh command the guard knows, so "
                                   f"it is an alias or an extension and what it runs "
                                   f"cannot be judged; run the command it stands for",
                            [], True)
        return Decision(True, f"`{noun}` outside a MOAT checkout")
    decision = None if reads else decision_write(noun, verb, rest, api_method(rest))

    def settle(label, where):
        """The verdict for a write whose targets are known."""
        foreign = [t for t in where if not owned(t)]
        ours = [t for t in where if owned(t)]
        if decision and (ours or strict):
            return Decision(False, f"{decision}, and granting or repairing one on a "
                                   f"review PR is a person's", where, True)
        if opaque and (ours or strict):
            return Decision(False, f"this call carries {opaque}", where, True)
        if foreign and strict:
            return Decision(False, f"{label} {', '.join(foreign)}", where, True)
        if foreign:
            return Decision(True, f"{label} {', '.join(foreign)}, outside a MOAT "
                                  f"checkout", where, True)
        return Decision(True, f"{label} {', '.join(where)}", where, True)

    def unresolved(why):
        """A write whose target nobody could name. Inside a checkout that is a
        refusal; outside one gh is left to act on it, unless it carries something
        that only matters on one of our own repositories."""
        if strict or opaque or decision:
            return Decision(False, why if not (opaque or decision) else
                            f"{why}, and this call carries {opaque or decision}",
                            [], True)
        return Decision(True, f"{why}; outside a MOAT checkout", [], True)

    if noun == "api":
        if reads:
            return Decision(True, "read-only api call", targets)
        if not targets:
            return unresolved("an api write whose target repo could not be "
                              "determined from the endpoint")
        return settle("api write to", targets)

    if reads:
        return Decision(True, f"`{noun} {verb}` is read-only")
    if noun == "repo" and verb == "fork":
        org = next((rest[i + 1] for i, a in enumerate(rest[:-1]) if a == "--org"),
                   next((a.split("=", 1)[1] for a in rest if a.startswith("--org=")),
                        ""))
        if strict or org.lower() == OWNED_ORG.lower():
            return Decision(False, "creating a fork is a human adoption decision",
                            [], True)

    label = f"`{noun} {verb}`".strip() + " targets"
    # A positional URL is what the command acts on, whatever --repo says.
    urls = url_targets(rest[1:])
    if any(not owned(t) for t in urls):
        return settle(label, urls)

    target = explicit_repo(argv) or global_target
    if not target:
        for a in rest[1:]:
            target = normalize(a)
            if target:
                break
    if not target:
        # gh's own order: --repo, then GH_REPO, then the clone it is standing in.
        target = normalize(env_repo if env_repo is not None
                           else os.environ.get("GH_REPO", ""))
    if not target and cwd is not False:
        target, remotes = cwd_remotes(cwd)
        mixed = [s for s in remotes if not owned(s)]
        if strict and target and owned(target) and mixed:
            return Decision(False, f"{label} no named repository, and this clone's "
                                   f"remotes include {', '.join(mixed)}; pass --repo "
                                   f"so the target is not left to gh's choice of "
                                   f"remote", [target] + mixed, True)

    if not target:
        return unresolved(f"{label[:-8]} writes, and no target repo could be resolved")
    return settle(label, [target])


REFUSAL = """moat: refused `gh {argv}`

{reason}.

The autonomy boundary reserves project adoption and every GitHub-visible action
against a repo we do not own for a person. That covers creating the organization fork,
PR and issue comments, reviews, edited bodies, and opening anything upstream. A
`/moat` decision on a review PR is a person's on any repository, from any directory.

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

    The full rule set binds there. Outside it the guard still runs, but only the
    refusals that do not depend on where the shell happens to be standing (see
    classify): these machines are used for other work, and a guard that broke
    `gh pr comment` on an unrelated repo would be uninstalled within the day. Any
    MOAT checkout counts -- the main clone, each of its worktrees, every fleet job's
    clone: the launcher is one per host, and binding only to the checkout it was
    installed from left every other one unguarded.
    """
    try:
        cwd = pathlib.Path(path or os.getcwd()).resolve()
    except OSError:
        return False
    return any(is_moat_checkout(d) for d in (cwd, *cwd.parents))


def delegate(argv, stdin_bytes=None):
    """Run the real gh and return its exit code.

    subprocess rather than os.exec: exec semantics differ on Windows, where the parent
    returns immediately and the caller sees a success it never waited for. Standard
    input the guard read in order to inspect it is handed on unchanged.
    """
    real = real_gh()
    if real is None:
        sys.stderr.write("moat: gh is not installed\n")
        return 127
    if stdin_bytes is not None:
        return subprocess.run([real] + list(argv), input=stdin_bytes).returncode
    return subprocess.run([real] + list(argv)).returncode


# ---- judging a shell command before it runs ---------------------------------
#
# The launcher only guards a `gh` that PATH resolves to it. On a host where the real
# binary sorts first, or for a command that names the binary by path, nothing above
# ever runs. A harness hook that sees the command text before the shell does has no
# such dependency, so `--check-command` (and `--hook`, which reads a PreToolUse
# event) finds the `gh` invocations in a command line and judges each one.
#
# It reads the text the way a shell would, far enough to tell an invocation from a
# mention: words inside quotes, a heredoc body, a commit message or an `echo` argument
# are data, not commands, and judging them made an earlier guard of this kind refuse
# ordinary prose. What it cannot evaluate -- a variable, a command substitution -- is
# marked rather than guessed at, and a write that depends on one is refused.

QDYN = "\ue000"      # stands for an expansion inside double quotes: one opaque word
UDYN = "\ue001"      # stands for an unquoted expansion: may be several words, or flags

_SHELLS = {"bash", "sh", "zsh", "dash", "ksh"}
_WRAPPERS = {"sudo", "command", "exec", "nohup", "time", "builtin", "then", "do",
             "else", "elif", "if", "while", "until", "!", "{", "}", "winpty",
             "start-process"}
_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_PS_HERE_RE = re.compile(r"@(['\"])\r?\n(.*?)\r?\n\1@", re.S)


class _Cmd:
    def __init__(self):
        self.words = []
        self.stdin = None            # heredoc or here-string text, when literal
        self.stdin_dynamic = False   # ... and whether the shell expands inside it
        self.piped = False           # standard input is another command's output
        self.redirects = []          # (operator, target word)


def _balanced(text, i, opener="(", closer=")"):
    """Index just past the `closer` matching an `opener` already consumed at i."""
    depth, n = 1, len(text)
    while i < n:
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == "'":
            j = text.find("'", i + 1)
            i = n if j < 0 else j + 1
            continue
        if c == opener:
            depth += 1
        elif c == closer:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return n


def _lex(text, nested):
    """Split shell text into simple commands. Command substitutions are appended to
    `nested` as text of their own and leave a marker in the word that held them."""
    cmds, cur = [], _Cmd()
    word, have = [], False
    pending = []                     # heredocs waiting for the end of this line
    redirect = None
    i, n = 0, len(text)

    def end_word():
        nonlocal word, have, redirect
        if have:
            w = "".join(word)
            if redirect is not None:
                cur.redirects.append((redirect, w))
                if redirect == "<<<":
                    cur.stdin = w.replace(QDYN, "").replace(UDYN, "")
                    cur.stdin_dynamic = (QDYN in w) or (UDYN in w)
                redirect = None
            else:
                cur.words.append(w)
        word, have = [], False

    def end_cmd(piped=False):
        nonlocal cur
        end_word()
        if cur.words or cur.redirects or cur.stdin is not None:
            cmds.append(cur)
        cur = _Cmd()
        cur.piped = piped

    def expansion(marker):
        """Consume a `$...` or backtick expansion at i; returns the new index."""
        nonlocal have
        have = True
        if text[i] == "`":
            j = text.find("`", i + 1)
            j = n if j < 0 else j
            nested.append(text[i + 1:j])
            word.append(marker)
            return j + 1
        if text.startswith("$(", i) and not text.startswith("$((", i):
            j = _balanced(text, i + 2)
            nested.append(text[i + 2:j - 1])
            word.append(marker)
            return j
        if text.startswith("${", i):
            word.append(marker)
            return _balanced(text, i + 2, "{", "}")
        m = re.match(r"\$(?:\(\(.*?\)\)|[A-Za-z_][A-Za-z0-9_:]*|[0-9@*#?$!-])",
                     text[i:], re.S)
        if m:
            word.append(marker)
            return i + m.end()
        word.append("$")
        return i + 1

    while i < n:
        c = text[i]
        if c == "\n":
            end_cmd()
            i += 1
            for strip_tabs, delim, quoted, owner in pending:
                lines = []
                while i <= n:
                    j = text.find("\n", i)
                    line = text[i:] if j < 0 else text[i:j]
                    i = n + 1 if j < 0 else j + 1
                    probe = line.rstrip("\r")
                    if (probe.lstrip("\t") if strip_tabs else probe) == delim:
                        break
                    lines.append(line)
                owner.stdin = "\n".join(lines)
                owner.stdin_dynamic = not quoted
            pending = []
            i = min(i, n)
            continue
        if c in " \t\r":
            end_word()
            i += 1
            continue
        if c == "\\":
            if text.startswith("\\\n", i) or text.startswith("\\\r\n", i):
                i += 2 if text[i + 1] == "\n" else 3
                continue
            # A Windows path separator in the middle of a word stays one: the
            # PowerShell tool spells the real binary `C:\...\gh.exe`.
            if have and i + 1 < n and (text[i + 1].isalnum() or text[i + 1] in "._-"):
                word.append("\\")
                i += 1
                continue
            if i + 1 < n:
                word.append(text[i + 1])
                have = True
            i += 2
            continue
        if c == "'":
            j = text.find("'", i + 1)
            j = n if j < 0 else j
            word.append(text[i + 1:j])
            have = True
            i = j + 1
            continue
        if c == '"':
            have = True
            i += 1
            while i < n and text[i] != '"':
                if text[i] == "\\" and i + 1 < n and text[i + 1] in '"\\$`\n':
                    if text[i + 1] != "\n":
                        word.append(text[i + 1])
                    i += 2
                elif text[i] in "$`":
                    i = expansion(QDYN)
                else:
                    word.append(text[i])
                    i += 1
            i += 1
            continue
        if c == "$" and text.startswith("$'", i):
            j = text.find("'", i + 2)
            j = n if j < 0 else j
            word.append(text[i + 2:j].replace("\\n", "\n").replace("\\t", "\t"))
            have = True
            i = j + 1
            continue
        if c in "$`":
            i = expansion(UDYN)
            continue
        if c == "#" and not have:
            j = text.find("\n", i)
            i = n if j < 0 else j
            continue
        if c in "<>":
            if have and "".join(word).isdigit():
                word, have = [], False          # the fd of `2>file`
            end_word()
            if text.startswith("<<<", i):
                redirect, i = "<<<", i + 3
                continue
            if text.startswith("<<", i):
                m = re.match(r"<<(-?)[ \t]*(?:'([^'\n]*)'|\"([^\"\n]*)\"|\\?([^\s;&|<>()]+))",
                             text[i:])
                if m:
                    delim = next(g for g in m.groups()[1:] if g is not None)
                    quoted = m.group(4) is None or text[i + m.start(4) - 1] == "\\"
                    pending.append((bool(m.group(1)), delim, quoted, cur))
                    if cur.stdin is None:
                        cur.stdin = ""
                    i += m.end()
                    continue
            m = re.match(r"[<>]{1,2}(?:&[0-9-]+|\|)?", text[i:])
            if "&" not in m.group(0):
                redirect = m.group(0)
            i += m.end()
            continue
        if c in ";&|()":
            if c == "&" and text.startswith("&>", i):
                end_word()
                redirect, i = "&>", i + 2 + (text[i + 2:i + 3] == ">")
                continue
            end_cmd(piped=(c == "|" and not text.startswith("||", i)))
            i += 2 if text[i:i + 2] in ("&&", "||", ";;", "|&") else 1
            continue
        word.append(c)
        have = True
        i += 1
    end_cmd()
    return cmds


def _base(word):
    """The command a word names, without its directory or Windows suffix."""
    name = re.split(r"[\\/]", word)[-1].lower()
    for suffix in (".exe", ".cmd", ".bat"):
        name = name.removesuffix(suffix)
    return name


class Invocation:
    def __init__(self, argv, cmd, env_repo, cwd, files):
        self.argv, self.cmd, self.env_repo, self.cwd, self.files = (
            argv, cmd, env_repo, cwd, files)


def gh_invocations(text, cwd=None, _state=None, _depth=0):
    """Every `gh` invocation a shell would run from this command text."""
    state = _state if _state is not None else {"cwd": cwd, "env_repo": None,
                                               "files": {}}
    out = []
    if _depth > 8:
        return out
    # A PowerShell here-string is a literal the POSIX reading below would mis-split.
    heres = []

    def stash(m):
        heres.append(m.group(2))
        return f"'\ue002{len(heres) - 1}\ue002'"
    text = _PS_HERE_RE.sub(stash, text)
    nested = []
    cmds = _lex(text, nested)

    def unstash(w):
        return re.sub("\ue002(\\d+)\ue002", lambda m: heres[int(m.group(1))], w)

    for cmd in cmds:
        words = [unstash(w) for w in cmd.words]
        env_repo = state["env_repo"]
        # Leading assignments and wrappers, down to the word that is the command.
        k = 0
        while k < len(words):
            w = words[k]
            if _ASSIGN_RE.match(w):
                if w.startswith("GH_REPO="):
                    env_repo = w.split("=", 1)[1]
                k += 1
            elif _base(w) in _WRAPPERS:
                k += 1
            elif _base(w) == "env":
                k += 1
                while k < len(words) and words[k].startswith("-"):
                    k += 1
            elif _base(w) == "timeout":
                k += 1
                while k < len(words) and (words[k].startswith("-")
                                          or re.match(r"^[0-9.]+[smhd]?$", words[k])):
                    k += 1
            else:
                break
        if k >= len(words):
            if k and all(_ASSIGN_RE.match(w) for w in words):
                state["env_repo"] = env_repo          # a bare `GH_REPO=x`
            continue
        name, args = _base(words[k]), words[k + 1:]
        if name == "export":
            for a in args:
                if a.startswith("GH_REPO="):
                    state["env_repo"] = a.split("=", 1)[1]
        elif name in ("cd", "pushd", "set-location"):
            dest = next((a for a in args if not a.startswith("-")), None)
            if dest is None or QDYN in dest or UDYN in dest or state["cwd"] is False:
                state["cwd"] = False
            else:
                state["cwd"] = os.path.normpath(os.path.join(
                    state["cwd"] or os.getcwd(), os.path.expanduser(dest)))
        elif name in ("cat", "tee") and cmd.stdin is not None \
                and not cmd.stdin_dynamic:
            # `cat > body.md <<'EOF'` followed by `--body-file body.md` in the same
            # command: the file does not exist yet, but its text is right here.
            dests = [t for op, t in cmd.redirects if op in (">", ">>", ">|")]
            dests += [a for a in args if not a.startswith("-")] if name == "tee" else []
            for d in dests:
                state["files"][d] = cmd.stdin
        elif name == "gh":
            out.append(Invocation(args, cmd, env_repo, state["cwd"],
                                  dict(state["files"])))
        elif name in _SHELLS | {"eval", "pwsh", "powershell", "cmd", "xargs",
                                "invoke-expression", "iex"}:
            script = None
            if name in ("eval", "invoke-expression", "iex"):
                script = " ".join(args)
            elif name == "xargs":
                script = " ".join(a for a in args if not a.startswith("-")) \
                    + " " + UDYN
            else:
                for j, a in enumerate(args):
                    if a.lower() in ("-c", "-command", "/c", "-lc", "-ic") \
                            and j + 1 < len(args):
                        script = " ".join(args[j + 1:]) if name == "cmd" \
                            else args[j + 1]
                        break
                if script is None and name in _SHELLS and cmd.stdin:
                    script = cmd.stdin
            if script:
                out += gh_invocations(script, _state=state, _depth=_depth + 1)
        # An unquoted heredoc is expanded by the shell before the command reads it.
        if cmd.stdin and cmd.stdin_dynamic and name != "gh":
            inner = []
            _lex('"' + cmd.stdin.replace('"', '\\"') + '"', inner)
            for sub in inner:
                out += gh_invocations(sub, _state=state, _depth=_depth + 1)
    for sub in nested:
        out += gh_invocations(unstash(sub), _state=state, _depth=_depth + 1)
    return out


_BODY_VALUE_FLAGS = {"-b", "--body", "-c", "--comment", "-n", "--notes", "-m",
                     "--message", "-R", "--repo"} | BODY_FILE_FLAGS | API_PARAM_FLAGS


def classify_invocation(inv, strict=True):
    """Judge one invocation found in a command line. What the shell would expand is
    not known here, so a write that depends on an expansion is refused with the way
    to spell it so that it can be judged."""
    argv = inv.argv
    dyn = [i for i, a in enumerate(argv) if QDYN in a or UDYN in a]
    plain = [a.replace(QDYN, "x").replace(UDYN, "x") for a in argv]

    def read_file(path):
        if path in inv.files:
            return inv.files[path]
        base = inv.cwd if isinstance(inv.cwd, str) else None
        if inv.cwd is False and not os.path.isabs(path):
            raise OSError("the working directory is not known")
        return _read_body_file(os.path.join(base, path) if base else path)

    stdin_text = None
    if inv.cmd.stdin is not None and not inv.cmd.stdin_dynamic \
            and not inv.cmd.piped:
        stdin_text = inv.cmd.stdin
    d = classify(plain, strict=strict, stdin_text=stdin_text,
                 env_repo=inv.env_repo if inv.env_repo is None
                 else inv.env_repo.replace(QDYN, "?").replace(UDYN, "?"),
                 cwd=inv.cwd, read_file=read_file)
    if not d.allow or not dyn:
        return d
    hint = ("the hook reads the command before the shell expands it; write the "
            "repository and the command literally and pass any body with "
            "--body-file <file>")
    stripped = _strip_global_flags(argv)
    # The noun, and the verb after it; `api` has an endpoint there instead.
    if any(QDYN in a or UDYN in a
           for a in stripped[:1 if stripped[:1] == ["api"] else 2]):
        return Decision(False, f"which gh command this runs depends on a shell "
                               f"expansion; {hint}", [], True)
    if any(argv[i].startswith(UDYN) for i in dyn) and (d.writes or
                                                       stripped[:1] == ["api"]):
        return Decision(False, f"an unquoted shell expansion could add flags or "
                               f"arguments to this call; {hint}", [], True)
    if not d.writes:
        return d
    explicit = bool(repo_flag_values(plain)) or bool(url_targets(plain))
    for i in dyn:
        a = argv[i]
        flag = a.split("=", 1)[0] if a.startswith("-") else None
        prev = argv[i - 1] if i else ""
        if flag in _BODY_VALUE_FLAGS or (flag and flag[:2] in ("-f", "-F", "-R",
                                                               "-b")
                                         and not flag.startswith("--")) \
                or prev in _BODY_VALUE_FLAGS:
            return Decision(False, f"a body, field or --repo value comes from a "
                                   f"shell expansion and cannot be checked; {hint}",
                            d.targets, True)
        if not a.startswith("-") and prev not in VALUE_FLAGS and not explicit:
            return Decision(False, f"the target of this write may come from a shell "
                                   f"expansion; {hint}", d.targets, True)
    return d


def check_command(text, strict=True, cwd=None):
    """[(invocation text, Decision)] for every refused `gh` call in a command line."""
    refused = []
    for inv in gh_invocations(text, cwd=cwd):
        d = classify_invocation(inv, strict=strict)
        if not d.allow:
            shown = " ".join(a.replace(QDYN, "$...").replace(UDYN, "$...")
                             for a in inv.argv)
            refused.append((shown, d))
    return refused


HOOK_REFUSAL = """moat: this command was not run. It contains a `gh` call the guard refuses:

    gh {argv}

{reason}.

The autonomy boundary reserves project adoption, every GitHub-visible action against
a repo we do not own, and every `/moat` decision on a review PR for a person.
"""


def report_refused(refused):
    for shown, d in refused:
        sys.stderr.write(HOOK_REFUSAL.format(argv=shown[:400], reason=d.reason))


def hook_main(raw):
    """A harness PreToolUse hook: the event is JSON on standard input, exit status 2
    blocks the tool call and standard error is what the agent is shown. An event
    that cannot be read blocks too -- a guard that cannot tell must not pass."""
    try:
        event = json.loads(raw)
        tool_input = event.get("tool_input") or {}
        command = tool_input.get("command")
    except (ValueError, AttributeError):
        sys.stderr.write("moat: gh guard hook could not read the tool event; "
                         "refusing rather than guessing\n")
        return 2
    if not isinstance(command, str) or not command.strip():
        return 0
    refused = check_command(command, strict=True, cwd=event.get("cwd") or None)
    report_refused(refused)
    return 2 if refused else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--explain", action="store_true",
                    help="print the decision instead of running gh")
    ap.add_argument("--outside", action="store_true",
                    help="with --explain or --check-command: judge as from outside "
                         "a MOAT checkout")
    ap.add_argument("--shim", action="store_true",
                    help="called as the PATH launcher: guard, then run the real gh")
    ap.add_argument("--check-command", metavar="TEXT",
                    help="judge every gh invocation in a shell command line; exit "
                         "13 and say why if one is refused")
    ap.add_argument("--hook", action="store_true",
                    help="as --check-command, reading a harness PreToolUse event "
                         "(JSON) on standard input; exit 2 on a refusal")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("args", nargs=argparse.REMAINDER)
    ns = ap.parse_args()

    if ns.self_test:
        return self_test()
    if ns.hook:
        return hook_main(sys.stdin.read())
    if ns.check_command is not None:
        refused = check_command(ns.check_command, strict=not ns.outside)
        report_refused(refused)
        return 13 if refused else 0

    argv = ns.args[1:] if ns.args[:1] == ["--"] else ns.args

    # The launcher judges every call. Outside a MOAT checkout the judgement is the
    # narrow one (see classify), but it is never skipped: a `/moat` line posted from
    # another directory is the same approval.
    strict = under_repo() if ns.shim else not ns.outside
    stdin_bytes = None
    if ns.shim:
        stripped = _strip_global_flags(argv)
        if stripped and reads_stdin(stripped[0], stripped[1:]):
            # The body arrives on standard input. Read it here so it can be
            # inspected, and hand the same bytes to gh afterwards.
            stdin_bytes = sys.stdin.buffer.read()
    d = classify(argv, strict=strict,
                 stdin_text=None if stdin_bytes is None
                 else stdin_bytes.decode("utf-8", errors="replace"))

    if ns.explain:
        print(("ALLOW" if d.allow else "REFUSE") + ": " + d.reason)
        return 0 if d.allow else 1

    if not d.allow:
        sys.stderr.write(REFUSAL.format(argv=" ".join(argv), reason=d.reason,
                                        real=real_gh() or "gh"))
        return 13

    return delegate(argv, stdin_bytes)


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
    # gh takes the LAST --repo, in every spelling it accepts.
    (["pr", "comment", "9", "--repo", f"{OWNED_ORG}/moat", "--repo",
      "torvalds/linux", "--body", "hi"], False),
    (["pr", "comment", "9", "--repo", "torvalds/linux", "--repo",
      f"{OWNED_ORG}/moat", "--body", "hi"], True),
    (["pr", "comment", "9", "-Rtorvalds/linux", "--body", "hi"], False),
    (["pr", "comment", "9", "-R=torvalds/linux", "--body", "hi"], False),
    (["pr", "comment", "9", f"-R{OWNED_ORG}/moat", "--body", "hi"], True),
    (["pr", "comment", "9", "--repo=torvalds/linux", "--body", "hi"], False),
    (["pr", "comment", "9", "--repo", "not a repo", "--body", "hi"], False),
    (["pr", "comment", "9", "--repo", f"github.com/{OWNED_ORG}/moat", "-b", "hi"],
     True),
    # A JSON or GraphQL string spells the line break as backslash-n.
    (["api", f"repos/{OWNED_ORG}/fork/issues/3/comments", "-f",
      "body=ok\\n/moat approve"], False),
    (["api", "graphql", "-f", 'query=mutation { addComment(input: {subjectId: '
      '"x", body: "ok\\n/moat approve"}) { clientMutationId } }'], False),
    # A review decision is a person's with or without the `/moat` words.
    (["pr", "review", "3", "--repo", f"{OWNED_ORG}/fork", "--approve"], False),
    (["pr", "review", "3", "--repo", f"{OWNED_ORG}/fork", "-r", "-b", "no"], False),
    (["pr", "review", "3", "--repo", f"{OWNED_ORG}/fork", "--comment", "-b",
      "addressed in abc1234"], True),
    (["api", "-X", "PUT", f"repos/{OWNED_ORG}/fork/pulls/3/reviews/9/dismissals",
      "-f", "message=x"], False),
    (["api", "-X", "DELETE", f"repos/{OWNED_ORG}/fork/issues/comments/77"], False),
    (["api", "-X", "PATCH", f"repos/{OWNED_ORG}/fork/issues/comments/77",
      "-f", "body=typo fixed"], True),
    (["pr", "comment", "3", "--repo", f"{OWNED_ORG}/fork", "--delete-last"], False),
    # An alias or an extension can wrap any write.
    (["alias", "list"], True),
    (["alias", "set", "co", "pr checkout"], False),
    (["alias", "set", "prs", "pr list --author @me"], True),
    (["alias", "set", "ok", "pr comment --body '/moat approve'"], False),
    (["alias", "set", "--shell", "x", "gh pr comment"], False),
    (["extension", "list"], True),
    (["extension", "install", "someone/gh-thing"], False),
    (["extension", "exec", "thing", "anything"], False),
    (["mycustomalias", "42"], False),
    (["co", "42", "--repo", f"{OWNED_ORG}/moat"], True),
]

# The judgement from outside a MOAT checkout (`strict=False`): this machine's other
# work keeps working, and only what is dangerous from any directory is refused.
OUTSIDE_CASES = [
    (["pr", "comment", "9", "--repo", "torvalds/linux", "--body", "hi"], True),
    (["pr", "create", "--repo", "torvalds/linux", "--title", "t", "--body", "b"],
     True),
    (["api", "repos/torvalds/linux/issues/1/comments", "-f", "body=hi"], True),
    (["repo", "fork", "torvalds/linux"], True),
    (["alias", "set", "co", "pr checkout"], True),
    (["extension", "install", "someone/gh-thing"], True),
    (["mycustomalias", "42"], True),
    (["pr", "review", "9", "--repo", "torvalds/linux", "--approve"], True),
    (["pr", "comment", "9", "--repo", "torvalds/linux", "--body-file", "-"], True),
    # Refused from anywhere.
    (["pr", "comment", f"https://github.com/{OWNED_ORG}/fork/pull/3",
      "--body", "/moat approve"], False),
    (["pr", "comment", "9", "--repo", "torvalds/linux", "--body",
      "/moat approve"], False),
    (["api", f"repos/{OWNED_ORG}/fork/issues/3/comments", "-f",
      "body=fine\n\n/moat approve"], False),
    (["pr", "review", "3", "--repo", f"{OWNED_ORG}/fork", "--approve"], False),
    (["pr", "comment", "3", "--repo", f"{OWNED_ORG}/fork", "--body-file", "-"],
     False),
    (["repo", "fork", "torvalds/linux", "--org", OWNED_ORG], False),
    (["repo", "fork", "torvalds/linux", f"--org={OWNED_ORG.lower()}"], False),
    (["alias", "set", "ok", "pr comment --body '/moat approve'"], False),
    (["mycustomalias", "--body", "/moat approve"], False),
]

# Shell command lines for --check-command / --hook, judged as inside a checkout.
COMMAND_CASES = [
    ("gh pr view 3 --repo torvalds/linux", True),
    ("gh pr comment 3 --repo torvalds/linux --body hi", False),
    ("cd /tmp && gh pr comment 3 --repo torvalds/linux --body hi", False),
    ('"/c/Program Files/GitHub CLI/gh" pr comment 3 -R torvalds/linux -b hi', False),
    ('& "C:\\Program Files\\GitHub CLI\\gh.exe" pr comment 3 -R torvalds/linux -b hi',
     False),
    ("GH_REPO=torvalds/linux gh pr comment 3 --body hi", False),
    ("x=1; echo ok | gh issue comment 3 --repo torvalds/linux -F -", False),
    ('bash -c "gh pr comment 3 --repo torvalds/linux --body hi"', False),
    ('echo "$(gh pr close 3 --repo torvalds/linux)"', False),
    (f"gh pr comment 3 --repo {OWNED_ORG}/fork --body hi", True),
    (f"gh pr comment 3 --repo {OWNED_ORG}/fork --body '/moat approve'", False),
    (f'gh pr comment 3 --repo {OWNED_ORG}/fork --body "$BODY"', False),
    ('gh pr comment "$URL" --body-file body.md', False),
    ('gh pr view "$URL" --json state', True),
    (f"gh pr comment 3 --repo {OWNED_ORG}/fork -F - <<'EOF'\nfixed in abc1234\nEOF",
     True),
    (f"gh pr comment 3 --repo {OWNED_ORG}/fork -F - <<'EOF'\nok\n/moat approve\nEOF",
     False),
    (f"cat > b.md <<'EOF'\n/moat approve\nEOF\n"
     f"gh pr comment 3 --repo {OWNED_ORG}/fork --body-file b.md", False),
    (f"cat > b.md <<'EOF'\nfixed in abc1234\nEOF\n"
     f"gh pr comment 3 --repo {OWNED_ORG}/fork --body-file b.md", True),
    # Mentions, not invocations.
    ("echo gh pr comment 3 --repo torvalds/linux --body hi", True),
    ("printf '%s\\n' 'gh pr create --repo torvalds/linux'", True),
    ('git commit -m "gh pr comment 3 --repo torvalds/linux --body /moat approve"',
     True),
    ("cat > notes.md <<'EOF'\ngh pr create --repo torvalds/linux\n/moat approve\nEOF",
     True),
    ("git commit -F - <<'EOF'\nfix\n\ngh api -X POST repos/torvalds/linux/issues\nEOF",
     True),
    ("# gh pr create --repo torvalds/linux\nls", True),
    ("grep -rn 'gh pr create' utils/ | head", True),
    ("git log --oneline | grep gh", True),
]


def self_test():
    bad = 0
    total = len(CASES) + len(OUTSIDE_CASES) + len(COMMAND_CASES)
    for strict, cases in ((True, CASES), (False, OUTSIDE_CASES)):
        for argv, want in cases:
            got = classify(argv, strict=strict).allow
            if got != want:
                bad += 1
                print(f"FAIL want={'allow' if want else 'refuse'} "
                      f"got={'allow' if got else 'refuse'} "
                      f"({'inside' if strict else 'outside'}): gh {' '.join(argv)}")
    for text, want in COMMAND_CASES:
        got = not check_command(text)
        if got != want:
            bad += 1
            print(f"FAIL want={'allow' if want else 'refuse'} "
                  f"got={'allow' if got else 'refuse'}: {text!r}")
    print(f"{total - bad}/{total} cases pass")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
