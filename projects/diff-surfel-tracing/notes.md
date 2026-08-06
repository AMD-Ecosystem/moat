# diff-surfel-tracing notes

A dependency fork: the tracer EnvGS's reflection path renders through.

## Why the platform row is empty

This has no test suite of its own. The code is exercised only through the project
that consumes it, so a GPU run against this repository alone would prove nothing.
The empty row is accurate, not a gap in the record.

The validation lives with **EnvGS**, `completed` on linux-gfx1100, linux-gfx90a, windows-gfx1101, windows-gfx1201.

## Port state

The `moat-port` branch predates this project being tracked here, so the port exists
but its provenance was not recorded: no plan, no dated validation entry, no note of
which commit was tested. Treat it as real work of unverified state rather than as a
validated port.
