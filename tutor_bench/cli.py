"""Command-line entry point for tutor-bench.

The benchmark subcommands (``run``, ``report``, ``view``, ``dataset``) are wired
up as the benchmark package is built out. Until then this entry point reports
that the CLI is not yet available.
"""

import sys


def main(argv: list[str] | None = None) -> None:
    """Dispatch a tutor-bench subcommand."""
    print("tutor-bench: the command-line interface is not available yet.", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
