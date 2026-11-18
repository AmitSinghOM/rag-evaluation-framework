"""``python -m rageval``: the same command line as the ``rageval`` script."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
