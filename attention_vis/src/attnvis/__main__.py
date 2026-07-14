"""attnvis CLI entry.

    python -m attnvis run ...
    python -m attnvis preflight <scene...>
    python -m attnvis fig <name> ...
"""
from __future__ import annotations

import sys


def _usage():
    print(__doc__)
    print("subcommands: run, preflight, fig")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        _usage()
        sys.exit(0)
    cmd = sys.argv[1]
    sys.argv = [f"attnvis {cmd}"] + sys.argv[2:]

    if cmd == "fig":
        rest = sys.argv[1:]
        if not rest:
            import pkgutil
            import attnvis.figs as _figs
            names = sorted(
                m.name[5:] for m in pkgutil.iter_modules(_figs.__path__)
                if m.name.startswith("make_")
            )
            print("usage: python -m attnvis fig <name> [args]")
            print("names:", ", ".join(names))
            sys.exit(1)
        name, sys.argv = rest[0], [f"attnvis fig {rest[0]}"] + rest[1:]
        import importlib
        importlib.import_module(f"attnvis.figs.make_{name}").main()
        sys.exit(0)

    if cmd == "preflight":
        from attnvis.registry import preflight as PF
        rest = sys.argv[1:]
        check_paths = "--no-check-paths" not in rest
        rest = [a for a in rest if not a.startswith("--")]
        if not rest:
            print("usage: python -m attnvis preflight <source:suite[:ep]> ...")
            sys.exit(1)
        rep = PF.check(rest, check_paths=check_paths)
        print(rep.render())
        sys.exit(0 if rep.ok else 2)

    if cmd == "run":
        import importlib
        importlib.import_module("attnvis.run").main()
        sys.exit(0)

    print(f"unknown subcommand {cmd!r}")
    _usage()
    sys.exit(1)


if __name__ == "__main__":
    main()
