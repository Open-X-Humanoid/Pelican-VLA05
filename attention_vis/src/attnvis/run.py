"""End-to-end PelicanVLA dump entry: preflight + dispatch dump_dense.

Usage:
  python -m attnvis run --scenes robotwin:adjust_bottle:0 --nframes 30 --exp demo
  python -m attnvis run --scenes robotwin:adjust_bottle:0 --preflight-only
  python -m attnvis run --scenes robotwin:adjust_bottle:0 --dry-run
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from attnvis import config
from attnvis.paths import PKG_ROOT
from attnvis.registry import preflight as PF

# Editable / src layout: put <repo>/src on PYTHONPATH for the subprocess.
_SRC_ROOT = str(PKG_ROOT.parent)

# Env vars forwarded into the dump subprocess (the PelicanVLA adapter reads
# these to locate Qwen3-VL / Cosmos). Kept inline: this repo only ships the
# PelicanVLA adapter.
_PELICANVLA_EXTRA_ENV = {
    "QWEN3_VL_PATH":         str(config.EXTERNAL["qwen3_vl"]),
    "COSMOS_TOKENIZER_PATH": str(config.EXTERNAL["cosmos"]),
}
_PELICANVLA_CAPTURE = "bottleneck on → two-hop / off → direct (auto by config)"


def _pythonpath() -> str:
    parts = [_SRC_ROOT]
    if os.environ.get("PYTHONPATH"):
        parts += os.environ["PYTHONPATH"].split(os.pathsep)
    seen, uniq = set(), []
    for p in parts:
        if p and p not in seen:
            seen.add(p)
            uniq.append(p)
    return os.pathsep.join(uniq)


def _dump_dense_args(args) -> list[str]:
    a = ["--exp", args.exp, "--nframes", str(args.nframes),
         "--scenes", *args.scenes]
    if args.ckpt:
        a += ["--ckpt", args.ckpt]
    if args.label:
        a += ["--label", args.label]
    if args.no_pngs:
        a += ["--no-pngs"]
    return a


def _extra_env(online: bool) -> dict:
    """Env vars added to the subprocess. HF/transformers default to offline
    because attnvis always uses local weights and some upstream loaders will
    otherwise hang on a network retry."""
    extra = dict(_PELICANVLA_EXTRA_ENV)
    if not online:
        extra.setdefault("HF_HUB_OFFLINE", "1")
        extra.setdefault("TRANSFORMERS_OFFLINE", "1")
    return extra


def build_command(args) -> tuple[list[str], dict, dict]:
    """Return (argv, full env, extra env to display). Does not execute."""
    extra = _extra_env(getattr(args, "online", False))
    env = dict(os.environ)
    env["PYTHONPATH"] = _pythonpath()
    env.update(extra)

    # Optional: run the adapter under a separate PelicanVLA venv; otherwise the
    # current interpreter is used.
    py = os.environ.get("ATTNVIS_PELICANVLA_PY", "")
    driver = ["-m", "attnvis.dump_dense", *_dump_dense_args(args)]
    cmd = [py or sys.executable, *driver]
    return cmd, env, extra


def main():
    ap = argparse.ArgumentParser(description="attnvis end-to-end entry (preflight + dump)")
    ap.add_argument("--scenes", nargs="+", required=True, help='"source:suite[:episode]" list')
    ap.add_argument("--exp", default="run_adhoc", help="Output experiment name")
    ap.add_argument("--nframes", type=int, default=30)
    ap.add_argument("--no-pngs", action="store_true",
                    help="Skip per-frame PNG / contact strip")
    ap.add_argument("--ckpt", default=None, help="Override the release checkpoint")
    ap.add_argument("--label", default=None, help="With --ckpt: output subdir name")
    ap.add_argument("--preflight-only", action="store_true", help="Only validate, do not run")
    ap.add_argument("--dry-run", action="store_true", help="Print the command, do not run")
    ap.add_argument("--no-check-paths", action="store_true",
                    help="Skip data/env path existence checks in preflight")
    ap.add_argument("--force", action="store_true",
                    help="Run even if preflight has blocking issues")
    ap.add_argument("--online", action="store_true",
                    help="Allow HF/transformers to go online (default is offline)")
    args = ap.parse_args()

    rep = PF.check(args.scenes, check_paths=not args.no_check_paths)
    print(rep.render(), flush=True)
    if not rep.ok and not args.force:
        print("\n[attnvis.run] preflight has blocking issues; aborting (use --force to bypass).",
              flush=True)
        sys.exit(2)
    if args.preflight_only:
        return

    cmd, env, extra = build_command(args)
    prefix = [f"PYTHONPATH={env['PYTHONPATH']}"] + [f"{k}={v}" for k, v in extra.items()]
    print(f"\n[attnvis.run] running (capture={_PELICANVLA_CAPTURE}):\n  "
          + " ".join(prefix + cmd) + "\n", flush=True)
    if args.dry_run:
        return

    try:
        rc = subprocess.run(cmd, env=env).returncode
    except FileNotFoundError as e:
        print(f"[attnvis.run] launch failed: {e}\n  Interpreter or command not found; "
              "use --dry-run to print the command and run it manually.", flush=True)
        sys.exit(3)
    if rc == 0:
        print(f"[attnvis.run] DONE. Artifacts in outputs/dense/{args.exp}/", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
