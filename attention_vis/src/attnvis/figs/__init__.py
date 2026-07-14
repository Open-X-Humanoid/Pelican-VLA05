"""attnvis.figs — built-in figure drivers.

One file per driver, runnable as:
    python -m attnvis.figs.make_<name> [args]
Or via the short command:
    python -m attnvis fig <name> [args]

Shipped drivers: `montage`. Everything else is up to the user — use the
`.npz` archives written by `dump_dense` together with `attnvis.figlib`
(`blend`, `pct_vmax`, `pct_range`, `save_cell`, `setup_cjk_font`).
"""
