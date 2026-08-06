import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import httpx
    import os
    import matplotlib.pyplot as plt

    SERVER = "https://atlas.taile454b.ts.net"
    API_KEY = os.environ.get("JERNERICS_API_KEY", "")

    def query(sql):
        r = httpx.post(
            f"{SERVER}/query",
            json={"sql": sql},
            headers={"authorization": f"Bearer {API_KEY}"} if API_KEY else {},
            timeout=15,
        )
        r.raise_for_status()
        d = r.json()
        return {"columns": d["columns"], "rows": d["rows"]}

    return mo, plt, query


@app.cell
def _(mo):
    refresh = mo.ui.refresh(
        options=["5s", "10s", "30s", "1m"],
        default_interval="5s",
        label="Auto-refresh",
    )
    refresh
    return (refresh,)


@app.cell
def _(mo, query):
    _s = query("SELECT DISTINCT study_name FROM tracked_values ORDER BY study_name")
    _names = [r[0] for r in _s["rows"]]
    study_selector = mo.ui.multiselect(
        options=_names,
        value=_names[-3:] if len(_names) >= 3 else _names,
        label="Studies",
    )
    study_selector
    return (study_selector,)


@app.cell
def _(mo):
    mo.md("""
    ## Training Metrics
    """)
    return


@app.cell
def _(query, refresh, study_selector):
    _ = refresh
    sel = study_selector.value or []
    mdata = {}
    if sel:
        _names = ",".join(f"'{s}'" for s in sel)
        try:
            _res = query(
                f"SELECT study_name, key, step, scalar_val "
                f"FROM tracked_values WHERE study_name IN ({_names}) "
                f"AND key IN ('loss','ce_loss','mse_loss','token_acc',"
                f"'grad_norm','val_r2','val_valid_rate','structure_ce') "
                f"ORDER BY study_name, key, step"
            )
            for _r in _res["rows"]:
                _k = (_r[0], _r[1])
                if _k not in mdata:
                    mdata[_k] = []
                try:
                    mdata[_k].append((int(_r[2]), float(_r[3])))
                except (ValueError, TypeError):
                    pass
        except Exception:
            pass
    return mdata, sel


@app.cell
def _(mdata, mo, plt, sel):
    METRICS = ["loss", "ce_loss", "structure_ce", "mse_loss", "token_acc", "val_r2"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    af = axes.flatten()
    for _i in range(6):
        metric = METRICS[_i]
        ax = af[_i]
        has = False
        for _k, _v in mdata.items():
            if _k[1] != metric or not _v:
                continue
            _v.sort()
            ax.plot([p[0] for p in _v], [p[1] for p in _v],
                    label=_k[0].split("_")[-1], alpha=0.8, linewidth=1.2)
            has = True
        ax.set_title(metric, fontsize=11)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
        if has:
            ax.legend(fontsize=7)
    plt.tight_layout()
    mo.mpl.interactive(fig) if sel and mdata else mo.md("No studies selected or no data yet.")
    return


@app.cell
def _(mdata, mo, sel):
    if not sel or not mdata:
        mo.md("")
    else:
        rows = []
        seen = set()
        for _k, _v in mdata.items():
            _short = _k[0].split("_")[-1] if "_" in _k[0] else _k[0]
            if _short not in seen:
                seen.add(_short)
                rows.append({"study": _short})
            if _v:
                _lv = _v[-1][1]
                for _r in rows:
                    if _r["study"] == _short:
                        _r[_k[1]] = round(_lv, 4) if isinstance(_lv, float) else _lv
        mo.ui.table(rows)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Inspection Reports
    """)
    return


@app.cell
def _(mo, query):
    _insp = query(
        "SELECT study_name, key, scalar_val FROM tracked_values "
        "WHERE key LIKE 'inspect_%' ORDER BY study_name, key"
    )
    pivot = {}
    for _r in _insp["rows"]:
        _sn = _r[0].split("_")[-1] if "_" in _r[0] else _r[0]
        if _sn not in pivot:
            pivot[_sn] = {}
        try:
            pivot[_sn][_r[1]] = float(_r[2])
        except (ValueError, TypeError):
            pass
    mo.ui.table([{"study": k, **v} for k, v in pivot.items()]) if pivot else mo.md("No inspection data.")
    return


@app.cell
def _(mo, query, refresh):
    _ = refresh
    _ov = query(
        "SELECT study_name, COUNT(DISTINCT key) as n_metrics, "
        "COUNT(*) as n_points, MAX(step) as max_step "
        "FROM tracked_values GROUP BY study_name ORDER BY study_name"
    )
    mo.ui.table([dict(zip(_ov["columns"], r)) for r in _ov["rows"]])
    return


if __name__ == "__main__":
    app.run()
