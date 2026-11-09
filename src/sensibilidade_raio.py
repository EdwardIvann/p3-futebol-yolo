#!/usr/bin/env python3
"""
Q5.2 — Sensibilidade da posse e do nº de passes ao raio de posse.

Reusa as funções de metricas.py SEM sobrescrever as saídas principais.

Uso
---
python sensibilidade_raio.py --saida ../saidas/jogo1 --raios 1 1.5 2 2.5 3
Saídas: sensibilidade_raio.csv · figuras/sensibilidade_raio.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from metricas import CORES_TIME, carregar_bola, carregar_jogadores, detectar_passes, posse_por_quadro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", required=True)
    ap.add_argument("--raios", type=float, nargs="+", default=[1, 1.5, 2, 2.5, 3])
    ap.add_argument("--min-quadros", type=int, default=3)
    ap.add_argument("--dist-min-passe", type=float, default=3.0)
    ap.add_argument("--dur-max-passe", type=float, default=4.0)
    a = ap.parse_args()
    pasta = Path(a.saida)
    fps = json.loads((pasta / "meta.json").read_text())["fps"]

    jog, bola = carregar_jogadores(pasta), carregar_bola(pasta)
    linhas = []
    for r in a.raios:
        p = posse_por_quadro(jog, bola, r, a.min_quadros)
        ps = detectar_passes(p, jog, fps, a.dist_min_passe, a.dur_max_passe)
        posse = p["time_posse"].value_counts(normalize=True)
        linhas.append({
            "raio_m": r,
            "posse_time0_pct": round(100 * posse.get(0.0, 0), 1),
            "posse_time1_pct": round(100 * posse.get(1.0, 0), 1),
            "quadros_com_dono_pct": round(100 * p["dono"].notna().mean(), 1),
            "passes": len(ps),
            "passes_certos": int(ps["sucesso"].sum()) if len(ps) else 0,
        })
    tab = pd.DataFrame(linhas)
    tab.to_csv(pasta / "sensibilidade_raio.csv", index=False)
    print(tab.to_string(index=False))

    (pasta / "figuras").mkdir(exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(tab["raio_m"], tab["posse_time0_pct"], "o-", color=CORES_TIME[0], label="time 0")
    ax1.plot(tab["raio_m"], tab["posse_time1_pct"], "o-", color=CORES_TIME[1], label="time 1")
    ax1.set_xlabel("raio de posse (m)"); ax1.set_ylabel("posse (%)"); ax1.legend()
    ax1.set_title("Posse × raio")
    ax2.bar(tab["raio_m"], tab["passes"], width=0.3, color="#94a3b8", label="detectados")
    ax2.bar(tab["raio_m"], tab["passes_certos"], width=0.3, color="#22c55e", label="certos")
    ax2.set_xlabel("raio de posse (m)"); ax2.set_ylabel("nº de passes"); ax2.legend()
    ax2.set_title("Passes × raio")
    fig.tight_layout()
    fig.savefig(pasta / "figuras" / "sensibilidade_raio.png", dpi=130)


if __name__ == "__main__":
    main()
