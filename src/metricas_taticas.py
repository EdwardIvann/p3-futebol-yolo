#!/usr/bin/env python3
"""
Métricas táticas extras (proposta do grupo), além de posse, passes e eXP.

Rode DEPOIS de metricas.py (usa tracks_campo.csv, posse_por_quadro.csv,
passes.csv e metricas_times.json).

Métricas por time
-----------------
1. Compactação     área do fecho convexo (m²), largura (y) e profundidade (x)
                   do bloco, separando "com bola" e "sem bola".
2. Linha defensiva altura (m a partir do próprio gol) do 2º jogador mais recuado
                   (o 1º costuma ser o goleiro).
3. Posse por terço % do tempo de posse com a bola no terço defensivo/médio/ofensivo.
4. Field tilt       % das posses no terço ofensivo que são do time (domínio territorial).
5. Pressão         nº médio de adversários a < --raio-pressao m do portador da bola.
6. Sequências      nº de posses, duração média, passes por posse e progressão (m/s)
                   da bola em direção ao gol durante a posse.
7. Recuperações    onde (x, normalizado) o time recupera a bola; altura média.

Limitação importante: a câmera de transmissão não mostra os 22 jogadores.
Métricas de bloco (1, 2) só são calculadas em quadros com >= --min-jogadores
visíveis do time — e mesmo assim tendem a SUBESTIMAR largura e área.

Uso
---
python metricas_taticas.py --saida ../saidas/jogo1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError

from campo import COMPRIMENTO, desenhar_campo
from metricas import CORES_TIME, carregar_jogadores

TERCOS = ("defensivo", "medio", "ofensivo")


def x_ataque(x, direcao: str):
    """Converte x para 'metros a partir do próprio gol' (0 = próprio gol, 105 = gol rival)."""
    return np.asarray(x) if direcao == "direita" else COMPRIMENTO - np.asarray(x)


def terco(xn):
    return np.select([xn < COMPRIMENTO / 3, xn < 2 * COMPRIMENTO / 3], [0, 1], 2)


# ── 1–2. Bloco: compactação e linha defensiva ─────────────────────────────

def bloco_por_quadro(jog, posse, direcao, min_jog):
    tp = posse.set_index("frame")["time_posse"]
    linhas = []
    for (f, t), g in jog[jog["time"] >= 0].groupby(["frame", "time"]):
        if len(g) < min_jog:
            continue
        pts = g[["x", "y"]].to_numpy()
        try:
            area = ConvexHull(pts).volume  # em 2D, "volume" = área
        except QhullError:
            area = np.nan
        xn = np.sort(x_ataque(g["x"], direcao[t]))
        linhas.append({
            "frame": f, "time": t, "n_visiveis": len(g), "area_m2": area,
            "largura_m": float(np.ptp(pts[:, 1])), "profundidade_m": float(np.ptp(pts[:, 0])),
            "linha_def_m": float(xn[1]) if len(xn) > 1 else np.nan,
            "com_bola": bool(tp.get(f, np.nan) == t),
        })
    return pd.DataFrame(linhas)


# ── 3–5. Território e pressão ──────────────────────────────────────────────

def territorio(posse, direcao):
    p = posse.dropna(subset=["time_posse", "x_bola"]).copy()
    p["time_posse"] = p["time_posse"].astype(int)
    p["xn"] = [x_ataque(x, direcao[t]) for x, t in zip(p["x_bola"], p["time_posse"])]
    p["terco"] = terco(p["xn"])
    return p


def pressao_portador(jog, posse, raio):
    d = posse.dropna(subset=["dono"])[["frame", "dono", "time_dono", "x_bola", "y_bola"]]
    m = d.merge(jog[["frame", "time", "x", "y"]], on="frame")
    m = m[m["time"] != m["time_dono"]]
    m["perto"] = np.hypot(m["x"] - m["x_bola"], m["y"] - m["y_bola"]) < raio
    por_quadro = m.groupby(["frame", "time_dono"])["perto"].sum().reset_index()
    # quadros em que nenhum adversário apareceu também contam como 0 de pressão
    por_quadro = d[["frame", "time_dono"]].merge(por_quadro, how="left").fillna({"perto": 0})
    return por_quadro


# ── 6–7. Sequências de posse e recuperações ────────────────────────────────

def sequencias(posse, passes, direcao, fps):
    p = posse.dropna(subset=["time_posse"]).copy()
    p["seq"] = (p["time_posse"] != p["time_posse"].shift()).cumsum()
    linhas = []
    for s, g in p.groupby("seq"):
        t = int(g["time_posse"].iloc[0])
        g = g.dropna(subset=["x_bola"])
        if len(g) < 2:
            continue
        xn = x_ataque(g["x_bola"], direcao[t])
        dur = (g["frame"].iloc[-1] - g["frame"].iloc[0]) / fps
        n_pass = 0
        if len(passes):
            n_pass = int(((passes["frame_ini"] >= g["frame"].iloc[0]) &
                          (passes["frame_ini"] <= g["frame"].iloc[-1]) &
                          (passes["time"] == t) & (passes["sucesso"] == 1)).sum())
        linhas.append({"seq": s, "time": t, "frame_ini": int(g["frame"].iloc[0]),
                       "duracao_s": dur, "passes_certos": n_pass,
                       "x_inicio_m": float(xn[0]), "x_fim_m": float(xn[-1]),
                       "x_campo": float(g["x_bola"].iloc[0]), "y_campo": float(g["y_bola"].iloc[0]),
                       "progressao_m": float(xn[-1] - xn[0]),
                       "progressao_m_s": float((xn[-1] - xn[0]) / dur) if dur > 0 else np.nan})
    seq = pd.DataFrame(linhas)
    # Recuperação = início de cada sequência, exceto a primeira do clipe
    rec = seq.iloc[1:][["time", "frame_ini", "x_inicio_m", "x_campo", "y_campo"]].rename(columns={"x_inicio_m": "x_recuperacao_m"})
    return seq, rec


# ── Figuras ────────────────────────────────────────────────────────────────

def figuras(pasta, bloco, terr, rec, fps, direcao):
    fig_dir = pasta / "figuras"
    fig_dir.mkdir(exist_ok=True)

    if len(bloco):
        fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        for t, g in bloco.groupby("time"):
            g = g.sort_values("frame")
            tt = g["frame"] / fps
            axs[0].plot(tt, g["area_m2"].rolling(25, min_periods=1).median(), color=CORES_TIME[t], label=f"time {t}")
            axs[1].plot(tt, g["linha_def_m"].rolling(25, min_periods=1).median(), color=CORES_TIME[t])
        axs[0].set_ylabel("área do bloco (m²)")
        axs[1].set_ylabel("linha defensiva (m do próprio gol)")
        axs[1].set_xlabel("tempo (s)")
        axs[0].legend()
        axs[0].set_title("Compactação e altura da linha defensiva (mediana móvel 1 s)")
        fig.tight_layout()
        fig.savefig(fig_dir / "compactacao_linha_def.png", dpi=130)
        plt.close(fig)

    if len(terr):
        tab = pd.crosstab(terr["time_posse"], terr["terco"], normalize="index") * 100
        tab = tab.reindex(columns=[0, 1, 2], fill_value=0)
        fig, ax = plt.subplots(figsize=(7, 3.5))
        esq = np.zeros(len(tab))
        for k, nome in enumerate(TERCOS):
            ax.barh([f"time {int(t)}" for t in tab.index], tab[k], left=esq, label=nome,
                    color=["#cbd5e1", "#64748b", "#0f172a"][k])
            esq += tab[k].to_numpy()
        ax.set_xlabel("% do tempo de posse")
        ax.set_title("Onde cada time tem a bola (terço do campo)")
        ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.45))
        fig.tight_layout()
        fig.savefig(fig_dir / "posse_por_terco.png", dpi=130)
        plt.close(fig)

    if len(rec):
        ax = desenhar_campo()
        for t, g in rec.groupby("time"):
            ax.scatter(g["x_campo"], g["y_campo"], s=80,
                       color=CORES_TIME[t], edgecolors="white", label=f"time {t}", zorder=3)
        ax.set_title("Onde cada time recuperou a bola")
        ax.legend()
        ax.figure.savefig(fig_dir / "recuperacoes.png", dpi=130, bbox_inches="tight")
        plt.close(ax.figure)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--raio-pressao", type=float, default=5.0, help="metros")
    ap.add_argument("--min-jogadores", type=int, default=6, help="mínimo visível por time para métricas de bloco")
    a = ap.parse_args()
    pasta = Path(a.saida)

    meta = json.loads((pasta / "meta.json").read_text())
    fps = meta["fps"]
    resumo_base = json.loads((pasta / "metricas_times.json").read_text())
    direcao = {int(k): v for k, v in resumo_base["direcao_ataque"].items()}

    jog = carregar_jogadores(pasta)
    posse = pd.read_csv(pasta / "posse_por_quadro.csv")
    passes = pd.read_csv(pasta / "passes.csv") if (pasta / "passes.csv").stat().st_size > 2 else pd.DataFrame()

    bloco = bloco_por_quadro(jog, posse, direcao, a.min_jogadores)
    terr = territorio(posse, direcao)
    press = pressao_portador(jog, posse, a.raio_pressao)
    seq, rec = sequencias(posse, passes, direcao, fps)

    bloco.to_csv(pasta / "bloco_por_quadro.csv", index=False)
    seq.to_csv(pasta / "sequencias_posse.csv", index=False)
    rec.to_csv(pasta / "recuperacoes.csv", index=False)

    ofensivo = terr[terr["terco"] == 2]["time_posse"].value_counts(normalize=True)
    res = {"parametros": {"raio_pressao_m": a.raio_pressao, "min_jogadores": a.min_jogadores},
           "quadros_bloco_validos": int(bloco["frame"].nunique()) if len(bloco) else 0}
    for t in (0, 1):
        b = bloco[bloco["time"] == t] if len(bloco) else bloco
        tr = terr[terr["time_posse"] == t]
        s = seq[seq["time"] == t] if len(seq) else seq
        r = rec[rec["time"] == t] if len(rec) else rec
        pr = press[press["time_dono"] == t]
        r2 = lambda v: None if v is None or pd.isna(v) else round(float(v), 2)
        res[f"time{t}"] = {
            "area_bloco_m2_com_bola": r2(b.loc[b["com_bola"], "area_m2"].median()) if len(b) else None,
            "area_bloco_m2_sem_bola": r2(b.loc[~b["com_bola"], "area_m2"].median()) if len(b) else None,
            "largura_m": r2(b["largura_m"].median()) if len(b) else None,
            "profundidade_m": r2(b["profundidade_m"].median()) if len(b) else None,
            "linha_defensiva_m_sem_bola": r2(b.loc[~b["com_bola"], "linha_def_m"].median()) if len(b) else None,
            "posse_por_terco_pct": {n: r2(100 * (tr["terco"] == k).mean()) if len(tr) else None
                                    for k, n in enumerate(TERCOS)},
            "field_tilt_pct": r2(100 * ofensivo.get(t, 0.0)),
            "pressao_media_sobre_portador": r2(pr["perto"].mean()) if len(pr) else None,
            "n_posses": int(len(s)),
            "duracao_media_posse_s": r2(s["duracao_s"].mean()) if len(s) else None,
            "passes_por_posse": r2(s["passes_certos"].mean()) if len(s) else None,
            "progressao_media_m_s": r2(s["progressao_m_s"].median()) if len(s) else None,
            "recuperacoes": int(len(r)),
            "altura_media_recuperacao_m": r2(r["x_recuperacao_m"].mean()) if len(r) else None,
        }

    (pasta / "metricas_taticas.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    figuras(pasta, bloco, terr, rec, fps, direcao)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
