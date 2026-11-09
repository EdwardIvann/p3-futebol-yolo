#!/usr/bin/env python3
"""
Quem cai no 3º cluster de cor com --k 3 (Q3.3) — adição da equipe (P3).

Lê <saida>/tracks.csv de uma rodada com --k 3, conta os tracks do cluster extra
(time = -2, OUTROS) e monta um mosaico com o recorte dos 24 tracks mais longos
desse cluster (número no canto = quadros do track).

Saídas: <saida>/cluster_extra.txt e figuras/jogo1/k3_cluster_extra.jpg

Uso (de src/):
    python inspecionar_k3.py --saida ../saidas/jogo1_k3 --video ../dados/videos/bra_kor_clip.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

OUTROS = -2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--figura", default=str(Path(__file__).resolve().parent.parent / "figuras" / "jogo1" / "k3_cluster_extra.jpg"))
    a = ap.parse_args()
    S = Path(a.saida)

    t = pd.read_csv(S / "tracks.csv")
    t = t[t["classe"] != "bola"]
    o = t[t["time"] == OUTROS]
    g = o.groupby("track_id").size().sort_values(ascending=False)
    linhas = [
        f"Tracks de pessoa: {t.track_id.nunique()} | no cluster extra (time = -2): {o.track_id.nunique()}",
        f"Detecções de pessoa no cluster extra: {100 * len(o) / len(t):.1f}%",
        f"Tracks do cluster extra com >= 30 quadros: {(g >= 30).sum()}",
    ]
    (S / "cluster_extra.txt").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    print("\n".join(linhas))

    cap = cv2.VideoCapture(a.video)
    crops = []
    for tid in g.index[:24]:
        s = o[o.track_id == tid]
        r = s.iloc[len(s) // 2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(r.frame))
        ok, f = cap.read()
        if not ok:
            continue
        c = cv2.resize(f[max(int(r.y1), 0):int(r.y2), max(int(r.x1), 0):int(r.x2)], (60, 120))
        cv2.putText(c, str(len(s)), (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        crops.append(c)
    while len(crops) % 12:
        crops.append(np.zeros((120, 60, 3), np.uint8))
    cv2.imwrite(a.figura, np.vstack([np.hstack(crops[i:i + 12]) for i in range(0, len(crops), 12)]))
    print(f"mosaico → {a.figura}")


if __name__ == "__main__":
    main()
