#!/usr/bin/env python3
"""
Vídeo de demonstração para a apresentação (1–2 min):
rastreado.mp4 + minimapa 2D (homografia) + barra de posse acumulada + último passe.

Uso
---
python video_minimapa.py --saida ../saidas/jogo1 --inicio-s 30 --dur-s 90
Saída: <saida>/demo_minimapa.mp4   (NÃO versionar — subir no Google Drive)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from campo import COMPRIMENTO, LARGURA

COR = {0: (40, 40, 214), 1: (216, 78, 29)}  # BGR: vermelho / azul (iguais ao metricas.py)


def desenhar_campo_cv(esc: float, margem: int = 8) -> np.ndarray:
    w, h = int(COMPRIMENTO * esc) + 2 * margem, int(LARGURA * esc) + 2 * margem
    img = np.full((h, w, 3), (60, 130, 60), np.uint8)
    P = lambda x, y: (int(margem + x * esc), int(margem + y * esc))
    br, lw = (235, 235, 235), 1
    cv2.rectangle(img, P(0, 0), P(COMPRIMENTO, LARGURA), br, lw)
    cv2.line(img, P(COMPRIMENTO / 2, 0), P(COMPRIMENTO / 2, LARGURA), br, lw)
    cv2.circle(img, P(COMPRIMENTO / 2, LARGURA / 2), int(9.15 * esc), br, lw)
    for x0, s in ((0, 1), (COMPRIMENTO, -1)):
        cv2.rectangle(img, P(x0, LARGURA / 2 - 20.16), P(x0 + s * 16.5, LARGURA / 2 + 20.16), br, lw)
        cv2.rectangle(img, P(x0, LARGURA / 2 - 9.16), P(x0 + s * 5.5, LARGURA / 2 + 9.16), br, lw)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", required=True)
    ap.add_argument("--inicio-s", type=float, default=0)
    ap.add_argument("--dur-s", type=float, default=90)
    ap.add_argument("--escala", type=float, default=4.0, help="pixels por metro no minimapa")
    a = ap.parse_args()
    pasta = Path(a.saida)

    meta = json.loads((pasta / "meta.json").read_text())
    fps = meta["fps"]
    jog = pd.read_csv(pasta / "tracks_campo.csv").dropna(subset=["x_m", "y_m"])
    jog = jog[jog["time"] >= 0]
    # time do track por votação, como no metricas.py
    jog["time"] = jog["track_id"].map(jog.groupby("track_id")["time"].agg(lambda s: s.value_counts().idxmax()))
    bola = pd.read_csv(pasta / "bola_campo.csv").dropna(subset=["x_m", "y_m"]).set_index("frame")
    posse = pd.read_csv(pasta / "posse_por_quadro.csv").set_index("frame")
    passes = pd.read_csv(pasta / "passes.csv") if (pasta / "passes.csv").stat().st_size > 2 else pd.DataFrame()

    por_frame = {f: g for f, g in jog.groupby("frame")}
    base = desenhar_campo_cv(a.escala)
    mh, mw = base.shape[:2]
    m = 8
    P = lambda x, y: (int(m + x * a.escala), int(m + y * a.escala))

    cap = cv2.VideoCapture(str(pasta / "rastreado.mp4"))
    f0 = int(a.inicio_s * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(pasta / "demo_minimapa.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    cont = {0: 0, 1: 0}
    tp_ant = posse.loc[:f0, "time_posse"].dropna() if len(posse) else pd.Series(dtype=float)
    for t in (0, 1):
        cont[t] = int((tp_ant == t).sum())

    for f in range(f0, f0 + int(a.dur_s * fps)):
        ok, frame = cap.read()
        if not ok:
            break
        mini = base.copy()
        for r in por_frame.get(f, pd.DataFrame()).itertuples():
            cv2.circle(mini, P(r.x_m, r.y_m), 5, COR[int(r.time)], -1)
            cv2.circle(mini, P(r.x_m, r.y_m), 5, (255, 255, 255), 1)
        if f in bola.index:
            b = bola.loc[f]
            cor_b = (255, 255, 255) if b.get("fonte", "detectada") == "detectada" else (0, 220, 255)
            cv2.circle(mini, P(b["x_m"], b["y_m"]), 4, cor_b, -1)
        if len(passes):
            ult = passes[(passes["frame_fim"] <= f) & (passes["frame_fim"] > f - 2 * fps)]
            for r in ult.itertuples():
                cv2.arrowedLine(mini, P(r.x0, r.y0), P(r.x1, r.y1),
                                (94, 197, 34) if r.sucesso else (68, 68, 239), 2, tipLength=0.15)

        tp = posse["time_posse"].get(f, np.nan) if len(posse) else np.nan
        if not pd.isna(tp):
            cont[int(tp)] += 1
        tot = max(cont[0] + cont[1], 1)
        p0 = cont[0] / tot

        # Cola minimapa no canto inferior direito
        x0, y0 = W - mw - 12, H - mh - 12
        frame[y0:y0 + mh, x0:x0 + mw] = cv2.addWeighted(frame[y0:y0 + mh, x0:x0 + mw], 0.15, mini, 0.85, 0)
        # Barra de posse
        bw, bh, bx, by = mw, 22, x0, y0 - 30
        cv2.rectangle(frame, (bx, by), (bx + int(bw * p0), by + bh), COR[0], -1)
        cv2.rectangle(frame, (bx + int(bw * p0), by), (bx + bw, by + bh), COR[1], -1)
        cv2.putText(frame, f"Posse  T0 {100*p0:.0f}%  |  T1 {100*(1-p0):.0f}%", (bx + 6, by + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        out.write(frame)

    cap.release()
    out.release()
    print(f"OK → {pasta/'demo_minimapa.mp4'}  (suba no Drive; não versione)")


if __name__ == "__main__":
    main()
