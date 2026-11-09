#!/usr/bin/env python3
"""
Q2.1 — Tabela comparando tamanhos de YOLO no MESMO subtrecho de 30 s.

Primeiro rode detectar_rastrear.py uma vez por modelo, cada um numa pasta:
    python detectar_rastrear.py --video V.mp4 --saida ../saidas/cmp_n --modelo yolov8n.pt --inicio 0 --fim 0.5 --sem-video
    python detectar_rastrear.py --video V.mp4 --saida ../saidas/cmp_m --modelo yolov8m.pt --inicio 0 --fim 0.5 --sem-video

Depois:
    python comparar_modelos.py ../saidas/cmp_n ../saidas/cmp_m --csv ../saidas/comparacao_modelos.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def resumir(pasta: Path) -> dict:
    meta = json.loads((pasta / "meta.json").read_text())
    tr = pd.read_csv(pasta / "tracks.csv")
    n = meta["n_frames"]
    pessoas = tr[tr["classe"] != "bola"]
    bola = tr[tr["classe"] == "bola"]
    return {
        "modelo": meta["modelo"], "imgsz": meta["imgsz"], "conf_bola": meta["conf_bola"],
        "device": meta["device"], "quadros": n,
        "fps_processamento": round(meta["fps_processamento"], 1),
        "pessoas_por_quadro": round(pessoas.groupby("frame").size().reindex(range(n), fill_value=0).mean(), 1),
        "ids_unicos": int(pessoas["track_id"].nunique()),
        "quadros_com_bola_pct": round(100 * bola["frame"].nunique() / n, 1) if n else 0,
        "conf_media_bola": round(float(bola.groupby("frame")["conf"].max().mean()), 3) if len(bola) else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pastas", nargs="+")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    tab = pd.DataFrame([resumir(Path(p)) for p in a.pastas])
    try:
        print(tab.to_markdown(index=False))  # requer o pacote tabulate
    except ImportError:
        print(tab.to_string(index=False))
    if a.csv:
        tab.to_csv(a.csv, index=False)


if __name__ == "__main__":
    main()
