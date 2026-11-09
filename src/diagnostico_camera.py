#!/usr/bin/env python3
"""
Diagnóstico da homografia por quadro do `homografia.py camera` (Q4.3) — adição da equipe (P3).

Projeta um retângulo fixo da imagem (o gramado na metade de baixo) com a
homografia de cada quadro em camera.npy e mede a área resultante em pixels do frame0 (camera.npy guarda Gt: quadro k → frame0;
sem movimento a área é a do próprio retângulo, 880 × 300 = 264 000 px²).
Quando a homografia degenera, todos os pontos caem no mesmo lugar e a área → 0.

Saída: <saida>/camera_degeneracao.csv (mediana por janela de 10 s) e um resumo no terminal.

Uso (de src/):
    python diagnostico_camera.py --saida ../saidas/jogo1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", required=True)
    ap.add_argument("--area-min", type=float, default=1.0, help="px²; abaixo disso a projeção é considerada degenerada")
    a = ap.parse_args()
    S = Path(a.saida)
    H = np.load(S / "camera.npy")
    fps = float(json.loads((S / "meta.json").read_text())["fps"])
    pts = np.array([[200, 400, 1], [1080, 400, 1], [1080, 700, 1], [200, 700, 1]], float).T

    area = np.full(len(H), np.nan)
    for k, h in enumerate(H):
        if not np.isfinite(h).all():
            continue
        p = h @ pts
        x, y = p[0] / p[2], p[1] / p[2]
        area[k] = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))  # shoelace

    t = np.arange(len(H)) / fps
    df = pd.DataFrame({"t_ini_s": (t // 10 * 10).astype(int), "area_px2": area})
    res = df.groupby("t_ini_s")["area_px2"].median().round(2).reset_index()
    res.columns = ["t_ini_s", "area_mediana_px2"]
    res.to_csv(S / "camera_degeneracao.csv", index=False)

    deg = res[res.area_mediana_px2 < a.area_min]
    primeiro = int(deg.t_ini_s.iloc[0]) if len(deg) else None
    sempre = primeiro is not None and (res[res.t_ini_s >= primeiro].area_mediana_px2 < a.area_min).all()
    print(res.to_string(index=False))
    print(f"\nPrimeira janela degenerada (< {a.area_min} px²): {primeiro} s"
          + (" — e continua degenerada até o fim" if sempre else ""))


if __name__ == "__main__":
    main()
