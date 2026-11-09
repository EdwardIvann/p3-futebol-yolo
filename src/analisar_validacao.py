#!/usr/bin/env python3
"""
Análise de erros da validação de passes — adição da equipe (P3).

Usa o MESMO casamento do validar_passes.py (guloso, mesmo time, tolerância em s)
e detalha onde o detector erra, para responder a Q5.3 ("erros dominantes"):

  * precisão / revocação por time;
  * falsos negativos (anotados não detectados) por resultado anotado
    (certo / interceptado / fora) e se a bola tinha posição no campo no trecho;
  * falsos positivos (detectados sem anotação) — se havia passe anotado do
    OUTRO time perto (troca de time) ou do mesmo time além da tolerância (tempo);
  * taxa de acerto de passe anotada × detectada, por time.

Saídas: analise_validacao_passes.txt e analise_validacao_passes.csv (um passe por linha).

Uso (de src/):
    python analisar_validacao.py --saida ../saidas/jogo1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from validar_passes import casar

NOME = {0: "Coreia", 1: "Brasil"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", required=True)
    ap.add_argument("--tolerancia", type=float, default=1.0)
    a = ap.parse_args()
    S = Path(a.saida)

    det = pd.read_csv(S / "passes.csv")
    gt_s = pd.read_csv(S / "passes_anotados.csv")
    gtd = pd.read_csv(S / "passes_anotados_detalhado.csv").sort_values("t_ini_s").reset_index(drop=True)
    gt = gt_s.sort_values("t_s").reset_index(drop=True)
    assert len(gt) == len(gtd) and np.allclose(gt.t_s, gtd.t_ini_s), "CSV simples e detalhado divergem"
    fps = json.loads((S / "meta.json").read_text())["fps"]
    bola = pd.read_csv(S / "bola_campo.csv").set_index("frame")
    bola_ok = bola["x_m"].notna()

    pares = casar(det, gt, a.tolerancia)
    det_ok = {i for i, _ in pares}
    gt_ok = {j for _, j in pares}

    linhas = []
    L = []
    p = L.append
    vp = len(pares)
    p(f"Tolerância {a.tolerancia:.1f} s, mesmo time. Detectados {len(det)} | Anotados {len(gt)} | Casados {vp}")
    p(f"Precisão {vp/len(det):.2f} | Revocação {vp/len(gt):.2f} | "
      f"F1 {2*vp/(len(det)+len(gt)):.2f}")
    p("")
    p("Por time:")
    for t in (1, 0):
        nd = int((det.time == t).sum()); ng = int((gt.time == t).sum())
        c = sum(1 for i, j in pares if det.loc[i, "time"] == t)
        p(f"  {NOME[t]}: detectados {nd}, anotados {ng}, casados {c} → precisão {c/nd:.2f}, revocação {c/ng:.2f}")
    p("")
    p("Taxa de acerto de passe (certos / total):")
    for t in (1, 0):
        g = gtd[gtd.time == t]; d = det[det.time == t]
        p(f"  {NOME[t]}: anotado {g.sucesso.sum()}/{len(g)} = {100*g.sucesso.mean():.1f}% | "
          f"detectado {d.sucesso.sum()}/{len(d)} = {100*d.sucesso.mean():.1f}%")
    p("")

    # falsos negativos
    fn = gtd.loc[[j for j in gtd.index if j not in gt_ok]].copy()

    def frac_bola(r):
        f0, f1 = int(round(r.t_ini_s * fps)), int(round(r.t_fim_s * fps))
        s = bola_ok.reindex(range(f0, f1 + 1), fill_value=False)
        return float(s.mean())
    fn["frac_bola_campo"] = fn.apply(frac_bola, axis=1)
    fn["dur_s"] = fn.t_fim_s - fn.t_ini_s
    ok_gt = gtd.loc[sorted(gt_ok)].copy()
    ok_gt["dur_s"] = ok_gt.t_fim_s - ok_gt.t_ini_s
    ok_gt["frac_bola_campo"] = ok_gt.apply(frac_bola, axis=1)
    p(f"Falsos negativos (anotados não detectados): {len(fn)} de {len(gt)}")
    for res, n in fn.resultado.value_counts().items():
        tot = int((gtd.resultado == res).sum())
        p(f"  {res}: {n} de {tot} anotados ({100*n/tot:.0f}% perdidos)")
    p(f"  bola com posição no campo em < 50% do trecho: {(fn.frac_bola_campo < .5).sum()} de {len(fn)}"
      f" (nos casados: {(ok_gt.frac_bola_campo < .5).sum()} de {len(ok_gt)})")
    p(f"  duração mediana do passe: perdidos {fn.dur_s.median():.2f} s × casados {ok_gt.dur_s.median():.2f} s")
    p(f"  passes < 0,7 s (toques curtos): perdidos {(fn.dur_s < .7).sum()} × casados {(ok_gt.dur_s < .7).sum()}")
    p("")

    # falsos positivos
    fp = det.loc[[i for i in det.index if i not in det_ok]].copy()

    def tipo_fp(r):
        dt = (gt.t_s - r.t_ini_s).abs()
        if ((dt <= a.tolerancia) & (gt.time != r.time)).any():
            return "time trocado (passe anotado do outro time ≤ tol.)"
        if ((dt <= 2 * a.tolerancia) & (gt.time == r.time)).any():
            return "tempo deslocado (mesmo time entre tol. e 2×tol.)"
        return "sem passe anotado perto (ruído de posse/bola)"
    fp["tipo"] = fp.apply(tipo_fp, axis=1) if len(fp) else []
    p(f"Falsos positivos (detectados sem anotação): {len(fp)} de {len(det)}")
    for k, n in fp.tipo.value_counts().items():
        p(f"  {k}: {n}")
    p("")
    acordo = [(det.loc[i, "sucesso"], gt.loc[j, "sucesso"]) for i, j in pares]
    disc = [x for x in acordo if x[0] != x[1]]
    p(f"Certo/errado nos casados: concordam {len(acordo)-len(disc)} de {len(acordo)}; "
      f"detector diz certo e anotação errado: {sum(1 for d, g in disc if d == 1)}; "
      f"o contrário: {sum(1 for d, g in disc if d == 0)}")

    for i, j in pares:
        linhas.append({"tipo": "casado", "t_det_s": round(det.loc[i, "t_ini_s"], 2), "t_anot_s": gt.loc[j, "t_s"],
                       "time": int(det.loc[i, "time"]), "sucesso_det": int(det.loc[i, "sucesso"]),
                       "resultado_anot": gtd.loc[j, "resultado"], "obs": ""})
    for j, r in fn.iterrows():
        linhas.append({"tipo": "falso_negativo", "t_det_s": None, "t_anot_s": r.t_ini_s, "time": int(r.time),
                       "sucesso_det": None, "resultado_anot": r.resultado,
                       "obs": f"bola no campo {100*r.frac_bola_campo:.0f}% do trecho; dur {r.dur_s:.2f} s"})
    for i, r in fp.iterrows():
        linhas.append({"tipo": "falso_positivo", "t_det_s": round(r.t_ini_s, 2), "t_anot_s": None,
                       "time": int(r.time), "sucesso_det": int(r.sucesso), "resultado_anot": None, "obs": r.tipo})
    pd.DataFrame(linhas).sort_values(["t_anot_s", "t_det_s"]).to_csv(S / "analise_validacao_passes.csv", index=False)
    txt = "\n".join(L)
    (S / "analise_validacao_passes.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
