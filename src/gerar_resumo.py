#!/usr/bin/env python3
"""
Gera saidas/<jogo>/RESUMO_PARA_SLIDES.md lendo SÓ os arquivos de saidas/ — adição da equipe (P3).

Assim todo número dos slides/README vem de um arquivo gerado (a SPEC penaliza
números divergentes). Rode de novo sempre que o pipeline for re-executado.

Uso: python gerar_resumo.py --saida ../saidas/jogo1
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def _j(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def _md(df):
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return df.to_string(index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default="../saidas/jogo1")
    a = ap.parse_args()
    S = Path(a.saida)
    raiz = S.parent
    L = []
    add = L.append

    vi, meta = _j(S / "video_info.json"), _j(S / "meta.json")
    times = _j(S / "times.json")
    nome = {0: times["0"]["time"], 1: times["1"]["time"]}

    add("# Resumo para os slides — Brasil × Coreia do Sul (Copa 2022, oitavas)\n")
    add("> Gerado automaticamente por `src/gerar_resumo.py` a partir de `saidas/`. Não editar à mão.\n")

    add("## 1. Vídeo e processamento")
    add(f"- URL: {vi['url']} — trecho {vi['intervalo_video']} do vídeo (placar 01:29 → 06:42)")
    add(f"- Resolução {vi['largura']}×{vi['altura']} · {vi['fps']:.0f} fps · {vi['n_quadros_lidos']} quadros · {vi['duracao_s']:.1f} s")
    add(f"- Detecção+rastreamento ({meta['modelo']}, imgsz {meta['imgsz']}, {meta['device']}): "
        f"{meta['fps_processamento']:.1f} fps → {meta['n_frames']/meta['fps_processamento']:.0f} s de processamento\n")

    add("## 2. Comparação de modelos (subtrecho de 30 s, 0:19–0:49 do clipe)")
    cm = pd.read_csv(raiz / "comparacao_modelos.csv")
    add(_md(cm[["modelo", "fps_processamento", "pessoas_por_quadro", "ids_unicos", "quadros_com_bola_pct", "conf_media_bola"]]))
    add("")
    ci = pd.read_csv(raiz / "comparacao_imgsz_conf.csv").sort_values(["imgsz", "conf_bola"])
    add("imgsz × conf-bola (yolov8m):\n")
    add(_md(ci[["imgsz", "conf_bola", "fps_processamento", "quadros_com_bola_pct", "conf_media_bola"]]))
    add("")

    add("## 3. Rastreamento e times")
    idm = pd.read_csv(S / "ids_por_minuto.csv")
    add(_md(idm))
    tr = _j(S / "trocas_id.json")
    add(f"\n- Esperado ≈ 22 jogadores + 3 árbitros; mediana de vida de um ID = {tr['mediana_duracao_id_s']} s; total de IDs = {tr['total_ids']}")
    for e in tr["exemplos"]:
        add(f"  - frame {e['frame']} ({e['t_s']} s, {e['novos_ids_no_segundo']} IDs novos em 1 s): {e['motivo']}")
    add(f"- Centróides (BGR, k=2): time 0 = {times['0']['centroide_bgr']} → **{nome[0]}** ({times['0']['uniforme']}); "
        f"time 1 = {times['1']['centroide_bgr']} → **{nome[1]}** ({times['1']['uniforme']})")
    k3 = _j(raiz / "jogo1_k3" / "meta.json")["times_bgr"]
    add(f"- k=3 (1º minuto): {k3}\n")

    add("## 4. Bola e homografia")
    log = (S / "bola_kalman_log.txt").read_text(encoding="utf-8-sig")
    det = re.search(r"\(([\d.]+)%\)", log).group(1)
    fin = re.search(r"Cobertura final\s*:\s*([\d.]+)%", log).group(1)
    acc = int(re.search(r"gating\s*:\s*(\d+)", log).group(1))
    add(f"- Bola: YOLO em {det}% dos quadros (com falsos positivos) → {100*acc/meta['n_frames']:.1f}% aceitos pelo gating → **{fin}%** após Kalman")
    segs = _j(S / "segmentos.json")
    dur = sum(s["dur_s"] for s in segs)
    add(f"- Câmera principal: {len(segs)} segmentos, {dur:.1f} s ({100*dur/vi['duracao_s']:.1f}% do clipe); replays/closes sem projeção")
    ca = pd.read_csv(S / "calibracao_ancoras.csv")
    add(f"- Âncoras de calibração: **{len(ca)}**; marcos manuais por âncora: {int(ca.n_marcos.min())}–{int(ca.n_marcos.max())}; "
        f"erro médio nos pontos manuais = **{ca.erro_medio_manual_m.mean():.2f} m**; resíduo mediano nas linhas detectadas = "
        f"**{ca.residuo_auto_mediano_m.median():.2f} m** ({int(ca.obs_auto.median())} pontos de linha por âncora)")
    dv = pd.read_csv(S / "deriva_entre_ancoras.csv")
    if len(dv):
        add(f"- Deriva entre âncoras consecutivas (validação cruzada): média {dv.deriva_media_m.mean():.2f} m, máx {dv.deriva_max_m.max():.2f} m")
    tc = pd.read_csv(S / "tracks_campo.csv")
    bc = pd.read_csv(S / "bola_campo.csv")
    add(f"- Projeção válida: jogadores {100*tc.x_m.notna().mean():.1f}% · bola {100*bc.x_m.notna().mean():.1f}% das linhas\n")

    add("## 5. Métricas")
    mt = _j(S / "metricas_times.json")
    mj = pd.read_csv(S / "metricas_jogadores.csv")
    rows = []
    for t in (0, 1):
        g = mj[(mj.time == t) & (mj.tempo_s >= 10)]   # tracks curtos têm velocidade ruidosa
        rows.append({"time": f"{t} ({nome[t]})", "posse_%": mt[f"time{t}"]["posse_pct"],
                     "passes": mt[f"time{t}"]["passes"], "certos": mt[f"time{t}"]["passes_certos"],
                     "precisao_%": mt[f"time{t}"]["precisao_pct"],
                     "dist_total_m": round(mt[f"time{t}"]["distancia_total"], 1),
                     "vel_max_kmh_tracks_10s": round(g.vel_max_kmh.max(), 1)})
    add(_md(pd.DataFrame(rows)))
    g10 = mj[mj.tempo_s >= 10]
    add(f"\n- Raio de posse {mt['raio_posse']} m; ataque: time 0 → {mt['direcao_ataque']['0']}, time 1 → {mt['direcao_ataque']['1']}")
    add(f"- Plausibilidade (tracks ≥ 10 s, n = {len(g10)}): mediana {np.median(g10.distancia / g10.tempo_s * 60):.0f} m/min; "
        f"vel. máx. {g10.vel_max_kmh.max():.1f} km/h; incluindo tracks curtos, máx. {mj.vel_max_kmh.max():.1f} km/h e "
        f"{int((mj.vel_max_kmh > 35).sum())} acima de 35 km/h")
    add("- Distância total por time soma TODOS os IDs (inclui fragmentos de trocas de ID e só cobre os 71% de câmera principal)")
    vis = tc[tc.x_m.notna() & tc.time.isin([0, 1])]
    pq = vis.groupby(["time", "frame"]).size().groupby("time").mean()
    add(f"- Jogadores visíveis por quadro (projeção válida, média): {nome[0]} {pq[0]:.2f} · {nome[1]} {pq[1]:.2f}")
    sr = pd.read_csv(S / "sensibilidade_raio.csv")
    add("\nSensibilidade ao raio de posse:\n")
    add(_md(sr))
    vp = S / "validacao_passes.txt"
    pa = S / "passes_anotados.csv"
    n_anot = max(len(pd.read_csv(pa)), 0) if pa.exists() else 0
    if vp.exists() and n_anot:
        add(f"\n- Validação manual ({n_anot} passes anotados):\n```\n{vp.read_text(encoding='utf-8-sig').strip()}\n```")
        av = S / "analise_validacao_passes.txt"
        if av.exists():
            add(f"\n- Análise de erros (`analise_validacao_passes.txt`):\n```\n{av.read_text(encoding='utf-8').strip()}\n```")
    else:
        add("\n- Validação manual: **PENDENTE** (aguardando `passes_anotados.csv` da equipe)")
    add("")

    add("## 6. eXP")
    rx = pd.read_csv(S / "resultados_xp.csv").rename(columns={"Unnamed: 0": "modelo"})
    pf = pd.read_csv(S / "passes_features.csv")
    add(f"{len(pf)} passes ({100*pf['sucesso'].mean():.1f}% certos):\n")
    add(_md(rx.round(3)))
    wi = pd.read_csv(S / "what_if.csv")
    add(f"\n- What-if: {len(wi)} passes errados reavaliados\n")

    add("## 7. Destaques táticos (`metricas_taticas.json`)")
    tt = _j(S / "metricas_taticas.json")
    add(f"- **Field tilt**: {nome[1]} {tt['time1']['field_tilt_pct']}% × {nome[0]} {tt['time0']['field_tilt_pct']}%")
    add(f"- **Linha defensiva sem bola** (m do próprio gol): {nome[1]} {tt['time1']['linha_defensiva_m_sem_bola']} × {nome[0]} {tt['time0']['linha_defensiva_m_sem_bola']}")
    add(f"- **Pressão média sobre o portador** (rivais a < 5 m): {nome[1]} sofre {tt['time1']['pressao_media_sobre_portador']}, "
        f"{nome[0]} sofre {tt['time0']['pressao_media_sobre_portador']}")
    add(f"- Altura média das recuperações: {nome[1]} {tt['time1']['altura_media_recuperacao_m']} m × {nome[0]} {tt['time0']['altura_media_recuperacao_m']} m")
    add(f"- Posse no terço ofensivo: {nome[1]} {tt['time1']['posse_por_terco_pct']['ofensivo']}% × {nome[0]} {tt['time0']['posse_por_terco_pct']['ofensivo']}%\n")

    add("## 8. Figuras para os slides")
    figs = sorted((raiz.parent / "figuras").rglob("*.*"))
    for f in figs:
        if f.suffix.lower() in (".png", ".jpg"):
            add(f"- `{f.relative_to(raiz.parent).as_posix()}`")
    (S / "RESUMO_PARA_SLIDES.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"→ {S / 'RESUMO_PARA_SLIDES.md'}")


if __name__ == "__main__":
    main()
