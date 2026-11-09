#!/usr/bin/env python3
"""
Homografia por SEGMENTOS de câmera principal + âncoras manuais + fluxo óptico (LK).
Adição da equipe (P3) — substitui `homografia.py camera/projetar` neste trecho.

Por que existe
--------------
No trecho Brasil × Coreia a transmissão corta muitas vezes (closes, replays, câmera
baixa de beira de campo). O `homografia.py camera` encadeia a homografia ATRAVÉS
desses cortes e, a partir de ~65 s, ela degenera (todos os pontos projetam no mesmo
lugar). Além disso, com uma única calibração no frame0, o erro de encadeamento e
de extrapolação passa de vários metros após poucos segundos de panorâmica.

Método
------
1. `segmentos`: divide o vídeo em segmentos de câmera principal (plano aberto),
   detectados pela altura mediana das caixas de pessoas em tracks.csv, e quebra
   nos cortes por histograma. Quadros fora dos segmentos ficam sem projeção (NaN).
2. `movimento`: dentro de cada segmento, M_t (quadro t → t-1) por Lucas-Kanade em
   cantos da imagem inteira (gramado + arquibancada), sem placar e sem pessoas,
   com checagem ida-e-volta + RANSAC. A câmera principal praticamente só gira e dá
   zoom, então uma homografia vale para a cena toda.
3. `calibrar`: cada âncora é um quadro com ≥ 4 marcos de campo.MARCOS marcados à
   mão (calib_segmentos/ancora_fNNNNN.json). Gera a grade de conferência.
4. `combinar`: cada quadro recebe H por encadeamento a partir da âncora anterior
   (para frente) e da seguinte (para trás); entre duas âncoras as duas estimativas
   são interpoladas linearmente no campo, o que limita a deriva do LK.
5. `projetar`: tracks_campo.csv e bola_campo.csv (mesmo formato do homografia.py).

Uso
---
python camera_segmentos.py segmentos --saida ../saidas/jogo1
python camera_segmentos.py movimento --saida ../saidas/jogo1
python camera_segmentos.py calibrar  --saida ../saidas/jogo1
python camera_segmentos.py combinar  --saida ../saidas/jogo1
python camera_segmentos.py projetar  --saida ../saidas/jogo1

Formato de ancora_fNNNNN.json: {"frame": 1234, "pixels": [[u, v], ...], "marcos": ["ga_dir_sup", ...]}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import ajuste_linhas as aj
from campo import COMPRIMENTO, LARGURA, MARCOS

# pontos de imagem usados para interpolar duas homografias
_REF = np.float64([[160, 260], [1120, 260], [160, 700], [1120, 700]])


def aplicar_h(H, pts):
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def _meta(pasta):
    return json.loads((pasta / "meta.json").read_text())


# ── 1. Segmentos ───────────────────────────────────────────────────────────

def cmd_segmentos(a):
    pasta = Path(a.saida)
    meta = _meta(pasta)
    n, fps = meta["n_frames"], meta["fps"]
    tr = pd.read_csv(pasta / "tracks.csv")
    p = tr[tr["classe"] != "bola"].copy()
    p["h"] = p["y2"] - p["y1"]
    g = p.groupby("frame").agg(hmed=("h", "median"), n=("h", "size")).reindex(range(n))
    aberto = ((g["hmed"] < a.h_max) & (g["n"] >= a.min_pessoas)).astype(float)
    aberto = aberto.rolling(int(fps) + 1, center=True, min_periods=1).mean() > 0.5
    # Replays: a transmissão esconde o placar. Exige o placar visível (comparado ao frame0).
    placar = _placar_visivel(meta, a.placar_ref, a.placar_tol)
    sem = (~placar).sum()
    print(f"Quadros sem placar (replays/transições): {sem} ({sem/fps:.1f} s)")
    aberto = aberto & pd.Series(placar, index=aberto.index)
    segs, ini = [], None
    for f, v in aberto.items():
        if v and ini is None:
            ini = f
        if not v and ini is not None:
            segs.append([ini, f - 1]); ini = None
    if ini is not None:
        segs.append([ini, n - 1])
    segs = [s for s in segs if (s[1] - s[0] + 1) >= a.min_dur * fps]
    segs = _quebrar_por_histograma(meta, segs, a.limiar_corte)

    out = []
    for i, (s0, s1) in enumerate(segs):
        out.append({"seg": i, "ini": s0, "fim": s1, "t_ini_s": round(s0 / fps, 2),
                    "t_fim_s": round(s1 / fps, 2), "dur_s": round((s1 - s0 + 1) / fps, 2)})
        print(f"seg {i:02d}: quadros {s0:5d}–{s1:5d}  ({s0/fps:6.1f}–{s1/fps:6.1f} s)")
    (pasta / "segmentos.json").write_text(json.dumps(out, indent=2))
    (pasta / "calib_segmentos").mkdir(exist_ok=True)
    tot = sum(s["dur_s"] for s in out)
    print(f"{len(out)} segmentos, {tot:.1f} s de câmera aberta ({100*tot*fps/n:.1f}% do clipe)")


def _placar_visivel(meta, caixa, tol):
    """True nos quadros em que o recorte `caixa` (x0 y0 x1 y1) do placar bate com o do frame0."""
    x0, y0, x1, y1 = caixa
    cap = cv2.VideoCapture(meta["video"])
    ok, fr = cap.read()
    ref = fr[y0:y1, x0:x1].astype(np.float32)
    vis = [True]
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        vis.append(float(np.abs(fr[y0:y1, x0:x1].astype(np.float32) - ref).mean()) < tol)
    return np.array(vis[:meta["n_frames"]] + [False] * max(0, meta["n_frames"] - len(vis)))


def _quebrar_por_histograma(meta, segs, limiar):
    cap = cv2.VideoCapture(meta["video"])
    fps = meta["fps"]
    novos = []
    for s0, s1 in segs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, s0)
        ant, ini = None, s0
        for f in range(s0, s1 + 1):
            ok, fr = cap.read()
            if not ok:
                break
            h = cv2.calcHist([cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)], [0, 1], None, [30, 32], [0, 180, 0, 256])
            cv2.normalize(h, h)
            if ant is not None and cv2.compareHist(ant, h, cv2.HISTCMP_CORREL) < limiar:
                if f - ini >= fps * 2:
                    novos.append([ini, f - 1])
                ini = f
            ant = h
        if s1 - ini >= fps * 2:
            novos.append([ini, s1])
    return novos


# ── 2. Movimento da câmera (LK) ────────────────────────────────────────────

def _mascara(shape, caixas, placar):
    h, w = shape[:2]
    m = np.full((h, w), 255, np.uint8)
    x0, y0, x1, y1 = placar
    m[y0:y1, x0:x1] = 0
    for bx1, by1, bx2, by2 in caixas.astype(int):
        m[max(by1 - 6, 0):by2 + 6, max(bx1 - 6, 0):bx2 + 6] = 0
    return m


def _passo(prev_g, cur_g, mask):
    """Homografia M (quadro atual → anterior) por LK + checagem ida-e-volta + RANSAC."""
    p0 = cv2.goodFeaturesToTrack(prev_g, maxCorners=1500, qualityLevel=0.005, minDistance=7, mask=mask)
    if p0 is None or len(p0) < 30:
        return None, 0
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_g, cur_g, p0, None, winSize=(21, 21), maxLevel=3)
    p0r, st2, _ = cv2.calcOpticalFlowPyrLK(cur_g, prev_g, p1, None, winSize=(21, 21), maxLevel=3)
    ok = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(p0 - p0r, axis=2).ravel() < 1.0)
    if ok.sum() < 30:
        return None, int(ok.sum())
    M, inl = cv2.findHomography(p1[ok], p0[ok], cv2.RANSAC, 1.0)
    if M is None:
        return None, 0
    return M, int(inl.sum())


def cmd_movimento(a):
    pasta = Path(a.saida)
    meta = _meta(pasta)
    tr = pd.read_csv(pasta / "tracks.csv")
    caixas = {f: g[["x1", "y1", "x2", "y2"]].to_numpy() for f, g in tr[tr["classe"] != "bola"].groupby("frame")}
    vazio = np.empty((0, 4))
    segs = json.loads((pasta / "segmentos.json").read_text())
    dmov = pasta / "calib_segmentos" / "movimento"
    dmov.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(meta["video"])
    log = []
    for s in segs:
        arq = dmov / f"seg_{s['seg']:02d}.npy"
        if arq.exists() and not a.refazer:
            continue
        M = np.tile(np.eye(3), (s["fim"] - s["ini"] + 1, 1, 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, s["ini"])
        g_ant, falhas = None, 0
        for k in tqdm(range(len(M)), desc=f"seg {s['seg']:02d}", leave=False):
            ok, fr = cap.read()
            if not ok:
                break
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            if g_ant is not None:
                m, inl = _passo(g_ant, g, _mascara(g.shape, caixas.get(s["ini"] + k - 1, vazio), a.placar))
                if m is None or inl < a.min_inliers:
                    falhas += 1
                else:
                    M[k] = m
            g_ant = g
        np.save(arq, M)
        log.append({"seg": s["seg"], "quadros": len(M), "falhas_lk": falhas})
        print(f"seg {s['seg']:02d}: {len(M)} quadros, {falhas} falhas de LK")
    if log:
        (pasta / "calib_segmentos" / "movimento_log.json").write_text(json.dumps(log, indent=2))


# ── 3. Âncoras ─────────────────────────────────────────────────────────────

def _ancoras(pasta):
    return sorted((json.loads(p.read_text(encoding="utf-8")) for p in (pasta / "calib_segmentos").glob("ancora_f*.json")),
                  key=lambda c: c["frame"])


def _linhas_modelo():
    L = [[(0, 0), (COMPRIMENTO, 0)], [(0, LARGURA), (COMPRIMENTO, LARGURA)],
         [(0, 0), (0, LARGURA)], [(COMPRIMENTO, 0), (COMPRIMENTO, LARGURA)],
         [(COMPRIMENTO / 2, 0), (COMPRIMENTO / 2, LARGURA)]]
    for x0, s in ((0, 1), (COMPRIMENTO, -1)):
        for d, hw in ((16.5, 20.16), (5.5, 9.16)):
            xa = x0 + s * d
            L += [[(x0, LARGURA / 2 - hw), (xa, LARGURA / 2 - hw)], [(xa, LARGURA / 2 - hw), (xa, LARGURA / 2 + hw)],
                  [(xa, LARGURA / 2 + hw), (x0, LARGURA / 2 + hw)]]
    out = [np.linspace(np.array(p0, float), np.array(p1, float), 80) for p0, p1 in L]
    t = np.linspace(0, 2 * np.pi, 160)
    out.append(np.c_[COMPRIMENTO / 2 + 9.15 * np.cos(t), LARGURA / 2 + 9.15 * np.sin(t)])
    for xp, s in ((11.0, 1), (COMPRIMENTO - 11.0, -1)):
        ang = np.linspace(-0.927, 0.927, 40)
        out.append(np.c_[xp + s * 9.15 * np.cos(ang), LARGURA / 2 + 9.15 * np.sin(ang)])
    return out


def desenhar_modelo(img, H_px_para_campo, cor=(255, 0, 255)):
    Hi = np.linalg.inv(H_px_para_campo)
    for p in _linhas_modelo():
        q = aplicar_h(Hi, p)
        # descarta pontos atrás da câmera / muito longe
        w = (Hi @ np.c_[p, np.ones(len(p))].T)[2]
        q = q[(w > 0) & (np.abs(q).max(axis=1) < 5000)]
        if len(q) > 1:
            cv2.polylines(img, [np.round(q).astype(np.int32)], False, cor, 2, cv2.LINE_AA)
    return img


def _ler_segmento(meta, s, guardar):
    """Lê os quadros do segmento e devolve {k: BGR} só para os índices em `guardar`."""
    cap = cv2.VideoCapture(meta["video"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, s["ini"])
    out = {}
    for k in range(s["fim"] - s["ini"] + 1):
        ok, fr = cap.read()
        if not ok:
            break
        if k in guardar:
            out[k] = fr
    return out


def _icp(H, fr, caixas, a):
    """Refino pelas linhas; devolve (H, aceito)."""
    Hn, info = aj.ajustar(fr, H, aj.mascara_invalida(fr.shape, caixas))
    if not info.get("ok"):
        return H, False
    d = np.linalg.norm(aplicar_h(Hn, _REF) - aplicar_h(H, _REF), axis=1).max()
    if d > a.max_corr or info["inliers_pct"] < 70:
        return H, False
    return Hn, True


def _propagar(H0, k0, k1, M, quadros, caixas_seg, a, st):
    """Encadeia H de k0 até k1 (inclusive), para frente ou para trás, com refino a cada a.icp_cada."""
    out = {k0: H0}
    H = H0.copy()
    passo = 1 if k1 >= k0 else -1
    for k in range(k0 + passo, k1 + passo, passo):
        H = H @ M[k] if passo == 1 else H @ np.linalg.inv(M[k + 1])
        if a.icp_cada and k in quadros and k % a.icp_cada == 0:
            H, ok = _icp(H, quadros[k], caixas_seg.get(k, np.empty((0, 4))), a)
            st["icp_ok" if ok else "icp_rej"] += 1
        out[k] = H
    return out


def _caixas(pasta):
    tr = pd.read_csv(pasta / "tracks.csv")
    return {f: g[["x1", "y1", "x2", "y2"]].to_numpy() for f, g in tr[tr["classe"] != "bola"].groupby("frame")}


def cmd_calibrar(a):
    """Resolve cada âncora: marcos + pontos em linhas (manuais) → refino automático pelas linhas."""
    pasta = Path(a.saida)
    meta = _meta(pasta)
    dcal = pasta / "calib_segmentos"
    segs = json.loads((pasta / "segmentos.json").read_text())
    caixas = _caixas(pasta)
    arqs = {json.loads(p.read_text(encoding="utf-8"))["frame"]: p for p in dcal.glob("ancora_f*.json")}
    linhas_csv = []
    for s in segs:
        fr_anc = sorted(f for f in arqs if s["ini"] <= f <= s["fim"])
        if not fr_anc:
            continue
        M = np.load(dcal / "movimento" / f"seg_{s['seg']:02d}.npy")
        ks = [f - s["ini"] for f in fr_anc]
        quadros = _ler_segmento(meta, s, set(range(0, s["fim"] - s["ini"] + 1, a.icp_cada or 10)) | set(ks))
        cx = {f - s["ini"]: c for f, c in caixas.items() if s["ini"] <= f <= s["fim"]}
        H_ant, k_ant = None, None
        for f, k in zip(fr_anc, ks):
            cal = json.loads(arqs[f].read_text(encoding="utf-8"))
            marcos = dict(zip(cal.get("marcos", []), [tuple(p) for p in cal.get("pixels", [])]))
            linhas = [(r, (u, v)) for u, v, r in cal.get("linhas", [])]
            if len(marcos) >= 4 and cal.get("usar_dlt", True):
                src = np.array(list(marcos.values()), float)
                dst = np.array([MARCOS[m] for m in marcos], float)
                H_init, _ = cv2.findHomography(src, dst, 0)
            elif "init_de" in cal:
                # estimativa inicial emprestada de outra âncora com enquadramento parecido
                H_init = np.array(json.loads(arqs[cal["init_de"]].read_text(encoding="utf-8"))["H"])
            elif H_ant is not None:
                st = {"icp_ok": 0, "icp_rej": 0}
                H_init = _propagar(H_ant, k_ant, k, M, quadros, cx, a, st)[k]
            else:
                raise SystemExit(f"âncora f{f}: primeira do segmento precisa de ≥ 4 marcos")
            H_man = aj.ajustar_manual(H_init, marcos, linhas) if (marcos or linhas) else H_init
            fr = quadros[k]
            H_fin, info = aj.ajustar(fr, H_man, aj.mascara_invalida(fr.shape, cx.get(k, np.empty((0, 4)))),
                                     marcos=marcos)
            if not info.get("ok"):
                H_fin = H_man
            em = [float(np.linalg.norm(aplicar_h(H_fin, [p])[0] - np.array(MARCOS[m]))) for m, p in marcos.items()]
            hv = np.ravel(H_fin / H_fin[2, 2])[:8]
            el = list(np.abs(aj._residuos(hv, linhas, np.empty((0, 2)), np.empty((0, 2)), 0))) if linhas else []
            cal["H"] = (H_fin / H_fin[2, 2]).tolist()
            cal["erro_marcos_m"] = [round(e, 3) for e in em]
            cal["erro_linhas_m"] = [round(float(e), 3) for e in el]
            cal["ajuste_linhas"] = {k2: (round(v, 3) if isinstance(v, float) else v) for k2, v in info.items()}
            arqs[f].write_text(json.dumps(cal, indent=2, ensure_ascii=False), encoding="utf-8")
            todos = em + el
            linhas_csv.append({"seg": s["seg"], "frame": f, "t_s": round(f / meta["fps"], 2),
                               "n_marcos": len(marcos), "n_pontos_linha": len(linhas),
                               "erro_medio_manual_m": round(float(np.mean(todos)), 3) if todos else None,
                               "erro_max_manual_m": round(float(np.max(todos)), 3) if todos else None,
                               "obs_auto": info.get("n_obs"),
                               "residuo_auto_mediano_m": round(info.get("residuo_mediano_m", float("nan")), 3),
                               "linhas_auto": ",".join(info.get("linhas", []))})
            print(f"âncora f{f:5d} (seg {s['seg']:02d}): manual {len(marcos)} marcos + {len(linhas)} pts-linha, "
                  f"erro manual médio {np.mean(todos) if todos else float('nan'):.2f} m | auto {info.get('n_obs')} obs, "
                  f"resíduo {info.get('residuo_mediano_m', float('nan')):.2f} m")
            img = desenhar_modelo(fr.copy(), H_fin)
            for (u, v) in marcos.values():
                cv2.circle(img, (int(u), int(v)), 6, (0, 255, 255), 2)
            for _, (u, v) in linhas:
                cv2.circle(img, (int(u), int(v)), 4, (255, 255, 0), -1)
            cv2.imwrite(str(dcal / f"grade_f{f:05d}.jpg"), img)
            H_ant, k_ant = H_fin, k
    pd.DataFrame(linhas_csv).to_csv(pasta / "calibracao_ancoras.csv", index=False)


# ── 4. Combinação ──────────────────────────────────────────────────────────

def _interp(Ha, Hb, w):
    """Interpola duas homografias pixel→campo pelos pontos de referência no campo."""
    pa, pb = aplicar_h(Ha, _REF), aplicar_h(Hb, _REF)
    return cv2.getPerspectiveTransform(_REF.astype(np.float32), ((1 - w) * pa + w * pb).astype(np.float32)).astype(np.float64)


def cmd_combinar(a):
    pasta = Path(a.saida)
    meta = _meta(pasta)
    n, fps = meta["n_frames"], meta["fps"]
    segs = json.loads((pasta / "segmentos.json").read_text())
    ancs = _ancoras(pasta)
    caixas = _caixas(pasta)
    Hs = np.full((n, 3, 3), np.nan)
    resumo, log = [], []
    for s in segs:
        an = [c for c in ancs if s["ini"] <= c["frame"] <= s["fim"] and "H" in c]
        if not an:
            continue
        M = np.load(pasta / "calib_segmentos" / "movimento" / f"seg_{s['seg']:02d}.npy")
        L = len(M)
        ks = [c["frame"] - s["ini"] for c in an]
        Ha = [np.array(c["H"]) for c in an]
        quadros = _ler_segmento(meta, s, set(range(0, L, a.icp_cada or 10)))
        cx = {f - s["ini"]: c for f, c in caixas.items() if s["ini"] <= f <= s["fim"]}
        st = {"icp_ok": 0, "icp_rej": 0}
        for k, H in _propagar(Ha[0], ks[0], 0, M, quadros, cx, a, st).items():
            Hs[s["ini"] + k] = H
        for k, H in _propagar(Ha[-1], ks[-1], L - 1, M, quadros, cx, a, st).items():
            Hs[s["ini"] + k] = H
        for j in range(len(ks) - 1):
            k0, k1 = ks[j], ks[j + 1]
            fw = _propagar(Ha[j], k0, k1, M, quadros, cx, a, st)
            bw = _propagar(Ha[j + 1], k1, k0, M, quadros, cx, a, st)
            for k in range(k0, k1 + 1):
                Hs[s["ini"] + k] = _interp(fw[k], bw[k], (k - k0) / (k1 - k0))
            d = np.linalg.norm(aplicar_h(fw[k1], _REF) - aplicar_h(Ha[j + 1], _REF), axis=1)
            resumo.append({"seg": s["seg"], "de_f": an[j]["frame"], "ate_f": an[j + 1]["frame"],
                           "dt_s": round((k1 - k0) / fps, 2), "deriva_media_m": round(float(d.mean()), 2),
                           "deriva_max_m": round(float(d.max()), 2)})
        log.append({"seg": s["seg"], "ancoras": len(an), **st})
        print(f"seg {s['seg']:02d}: {len(an)} âncora(s), {L} quadros, refino aceito {st['icp_ok']} / rejeitado {st['icp_rej']}")
    np.save(pasta / "homografias_campo.npy", Hs)
    pd.DataFrame(resumo).to_csv(pasta / "deriva_entre_ancoras.csv", index=False)
    (pasta / "calib_segmentos" / "combinar_log.json").write_text(json.dumps(log, indent=2))
    if resumo:
        print(pd.DataFrame(resumo).to_string(index=False))
    ok = np.isfinite(Hs[:, 0, 0])
    print(f"Quadros com homografia: {ok.sum()} / {n} ({100*ok.mean():.1f}%) → homografias_campo.npy")


# ── 5. Projeção ────────────────────────────────────────────────────────────

def cmd_projetar(a):
    pasta = Path(a.saida)
    Hs = np.load(pasta / "homografias_campo.npy")

    def para_campo(df):
        df = df.copy()
        xy = np.full((len(df), 2), np.nan)
        for f, ii in df.groupby("frame").indices.items():
            if f >= len(Hs) or not np.isfinite(Hs[f, 0, 0]):
                continue
            xy[ii] = aplicar_h(Hs[f], df.iloc[ii][["px", "py"]].to_numpy())
        df["x_m"], df["y_m"] = xy[:, 0], xy[:, 1]
        fora = (df["x_m"] < -5) | (df["x_m"] > COMPRIMENTO + 5) | (df["y_m"] < -5) | (df["y_m"] > LARGURA + 5)
        df.loc[fora, ["x_m", "y_m"]] = np.nan
        return df

    tracks = pd.read_csv(pasta / "tracks.csv")
    jog = para_campo(tracks[tracks["classe"] != "bola"])
    jog.to_csv(pasta / "tracks_campo.csv", index=False)
    print(f"Jogadores projetados: {jog['x_m'].notna().mean()*100:.1f}% válidos → tracks_campo.csv")
    if (pasta / "bola.csv").exists():
        bola = para_campo(pd.read_csv(pasta / "bola.csv"))
        bola.to_csv(pasta / "bola_campo.csv", index=False)
        print(f"Bola projetada: {bola['x_m'].notna().mean()*100:.1f}% válidos → bola_campo.csv")


def main():
    p = argparse.ArgumentParser(description="Homografia por segmentos de câmera principal")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("segmentos"); s.add_argument("--saida", required=True)
    s.add_argument("--h-max", type=float, default=75, help="altura mediana máx. das caixas (px) p/ plano aberto")
    s.add_argument("--min-pessoas", type=int, default=6)
    s.add_argument("--min-dur", type=float, default=4.0, help="duração mínima do segmento (s)")
    s.add_argument("--limiar-corte", type=float, default=0.6, help="correlação de histograma abaixo = corte")
    s.add_argument("--placar-ref", type=int, nargs=4, default=[165, 38, 200, 66],
                   help="recorte do placar (bandeira) comparado ao frame0 p/ detectar replays")
    s.add_argument("--placar-tol", type=float, default=25.0)
    m = sub.add_parser("movimento"); m.add_argument("--saida", required=True)
    m.add_argument("--min-inliers", type=int, default=40)
    m.add_argument("--placar", type=int, nargs=4, default=[0, 0, 560, 80], help="x0 y0 x1 y1 do placar")
    m.add_argument("--refazer", action="store_true")
    for nome in ("calibrar", "combinar"):
        q = sub.add_parser(nome); q.add_argument("--saida", required=True)
        q.add_argument("--icp-cada", type=int, default=10, help="refino pelas linhas a cada N quadros (0 = desliga)")
        q.add_argument("--max-corr", type=float, default=3.0, help="correção máxima aceita no refino (m)")
    sub.add_parser("projetar").add_argument("--saida", required=True)
    a = p.parse_args()
    {"segmentos": cmd_segmentos, "movimento": cmd_movimento, "calibrar": cmd_calibrar,
     "combinar": cmd_combinar, "projetar": cmd_projetar}[a.cmd](a)


if __name__ == "__main__":
    main()
