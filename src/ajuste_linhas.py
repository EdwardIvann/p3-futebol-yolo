"""
Ajuste fino de uma homografia pixel→campo pelas LINHAS BRANCAS do campo — adição da equipe (P3).

Dada uma estimativa inicial H (ex.: encadeada por fluxo óptico a partir de uma
âncora), projeta as linhas do modelo FIFA na imagem, procura a linha branca real
ao longo da normal de cada ponto amostrado (pico de brilho em relação ao gramado
vizinho) e re-estima H por mínimos quadrados robustos:
  - ponto em reta conhecida (ex.: lateral y = 0)  → resíduo 1D (distância à reta, em m);
  - ponto no círculo central / meia-lua           → resíduo |p - c| - r;
  - marcos pontuais (opcional, com peso maior)    → resíduo 2D.
Itera algumas vezes (nova projeção → nova busca), como um ICP de linhas.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import least_squares

from campo import COMPRIMENTO as C, LARGURA as L, MARCOS

_GA, _PA = 20.16, 9.16
# (tipo, parâmetros, intervalo do parâmetro livre em metros)
RETAS = {
    "lateral_sup": ("y", 0.0, (0, C)), "lateral_inf": ("y", L, (0, C)),
    "fundo_esq": ("x", 0.0, (0, L)), "fundo_dir": ("x", C, (0, L)),
    "meio": ("x", C / 2, (0, L)),
    "ga_esq": ("x", 16.5, (L / 2 - _GA, L / 2 + _GA)), "ga_dir": ("x", C - 16.5, (L / 2 - _GA, L / 2 + _GA)),
    "ga_esq_sup": ("y", L / 2 - _GA, (0, 16.5)), "ga_esq_inf": ("y", L / 2 + _GA, (0, 16.5)),
    "ga_dir_sup": ("y", L / 2 - _GA, (C - 16.5, C)), "ga_dir_inf": ("y", L / 2 + _GA, (C - 16.5, C)),
    "pa_esq": ("x", 5.5, (L / 2 - _PA, L / 2 + _PA)), "pa_dir": ("x", C - 5.5, (L / 2 - _PA, L / 2 + _PA)),
    "pa_esq_sup": ("y", L / 2 - _PA, (0, 5.5)), "pa_esq_inf": ("y", L / 2 + _PA, (0, 5.5)),
    "pa_dir_sup": ("y", L / 2 - _PA, (C - 5.5, C)), "pa_dir_inf": ("y", L / 2 + _PA, (C - 5.5, C)),
}
CIRCULOS = {  # centro, raio, intervalo angular (rad) visível no campo
    "central": ((C / 2, L / 2), 9.15, (0, 2 * np.pi)),
    "meia_lua_esq": ((11.0, L / 2), 9.15, (-0.927, 0.927)),
    "meia_lua_dir": ((C - 11.0, L / 2), 9.15, (np.pi - 0.927, np.pi + 0.927)),
}


def aplicar_h(H, pts):
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def _amostras_modelo(passo_m=1.0):
    """Pontos (em metros) amostrados sobre cada linha do modelo, com o rótulo da linha."""
    out = []
    for nome, (eixo, v, (a, b)) in RETAS.items():
        t = np.arange(a, b + 1e-6, passo_m)
        pts = np.c_[np.full_like(t, v), t] if eixo == "x" else np.c_[t, np.full_like(t, v)]
        out += [(nome, p) for p in pts]
    for nome, (c, r, (a0, a1)) in CIRCULOS.items():
        n = int(abs(a1 - a0) * r / passo_m)
        for th in np.linspace(a0, a1, max(n, 8)):
            out.append((nome, np.array([c[0] + r * np.cos(th), c[1] + r * np.sin(th)])))
    return out


_AMOSTRAS = None


def _brilho(gray_f, pts):
    """Interpolação bilinear do brilho em pontos (N×2)."""
    m = cv2.remap(gray_f, pts[:, 0].astype(np.float32).reshape(-1, 1), pts[:, 1].astype(np.float32).reshape(-1, 1),
                  cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return m.ravel()


def buscar_linhas(frame_bgr, H, invalido=None, raio_px=12, contraste_min=18.0):
    """Para cada ponto do modelo projetado na imagem, acha a linha branca ao longo da normal.

    Retorna lista de (rotulo, (u, v)) com os pixels encontrados.
    """
    global _AMOSTRAS
    if _AMOSTRAS is None:
        _AMOSTRAS = _amostras_modelo()
    h, w = frame_bgr.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), (3, 3), 0).astype(np.float32)
    Hi = np.linalg.inv(H)
    achados = []
    eps = 0.3
    for rot, p in _AMOSTRAS:
        # projeta o ponto e um vizinho ao longo da linha para obter a tangente na imagem
        if rot in RETAS:
            eixo = RETAS[rot][0]
            d = np.array([0.0, eps]) if eixo == "x" else np.array([eps, 0.0])
        else:
            c = np.array(CIRCULOS[rot][0]); r = p - c
            d = np.array([-r[1], r[0]]) / np.linalg.norm(r) * eps
        q = aplicar_h(Hi, np.vstack([p, p + d]))
        w3 = (Hi @ np.r_[p, 1.0])[2]
        if w3 <= 0 or not np.all(np.isfinite(q)):
            continue
        u, v = q[0]
        if not (raio_px + 2 <= u < w - raio_px - 2 and raio_px + 2 <= v < h - raio_px - 2):
            continue
        if invalido is not None and invalido[int(v), int(u)]:
            continue
        t = q[1] - q[0]
        nt = np.linalg.norm(t)
        if nt < 1e-6:
            continue
        nrm = np.array([-t[1], t[0]]) / nt
        s = np.arange(-raio_px, raio_px + 1, 0.5)
        perfil = _brilho(gray, q[0] + s[:, None] * nrm)
        k = int(np.argmax(perfil))
        if k < 4 or k > len(s) - 5:
            continue
        fundo = np.median(np.r_[perfil[:6], perfil[-6:]])
        if perfil[k] - fundo < contraste_min:
            continue
        # refino sub-pixel (parábola)
        a, b, c3 = perfil[k - 1], perfil[k], perfil[k + 1]
        den = a - 2 * b + c3
        off = 0.5 * (a - c3) / den if abs(den) > 1e-6 else 0.0
        achados.append((rot, tuple(q[0] + (s[k] + off * 0.5) * nrm)))
    return achados


_REF = np.float64([[160, 260], [640, 260], [1120, 260], [160, 700], [640, 700], [1120, 700], [640, 480]])


def _residuos(hv, obs, marcos_px, marcos_m, peso_marco, prior=None, peso_prior=0.0):
    H = np.append(hv, 1.0).reshape(3, 3)
    res = []
    if prior is not None and peso_prior > 0:
        res += list((peso_prior * (aplicar_h(H, _REF) - prior)).ravel())
    if obs:
        rots, px = zip(*obs)
        P = aplicar_h(H, np.array(px))
        for rot, (x, y) in zip(rots, P):
            if rot in RETAS:
                eixo, v, _ = RETAS[rot]
                res.append((x - v) if eixo == "x" else (y - v))
            else:
                c, r, _ = CIRCULOS[rot]
                res.append(np.hypot(x - c[0], y - c[1]) - r)
    if len(marcos_px):
        P = aplicar_h(H, marcos_px)
        res += list((peso_marco * (P - marcos_m)).ravel())
    return np.array(res)


def ajustar(frame_bgr, H0, invalido=None, marcos=None, iteracoes=4, raios=(14, 9, 6, 5), peso_marco=3.0,
            peso_prior=0.15):
    """Refina H0 (pixel→campo). Retorna (H, info)."""
    marcos = marcos or {}
    mpx = np.array([v for v in marcos.values()], float).reshape(-1, 2)
    mm = np.array([MARCOS[k] for k in marcos], float).reshape(-1, 2)
    H = H0 / H0[2, 2]
    prior = aplicar_h(H, _REF)   # a estimativa inicial segura as direções mal observadas
    info = {}
    for it in range(iteracoes):
        obs = buscar_linhas(frame_bgr, H, invalido, raio_px=raios[min(it, len(raios) - 1)])
        if len(obs) < 12:
            info = {"ok": False, "n_obs": len(obs)}
            return H0, info
        r = least_squares(_residuos, H.ravel()[:8], loss="soft_l1", f_scale=0.3,
                          args=(obs, mpx, mm, peso_marco, prior, peso_prior))
        H = np.append(r.x, 1.0).reshape(3, 3)
        res = _residuos(r.x, obs, np.empty((0, 2)), np.empty((0, 2)), 0)
        linhas = sorted({o[0] for o in obs})
        info = {"ok": True, "n_obs": len(obs), "linhas": linhas,
                "residuo_mediano_m": float(np.median(np.abs(res))),
                "inliers_pct": float(np.mean(np.abs(res) < 0.5) * 100)}
    return H, info


def ajustar_manual(H_init, marcos, pontos_linha, peso_marco=1.0, peso_prior=0.02):
    """Estima H a partir de marcos pontuais + pontos clicados sobre linhas/círculos conhecidos.

    pontos_linha: lista de (rotulo, (u, v)) com rótulos de RETAS ou CIRCULOS.
    H_init: estimativa inicial (ex.: encadeada da âncora anterior ou DLT dos marcos).
    """
    mpx = np.array([v for v in marcos.values()], float).reshape(-1, 2)
    mm = np.array([MARCOS[k] for k in marcos], float).reshape(-1, 2)
    H = H_init / H_init[2, 2]
    prior = aplicar_h(H, _REF)
    r = least_squares(_residuos, H.ravel()[:8], loss="soft_l1", f_scale=0.5,
                      args=(list(pontos_linha), mpx, mm, peso_marco, prior, peso_prior))
    return np.append(r.x, 1.0).reshape(3, 3)


def mascara_invalida(shape, caixas, placar=(0, 0, 560, 80), margem=4):
    inv = np.zeros(shape[:2], bool)
    x0, y0, x1, y1 = placar
    inv[y0:y1, x0:x1] = True
    for bx1, by1, bx2, by2 in np.asarray(caixas, int).reshape(-1, 4):
        inv[max(by1 - margem, 0):by2 + margem, max(bx1 - margem, 0):bx2 + margem] = True
    return inv
