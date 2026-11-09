#!/usr/bin/env python3
"""
Anotador de passes no navegador (localhost) — adição da equipe (P3).

Serve uma página com o vídeo do clipe, a linha do tempo com as janelas de câmera
principal e botões para marcar cada passe em dois cliques: SAÍDA (time que passa)
e CHEGADA (time que recebe, ou bola fora). O resultado (certo / interceptado /
fora) sai automaticamente. Tudo é salvo em disco a cada alteração:

  saidas/jogo1/passes_anotados.csv            t_s,time,sucesso   (formato do validar_passes.py)
  saidas/jogo1/passes_anotados_detalhado.csv  saída, chegada, time que recebeu, resultado, obs

A página NÃO mostra os passes detectados pelo sistema (anotação independente).

Uso (da raiz do projeto):
    .venv\\Scripts\\python.exe src\\anotador_web\\servidor.py
    → abra http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent

CAMPOS_DET = ["id", "t_ini_s", "t_fim_s", "time", "time_recebe", "resultado", "sucesso", "seg", "obs"]


def ler_anotacoes(pasta: Path) -> list[dict]:
    det = pasta / "passes_anotados_detalhado.csv"
    simples = pasta / "passes_anotados.csv"
    out = []
    if det.exists():
        with open(det, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                out.append({
                    "id": r["id"], "t_ini": float(r["t_ini_s"]),
                    "t_fim": float(r["t_fim_s"]) if r["t_fim_s"] else None,
                    "time": int(r["time"]),
                    "time_recebe": int(r["time_recebe"]) if r["time_recebe"] not in ("", None) else None,
                    "resultado": r["resultado"], "obs": r.get("obs", ""),
                })
    elif simples.exists():
        with open(simples, encoding="utf-8") as f:
            for i, r in enumerate(csv.DictReader(f)):
                out.append({"id": f"importado{i}", "t_ini": float(r["t_s"]), "t_fim": None,
                            "time": int(r["time"]), "time_recebe": None,
                            "resultado": "certo" if int(r["sucesso"]) else "interceptado", "obs": "importado"})
    return out


def salvar_anotacoes(pasta: Path, anot: list[dict], segs: list[dict], fps: float):
    anot = sorted(anot, key=lambda a: a["t_ini"])

    def seg_de(t):
        f = round(t * fps)
        s = next((s for s in segs if s["ini"] <= f <= s["fim"]), None)
        return "" if s is None else s["seg"]

    tmp = pasta / "passes_anotados.csv.tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "time", "sucesso"])
        for a in anot:
            if a.get("resultado") in ("certo", "interceptado", "fora"):
                w.writerow([f"{a['t_ini']:.2f}", a["time"], 1 if a["resultado"] == "certo" else 0])
    os.replace(tmp, pasta / "passes_anotados.csv")

    tmp = pasta / "passes_anotados_detalhado.csv.tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS_DET)
        w.writeheader()
        for a in anot:
            w.writerow({"id": a["id"], "t_ini_s": f"{a['t_ini']:.2f}",
                        "t_fim_s": "" if a.get("t_fim") is None else f"{a['t_fim']:.2f}",
                        "time": a["time"], "time_recebe": "" if a.get("time_recebe") is None else a["time_recebe"],
                        "resultado": a.get("resultado", ""), "sucesso": 1 if a.get("resultado") == "certo" else 0,
                        "seg": seg_de(a["t_ini"]), "obs": (a.get("obs") or "").replace("\n", " ")})
    os.replace(tmp, pasta / "passes_anotados_detalhado.csv")


def criar_handler(pasta: Path, video: Path):
    segs = json.loads((pasta / "segmentos.json").read_text())
    meta = json.loads((pasta / "meta.json").read_text())
    fps = float(meta["fps"])
    times = json.loads((pasta / "times.json").read_text(encoding="utf-8-sig"))

    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            if "/video" not in (args[0] if args else ""):
                sys.stderr.write("%s\n" % (fmt % args))

        def _json(self, obj, code=200):
            b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b)

        def _arquivo(self, p: Path):
            b = p.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", (mimetypes.guess_type(p.name)[0] or "text/plain") + "; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b)

        def _video(self):
            tam = video.stat().st_size
            rng = self.headers.get("Range")
            ini, fim = 0, tam - 1
            if rng:
                m = re.match(r"bytes=(\d*)-(\d*)", rng)
                if m:
                    if m.group(1):
                        ini = int(m.group(1))
                        fim = int(m.group(2)) if m.group(2) else tam - 1
                    elif m.group(2):
                        ini = tam - int(m.group(2))
                fim = min(fim, tam - 1)
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {ini}-{fim}/{tam}")
            else:
                self.send_response(200)
            n = fim - ini + 1
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(n))
            self.end_headers()
            with open(video, "rb") as f:
                f.seek(ini)
                resta = n
                try:
                    while resta > 0:
                        bloco = f.read(min(1 << 20, resta))
                        if not bloco:
                            break
                        self.wfile.write(bloco)
                        resta -= len(bloco)
                except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                    pass

        def do_GET(self):
            caminho = self.path.split("?")[0]
            if caminho in ("/", "/index.html"):
                return self._arquivo(AQUI / "index.html")
            if caminho == "/video":
                return self._video()
            if caminho == "/api/config":
                return self._json({"fps": fps, "duracao_s": meta["n_frames"] / fps, "segmentos": segs,
                                   "offset_youtube_s": 251,
                                   "times": {"0": times["0"]["time"], "1": times["1"]["time"]}})
            if caminho == "/api/anotacoes":
                return self._json(ler_anotacoes(pasta))
            self.send_error(404)

        def do_POST(self):
            if self.path != "/api/anotacoes":
                return self.send_error(404)
            n = int(self.headers.get("Content-Length", 0))
            try:
                anot = json.loads(self.rfile.read(n).decode("utf-8"))
                salvar_anotacoes(pasta, anot, segs, fps)
            except Exception as e:  # noqa: BLE001
                return self._json({"ok": False, "erro": str(e)}, 400)
            self._json({"ok": True, "n": len(anot)})

    return H


def main():
    ap = argparse.ArgumentParser(description="Anotador de passes (localhost)")
    ap.add_argument("--saida", default=str(RAIZ / "saidas" / "jogo1"))
    ap.add_argument("--video", default=None, help="padrão: <saida>/rastreado_h264.mp4")
    ap.add_argument("--porta", type=int, default=8765)
    a = ap.parse_args()
    pasta = Path(a.saida).resolve()
    video = Path(a.video).resolve() if a.video else pasta / "rastreado_h264.mp4"
    if not video.exists():
        sys.exit(f"Vídeo não encontrado: {video}")
    srv = ThreadingHTTPServer(("127.0.0.1", a.porta), criar_handler(pasta, video))
    print(f"Anotador em http://127.0.0.1:{a.porta}  (Ctrl+C para sair)\n  vídeo: {video}\n  salva em: {pasta}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
