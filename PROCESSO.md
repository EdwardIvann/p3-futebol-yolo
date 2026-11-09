# Como o processo foi feito — P3 Futebol Analytics (Brasil × Coreia do Sul)

Registro de execução: o que foi feito em cada etapa, as decisões tomadas, os desvios em relação à SPEC e como reproduzir/conferir. Execução em 09/11/2026, com apoio do Claude Code; as decisões marcadas **[equipe]** foram tomadas pela equipe.

---

## 1. Ambiente

| Item | Valor |
|---|---|
| Máquina | Windows 11 · Ryzen 5 5700X3D · RTX 4070 Super (driver 621.01) |
| Python | 3.13.17 (único instalado; a SPEC pedia 3.10–3.12, mas o PyTorch cu124 já tem wheels 3.13) |
| PyTorch | 2.6.0+cu124 — `torch.cuda.is_available() = True` |
| FFmpeg | 9.0.2 (instalado via `winget install Gyan.FFmpeg`) |
| Dependências | `.venv` + `src/requirements.txt` + `tabulate` |

## 2. Aquisição (Etapa 1)

- `baixar_video.py` com `--inicio 4.1834 --fim 9.4` (04:11 → 09:24); o yt-dlp reencodou o trecho em H.264 720p.
- Renomeado para `dados/videos/bra_kor_clip.mp4`. Medido com OpenCV (`video_info.json`): 1280×720, 30 fps, 9390 quadros, 313,0 s. Conferido que o último quadro é a bola entrando no gol.

## 3. Comparação de modelos (Etapa 2)

- 4 tamanhos de YOLOv8 + grade imgsz {640, 1280} × conf-bola {0,05; 0,10; 0,25} no subtrecho 0:19–0:49.
- Decisão: **yolov8m, imgsz 1280, conf-bola 0,10** (melhor cobertura de bola; yolov8l não melhorou e é 30% mais lento).
- Falsos positivos (Q2.3): candidatos de bola de menor confiança inspecionados em mosaico; os 3 salvos em `figuras/falsos_positivos_*.jpg` são roupas amarelas na torcida e um colete.

## 4. Detecção completa e times (Etapa 3)

- `detectar_rastrear.py` no clipe inteiro (26,1 fps) + execução `--k 3` no 1º minuto.
- Mapeamento dos times pelos centróides BGR → `times.json` (time 0 = Coreia/vermelho, time 1 = Brasil/amarelo). Conferido visualmente em quadros do `rastreado.mp4` (vermelhos = T0, amarelos = T1). **[equipe] confirmar assistindo ao vídeo.**
- Trocas de ID: IDs únicos por minuto (`ids_por_minuto.csv`) e picos de "nascimento" de IDs; os 3 picos foram inspecionados no vídeo e explicados em `trocas_id.json`.

## 5. Bola e homografia (Etapa 4) — principal desvio da SPEC

### 5.1 O que a SPEC previa e por que não funcionou
1. Calibração no frame0 com `meio_sup, centro, meio_inf, circulo_*` — **a linha do meio não aparece no frame0**. Usamos 5 outros marcos visíveis (canto, grande área, meia-lua, ponto direito do círculo) e adicionamos esses marcos a `campo.py`. Erro de reprojeção 0,02–0,25 m.
2. `homografia.py camera` + `projetar`: a homografia **degenera**: todos os jogadores projetam no mesmo ponto. Medido depois com `diagnostico_camera.py` → `camera_degeneracao.csv`: área projetada de um retângulo fixo cai a 59 px² na janela de 70–80 s e a 0 de 80 s até o fim. Causas: (a) o encadeamento Gt atravessa closes e replays; (b) ORB restrito ao gramado tem poucos pontos estáveis — saltos de 3–11 m entre quadros consecutivos já no 1º minuto.
3. O trecho tem **muito mais cortes do que a SPEC listava**: além dos closes conhecidos, câmera baixa de beira de campo (66 s, 81 s, 248 s, 295 s), closes em árbitro/jogadores e 3 replays (95,8–100,6 s, 200,9–210,1 s, 259,3–265,4 s).

**[equipe]** Entre "âncoras manuais em todos os segmentos" e "só 2 segmentos", a equipe escolheu **âncoras manuais**.

### 5.2 Solução implementada (`src/camera_segmentos.py` + `src/ajuste_linhas.py`)
1. **Segmentos de câmera principal** (`segmentos`): quadros com altura mediana das caixas < 75 px e ≥ 6 pessoas (plano aberto), **com o placar visível** (comparado ao frame0 — o placar some nos replays), quebrados em cortes por histograma (correlação < 0,6). Resultado: 12 segmentos, 222,3 s (71,0%).
2. **Movimento da câmera** (`movimento`): homografia quadro a quadro por fluxo óptico Lucas-Kanade em cantos da imagem inteira (gramado + arquibancada; sem placar e sem pessoas), checagem ida-e-volta + RANSAC. Como a câmera principal praticamente só gira e dá zoom, uma homografia vale para a cena toda. 0 falhas.
3. **Âncoras** (`calibrar`): 16 quadros com marcos marcados à mão (`calib_segmentos/ancora_f*.json`): cada marco foi localizado em zoom 4× com grade de pixels. Para âncoras só com marcos colineares (linha do meio + círculo), usamos também pontos sobre linhas/círculo e uma estimativa inicial de outra âncora parecida (`init_de`). Depois, o **ajuste pelas linhas** (`ajuste_linhas.py`) projeta o modelo FIFA, procura a linha branca real ao longo da normal (pico de brilho) e re-estima H por mínimos quadrados robustos (reta: distância 1D; círculo: |p−c|−r), com um prior fraco na estimativa inicial para não cair em soluções degeneradas.
4. **Propagação** (`combinar`): de cada âncora para frente e para trás pelo LK, com o ajuste pelas linhas a cada 10 quadros (aceito só se a correção < 3 m e ≥ 70% de inliers). Entre duas âncoras, as duas estimativas são interpoladas linearmente no campo.
5. **Projeção** (`projetar`): mesmo formato de `tracks_campo.csv` / `bola_campo.csv` do script original.

### 5.3 Validação da homografia
- Grade de cada âncora conferida visualmente (`figuras/jogo1/calibracao/`).
- Inspeção visual do modelo do campo sobreposto em quadros longe das âncoras (início/fim de cada segmento). Três problemas encontrados e corrigidos com âncoras extras: início do seg 08 (f6523), meio do seg 08 (f6900, onde o ajuste tinha aceitado uma correção errada) e início do seg 11 (f8901).
- **Validação cruzada**: propagar a partir de uma âncora até a seguinte e comparar com a calibração independente desta → deriva média 0,23 m, máx. 0,72 m (`deriva_entre_ancoras.csv`).
- Resíduo mediano nas linhas detectadas: 0,03 m; erro médio nos pontos manuais: 0,21 m (`calibracao_ancoras.csv`).
- Uma âncora (f6900) teve a primeira tentativa degenerada (DLT com 4 pontos mal condicionados e pontos da lateral lidos ~6 px errados); refeita com estimativa inicial de outra âncora.

### 5.4 Abordagens testadas e descartadas
- **ECC** (alinhamento por correlação entre modelo renderizado e máscara de linhas): rejeitava 593 de 594 correções — máscara de linhas ruim e bacia de convergência estreita.
- **LK com uma única âncora por segmento grande**: suave (0,13 m/quadro) mas deriva vários metros em ~5 s e herdava a extrapolação ruim da calibração do frame0.

## 6. Métricas (Etapa 5)
- `metricas.py --ataque-time0 esquerda` (Brasil ataca para a direita no 1º tempo).
- Plausibilidade: mediana 157 m/min (acima de 100–130; medimos só jogo corrido em câmera aberta, e o tremor do box soma distância); vel. máx. 26,8 km/h (tracks ≥ 10 s); nenhum > 35 km/h.
- Planilhas de apoio para a validação manual: `apoio_anotacao_passes.csv` (os 38 passes detectados com tempo do clipe e do YouTube) e `janelas_para_anotar.csv` (segmentos de câmera principal).
- **Validação manual dos passes:** a anotação foi feita no anotador web (`src/anotador_web/servidor.py` → http://127.0.0.1:8765), que mostra só as janelas de câmera principal e marca cada passe em 2 cliques (saída: time que passa; chegada: time que recebe ou "fora"; o resultado certo/interceptado/fora sai sozinho). Os passes detectados ficam escondidos para a anotação ser independente. Resultado: 65 passes anotados (`passes_anotados.csv` + `passes_anotados_detalhado.csv`; cópias em `saidas/jogo1/backup_anotacao/`, fora do git).
- `validar_passes.py` (tolerância 1 s, mesmo time) → precisão 0,63, revocação 0,37, F1 0,47; `analisar_validacao.py` separa os erros (`analise_validacao_passes.txt/.csv`): 41 falsos negativos (10 de 11 interceptações perdidas) e 14 falsos positivos (5 time trocado, 4 tempo deslocado, 5 sem passe perto).
- O servidor do anotador caiu duas vezes quando rodava pelo painel do app (o app encerra o processo); nada se perdeu porque cada passe é gravado em disco na hora. Depois ele foi rodado num processo próprio.

## 7. eXP (Etapa 6)
- `xpass.py --what-if`; superfície escolhida: passe da construção do gol (f9079, passador #14392).

## 8. Vídeo e figuras (Etapa 7)
- `video_minimapa.py --inicio-s 200 --dur-s 113` (termina no gol); vídeos reencodados para H.264 (`*_h264.mp4`) para abrir em navegador/PowerPoint.
- Figuras copiadas para `figuras/jogo1/` (incluindo as grades de todas as âncoras).

## 9. README e resumo (Etapa 8)
- `src/gerar_resumo.py` gera `saidas/jogo1/RESUMO_PARA_SLIDES.md` lendo só arquivos de `saidas/`; os números do README foram copiados de lá.
- Respostas analíticas deixadas como **RASCUNHO —** para a equipe reescrever.

## 10. Conferência antes de versionar
Antes de qualquer commit/push (Etapa 9), revisão conjunta com a equipe:
1. Reexecutar as etapas 5–8 (`metricas` → `gerar_resumo`) e conferir com um script local (fora do repositório) se os arquivos do checklist existem, se nenhum vídeo/peso escapa do `.gitignore` e se os números-chave do README batem com `saidas/`.
2. Conferir visualmente `rastreado_h264.mp4` e `demo_minimapa_h264.mp4`.
3. Só então `git init`, revisar `git status` e a mensagem de commit com a equipe.

## Pendências da equipe
- [x] Confirmar as cores dos times no `rastreado_h264.mp4` (confirmado: Brasil = T1, Coreia = T0; erros esperados na torcida e nos goleiros)
- [x] Anotar `saidas/jogo1/passes_anotados.csv` (65 passes) e rodar `validar_passes.py` + `analisar_validacao.py`
- [ ] Subir os vídeos no Google Drive e colocar o link no README
- [x] Nomes e e-mails @cesar.school da equipe
- [ ] Link dos slides (a equipe faz os slides por último, com tudo pronto)
- [x] Fora do repositório (via `.gitignore`, mantidos localmente): `.claude/`, `src/anotar_passes.py` (1º anotador, OpenCV, substituído pelo web), `saidas/jogo1_marcos_sugeridos.jpg`
- [x] Repositório criado (privado, vazio): https://github.com/EdwardIvann/p3-futebol-yolo
- [ ] Adicionar o professor (e a equipe) como colaboradores no GitHub
- [ ] Commit único no final, depois de revisar `git status` com a equipe
- [x] Link do Drive no README
- [x] Versão oficial das respostas (antes RASCUNHO), escrita com IA a partir dos arquivos de `saidas/` — **aguardando validação da equipe**. Para dar fonte a números novos foram criados `inspecionar_k3.py` (Q3.3 → `jogo1_k3/cluster_extra.txt`, `figuras/jogo1/k3_cluster_extra.jpg`), `diagnostico_camera.py` (Q4.3 → `camera_degeneracao.csv`) e a linha "jogadores visíveis por quadro" no RESUMO (Q5.1)
- [x] `SPEC.md` fora do repositório (`.gitignore`)
