# P3 — Futebol Analytics com YOLO

Visão Computacional · CESAR School · 2026.2

Pipeline que parte de um trecho de jogo do YouTube, detecta e rastreia jogadores e bola (YOLO + ByteTrack), separa os times por cor, projeta tudo no campo em metros (homografia) e calcula métricas de jogo: posse, passes, eXP (expected pass) e um conjunto de **métricas táticas** propostas pelo grupo.

> **Resumo numérico único:** [`saidas/jogo1/RESUMO_PARA_SLIDES.md`](saidas/jogo1/RESUMO_PARA_SLIDES.md), gerado por `src/gerar_resumo.py` só a partir dos arquivos de `saidas/`. Todos os números abaixo vêm de lá.
> **Como o processo foi feito (decisões e desvios):** [`PROCESSO.md`](PROCESSO.md).

## Equipe

| Nome | E-mail |
|---|---|
| Ivan Edward | `iers@cesar.school` |
| Davi Melo | `dam3@cesar.school` |
| Elizabete Barbosa | `eba@cesar.school` |

## Links

- **Vídeo original:** https://www.youtube.com/watch?v=JIQnnYy7qYk (FIFA, partida completa Brasil × Coreia do Sul, oitavas da Copa 2022) — intervalo usado: **04:11 → 09:24** do vídeo (placar 01:29 → 06:42; termina no gol de Vinícius Jr.)
- **Vídeos gerados (Google Drive):** https://drive.google.com/drive/folders/1at7bCFBhKabAbtkR-KcMoMbRg1WRYp8K?usp=sharing (`rastreado_h264.mp4` = `rastreado.mp4` e `demo_minimapa_h264.mp4` = `demo_minimapa.mp4`, convertidos para H.264 para tocar no navegador e no PowerPoint — mesmos quadros e duração)
- **Slides:** LINK_DOS_SLIDES

> Vídeos **não** são versionados aqui (ver `.gitignore`).

## Pipeline

```
YouTube ─► baixar_video.py ─► bra_kor_clip.mp4 (Drive)
              │
   detectar_rastrear.py   YOLO (pessoa + bola) · ByteTrack · K-Means de cor
              │
   bola_kalman.py         Kalman + gating na bola
   camera_segmentos.py    (grupo) segmentos de câmera principal · âncoras · LK · refino pelas linhas → pixel → metro
     └ ajuste_linhas.py   (grupo) ajuste da homografia às linhas brancas do campo
              │
   metricas.py            posse · passes · distância/velocidade
   metricas_taticas.py    compactação · linha defensiva · território · pressão · recuperações
   sensibilidade_raio.py  posse/passes × raio de posse
   validar_passes.py      precisão/revocação/F1 contra anotação manual
   xpass.py               eXP (baseline · heurístico · logístico) · superfície · what-if · PAx
              │
   video_minimapa.py      vídeo da apresentação: rastreado + minimapa + posse
   gerar_resumo.py        (grupo) RESUMO_PARA_SLIDES.md a partir de saidas/
```

## Como rodar

Ambiente usado: Windows 11, Ryzen 5 5700X3D, RTX 4070 Super, Python 3.13, torch 2.6.0+cu124, FFmpeg 9.0.2.

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r src\requirements.txt tabulate
cd src
```

### Comandos exatos executados (a partir de `src\`)

```powershell
# Etapa 1 — aquisição (renomeado depois para ..\dados\videos\bra_kor_clip.mp4)
python baixar_video.py "https://www.youtube.com/watch?v=JIQnnYy7qYk" --inicio 4.1834 --fim 9.4 --saida ..\dados\videos
$V="..\dados\videos\bra_kor_clip.mp4"

# Etapa 2 — comparação de modelos (subtrecho 0:19–0:49 do clipe)
foreach ($m in "yolov8n","yolov8s","yolov8m","yolov8l") {
  python detectar_rastrear.py --video $V --saida ..\saidas\cmp_$m --modelo "$m.pt" --inicio 0.3167 --fim 0.8167 --sem-video }
python comparar_modelos.py ..\saidas\cmp_yolov8n ..\saidas\cmp_yolov8s ..\saidas\cmp_yolov8m ..\saidas\cmp_yolov8l --csv ..\saidas\comparacao_modelos.csv
foreach ($i in 640,1280) { foreach ($c in 0.05,0.10,0.25) {
  python detectar_rastrear.py --video $V --saida ..\saidas\cmp_img${i}_cb$c --modelo yolov8m.pt --imgsz $i --conf-bola $c --inicio 0.3167 --fim 0.8167 --sem-video }}
python comparar_modelos.py (Get-ChildItem ..\saidas -Directory -Filter "cmp_img*").FullName --csv ..\saidas\comparacao_imgsz_conf.csv

# Etapa 3 — detecção completa + times
python detectar_rastrear.py --video $V --saida ..\saidas\jogo1 --modelo yolov8m.pt --imgsz 1280
python detectar_rastrear.py --video $V --saida ..\saidas\jogo1_k3 --modelo yolov8m.pt --k 3 --inicio 0 --fim 1
python inspecionar_k3.py --saida ..\saidas\jogo1_k3 --video $V      # cluster_extra.txt + figuras\jogo1\k3_cluster_extra.jpg

# Etapa 4 — bola e homografia
python bola_kalman.py --saida ..\saidas\jogo1
python homografia.py calibrar --saida ..\saidas\jogo1 --sem-janela   # calibração do frame0 (referência / Q4.2)
python homografia.py camera   --saida ..\saidas\jogo1                # abordagem da disciplina — degenera (ver Q4.3)
python diagnostico_camera.py  --saida ..\saidas\jogo1                # camera_degeneracao.csv
python camera_segmentos.py segmentos --saida ..\saidas\jogo1
python camera_segmentos.py movimento --saida ..\saidas\jogo1
python camera_segmentos.py calibrar  --saida ..\saidas\jogo1          # lê calib_segmentos\ancora_f*.json
python camera_segmentos.py combinar  --saida ..\saidas\jogo1
python camera_segmentos.py projetar  --saida ..\saidas\jogo1          # tracks_campo.csv, bola_campo.csv

# Etapa 5 — métricas
python metricas.py --saida ..\saidas\jogo1 --ataque-time0 esquerda
python sensibilidade_raio.py --saida ..\saidas\jogo1 --raios 1 1.5 2 2.5 3
python metricas_taticas.py --saida ..\saidas\jogo1
python validar_passes.py --saida ..\saidas\jogo1 --anotado ..\saidas\jogo1\passes_anotados.csv > ..\saidas\jogo1\validacao_passes.txt
python analisar_validacao.py --saida ..\saidas\jogo1   # erros por tipo → analise_validacao_passes.txt/.csv

# Etapa 6 — eXP
python xpass.py --saida ..\saidas\jogo1 --what-if --frame 9079 --passador 14392

# Etapa 7 — vídeo e figuras
python video_minimapa.py --saida ..\saidas\jogo1 --inicio-s 200 --dur-s 113
ffmpeg -i ..\saidas\jogo1\demo_minimapa.mp4 -c:v libx264 -preset veryfast -crf 23 ..\saidas\jogo1\demo_minimapa_h264.mp4
ffmpeg -i ..\saidas\jogo1\rastreado.mp4    -c:v libx264 -preset veryfast -crf 23 ..\saidas\jogo1\rastreado_h264.mp4

# Etapa 8 — resumo
python gerar_resumo.py --saida ..\saidas\jogo1
```

## Estrutura

```
src/                     scripts (base do repositório da disciplina + scripts do grupo)
notebooks/               pipeline no Colab (versão da disciplina)
saidas/comparacao_*.csv  Q2
saidas/jogo1/            tracks.csv, bola.csv, tracks_campo.csv, passes.csv, metricas_*.{csv,json},
                         segmentos.json, calib_segmentos/ (âncoras + grades), deriva_entre_ancoras.csv,
                         resultados_xp.csv, passes_anotados.csv, metricas_taticas.json, RESUMO_PARA_SLIDES.md
figuras/                 falsos positivos (Q2.3) · jogo1/ (mapas, passes, posse, calibração, eXP, táticas)
```

Scripts escritos pelo grupo: `camera_segmentos.py`, `ajuste_linhas.py`, `gerar_resumo.py`, `metricas_taticas.py`, `sensibilidade_raio.py`, `comparar_modelos.py`, `video_minimapa.py`, `analisar_validacao.py`, `inspecionar_k3.py`, `diagnostico_camera.py` e o anotador de passes `anotador_web/` (`servidor.py` + `index.html`). Alteração em script da disciplina: `campo.py` ganhou 6 marcos (`circulo_dir`, `circulo_esq`, `meia_lua_{dir,esq}_{sup,inf}`). Os demais vêm de [eron93br/vc-20262-p3](https://github.com/eron93br/vc-20262-p3) sem alteração.

---

## Respostas

> Todos os números vêm de arquivos em `saidas/` (indicados em cada resposta; resumo em `RESUMO_PARA_SLIDES.md`).

### Parte 1 — Aquisição
**Q1.1** Escolhemos Brasil × Coreia do Sul (oitavas de final da Copa de 2022), publicado no canal oficial da FIFA, no trecho 04:11–09:24 do vídeo. Três motivos: é jogo corrido do 1º tempo, quase todo na câmera aberta da transmissão; os uniformes têm alto contraste (amarelo × vermelho), o que favorece a separação por cor; e o trecho termina no gol de Vinícius Jr. (≈ 9:23), um lance completo para o minimapa e para a superfície de eXP. O trecho não é "limpo": há closes em 5:43–5:51, 6:17–6:34 e 7:33–7:48 do vídeo, outros closes curtos, uma câmera baixa de beira de campo e 3 replays (95,8–100,6 s, 200,9–210,1 s e 259,3–265,4 s do clipe). Por isso só 71,0% do clipe (222,3 s em 12 segmentos, `segmentos.json`) é câmera principal, e todas as métricas em metros usam apenas esses segmentos.

**Q1.2** 1280×720 · 30 fps · 9390 quadros · 313,0 s (`video_info.json`). Detecção + rastreamento com yolov8m (imgsz 1280, RTX 4070 Super): 26,1 fps → ≈ 360 s (`meta.json`).

### Parte 2 — Detecção
**Q2.1** (`saidas/comparacao_modelos.csv`, subtrecho de 30 s, imgsz 1280, conf-bola 0,10):

| modelo | fps proc. | pessoas/quadro | IDs únicos | % quadros com bola | conf. média bola |
|---|---|---|---|---|---|
| yolov8n | 31,4 | 18,5 | 166 | 81,4 | 0,465 |
| yolov8s | 33,2 | 19,6 | 234 | 67,4 | 0,313 |
| yolov8m | 27,8 | 18,6 | 175 | 87,8 | 0,429 |
| yolov8l | 19,5 | 18,6 | 174 | 79,9 | 0,475 |

Escolhido: **yolov8m** (mais quadros com bola; o l é 30% mais lento sem ganho de bola). Com imgsz 1280 o gargalo é CPU/ByteTrack, por isso n e s têm fps parecido.

**Q2.2** A bola é o objeto mais difícil por quatro razões: (1) na câmera aberta em 720p ela ocupa poucos pixels, e com `imgsz` 640 a imagem ainda é reduzida antes de entrar na rede (ver Q2.3); (2) em passes rápidos e lançamentos sai borrada pelo movimento; (3) fica ocluída por pernas e corpos justamente nos momentos de disputa e domínio; (4) usamos um modelo COCO genérico, cuja classe *sports ball* não foi treinada com transmissões de futebol, e que confunde a bola com cabeças, placas e detalhes da arquibancada. Ahmad Yousef et al. relatam a mesma dificuldade: detectar a bola é bem mais difícil que detectar jogadores e exige tratamento específico.

**Q2.3** (`saidas/comparacao_imgsz_conf.csv`, yolov8m):

| imgsz | conf-bola | % quadros com bola | conf. média |
|---|---|---|---|
| 640 | 0,05 / 0,10 / 0,25 | 36,4 / 24,6 / 13,7 | 0,258 / 0,349 / 0,499 |
| 1280 | 0,05 / 0,10 / 0,25 | 95,9 / 87,8 / 67,4 | 0,399 / 0,429 / 0,505 |

Dobrar o `imgsz` de 640 para 1280 multiplica por 2,6 a 4,9 a fração de quadros com bola, conforme o limiar: em 640 a imagem de 1280 px é reduzida à metade e a bola fica pequena demais para a rede. Baixar o limiar de confiança aumenta a cobertura, mas traz falsos positivos. Os 3 exemplos em `figuras/falsos_positivos_*.jpg` (conf ≈ 0,05) são camisas amarelas de torcedores (f480 e f834) e o colete de um segurança (f758). Por isso usamos limiar baixo (0,10) combinado com o filtro de Kalman com gating (Q4.1), que aceita só os candidatos coerentes com a trajetória da bola.

### Parte 3 — Rastreamento e times
**Q3.1** (`ids_por_minuto.csv`, `trocas_id.json`) IDs únicos por minuto: 358, 358, 290, 325, 436 (esperado ≈ 25); IDs com > 1 s: 109–161 por minuto; mediana de vida de um ID = 0,5 s; 1775 IDs no total. Momentos:
- frame 2985 (99,5 s): saída de close/replay (wipe) — ByteTrack perde todos os tracks e recria (30 IDs novos em 1 s);
- frame 6270 (209,0 s): close com banco de reservas no quadro — pessoas fora de campo viram "jogadores" (25 IDs novos);
- frame 8880 (296,0 s): câmera baixa antes do gol — torcedores/fotógrafos e oclusão (40 IDs novos).

A maior parte das "trocas" vem dos cortes da transmissão, e não do rastreador em câmera aberta. O ByteTrack associa caixas por movimento e IoU entre quadros consecutivos; num corte a cena muda inteira, nenhuma caixa casa e todos os IDs são recriados. Além disso, os closes colocam no quadro pessoas que não são jogadores (banco, torcida, fotógrafos). Por isso não calculamos métricas por jogador individual e trabalhamos com métricas por time.

**Q3.2** Centróides (`meta.json → times_bgr`, k=2): time 0 = BGR (86, 89, 176) → vermelho → **Coreia do Sul**; time 1 = BGR (46, 229, 217) → amarelo → **Brasil** (`times.json`). Conferimos no `rastreado_h264.mp4`: as cores dos times estão corretas. Os erros que aparecem são os esperados com k = 2. Como só existem dois grupos de cor, goleiros e árbitros são forçados para o time de cor mais parecida, e o mesmo goleiro pode trocar de time quando o ID dele muda. Nos closes, torcedores e pessoas fora de campo também recebem um dos dois times; esses trechos ficam fora das métricas porque não são câmera principal. O espaço de cor do classificador (S·cos H, S·sin H, V) trata o matiz como ângulo, o que evita a descontinuidade do vermelho em H ≈ 0/180: a média simples de H entre um pixel com H = 2 e outro com H = 178 daria ≈ 90 (ciano), e as camisas coreanas se espalhariam entre os grupos. A cor é medida só na faixa da camisa (10%–55% da altura do box) e sem os pixels de grama.

**Q3.3** `--k 3` (1º minuto, `saidas/jogo1_k3/meta.json`): centróides (76, 76, 190) vermelho, (46, 229, 217) amarelo e (67, 79, 66) escuro. O 3º cluster ficou com 160 dos 358 IDs de pessoa e 9,6% das detecções (`saidas/jogo1_k3/cluster_extra.txt`). O mosaico dos 24 IDs mais longos desse cluster (`figuras/jogo1/k3_cluster_extra.jpg`) mostra quem caiu nele: **árbitros (de preto), um assistente com bandeira, o goleiro de lilás, membros da comissão de terno, seguranças de colete e torcedores**. Ou seja, o k = 3 funciona como um "lixo" útil: tira dos dois times boa parte de quem não é jogador de linha. Não separa goleiro de árbitro (para isso seria preciso k maior ou um detector com essas classes), mas limpa as métricas de time.

### Parte 4 — Bola e homografia
**Q4.1** (`bola_kalman_log.txt`) YOLO encontrou candidatos em 79,3% dos quadros (inclui falsos positivos); o gating aceitou 4848 quadros (51,6%); o Kalman preencheu 3781 → **cobertura final 91,9%**.

**Q4.2** Calibração do frame0 (`figuras/jogo1/calibracao_grade_frame0.jpg`): a linha do meio não aparece, então usamos 5 marcos — `canto_sup_dir` 0,16 m · `ga_dir_sup` 0,25 m · `meia_lua_dir_sup` 0,05 m · `meia_lua_dir_inf` 0,02 m · `circulo_dir` 0,10 m. Na versão final usamos **16 âncoras** (`calibracao_ancoras.csv`, grades em `figuras/jogo1/calibracao/`): erro médio nos pontos manuais 0,21 m; resíduo mediano nas linhas brancas detectadas automaticamente 0,03 m.

**Q4.3** O `homografia.py camera` da disciplina (H0·Gt, com Gt encadeado por ORB + RANSAC entre quadros-chave) **degenera neste vídeo**. Em `camera_degeneracao.csv`, um retângulo fixo do gramado (264 000 px²) projetado por Gt tem área mediana de 59 px² na janela de 70–80 s e de 0 px² de 80 s até o fim: todos os pontos passam a cair no mesmo lugar. Causas: Gt é encadeado atravessando closes e replays (o próprio script avisa 15 cortes, `camera_cortes.json`, e registra 699 falhas de casamento em `camera_log.txt`), e o gramado tem poucos pontos ORB estáveis, então cada erro se acumula no encadeamento. Solução do grupo (`camera_segmentos.py`): (1) segmentar o clipe em trechos de câmera principal (caixas pequenas + placar visível, que some nos replays); (2) uma ou mais âncoras calibradas por segmento; (3) movimento quadro a quadro por fluxo óptico LK na imagem inteira (a câmera só gira/dá zoom); (4) a cada 10 quadros, re-ancoragem automática ajustando a homografia às linhas brancas do campo (`ajuste_linhas.py`). Validação: a deriva ao propagar de uma âncora até a seguinte foi de 0,23 m em média (máx. 0,72 m, `deriva_entre_ancoras.csv`). Closes/replays ficam sem projeção (29% do clipe).

**Q4.4** A homografia só vale para pontos no plano do gramado. Por isso projetamos o **pé** do jogador (centro da base do box), que toca o chão. O centro do box fica perto da cintura, fora do plano, e a homografia o jogaria para trás, mais longe da câmera, com erro que cresce com a altura do jogador na imagem. Pela mesma razão, a bola no ar viola a hipótese de plano: em lançamentos ela aparece mais longe do que está e com velocidade exagerada. Como não temos a altura da bola, aceitamos esse erro nos passes altos. A bola tem posição válida no campo em 59,0% das linhas de `bola_campo.csv`; o resto são closes, replays ou quadros sem bola.

### Parte 5 — Métricas
**Q5.1** (`metricas_times.json`, `metricas_jogadores.csv`; raio de posse 2 m)

| time | posse | passes (certos) | precisão | distância total* | vel. máx.** |
|---|---|---|---|---|---|
| 0 Coreia do Sul | 44,7% | 15 (11) | 73,3% | 5416,4 m | 26,8 km/h |
| 1 Brasil | 55,3% | 23 (18) | 78,3% | 4014,0 m | 25,7 km/h |

\* soma de todos os IDs, só nos 71% de câmera principal. \** tracks ≥ 10 s (tracks curtos têm velocidade ruidosa).
Plausibilidade (tracks ≥ 10 s, n = 108): mediana 157 m/min e vel. máx. 26,8 km/h; incluindo tracks curtos, máx. 34,7 km/h e nenhum jogador > 35 km/h. As velocidades são plausíveis. A distância por minuto fica acima dos 100–130 m/min de referência por dois motivos: medimos só jogo corrido em câmera aberta, sem as paradas de bola, e o tremor do box de um quadro para o outro soma distância que o jogador não percorreu. A distância total por time não é comparável a dados oficiais: é a soma de todos os IDs (que trocam nos cortes) e cobre só o tempo e a parte do campo que a câmera mostra. A Coreia aparece com mais distância que o Brasil porque teve mais jogadores visíveis, defendendo no campo que a câmera enquadrava: em média 9,43 coreanos contra 6,69 brasileiros por quadro (`RESUMO_PARA_SLIDES.md`). Não é porque correu mais.

**Q5.2** (`sensibilidade_raio.csv`, `figuras/jogo1/sensibilidade_raio.png`): de 1 a 3 m a posse do Brasil varia entre 55,3% e 62,1% e os passes entre 38 e 44; quadros com dono vão de 23,3% a 51,3%. A conclusão principal, "o Brasil teve mais posse", é robusta ao raio: vale em todos os valores testados. O número de passes também é estável (38–39), exceto em 2,5 m (44). Já a fração de quadros com dono depende muito do raio: mais que dobra de 1 para 3 m. Com raio pequeno, a bola fica "sem dono" em conduções e domínios; com raio grande, um jogador perto da bola "rouba" a posse sem tocar nela. Ficamos com 2 m, um meio-termo entre esses dois erros. Nele a posse do Brasil (55,3%) é o valor mais conservador da faixa, então a vantagem brasileira não é exagerada pela escolha do raio.

**Q5.3** Validação manual (`passes_anotados.csv`, `validacao_passes.txt`, `analise_validacao_passes.txt`): a equipe anotou **65 passes** nas 12 janelas de câmera principal com o anotador web (`src/anotador_web/`: saída e chegada de cada passe, sem ver os passes detectados). Com tolerância de 1 s e mesmo time: 38 detectados, 24 casados → **precisão = 0,63, revocação = 0,37, F1 = 0,47**; nos casados, certo/errado concorda em 20 de 24 (0,83). Erros dominantes:
- **Falsos negativos (41 de 65)** — o detector perde 10 de 11 interceptações anotadas e 30 de 53 passes certos; em 12 dos 41 perdidos a bola tem posição no campo em < 50% do trecho (nos casados, só 1 de 24).
- **Falsos positivos (14 de 38)** — 5 com o time trocado (há passe anotado do outro time a ≤ 1 s), 4 com o tempo deslocado (mesmo time, entre 1 e 2 s) e 5 sem passe anotado perto.

A revocação baixa vem sobretudo da bola: sem posição projetada não há dono, e o passe não existe para o detector. As interceptações quase não aparecem porque a posse só conta depois de alguns quadros seguidos com o mesmo jogador mais perto da bola; numa disputa o "mais perto" alterna e a posse curta é descartada. A proporção certo/errado, porém, fica próxima da anotada: Brasil 84,4% anotado × 78,3% detectado, Coreia 75,0% × 73,3%. O sistema subconta passes, mas não distorce muito a precisão de passe.

**Q5.4** Três decisões e limitações afetam as métricas:
- **Direção de ataque:** foi fixada com `--ataque-time0 esquerda`. No 1º tempo o Brasil ataca para a direita, e o gol do fim do trecho sai no gol da direita. Conferimos isso no vídeo em vez de deixar o script inferir. Todas as métricas territoriais (terços, field tilt, linha defensiva) dependem dessa direção.
- **Trocas de ID:** quando o mesmo jogador troca de ID com a bola no pé, a posse "muda de dono" e isso pode virar um passe falso. A distância mínima do passe filtra parte desses casos, porque a bola quase não se desloca. As trocas também inflam a distância total por time.
- **Jogadores fora do quadro:** somem das métricas de bloco (compactação, linha defensiva). Por isso só calculamos bloco em quadros com ≥ 6 jogadores visíveis do time.

### Métricas táticas propostas (`metricas_taticas.json`)

| Métrica | Como é calculada | Para que serve | Limitação no vídeo |
|---|---|---|---|
| Compactação (área, largura, profundidade) | fecho convexo dos jogadores visíveis do time | bloco curto/longo; com × sem bola | câmera não mostra os 22 → subestima |
| Altura da linha defensiva | 2º jogador mais recuado (m do próprio gol) | linha alta × bloco baixo | goleiro fora do quadro desloca o "2º" |
| Posse por terço / field tilt | posição da bola durante a posse de cada time | domínio territorial além da % de posse | depende da direção de ataque correta |
| Pressão no portador | nº de rivais a < 5 m do dono da bola | intensidade da marcação | oclusão e IDs perdidos |
| Sequências de posse | duração, passes por posse, progressão (m/s) | estilo: posse longa × transição rápida | posse "quebra" quando a bola some |
| Recuperações | onde o time retoma a bola | pressão alta × recuo | depende da qualidade da posse |

Resultados: field tilt Brasil 79,87% × Coreia 20,13%; linha defensiva sem bola a 55,9 m (Brasil) × 30,39 m (Coreia) do próprio gol; posse no terço ofensivo 30,54% × 9,53%; recuperações a 64,68 m × 34,06 m; pressão sofrida pelo portador 0,52 (Brasil) × 0,2 (Coreia). As métricas táticas contam uma história coerente com o jogo: um Brasil dominante, com a linha defensiva alta (quase no meio-campo) e recuperando a bola no campo de ataque, contra uma Coreia recuada em bloco baixo. O Brasil fica com a bola na maior parte do tempo no campo coreano (field tilt quase 4 × 1), muito mais do que a posse (55,3%) sozinha sugere. A pressão maior sobre o portador brasileiro é consequência desse bloco baixo: com a Coreia toda perto da própria área, sempre há defensores perto de quem tem a bola.

### Parte 6 — eXP
**Q6.1** O eXP usa, para cada passe, distância, adversários na linha do passe, pressão sobre o passador, espaço livre no destino, progressão em x e distância do destino ao gol (`passes_features.csv`). Comparado à representação de Gomes et al., que usa a posição e a velocidade de todos os jogadores em várias camadas espaciais, falta: (1) os 22 jogadores (a câmera mostra só parte do campo); (2) velocidades e direções confiáveis (as trocas de ID quebram as trajetórias); (3) a altura da bola (não distinguimos passe rasteiro de lançamento); (4) o contexto do lance (bola parada, contra-ataque, placar).

**Q6.2** (`resultados_xp.csv`, 38 passes, 76,3% certos)

| modelo | AUC-ROC | Brier | Log-Loss |
|---|---|---|---|
| baseline (distância) | 0,261 | 0,203 | 0,617 |
| heurístico | 0,789 | 0,157 | 0,479 |
| logístico | 0,774 | 0,159 | 0,469 |

O heurístico e o logístico ficam próximos e bem acima do baseline nas três métricas. Mas, com 38 passes e só 9 errados, a incerteza é grande: cada passe errado classificado de outra forma muda a AUC em vários pontos, então não dá para dizer qual dos dois é melhor. O baseline (regressão logística só com a distância, em validação cruzada) tem AUC < 0,5, pior que o acaso. Com poucos dados e uma feature fraca, cada dobra da validação cruzada aprende uma inclinação diferente e a ordenação se inverte: neste trecho a distância sozinha não separa passes certos de errados. O que separa é o contexto. Nas medianas de `passes_features.csv`, os passes errados saem com 1 adversário pressionando o passador (0 nos certos) e chegam a um destino com 2,85 m de espaço livre (10 m nos certos). É exatamente o que os modelos heurístico e logístico usam e o baseline ignora.

**Q6.3** Na média o modelo logístico está bem calibrado: prevê 0,787 de chance de acerto e a taxa observada é 0,763 (`resultados_xp.csv`), uma superestimação leve. A curva de calibração (`figuras/jogo1/calibracao_xp.png`) tem poucos passes por faixa de probabilidade, então a forma da curva não é confiável. Com mais trechos, cada faixa teria passes suficientes para avaliar a calibração ponto a ponto.

**Q6.4** A superfície (`figuras/jogo1/superficie_xp_f9079.png`) é do passe da construção do gol (302,6 s do clipe). Ela mostra para onde o passador poderia passar com segurança naquele instante. A defesa coreana concentrada na área deixa "buracos" de eXP baixo em volta de cada defensor, e a probabilidade cai quase a zero dentro da área, perto do goleiro. As zonas abertas nos lados e à frente da área têm eXP alto. A leitura é a de um bloco baixo: passar para fora da área é fácil, e o difícil é entrar nela.

**Q6.5** No what-if (`what_if.csv`) reavaliamos os 9 passes errados, procurando um destino próximo com eXP maior. O ganho médio foi de 0,384, e 89% deles tinham uma opção com ganho > 0,10: na maioria dos erros havia um destino mais seguro a 6 m do escolhido (o passo da busca do what-if). O PAx (passes acima do esperado, `jogadores_xp.csv`) é calculado por ID. Como os IDs se fragmentam nos cortes, a leitura por jogador não é confiável neste vídeo; ela exigiria re-identificação (número da camisa, por exemplo).

## Limitações e próximos passos
1. **Cobertura:** 29% do clipe (closes e replays) fica sem projeção no campo; um trecho curto de câmera principal mais fechada também foi excluído pela heurística de segmentação (caixas de jogadores maiores que o limite).
2. **Identidade:** as trocas de ID nos cortes impedem métricas confiáveis por jogador. Próximo passo: re-identificação por aparência ou número da camisa.
3. **Detector genérico:** o modelo COCO não distingue jogador, goleiro, árbitro e torcedor. Um YOLO ajustado com dados de futebol (classes jogador/goleiro/árbitro/bola) reduziria falsos positivos e melhoraria a bola.
4. **Calibração manual:** a homografia exigiu 16 âncoras marcadas à mão. Próximo passo: detectar automaticamente linhas e keypoints do campo.
5. **Passes:** o detector perde a maior parte das interceptações e passes com a bola oculta (revocação 0,37), e o eXP foi treinado com só 38 passes. Próximo passo: juntar mais trechos (`xpass.py --extra`) e usar a validação manual para ajustar os limiares de posse.

## Declaração de uso de IA
Usamos IA generativa (Claude Code) como apoio em quatro frentes:
- **Execução do pipeline:** rodar e encadear os scripts.
- **Código:** os scripts do grupo (`camera_segmentos.py`, `ajuste_linhas.py`, `gerar_resumo.py`, o anotador web e os demais listados acima).
- **Calibração:** localizar os marcos nas âncoras de calibração, conferidos pela equipe nas grades.
- **Redação:** a primeira versão das respostas, sempre a partir dos arquivos de `saidas/`.

A escolha do vídeo e do trecho, a decisão pelas âncoras manuais, a conferência visual dos vídeos e das cores dos times, a anotação manual dos 65 passes (ground truth da Q5.3) e a revisão e validação final das respostas são da equipe.

## Referências
- Ahmad Yousef et al. (2025). *Real-Time Football Match Analysis: Leveraging YOLO for Enhanced Object Detection and Possession Tracking*. Procedia Computer Science 272.
- Gomes et al. (2026). *Modelagem de Passes no Futebol via Componentes do Expected Possession Value com Redes Totalmente Convolucionais*. REIC 24:1.
- Fernández & Bornn (2020). *SoccerMap: A Deep Learning Architecture for Visually-Interpretable Analysis in Soccer*.
- Repositório da disciplina: https://github.com/eron93br/vc-20262-p3
