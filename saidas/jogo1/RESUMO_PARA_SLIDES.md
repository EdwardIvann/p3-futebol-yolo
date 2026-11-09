# Resumo para os slides — Brasil × Coreia do Sul (Copa 2022, oitavas)

> Gerado automaticamente por `src/gerar_resumo.py` a partir de `saidas/`. Não editar à mão.

## 1. Vídeo e processamento
- URL: https://www.youtube.com/watch?v=JIQnnYy7qYk — trecho 04:11-09:24 do vídeo (placar 01:29 → 06:42)
- Resolução 1280×720 · 30 fps · 9390 quadros · 313.0 s
- Detecção+rastreamento (yolov8m.pt, imgsz 1280, cuda): 26.1 fps → 360 s de processamento

## 2. Comparação de modelos (subtrecho de 30 s, 0:19–0:49 do clipe)
| modelo     |   fps_processamento |   pessoas_por_quadro |   ids_unicos |   quadros_com_bola_pct |   conf_media_bola |
|:-----------|--------------------:|---------------------:|-------------:|-----------------------:|------------------:|
| yolov8n.pt |                31.4 |                 18.5 |          166 |                   81.4 |             0.465 |
| yolov8s.pt |                33.2 |                 19.6 |          234 |                   67.4 |             0.313 |
| yolov8m.pt |                27.8 |                 18.6 |          175 |                   87.8 |             0.429 |
| yolov8l.pt |                19.5 |                 18.6 |          174 |                   79.9 |             0.475 |

imgsz × conf-bola (yolov8m):

|   imgsz |   conf_bola |   fps_processamento |   quadros_com_bola_pct |   conf_media_bola |
|--------:|------------:|--------------------:|-----------------------:|------------------:|
|     640 |        0.05 |                28.5 |                   36.4 |             0.258 |
|     640 |        0.1  |                29.3 |                   24.6 |             0.349 |
|     640 |        0.25 |                28.6 |                   13.7 |             0.499 |
|    1280 |        0.05 |                28.3 |                   95.9 |             0.399 |
|    1280 |        0.1  |                28.7 |                   87.8 |             0.429 |
|    1280 |        0.25 |                27.8 |                   67.4 |             0.505 |

## 3. Rastreamento e times
|   minuto |   t_ini_s |   quadros |   ids_unicos |   ids_com_mais_de_1s |   pessoas_por_quadro |
|---------:|----------:|----------:|-------------:|---------------------:|---------------------:|
|        0 |         0 |      1800 |          358 |                  109 |                 18.2 |
|        1 |        60 |      1795 |          358 |                  147 |                 12   |
|        2 |       120 |      1784 |          290 |                  134 |                  9.1 |
|        3 |       180 |      1795 |          325 |                  161 |                 13.2 |
|        4 |       240 |      1796 |          436 |                  158 |                 18   |
|        5 |       300 |       390 |           71 |                   34 |                 15.8 |

- Esperado ≈ 22 jogadores + 3 árbitros; mediana de vida de um ID = 0.5 s; total de IDs = 1775
  - frame 2985 (99.5 s, 30 IDs novos em 1 s): Close em jogador e transicao (wipe) de volta para a camera aberta: o ByteTrack perde todos os tracks no corte e recria IDs.
  - frame 6270 (209.0 s, 25 IDs novos em 1 s): Close de jogador caido com banco de reservas e comissao tecnica no quadro: pessoas fora de campo detectadas como 'person' e atribuidas a um time (T0).
  - frame 8880 (296.0 s, 40 IDs novos em 1 s): Camera baixa de beira de campo no lance antes do gol: torcedores e fotografos atras do jogador geram dezenas de deteccoes curtas; oclusao entre jogadores.
- Centróides (BGR, k=2): time 0 = [86, 89, 176] → **Coreia do Sul** (vermelho); time 1 = [46, 229, 217] → **Brasil** (amarelo)
- k=3 (1º minuto): {'0': [76, 76, 190], '1': [46, 229, 217], '-2': [67, 79, 66]}

## 4. Bola e homografia
- Bola: YOLO em 79.3% dos quadros (com falsos positivos) → 51.6% aceitos pelo gating → **91.9%** após Kalman
- Câmera principal: 12 segmentos, 222.3 s (71.0% do clipe); replays/closes sem projeção
- Âncoras de calibração: **16**; marcos manuais por âncora: 4–6; erro médio nos pontos manuais = **0.21 m**; resíduo mediano nas linhas detectadas = **0.03 m** (138 pontos de linha por âncora)
- Deriva entre âncoras consecutivas (validação cruzada): média 0.23 m, máx 0.72 m
- Projeção válida: jogadores 81.8% · bola 59.0% das linhas

## 5. Métricas
| time              |   posse_% |   passes |   certos |   precisao_% |   dist_total_m |   vel_max_kmh_tracks_10s |
|:------------------|----------:|---------:|---------:|-------------:|---------------:|-------------------------:|
| 0 (Coreia do Sul) |      44.7 |       15 |       11 |         73.3 |         5416.4 |                     26.8 |
| 1 (Brasil)        |      55.3 |       23 |       18 |         78.3 |         4014   |                     25.7 |

- Raio de posse 2.0 m; ataque: time 0 → esquerda, time 1 → direita
- Plausibilidade (tracks ≥ 10 s, n = 108): mediana 157 m/min; vel. máx. 26.8 km/h; incluindo tracks curtos, máx. 34.7 km/h e 0 acima de 35 km/h
- Distância total por time soma TODOS os IDs (inclui fragmentos de trocas de ID e só cobre os 71% de câmera principal)
- Jogadores visíveis por quadro (projeção válida, média): Coreia do Sul 9.43 · Brasil 6.69

Sensibilidade ao raio de posse:

|   raio_m |   posse_time0_pct |   posse_time1_pct |   quadros_com_dono_pct |   passes |   passes_certos |
|---------:|------------------:|------------------:|-----------------------:|---------:|----------------:|
|      1   |              38.9 |              61.1 |                   23.3 |       38 |              28 |
|      1.5 |              37.9 |              62.1 |                   36.6 |       39 |              29 |
|      2   |              44.7 |              55.3 |                   44.7 |       38 |              29 |
|      2.5 |              42.7 |              57.3 |                   48.1 |       44 |              28 |
|      3   |              43.8 |              56.2 |                   51.3 |       39 |              27 |

- Validação manual (65 passes anotados):
```
Detectados: 38 | Anotados: 65 | Casados: 24
Precisão  : 0.63
Revocação : 0.37
F1        : 0.47
Acordo certo/errado nos casados: 0.83
```

- Análise de erros (`analise_validacao_passes.txt`):
```
Tolerância 1.0 s, mesmo time. Detectados 38 | Anotados 65 | Casados 24
Precisão 0.63 | Revocação 0.37 | F1 0.47

Por time:
  Brasil: detectados 23, anotados 45, casados 16 → precisão 0.70, revocação 0.36
  Coreia: detectados 15, anotados 20, casados 8 → precisão 0.53, revocação 0.40

Taxa de acerto de passe (certos / total):
  Brasil: anotado 38/45 = 84.4% | detectado 18/23 = 78.3%
  Coreia: anotado 15/20 = 75.0% | detectado 11/15 = 73.3%

Falsos negativos (anotados não detectados): 41 de 65
  certo: 30 de 53 anotados (57% perdidos)
  interceptado: 10 de 11 anotados (91% perdidos)
  fora: 1 de 1 anotados (100% perdidos)
  bola com posição no campo em < 50% do trecho: 12 de 41 (nos casados: 1 de 24)
  duração mediana do passe: perdidos 1.44 s × casados 1.45 s
  passes < 0,7 s (toques curtos): perdidos 2 × casados 0

Falsos positivos (detectados sem anotação): 14 de 38
  sem passe anotado perto (ruído de posse/bola): 5
  time trocado (passe anotado do outro time ≤ tol.): 5
  tempo deslocado (mesmo time entre tol. e 2×tol.): 4

Certo/errado nos casados: concordam 20 de 24; detector diz certo e anotação errado: 1; o contrário: 3
```

## 6. eXP
38 passes (76.3% certos):

| modelo               |   AUC-ROC (↑) |   Brier (↓) |   Log-Loss (↓) |   Avg Precision (↑) |   média prevista |   taxa observada |
|:---------------------|--------------:|------------:|---------------:|--------------------:|-----------------:|-----------------:|
| heuristico           |         0.789 |       0.157 |          0.479 |               0.937 |            0.803 |            0.763 |
| baseline (distância) |         0.261 |       0.203 |          0.617 |               0.65  |            0.776 |            0.763 |
| logistico            |         0.774 |       0.159 |          0.469 |               0.929 |            0.787 |            0.763 |

- What-if: 9 passes errados reavaliados

## 7. Destaques táticos (`metricas_taticas.json`)
- **Field tilt**: Brasil 79.87% × Coreia do Sul 20.13%
- **Linha defensiva sem bola** (m do próprio gol): Brasil 55.9 × Coreia do Sul 30.39
- **Pressão média sobre o portador** (rivais a < 5 m): Brasil sofre 0.52, Coreia do Sul sofre 0.2
- Altura média das recuperações: Brasil 64.68 m × Coreia do Sul 34.06 m
- Posse no terço ofensivo: Brasil 30.54% × Coreia do Sul 9.53%

## 8. Figuras para os slides
- `figuras/falsos_positivos_colete_f758.jpg`
- `figuras/falsos_positivos_torcida_f480.jpg`
- `figuras/falsos_positivos_torcida_f834.jpg`
- `figuras/jogo1/calibracao/grade_f00000.jpg`
- `figuras/jogo1/calibracao/grade_f00300.jpg`
- `figuras/jogo1/calibracao/grade_f02290.jpg`
- `figuras/jogo1/calibracao/grade_f02684.jpg`
- `figuras/jogo1/calibracao/grade_f03607.jpg`
- `figuras/jogo1/calibracao/grade_f04294.jpg`
- `figuras/jogo1/calibracao/grade_f04809.jpg`
- `figuras/jogo1/calibracao/grade_f05306.jpg`
- `figuras/jogo1/calibracao/grade_f05773.jpg`
- `figuras/jogo1/calibracao/grade_f06530.jpg`
- `figuras/jogo1/calibracao/grade_f06900.jpg`
- `figuras/jogo1/calibracao/grade_f07273.jpg`
- `figuras/jogo1/calibracao/grade_f07479.jpg`
- `figuras/jogo1/calibracao/grade_f08102.jpg`
- `figuras/jogo1/calibracao/grade_f08905.jpg`
- `figuras/jogo1/calibracao/grade_f09164.jpg`
- `figuras/jogo1/calibracao_grade_frame0.jpg`
- `figuras/jogo1/calibracao_xp.png`
- `figuras/jogo1/compactacao_linha_def.png`
- `figuras/jogo1/k3_cluster_extra.jpg`
- `figuras/jogo1/mapa_calor_times.png`
- `figuras/jogo1/mapa_passes.png`
- `figuras/jogo1/posse_linha_tempo.png`
- `figuras/jogo1/posse_por_terco.png`
- `figuras/jogo1/recuperacoes.png`
- `figuras/jogo1/sensibilidade_raio.png`
- `figuras/jogo1/superficie_xp_f376.png`
- `figuras/jogo1/superficie_xp_f9079.png`
