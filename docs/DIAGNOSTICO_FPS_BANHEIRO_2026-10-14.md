# Diagnostico da demora para reconhecer Banheiro

## Plano

1. Confirmar que checkpoint, limiar de confianca e janela temporal nao mudaram nas alteracoes recentes.
2. Comparar o mesmo video de Banheiro no pipeline JPEG + MediaPipe + Transformer, primeiro com amostragem proxima de 15 FPS e depois com todos os quadros (~30 FPS).
3. Relacionar as predicoes ao tamanho da janela de 48 quadros e registrar o que ainda precisa ser medido na Canon R100 ao vivo.

## Codigo conferido

- O backend continua carregando `Modelos salvos/transformer_minds_8_classes_pose_face/best.pt`, usando `confidence_threshold=0.3`, 48 quadros na primeira janela e avancos de 24 quadros.
- O navegador passou a pedir 30 FPS e capturar cada quadro novo, sem o antigo temporizador de 15 FPS. O WebSocket mantem ate dois quadros em transito; quadros adicionais sao descartados se o backend nao acompanhar.
- A ultima correcao adicionou os landmarks brutos ao `frame_ack` para desenhar o overlay. Ela nao modifica as features normalizadas nem os pesos passados ao classificador. Pode adicionar algum custo de serializacao e desenho, mas este custo nao foi isolado numa medicao ao vivo.

## Teste reproduzivel

Na raiz do projeto, com a `.venv` funcional:

```bash
./.venv/Scripts/python.exe scripts/smoke_web_inference.py \
  --video data/raw/minds_libras/08BanheiroSinalizador11-1.mp4 --fps 15
./.venv/Scripts/python.exe scripts/smoke_web_inference.py \
  --video data/raw/minds_libras/08BanheiroSinalizador11-1.mp4 --fps 25
```

O script arredonda `source_fps / --fps` para um passo inteiro. Como o arquivo tem aproximadamente 30,2 FPS, `--fps 15` usa um quadro a cada dois (~15 FPS) e `--fps 25` usa todos os quadros (~30 FPS). O segundo comando nao mede exatamente 25 FPS.

| Entrada | Quadros processados | Primeiro palpite | Palpites seguintes |
|---|---:|---|---|
| Aproximadamente 15 FPS | 71 | Banheiro, 79,2% | Nenhum antes do fim do video |
| Aproximadamente 30 FPS | 141 | Acontecer, 96,5% | Banheiro, 56,1%; Banheiro, 79,4%; Banheiro, 84,2% |

Na primeira predicao, 48 quadros representam aproximadamente 3,2 segundos a 15 FPS e 1,6 segundo a 30 FPS. O modelo foi treinado com 48 quadros amostrados uniformemente ao longo de cada video. Portanto, a primeira janela mais curta pode nao conter o sinal completo. Isso explica a troca de palpite observada nesse arquivo, mas nao prova que a causa no uso ao vivo seja somente o FPS.

O MediaPipe detectou pose e face em todos os quadros amostrados do arquivo nos dois testes. Nao houve falha de rastreamento nesse video.

## Correcao de interpretacao apos o relato do usuario

O usuario informou que a webcam com mais FPS reconhecia melhor antes da copia dos arquivos do RAR. Portanto, o experimento com um video do dataset nao explica a regressao temporal relatada e nao justifica reduzir o FPS como solucao. A comparacao do codigo identificou trabalho adicional de renderizacao da pagina a cada `frame_ack`, corrigido em `PLANO_REGRESSAO_WEBCAM_POS_RAR_2026-10-14.md`. O impacto dessa correcao na webcam real ainda precisa ser medido.

## Limites e proximo diagnostico

A pagina mostra FPS da camera, nao FPS efetivamente processado pelo backend. A CPU, o descarte de quadros, a duracao do gesto, o enquadramento e o status `poor_tracking`/`uncertain` podem afetar a Canon ao vivo. O arquivo testado pertence ao dataset; nao mede generalizacao para o usuario. Antes de mudar permanentemente a taxa enviada ou o tamanho da janela, medir na webcam quadros processados por segundo, tempo para completar 48 quadros e status das predicoes. Testar mais de um sinal e repetir as tentativas com protocolo fixo.
