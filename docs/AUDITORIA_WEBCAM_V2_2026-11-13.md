# Auditoria: validação v2 alta, webcam ruim

Data: 13/11/2026. Escopo: `ml/features/pipeline_v2.py`, `temporal_v2.py`, `ml/data/dataset_v2.py`, `augmentation_v2.py`, `backend/services/video_stream*.py`, `prediction.py`, `ml/inference/predictor.py` e `frontend/src/hooks/useCameraFrames.ts`, no commit `05a101e`.

## Achados

1. **O 80% é o pico de uma curva ruidosa.** No `history.csv` do `transformer_v2_mixed23_seed42`, a validação top 1 das épocas 40–94 tem média 0,60 (desvio 0,09). O checkpoint escolhido é o máximo (época 69). Na época 81 a validação era 0,49. A expectativa realista é ~60–65% nesse split; o teste ainda não foi avaliado.
2. **A validação usa uma janela "oráculo".** `window_end(training=False)` centraliza a janela no trecho de maior movimento, escolhido com o vídeo inteiro. Ao vivo, a janela termina no instante atual a cada 0,4 s: pega sinal parcial ou só repouso. O modelo nunca viu essas janelas e não tem classe de repouso. Por isso emite palavras com a pessoa parada, e a EMA (α 0,6), o limiar 0,30 e o cooldown de 1 s fazem a primeira palavra errada bloquear a certa.
3. **Profundidade absoluta nas features.** O z da pose tem origem no quadril, o da mão no pulso e o da face no rosto. A normalização subtrai o z dos ombros de todas as partes, então o z global de mãos e rosto vira "menos a profundidade dos ombros". Com o quadril fora do quadro (webcam), esse valor muda sem relação com o sinal. Alterar só o z da pose muda 86 das 725 features (z dos 42 pontos das mãos e dos 40 da face, mais 4 distâncias).
4. **Escala 3D.** A largura dos ombros usa z, instável com enquadramento de meio corpo.
5. **Pontos extrapolados.** Cotovelos e punhos da pose fora do quadro continuam como válidos, porque a máscara é da pose inteira. A visibilidade é descartada sem ser usada como filtro.
6. **Cadência.** O backend processa ~6–12 fps; o augmentation com descarte de até 30% de vídeos de 30 fps nunca fica abaixo de ~21 fps.
7. **Proporção no frontend.** `drawImage(video, 0, 0, 640, 480)` estica a imagem se a câmera entregar 16:9, e o backend usa a proporção da imagem esticada.
8. **Validação das palavras do grupo.** 10/10 com um vídeo por classe tem intervalo de confiança de 95% com limite inferior ~69%. Se os IDs `Sinalizador01–07` não forem sete pessoas reais distintas, o split por pessoa não impede vazamento.

## Correções (retrocompatíveis)

Os padrões de `PipelineV2Config` reproduzem o checkpoint atual (`test_profundidade_da_pose_nao_altera_features_revisadas` confere). A config `configs/transformer_v2_mixed23_webcam.yaml` liga:

- `global_depth: false`: zera o z global; a forma local da mão mantém z.
- `scale_2d: true`: largura dos ombros em x,y.
- `point_visibility: true`: zera cotovelos, punhos e dedos da pose com visibilidade < 0,5.
- `segmentation: true`: `ml/features/segmentation_v2.py` define início e fim do sinal pelo deslocamento dos punhos em relação aos ombros, em larguras de ombro por segundo, igual a qualquer FPS. O dataset centraliza a janela nesse trecho (±0,25 s no treino). A sessão da webcam espera o repouso, classifica uma vez a mesma janela centrada e não herda EMA nem cooldown.
- `augmentation_v2_webcam.yaml`: `low_fps_prob: 0.5` com 6–15 fps e intervalos irregulares.
- Frontend: a captura preserva `videoWidth/videoHeight`.

## Comandos

```bash
./.venv/Scripts/python.exe scripts/calibrate_segmentation_v2.py --data-dir data/landmarks/mixed_23
./.venv/Scripts/python.exe scripts/train.py --config configs/transformer_v2_mixed23_webcam.yaml \
  --data-dir data/landmarks/mixed_23 --processed-dir data/processed/mixed_23 \
  --experiment transformer_v2_mixed23_webcam_seed42 --num-workers 0 --skip-test
./.venv/Scripts/python.exe scripts/evaluate_live_protocol_v2.py \
  --checkpoint experiments/transformer_v2_mixed23_webcam_seed42/best.pt \
  --data-dir data/landmarks/mixed_23 --processed-dir data/processed/mixed_23 --split val --fps 9 15
```

Para testar a segmentação ao vivo já com o checkpoint atual, sem retreinar, inicie o backend com `SINALIZA_LIVE_SEGMENTATION=true` (as features continuam as do treino). Rode o `evaluate_live_protocol_v2.py` também no checkpoint atual para ter a linha de base no protocolo da webcam. Compare `primeira_correta` e `palavras_erradas_por_minuto`, não só o top 1 da janela centrada.
