# Transformer MINDS-Libras — 8 classes, pose e face

## Identidade e resultado

- Experimento original: `experiments/transformer_minds_school8_pose_face_seed42/`.
- Checkpoint arquivado: `best.pt`, **época 94** nos logs (`epoch=93` dentro do `.pt`, com índice iniciado em zero), selecionado pela menor `val_loss` (aproximadamente 0,1349). O treino parou por early stopping na época 109.
- Dataset: MINDS-Libras, 8 classes, 320 vídeos; classes e índices em `data_split/label_map.json` e lista fechada em `config/minds_libras_school8.json`.
- Validação (Sinalizador11): **40/40, top-1 100%, F1 macro 1,000**.
- Teste (Sinalizador12): **37/40, top-1 92,5%, F1 macro 0,923**. Os erros foram `Medo` → `Aluno` (1) e `Vontade` → `Conhecer` (2). Matrizes e previsões por vídeo estão em `records/`.

São sinais isolados de um dataset pequeno. O Sinalizador12 já havia sido consultado em experimentos anteriores; a métrica é exploratória e não prova desempenho em pessoas, câmeras ou frases novas.

## Dados e pré-processamento

- Split por pessoa, sem vídeos sintéticos: treino Sinalizador01, 02, 05, 06, 08 e 10 (**240 vídeos; 30 por classe**); validação Sinalizador11 (**40; 5 por classe**); teste Sinalizador12 (**40; 5 por classe**).
- Os vídeos passaram pelo extrator MediaPipe do projeto e geraram `.npz` em `data/landmarks/minds_libras_school8/`. O modelo recebeu sequências amostradas em **48 frames** e **697 features por frame** após o pipeline de normalização/derivação.
- `feature_mode=pose_face`: os pontos detalhados e máscaras das duas mãos são zerados de forma consistente; pose (inclusive braços/pulsos) e face são preservadas. A normalização espacial usa centro e distância entre ombros conforme o código do projeto.
- Augmentation dinâmica **somente no treino**: ruído gaussiano (`std=0,002`), escala entre 0,98 e 1,02 e rotação de até 2°, cada transformação com probabilidade 0,7. Validação e teste usam os landmarks limpos.

## Rede e otimização

- Transformer temporal 1-D com embeddings de dimensão 64, **2 camadas**, **4 cabeças de atenção**, feedforward de 128, codificação posicional aprendível, token CLS, LayerNorm e dropout **0,45**. Classificador oculto de 64. Total: **121.018 parâmetros**.
- Loss: CrossEntropy padrão, sem pesos de classe e sem label smoothing (classes equilibradas).
- Otimizador **Adam**, `learning_rate=3e-4`, `weight_decay=1e-4`; batch 16; gradiente limitado a 1,0; mixed precision no treino CUDA.
- `ReduceLROnPlateau` pela `val_loss`, paciência 5 e fator 0,5; early stopping pela `val_loss`, paciência 15; máximo de 120 épocas; seed 42.

Os YAMLs em `config/` mantêm a cadeia `_base_` usada no treino. `records/run_record.json` guarda a configuração efetiva, ambiente, hashes dos dados e comando original; `records/history.csv` guarda todas as épocas. `data_split/manifest.json` e `splits.json` fixam a ordem e a identidade dos exemplos.

## Como reproduzir no Git Bash

Execute na raiz do projeto, mantendo `data/landmarks/minds_libras_school8/` disponível. O nome novo evita sobrescrever o experimento original:

```bash
./.venv/Scripts/python.exe scripts/train.py \
  --config "Modelos salvos/transformer_minds_8_classes_pose_face/config/transformer_minds_pose_face.yaml" \
  --data-dir data/landmarks/minds_libras_school8 \
  --processed-dir "Modelos salvos/transformer_minds_8_classes_pose_face/data_split" \
  --experiment transformer_minds_school8_pose_face_repro_seed42 \
  --num-workers 0 --skip-test
```

Para auditar **o checkpoint arquivado** na validação, sem modificar esta pasta:

```bash
./.venv/Scripts/python.exe scripts/audit_checkpoint_val.py \
  --checkpoint "Modelos salvos/transformer_minds_8_classes_pose_face/best.pt" \
  --data-dir data/landmarks/minds_libras_school8 \
  --processed-dir "Modelos salvos/transformer_minds_8_classes_pose_face/data_split" \
  --split val --output-dir experiments/audit_modelo_salvo_8_classes
```

Reproduzir exatamente os landmarks a partir dos MP4 requer o extrator e a versão do código deste projeto. A execução em CUDA pode não ser bit a bit determinística; compare métricas e matrizes, não apenas o hash de um treino novo.
