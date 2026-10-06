# Transformer MINDS-Libras — 20 classes, pose e face

## Identidade e resultado

- Experimento original: `experiments/transformer_minds_all20_pose_face_seed42/`.
- Checkpoint arquivado: `best.pt`, **época 35** nos logs (`epoch=34` dentro do `.pt`, com índice iniciado em zero), selecionado pela menor `val_loss` (aproximadamente 1,5430). O treino parou por early stopping na época 50.
- Dataset: MINDS-Libras completo, 20 classes, 800 vídeos; classes e índices em `data_split/label_map.json` e lista fechada em `config/minds_libras_all20.json`.
- Validação (Sinalizador11): **37/100, top-1 37%, F1 macro 0,286** no checkpoint arquivado. A época 47 chegou a 50% top-1, mas teve `val_loss` maior e **não** é o checkpoint `best.pt`.
- **Teste ainda não executado para este experimento.** Não há `test_metrics.json` nesta pasta. O split de teste está reservado em `data_split/splits.json`.

Este checkpoint cobre mais palavras que o de 8 classes, mas sua validação é inferior. Onze das vinte classes tiveram 0/5 acertos na validação. As matrizes e previsões estão em `records/val_metrics.json` e `records/val_predictions.csv`. O desempenho em pessoas novas e tradução de frases permanece desconhecido.

## Dados e pré-processamento

- Split por pessoa, sem vídeos sintéticos: treino Sinalizador01, 02, 05, 06, 08 e 10 (**600 vídeos; 30 por classe**); validação Sinalizador11 (**100; 5 por classe**); teste Sinalizador12 (**100; 5 por classe**, reservado).
- Os vídeos passaram pelo extrator MediaPipe do projeto e geraram `.npz` em `data/landmarks/minds_libras_all20/`. O modelo recebeu sequências amostradas em **48 frames** e **697 features por frame** após o pipeline de normalização/derivação.
- `feature_mode=pose_face`: os pontos detalhados e máscaras das duas mãos são zerados; pose (inclusive braços/pulsos) e face são preservadas. A normalização espacial usa centro e distância entre ombros conforme o código do projeto.
- Augmentation dinâmica **somente no treino**: ruído gaussiano (`std=0,002`), escala entre 0,98 e 1,02 e rotação de até 2°, cada transformação com probabilidade 0,7. Validação e teste usam dados limpos.

## Rede e otimização

- Transformer temporal 1-D com embeddings de dimensão 64, **2 camadas**, **4 cabeças de atenção**, feedforward de 128, codificação posicional aprendível, token CLS, LayerNorm e dropout **0,45**. Classificador oculto de 64. Total: **121.798 parâmetros**.
- Loss: CrossEntropy padrão, sem pesos de classe e sem label smoothing (classes equilibradas).
- Otimizador **Adam**, `learning_rate=3e-4`, `weight_decay=1e-4`; batch 16; gradiente limitado a 1,0; mixed precision no treino CUDA.
- `ReduceLROnPlateau` pela `val_loss`, paciência 5 e fator 0,5; early stopping pela `val_loss`, paciência 15; máximo de 120 épocas; seed 42.

Os YAMLs em `config/` mantêm a cadeia `_base_` usada no treino. `records/run_record.json` guarda a configuração efetiva, ambiente, hashes dos dados e comando original; `records/history.csv` guarda todas as épocas. `data_split/manifest.json` e `splits.json` fixam a ordem e a identidade dos exemplos.

## Como reproduzir no Git Bash

Execute na raiz do projeto, mantendo `data/landmarks/minds_libras_all20/` disponível. O nome novo evita sobrescrever o experimento original:

```bash
./.venv/Scripts/python.exe scripts/train.py \
  --config "Modelos salvos/transformer_minds_20_classes_pose_face/config/transformer_minds_pose_face.yaml" \
  --data-dir data/landmarks/minds_libras_all20 \
  --processed-dir "Modelos salvos/transformer_minds_20_classes_pose_face/data_split" \
  --experiment transformer_minds_all20_pose_face_repro_seed42 \
  --num-workers 0 --skip-test
```

Para auditar **o checkpoint arquivado** na validação, sem modificar esta pasta:

```bash
./.venv/Scripts/python.exe scripts/audit_checkpoint_val.py \
  --checkpoint "Modelos salvos/transformer_minds_20_classes_pose_face/best.pt" \
  --data-dir data/landmarks/minds_libras_all20 \
  --processed-dir "Modelos salvos/transformer_minds_20_classes_pose_face/data_split" \
  --split val --output-dir experiments/audit_modelo_salvo_20_classes
```

O teste Sinalizador12 só deve ser avaliado após fixar o checkpoint e o protocolo; a métrica do teste não deve guiar mudanças posteriores de hiperparâmetros. Reproduzir exatamente os landmarks requer o extrator e a versão do código deste projeto. A execução em CUDA pode não ser bit a bit determinística.
