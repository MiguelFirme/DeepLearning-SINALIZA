# Experimento: cinco palavras restantes com dois Transformers

## Plano registrado antes da execução

1. Usar Brincar, Cansado, Certo, Comer e Diretor, as cinco pastas ainda não usadas do lote de 10 palavras em `data/raw/`. Cada classe tem um vídeo de `Sinalizador01` a `Sinalizador07`; manter os originais intactos.
2. Extrair MediaPipe Holistic para 35 vídeos no formato `folder_signer`, conferir leitura e presença dos landmarks. Construir um vocabulário fechado de 5 classes e o mesmo holdout por pessoa do experimento anterior: 01 a 05 treino, 06 validação, 07 teste.
3. Treinar `pose_face` e `full` com os mesmos YAMLs, arquitetura, semente e técnicas do teste das cinco primeiras palavras. Selecionar os checkpoints por `val_loss`; salvar logs e métricas.
4. Comparar cada modelo e a fusão sequencial predefinida nas duas ordens, top 3 e top 5, peso 0,65 para pose e face e 0,35 para o modelo completo. Usar o teste apenas após fixar os parâmetros, sem escolher a melhor configuração pelos cinco exemplos de teste.
5. Arquivar cada `best.pt`, configuração, mapa de classes, splits, hashes, logs, histórico e README em `Modelos salvos/`. Registrar acertos por palavra e limites da análise aqui.

## Limites previstos

Validação e teste têm um único vídeo por classe; um acerto equivale a 20 pontos percentuais. Comparações com as cinco primeiras palavras são exploratórias porque são classes diferentes. Este experimento treina cinco classes separadamente; não amplia automaticamente os modelos anteriores para dez classes.

## Execução e resultados

Executado em 22/10/2026. O extrator MediaPipe Holistic processou **35/35 vídeos sem erros**. O manifesto `folder_signer` identificou cinco classes e sete sinalizadores. `build_dataset.py` confirmou ausência de vazamento: 25 amostras de treino (01–05), 5 de validação (06) e 5 de teste (07), uma de cada classe por pessoa.

Presença média de cada mão nos cinco vídeos de cada sinalizador, de acordo com as máscaras extraídas: 01 (26%/33%), 02 (22%/31%), 03 (40%/47%), 04 (23%/33%), 05 (89%/88%), 06 (42%/57%), 07 (100%/99%). Pose e face tiveram presença de 100% nesse resumo. A diferença na detecção de mãos entre pessoas limita a interpretação causal dos resultados.

Os dois modelos têm **120.823 parâmetros** e foram treinados com os mesmos YAMLs do experimento das cinco primeiras palavras: Transformer de 2 camadas, dimensão 64, 4 cabeças, dropout 0,45, Adam com `weight_decay=1e-4`, augmentation leve somente no treino, `ReduceLROnPlateau`, early stopping por `val_loss` com paciência 15, semente 42. O único fator alterado entre eles foi o modo de entrada `pose_face` ou `full`. O modelo `pose_face` escolheu a época **96/111**; o `full` escolheu **120/120**, atingindo o limite de épocas.

| Método | Validação: 5 vídeos | Teste: 5 vídeos | F1 macro teste |
|---|---:|---:|---:|
| `pose_face` | 4/5 (80%) | 4/5 (80%) | 0,733 |
| **`full`** | 4/5 (80%) | **5/5 (100%)** | **1,000** |
| `pose_face` → `full`, top 3 | 4/5 (80%) | 4/5 (80%) | 0,733 |
| `pose_face` → `full`, top 5 | 4/5 (80%) | 4/5 (80%) | 0,733 |
| `full` → `pose_face`, top 3 | 4/5 (80%) | 4/5 (80%) | 0,733 |
| `full` → `pose_face`, top 5 | 4/5 (80%) | 4/5 (80%) | 0,733 |

A fusão reutilizou **0,65 × P(pose_face) + 0,35 × P(full)**. No vídeo de teste de **Brincar**, `pose_face` previu Cansado e `full` previu Brincar; a fusão manteve Cansado. Os demais quatro vídeos foram classificados corretamente por ambos. Na validação, ambos erraram Brincar como Cansado. Com top 5 entre cinco classes, todas são candidatas e a ordem não afeta o resultado. A fusão não melhorou o modelo individual `full` neste lote. Alterar seu peso agora para aproveitar o vídeo de teste seria ajuste sobre o próprio teste; por isso o peso foi mantido.

**Interpretação:** 5/5 no teste é um resultado promissor, mas vem de um único vídeo de cada classe, todos do mesmo sinalizador. Um erro mudaria 100% para 80%. A validação do modelo completo foi 4/5, e o checkpoint chegou ao limite de 120 épocas. Não há evidência suficiente para prometer 100% com outras pessoas ou para trocar automaticamente o modelo da aplicação. Estes modelos classificam apenas as cinco classes deste lote, não as dez classes novas conjuntamente.

### Artefatos

- Checkpoints e reprodução: `Modelos salvos/transformer_novas_5_restantes_pose_face/` e `Modelos salvos/transformer_novas_5_restantes_full/` (`best.pt`, SHA-256, configurações, split, logs, histórico, métricas e README).
- Landmarks/CSV: `data/landmarks/novas_5_restantes/`; split: `data/processed/novas_5_restantes/`.
- Métricas, matrizes de confusão e predições por vídeo nas duas ordens: `experiments/fusion_novas_5_restantes/`.
- Logs completos: `logs/experiments/transformer_novas_5_restantes_{pose_face,full}_seed42/`.

### Reproduzir no Git Bash do VS Code

Na raiz do projeto, com os MP4 originais em `data/raw/<classe>/` e a `.venv` funcionando com Python 3.11:

```bash
./.venv/Scripts/python.exe scripts/extract_landmarks.py \
  --input data/raw --output data/landmarks/novas_5_restantes \
  --dataset-format folder_signer \
  --include-classes Brincar Cansado Certo Comer Diretor --workers 2

./.venv/Scripts/python.exe scripts/build_dataset.py \
  --landmarks data/landmarks/novas_5_restantes \
  --annotations data/landmarks/novas_5_restantes/annotations.csv \
  --output data/processed/novas_5_restantes --strategy signer_holdout \
  --class-list configs/novas_5_restantes.json \
  --train-signers Sinalizador01 Sinalizador02 Sinalizador03 Sinalizador04 Sinalizador05 \
  --val-signer Sinalizador06 --test-signer Sinalizador07

./.venv/Scripts/python.exe scripts/train.py \
  --config configs/transformer_minds_pose_face.yaml \
  --data-dir data/landmarks/novas_5_restantes \
  --processed-dir data/processed/novas_5_restantes \
  --experiment transformer_novas_5_restantes_pose_face_seed42 --num-workers 0

./.venv/Scripts/python.exe scripts/train.py \
  --config configs/transformer_minds_full.yaml \
  --data-dir data/landmarks/novas_5_restantes \
  --processed-dir data/processed/novas_5_restantes \
  --experiment transformer_novas_5_restantes_full_seed42 --num-workers 0

./.venv/Scripts/python.exe scripts/evaluate_two_stage_fusion.py \
  --pose-checkpoint experiments/transformer_novas_5_restantes_pose_face_seed42/best.pt \
  --full-checkpoint experiments/transformer_novas_5_restantes_full_seed42/best.pt \
  --pose-split-dir data/processed/novas_5_restantes \
  --full-split-dir data/processed/novas_5_restantes \
  --data-dir data/landmarks/novas_5_restantes \
  --split test --first pose_face --top-k 3 5 --pose-weight 0.65 \
  --output-dir experiments/fusion_novas_5_restantes
```

Para repetir a validação ou inverter a ordem, mude `--split val` ou `--first full`.

**Ambiente desta execução:** o `.venv/Scripts/python.exe` local continua indisponível porque seu `pyvenv.cfg` aponta para um Python 3.11 removido. O treino foi feito com um Python 3.11 portátil temporário em `C:\Users\migue\AppData\Local\Temp\sinaliza_python311_embed\python.exe`, usando os pacotes existentes da `.venv` e CUDA. Os comandos acima exigem reparar a `.venv` caso ela ainda esteja nessa condição.
