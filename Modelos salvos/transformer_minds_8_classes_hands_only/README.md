# Transformer MINDS-Libras — 8 classes, somente mãos

## Identidade e resultado

- Experimento: `experiments/transformer_minds_school8_hands_only_seed42/`.
- Modelo treinado: **`best.pt`**. Melhor época: **1** nos logs (`epoch=0` dentro do checkpoint, índice iniciado em zero). Seleção pela menor `val_loss` (2,0803); early stopping na época 16.
- Dataset: MINDS-Libras, 8 classes, 320 vídeos. Os nomes e índices estão em `data_split/label_map.json`.
- Validação (Sinalizador11): **5/40, top-1 12,5%, F1 macro 0,0278**. Todos os vídeos foram classificados como `Acontecer`.
- Teste (Sinalizador12): **6/40, top-1 15%, F1 macro 0,0648**. Houve 38 previsões de `Acontecer` e 2 de `Conhecer`.

O modelo foi salvo para análise e reprodução. **Ele não está ativo no backend e não deve receber peso em uma fusão com o modelo de pose e face neste estado.** O desempenho de validação está no nível de uma escolha aleatória entre oito classes equilibradas. O Sinalizador12 já havia sido consultado em experimentos anteriores, de modo que sua métrica é exploratória.

## Dados e preparação

- Mesmo split do modelo de pose e face: treino Sinalizador01, 02, 05, 06, 08 e 10 (**240 vídeos, 30 por classe**); validação Sinalizador11 (**40, 5 por classe**); teste Sinalizador12 (**40, 5 por classe**). Não há vídeos sintéticos.
- Os `.npz` originais precisam estar em `data/landmarks/minds_libras_school8/`; não são incluídos neste arquivo de modelo.
- Cada vídeo é amostrado em **48 frames**. `feature_mode=hands_only` zera coordenadas e máscaras de pose e face **antes** da augmentation e da normalização, tanto no treino quanto na inferência. Permanecem os 21 pontos de cada mão em coordenadas MediaPipe; a máscara de detecção controla o pré-processamento de pontos ausentes e das velocidades. Também é calculada a distância entre mãos quando ambas aparecem. Sem pose válida, a normalização por ombros não é aplicada. A entrada final conserva **697 posições por frame** para manter a mesma arquitetura; as posições de pose e face e suas derivadas ficam zeradas.
- Augmentation dinâmica somente no treino: ruído gaussiano (`std=0,002`), escala de 0,98 a 1,02 e rotação até 2°, cada transformação com probabilidade 0,7. Validação e teste não usam augmentation.

## Rede e otimização

Mesma arquitetura e configuração de treino do modelo arquivado em `Modelos salvos/transformer_minds_8_classes_pose_face/`; a única diferença de configuração é `data.feature_mode`.

- Transformer temporal 1-D, dimensão 64, **2 camadas**, **4 cabeças**, feedforward 128, token CLS, LayerNorm, dropout **0,45**, classificador oculto 64; **121.018 parâmetros**.
- CrossEntropy padrão, sem pesos de classe; Adam, `learning_rate=3e-4`, `weight_decay=1e-4`; batch 16, limite de gradiente 1,0 e mixed precision em CUDA.
- `ReduceLROnPlateau` pela `val_loss` (paciência 5, fator 0,5) e early stopping pela `val_loss` (paciência 15); máximo de 120 épocas; seed 42.

`config/` contém a cadeia YAML completa; `data_split/` fixa amostras e classes; `records/` contém logs, configuração efetiva, histórico, métricas, previsões por vídeo e hashes dos arquivos de código relevantes. `checksums.sha256` permite conferir a integridade desta cópia. O código de seleção da modalidade está em `ml/features/feature_modes.py` e é necessário para reproduzir a entrada do modelo.

## Como reproduzir no Git Bash

Na raiz do projeto, com os `.npz` disponíveis:

```bash
./.venv/Scripts/python.exe scripts/train.py \
  --config "Modelos salvos/transformer_minds_8_classes_hands_only/config/transformer_minds_hands_only.yaml" \
  --data-dir data/landmarks/minds_libras_school8 \
  --processed-dir "Modelos salvos/transformer_minds_8_classes_hands_only/data_split" \
  --experiment transformer_minds_school8_hands_only_repro_seed42 \
  --num-workers 0 --skip-test
```

Para auditar **este checkpoint** na validação sem modificar a pasta arquivada:

```bash
./.venv/Scripts/python.exe scripts/audit_checkpoint_val.py \
  --checkpoint "Modelos salvos/transformer_minds_8_classes_hands_only/best.pt" \
  --data-dir data/landmarks/minds_libras_school8 \
  --processed-dir "Modelos salvos/transformer_minds_8_classes_hands_only/data_split" \
  --split val --output-dir experiments/audit_modelo_maos_salvo
```

Treinos em CUDA podem variar ligeiramente entre execuções. A auditoria de cobertura das mãos está em `docs/AUDITORIA_MAOS_MINDS_8_CLASSES.md`: a detecção de pelo menos uma mão foi muito menor no treino que na validação, uma hipótese importante para investigar o fracasso deste modelo.
