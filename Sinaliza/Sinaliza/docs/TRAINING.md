# Guia de Treinamento — Sinaliza

## Visão Geral do Pipeline

```
Download → Extração Landmarks → Validação → Build Dataset → Treino → Avaliação
```

## 1. Download do Dataset

```bash
python scripts/download_dataset.py --output data/raw
```

Requer `~/.kaggle/kaggle.json` configurado. O dataset V-Librasil tem ~4088 vídeos em ~1364 classes.

## 2. Extração de Landmarks

```bash
python scripts/extract_landmarks.py \
  --input data/raw/v-librasil \
  --output data/landmarks \
  --workers 4
```

Para cada vídeo, o MediaPipe Holistic extrai 346 features por frame:
- Mão esquerda: 21 × 3 = 63
- Mão direita: 21 × 3 = 63
- Pose (torso): 25 × 4 = 100
- Face (seleção): 40 × 3 = 120

Saída: arquivos `.npz` com arrays `landmarks` e metadata.

## 3. Validação

```bash
python scripts/validate_landmarks.py --input data/landmarks
```

Verifica integridade dos `.npz`, detecta outliers, e gera estatísticas.

## 4. Construção do Dataset

```bash
python scripts/build_dataset.py \
  --input data/landmarks \
  --output data/processed \
  --train-ratio 0.75 \
  --val-ratio 0.15
```

Gera splits estratificados e `label_map.json`.

## 5. Treinamento

### Configuração via YAML

Cada modelo tem seu arquivo de configuração em `configs/`:

```bash
python scripts/train.py --config configs/bilstm.yaml
python scripts/train.py --config configs/transformer.yaml
python scripts/train.py --config configs/tcn.yaml
```

### Parâmetros Principais

| Parâmetro | Default | Descrição |
|-----------|---------|-----------|
| `learning_rate` | 0.001 | Taxa de aprendizado |
| `epochs` | 100 | Épocas máximas |
| `batch_size` | 32 | Tamanho do batch |
| `target_frames` | 48 | Frames por sequência |
| `patience` | 15 | Early stopping |
| `warmup_epochs` | 5 | Warmup do scheduler |
| `label_smoothing` | 0.1 | Suavização dos labels |
| `gradient_clip` | 1.0 | Clipping de gradientes |

### Augmentação de Dados

As seguintes técnicas são aplicadas (sem espelhamento horizontal):

- **Speed Warp** — varia velocidade do sinal
- **Temporal Crop** — corta início/fim aleatório
- **Gaussian Noise** — ruído nos landmarks
- **Scale Jitter** — variação de escala
- **Rotation Jitter** — rotação leve
- **Frame Drop** — remove frames aleatórios
- **Channel Drop** — zera canais aleatórios
- **Temporal Shift** — desloca sequência no tempo
- **Cutout** — zera blocos de frames

### Mixed Precision (AMP)

Ativado por padrão. Reduz uso de memória GPU e acelera treino.

## 6. Avaliação

```bash
python scripts/evaluate.py \
  --checkpoint artifacts/checkpoints/best_bilstm.pt \
  --config configs/bilstm.yaml \
  --output artifacts/evaluation
```

Gera:
- Acurácia Top-1/3/5
- F1 macro/weighted
- Matriz de confusão (imagem)
- Pares mais confusos
- Relatório Markdown

## 7. Comparação de Modelos

```bash
python scripts/compare_models.py \
  --checkpoints artifacts/checkpoints/best_*.pt \
  --output artifacts/comparison
```

## 8. Hyperparameter Tuning

```bash
python scripts/tune_hyperparams.py \
  --config configs/bilstm.yaml \
  --n-trials 50 \
  --output artifacts/tuning
```

Usa Optuna para busca bayesiana de hiperparâmetros.

## 9. Export para Produção

```bash
python scripts/export_model.py \
  --checkpoint artifacts/checkpoints/best_bilstm.pt \
  --output artifacts/sinaliza_model.pt
```

## Treino no Google Colab

Utilize o notebook `notebooks/train_colab.ipynb` para treinar no Colab com GPU gratuita.

## Dicas

1. Comece com Bi-LSTM — mais rápido de treinar e debug
2. Use `target_frames=48` como baseline
3. Monitore overfitting com a curva de validação
4. Para classes raras, use `class_weight_method: "effective"`
5. O Transformer se beneficia mais de warmup (8-10 épocas)
