# Requisitos de Hardware — Sinaliza

## Treinamento

### Mínimo (CPU)

- **CPU**: 4 cores
- **RAM**: 8 GB
- **Armazenamento**: 10 GB livres
- **Tempo estimado**: ~4-8h por modelo (100 épocas)

### Recomendado (GPU)

- **GPU**: NVIDIA com 4+ GB VRAM (GTX 1650+, RTX 3060+)
- **CPU**: 8 cores
- **RAM**: 16 GB
- **Armazenamento**: 20 GB (SSD recomendado)
- **Tempo estimado**: ~30-90 min por modelo

### Google Colab (Gratuito)

- GPU T4 (15 GB VRAM) — suficiente para todos os modelos
- Use o notebook `notebooks/train_colab.ipynb`
- Upload do dataset processado (~2 GB de .npz)

## Inferência

### Backend

- **CPU**: 2+ cores (CPU moderno)
- **RAM**: 4 GB
- **GPU**: opcional (melhora latência)
- **Latência esperada**: ~20-50ms (CPU), ~5-15ms (GPU)

### Frontend

- **Navegador**: Chrome 90+, Firefox 85+, Edge 90+
- **Webcam**: qualquer USB ou integrada
- **CPU**: necessário para MediaPipe (executa no browser)
- **RAM browser**: ~500 MB (MediaPipe + React)

## Estimativa de Espaço

| Item | Tamanho |
|------|---------|
| Dataset bruto (vídeos) | ~5 GB |
| Landmarks (.npz) | ~2 GB |
| Dataset processado | ~1.5 GB |
| Modelo treinado (.pt) | ~5-50 MB |
| Checkpoints (top 3) | ~150 MB |
| Docker images | ~3 GB |

## Consumo de GPU (durante treino)

| Modelo | VRAM (batch=32) | VRAM (batch=16) |
|--------|----------------|----------------|
| Bi-LSTM | ~2.5 GB | ~1.5 GB |
| Transformer | ~3.0 GB | ~2.0 GB |
| TCN | ~2.0 GB | ~1.2 GB |

## Notas

1. Mixed precision (AMP) reduz VRAM em ~40%
2. `num_workers` para DataLoader: use N = min(4, cores_disponíveis)
3. Em CPU, reduza `batch_size` para 16 para evitar thrashing
4. A extração de landmarks (MediaPipe) é CPU-only
