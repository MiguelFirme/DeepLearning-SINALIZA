# Arquitetura Técnica — Sinaliza

## Diagrama de Alto Nível

```
┌──────────────┐     WebSocket      ┌──────────────┐
│   Frontend   │ ◄──────────────► │   Backend    │
│  React + TS  │     REST API      │   FastAPI    │
│              │ ◄──────────────► │              │
└──────┬───────┘                    └──────┬───────┘
       │                                    │
  MediaPipe                          SignPredictor
  Holistic                           (PyTorch)
       │                                    │
   landmarks                          modelo .pt
   346 feat/frame                   label_map.json
```

## Pipeline de Features

### 1. Extração de Landmarks (MediaPipe Holistic)

De cada frame de vídeo, o MediaPipe extrai pontos 3D:

| Região | Landmarks | Coordenadas | Subtotal |
|--------|-----------|-------------|----------|
| Mão esquerda | 21 | x, y, z | 63 |
| Mão direita | 21 | x, y, z | 63 |
| Pose (torso) | 25 | x, y, z, visibility | 100 |
| Face (seleção) | 40 | x, y, z | 120 |
| **Total** | **107** | — | **346** |

A seleção facial inclui apenas contorno de lábios, sobrancelhas e contorno do rosto (40 landmarks), relevantes para expressão em Libras.

### 2. Normalização

- Centralização nos ombros (landmarks 11/12 do pose)
- Escala normalizada pela distância inter-ombros
- Features derivadas opcionais: velocidade, aceleração, distâncias entre mãos

### 3. Processamento de Sequência

- Target: 48 frames por amostra
- Sequências longas: amostragem uniforme
- Sequências curtas: padding com zeros + máscara
- Interpolação opcional para suavizar

## Modelos

### A. Bi-LSTM

```
Input (B, T, 346)
  → LayerNorm → Linear(346, proj_dim)
  → BiLSTM × N camadas
  → Temporal Attention (query automática)
  → Classifier (Linear → ReLU → Dropout → Linear)
  → Logits (B, C)
```

### B. Transformer 1D

```
Input (B, T, 346)
  → Linear(346, d_model)
  → [CLS] token + Positional Encoding (aprendido)
  → TransformerEncoder × N camadas (PreNorm)
  → CLS token → LayerNorm
  → Classifier
  → Logits (B, C)
```

### C. TCN (Temporal Convolutional Network)

```
Input (B, T, 346) → transpose → (B, 346, T)
  → TCN Block × N (Conv1d causal dilatado + residual)
  → Global Average Pooling + Global Max Pooling → concat
  → Classifier
  → Logits (B, C)
```

## Treinamento

- **Otimizador**: AdamW (weight_decay=0.01)
- **Scheduler**: Cosine Annealing com warmup linear
- **Loss**: Label Smoothing Cross-Entropy (ε=0.1)
- **Pesos de classe**: Effective number weighting (β=0.999)
- **AMP**: Mixed precision com GradScaler
- **Early Stopping**: patience=15, monitor val_acc
- **Gradient Clipping**: max_norm=1.0

## Inferência em Tempo Real

### Fluxo WebSocket

```
Frontend                         Backend
   │                                │
   ├── landmarks[] ──────────────►  │
   │                            predict()
   │                                │
   │  ◄───── {sign, confidence} ── │
   │                                │
   ├── landmarks[] ──────────────►  │
   │         ...                    │
```

### Smoothing de Predições

- **EMA** (default): média móvel exponencial das probabilidades
- **Majority**: voto majoritário em janela deslizante
- **Cooldown**: após detectar sinal, pausa N frames para evitar repetição

## Backend (FastAPI)

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/health` | GET | Health check |
| `/model/info` | GET | Info do modelo |
| `/predict` | POST | Predição batch (REST) |
| `/predict/reset` | POST | Reset do smoothing |
| `/dictionary` | GET | Dicionário de sinais |
| `/ws/predict` | WS | Predição real-time |

## Frontend (React)

### Páginas

1. **Home** — Webcam + tradução em tempo real + painel de confiança
2. **Dictionary** — Dicionário com categorias + busca
3. **Settings** — Sobre + contato

### Hooks Principais

- `useMediaPipe` — gerencia webcam + extração de landmarks
- `useWebSocket` — conexão WS para predição em tempo real

## Decisões de Design

1. **Sem espelhamento horizontal** — lateralidade é significativa em Libras
2. **48 frames como default** — bom equilíbrio entre contexto e eficiência
3. **346 features/frame** — inclui face selecionada para expressões
4. **EMA smoothing** — suaviza predições sem introduzir latência
5. **Label smoothing** — combate overfitting em dataset pequeno (~3 amostras/classe)
6. **3 modelos** — permite comparação acadêmica de arquiteturas
