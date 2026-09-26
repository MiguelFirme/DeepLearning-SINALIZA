# ✋ Sinaliza — Tradutor de Libras Inteligente

<p align="center">
  <strong>Tradução em tempo real de Libras (Língua Brasileira de Sinais) para Português</strong><br>
  Projeto acadêmico · Visão Computacional · Deep Learning
</p>

---

## Sobre o Projeto

O **Sinaliza** é uma plataforma que utiliza inteligência artificial para traduzir sinais isolados de Libras captados via webcam em texto em Português, em tempo real. O projeto nasceu na **EMEB Polo de Surdos Profª Maria de Lourdes Carneiro** em Criciúma/SC, com o objetivo de promover a inclusão e a autonomia da comunidade surda.

### Funcionalidades

- **Tradução em tempo real** — webcam → landmarks → modelo → texto
- **Três arquiteturas de modelo** — Bi-LSTM, Transformer 1D, TCN
- **Dicionário de sinais** — busca por categoria e texto
- **Text-to-Speech** — leitura em voz do sinal detectado
- **Pipeline ML completo** — download, extração, treino, avaliação, comparação

---

## Arquitetura

```
Webcam → MediaPipe Holistic → 346 landmarks/frame
       → Normalização (ombros) → Sequência (48 frames)
       → Modelo (BiLSTM | Transformer | TCN)
       → Top-K predições → Smoothing EMA
       → Texto em Português
```

### Stack Tecnológica

| Camada      | Tecnologia                                      |
|-------------|------------------------------------------------|
| ML          | PyTorch, MediaPipe, NumPy, scikit-learn         |
| Backend     | FastAPI, Uvicorn, WebSocket, Pydantic           |
| Frontend    | React 18, TypeScript, Vite, Lucide Icons        |
| Infra       | Docker, Docker Compose                          |
| Dataset     | V-Librasil (Kaggle) — ~4088 vídeos, ~1364 classes |

---

## Estrutura do Projeto

```
Sinaliza/
├── ml/                          # Pipeline de Machine Learning
│   ├── features/                # Extração e normalização de landmarks
│   ├── models/                  # BiLSTM, Transformer, TCN
│   ├── data/                    # Dataset, augmentação, split
│   ├── training/                # Trainer, losses, métricas
│   ├── inference/               # Preditor com smoothing
│   └── evaluation/              # Avaliação e relatórios
├── backend/                     # API FastAPI
│   ├── routers/                 # Endpoints REST + WebSocket
│   ├── services/                # Serviço de predição
│   ├── schemas/                 # Modelos Pydantic
│   └── middleware/              # Logging
├── frontend/                    # React + TypeScript
│   └── src/
│       ├── components/          # Navbar, WebcamView, TranslationDisplay
│       ├── pages/               # Home, Dictionary, Settings
│       ├── hooks/               # useMediaPipe, useWebSocket
│       └── services/            # Cliente API
├── scripts/                     # Scripts de pipeline
├── configs/                     # Configurações YAML por modelo
├── tests/                       # Testes unitários
├── docs/                        # Documentação técnica
└── notebooks/                   # Notebooks para Colab
```

---

## Início Rápido

### Pré-requisitos

- Python 3.10+
- Node.js 18+
- GPU com CUDA (recomendado para treino, CPU funciona)
- Conta Kaggle (para download do dataset)

### 1. Instalação

```bash
git clone https://github.com/seu-usuario/sinaliza.git
cd sinaliza

# Backend
pip install -r requirements.txt

# Frontend
cd frontend && npm install && cd ..
```

### 2. Download do Dataset

```bash
# Configure ~/.kaggle/kaggle.json primeiro
python scripts/download_dataset.py --output data/raw
```

### 3. Pipeline de Preparação

```bash
# Extrair landmarks com MediaPipe
python scripts/extract_landmarks.py \
  --input data/raw/v-librasil \
  --output data/landmarks \
  --workers 4

# Validar integridade
python scripts/validate_landmarks.py --dir data/landmarks

# Construir splits de treino/val/teste
python scripts/build_dataset.py \
  --landmarks data/landmarks \
  --output data/processed
```

### 4. Treinamento

#### Experimento recomendado: dois articuladores contra o terceiro

Comece com 50 classes. O comando executa três modelos independentes, cada um
reservando um articulador diferente para o teste:

```bash
python scripts/run_signer_cv.py \
  --config configs/bilstm.yaml \
  --max-classes 50 \
  --epochs 50 \
  --num-workers 0
```

Antes de treinar, é possível apenas conferir e gerar os splits:

```bash
python scripts/run_signer_cv.py --max-classes 50 --prepare-only
```

Para executar somente uma rodada, use `--round 1`, `--round 2` ou `--round 3`.
Se o processo for interrompido, execute o mesmo comando novamente: rodadas com
`test_metrics.json` são reutilizadas. Use `--force` somente para treiná-las de
novo.

As saídas ficam em:

```text
experiments/bilstm_signer_cv_50classes_50epochs_seed42/
├── run_state.json
├── summary.json
├── summary.csv
├── articulador1/
├── articulador2/
└── articulador3/
```

Consulte o procedimento completo em
[`docs/TRAINING.md`](docs/TRAINING.md) e o histórico da implementação em
[`docs/SIGNER_CV_EXECUTION.md`](docs/SIGNER_CV_EXECUTION.md).

#### Treinamento manual de um split já preparado

```bash
# Treinar Bi-LSTM
python scripts/train.py --config configs/bilstm.yaml

# Treinar Transformer
python scripts/train.py --config configs/transformer.yaml

# Treinar TCN
python scripts/train.py --config configs/tcn.yaml
```

### 5. Avaliação e Comparação

```bash
# Avaliar um modelo
python scripts/evaluate.py \
  --checkpoint artifacts/checkpoints/best_bilstm.pt \
  --config configs/bilstm.yaml

# Comparar os 3 modelos
python scripts/compare_models.py \
  --checkpoints artifacts/checkpoints/best_*.pt \
  --output artifacts/comparison
```

### 6. Executar Aplicação

```bash
# Backend
uvicorn backend.main:app --reload --port 8000

# Frontend (outro terminal)
cd frontend && npm run dev
```

Acesse **http://localhost:5173**

---

## Docker

```bash
docker compose up --build
```

| Serviço  | URL                      |
|----------|--------------------------|
| Frontend | http://localhost:5173     |
| Backend  | http://localhost:8000     |
| API Docs | http://localhost:8000/docs|

---

## Modelos

| Modelo       | Descrição                                   | Pontos Fortes                        |
|-------------|---------------------------------------------|--------------------------------------|
| **Bi-LSTM** | LSTM bidirecional + atenção temporal         | Captura dependências longas          |
| **Transformer** | Encoder com CLS token + positional encoding | Atenção multi-head, paralelizável |
| **TCN**     | Conv 1D causal dilatada                      | Eficiente, campo receptivo grande    |

### Features por Frame (346 dimensões)

| Componente | Landmarks | Dims/Landmark | Total |
|-----------|-----------|---------------|-------|
| Mão esquerda | 21 | 3 (x,y,z) | 63 |
| Mão direita | 21 | 3 (x,y,z) | 63 |
| Pose (torso) | 25 | 4 (x,y,z,vis) | 100 |
| Face (seleção) | 40 | 3 (x,y,z) | 120 |
| **Total** | **107** | — | **346** |

Após normalização, são acrescentadas 346 velocidades e 5 distâncias, formando
697 features de entrada por frame.

---

## API

### REST Endpoints

| Método | Rota             | Descrição                         |
|--------|------------------|-----------------------------------|
| GET    | `/health`        | Status da API e modelo            |
| GET    | `/model/info`    | Informações do modelo carregado   |
| POST   | `/predict`       | Predição a partir de landmarks    |
| POST   | `/predict/reset` | Reset do estado de smoothing      |
| GET    | `/dictionary`    | Dicionário com busca e categorias |

### WebSocket

`ws://localhost:8000/ws/predict` — enviar landmarks em tempo real, receber predições.

---

## Testes

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests/ -v
```

---

## Documentação Adicional

- [Guia de Treinamento](docs/TRAINING.md)
- [Arquitetura Técnica](docs/ARCHITECTURE.md)
- [Requisitos de Hardware](docs/HARDWARE.md)

---

## Dataset

O projeto utiliza o **V-Librasil** disponível no Kaggle:

- ~4.088 vídeos
- ~1.364 classes de sinais
- ~3 amostras por classe
- Resolução variada

> **Importante**: Não utilizar espelhamento horizontal na augmentação — a lateralidade é significativa em Libras.

---

## Licença

Este projeto é distribuído sob a licença MIT. Veja [LICENSE](LICENSE) para mais detalhes.

---

## Créditos

- **Origem**: EMEB Polo de Surdos Profª Maria de Lourdes Carneiro — Criciúma, SC
- **Dataset**: V-Librasil (Kaggle)
- **MediaPipe**: Google
- **Framework**: PyTorch
