# Ativação da fusão de 25 classes como padrão da aplicação

Data: 04/11/2026. [Plano](PLANO_ATIVACAO_FUSAO25_PADRAO_2026-11-04.md).

## O que mudou

`backend/config.py` define `fusion_manifest_path="configs/fusion25_frozen.json"` por padrão. Ao iniciar o FastAPI a partir da raiz do repositório, `PredictionService.load_model()` lê esse manifesto e carrega dois Transformers já arquivados:

1. `Modelos salvos/transformer_mixed25_pose_face/best.pt`: pose + 40 pontos faciais, sem detalhes das mãos; propõe as cinco classes candidatas.
2. `Modelos salvos/transformer_mixed25_full/best.pt`: pose + 40 pontos faciais + mãos; reordena as candidatas pela regra congelada `0,22 × pose_face + 0,78 × full`.

Os dois modelos têm o mesmo mapa de 25 classes e recebem janelas de 48 quadros. O carregador verifica os hashes dos checkpoints e do mapa de classes. O modelo antigo de oito classes continua arquivado e pode ser carregado explicitamente via `PredictionService.load_model(model_path=...)`; a variável `SINALIZA_FUSION_MANIFEST_PATH` permite selecionar outro manifesto para experimentos. Nenhuma variável precisa ser definida para o uso normal.

O endpoint `/api/model/info` agora informa `model_type=transformer_fusion`, 25 classes, dispositivo e quantidade total de parâmetros da dupla. O WebSocket anuncia 25 classes e `feature_mode=fusion`. A imagem do backend no `Dockerfile.backend` passou a incluir os dois checkpoints de 25 classes; o healthcheck do Compose usa `/api/health` e confere que o modelo está carregado. A configuração Docker não pôde ser executada aqui porque o comando `docker` não está instalado neste ambiente.

## Como iniciar no Git Bash do VS Code

A partir da raiz `DeepLearning-SINALIZA`, em terminais separados:

```bash
./.venv/Scripts/python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

```bash
cd frontend
npm run dev
```

Abra `http://localhost:5173`. Em `http://127.0.0.1:8000/api/health`, procure `"model_loaded": true` e `"num_classes": 25`. Em `/api/model/info`, o `model_type` deve ser `transformer_fusion`. O diretório de trabalho deve ser a raiz do repositório, pois o manifesto contém caminhos relativos a ela.

## Verificação e limites

`./.venv/Scripts/python.exe -m pytest tests -q`: **122 testes passaram**. Isso inclui inicialização real dos checkpoints, checagem das classes e feature modes, transporte de 48 quadros pelo WebSocket, compatibilidade com a avaliação offline e regressão do código de fusão. Houve dois avisos de depreciação em bibliotecas, sem falhas. `git diff --check` também passou.

A regra original atingiu 50/85 acertos na validação e 56/85 no teste exploratório já consultado em experimentos anteriores. Essas métricas não demonstram desempenho em pessoas inéditas. O classificador não tem classe treinada de “nenhum sinal”; o limiar de confiança continua em 0,30, e uma pessoa parada ainda pode receber uma palavra incorreta. O dicionário da interface é um catálogo independente do vocabulário do checkpoint.
