# Ativação do Transformer v2 de 41 classes

Em 17/11/2026, o backend passou a selecionar por padrão
`Modelos salvos/transformer_v2_grupo41_full/best.pt`. É um único Transformer
com pose, face e mãos, 725 features, 15 FPS e janela de 3,2 s (48 amostras).
Não houve novo treino nem avaliação do conjunto de teste nesta ativação.

## Origem e preservação

- Experimento: `transformer_v2_grupo_expandido_20261117_seed42`, concluído.
- Checkpoint selecionado: época 80, pelo F1 macro de validação; parada na época 105.
- Validação já registrada: 70/93 (75,27%), F1 macro 0,7512. Não mede a webcam.
- Dados: 13 classes MINDS + 28 do grupo; 529/93/93 vídeos por split, separados por pessoa/origem.
- SHA-256 de `best.pt`: `326bca14a17846a193811bf16b666ff06bfd1397440be8881cb14a4b5f944cd4`.

`scripts/archive_transformer_v2_grupo41.py` verificou os hashes registrados do
treino, os metadados, a configuração e o mapa de 41 classes antes de copiar os
pesos. O arquivo arquivado preserva o conteúdo do checkpoint original.
`config/`, `data_split/`, `records/`, README e `checksums.sha256` acompanham os
pesos. Os modelos anteriores permanecem disponíveis.

O primeiro teste detectou opções adicionais nos metadados do treino. A dataclass
agora aceita explicitamente os valores equivalentes à geometria original:
`global_depth=true`, `scale_2d=false`, `point_visibility=false`,
`segmentation=false`, `visibility_threshold=0.5`. Valores experimentais distintos
são recusados, evitando carregamento com pré-processamento incompatível.
Nenhuma coordenada ou peso foi alterado para resolver essa compatibilidade.

`backend/config.py` define o novo caminho e mantém o manifesto de fusão vazio.
O Dockerfile inclui o arquivo e o healthcheck do Compose espera 41 classes.

## Iniciar no Git Bash

Na raiz do projeto, pare o backend anterior com Ctrl+C e execute:

```bash
unset SINALIZA_MODEL_PATH SINALIZA_FUSION_MANIFEST_PATH
./.venv/Scripts/python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Não há `.env` local nesta ativação; em outra máquina, remova também seus valores
antigos de `SINALIZA_MODEL_PATH` e `SINALIZA_FUSION_MANIFEST_PATH` se existirem.
Os overrides continuam tendo prioridade para permitir escolher outros modelos.

Em outro terminal:

```bash
cd frontend
npm run dev
```

Confira `/api/health`: `model_loaded: true`, `num_classes: 41`.
Em `/api/model/info`: `model_type: transformer_v2`, `pipeline_version: 2`,
`feature_dim: 725`, `target_fps: 15`, `window_seconds: 3.2`.
A lista inclui Escola, Diretoria, Dor e Querer; Cinco e Espelho continuam fora.
O mapa completo está em `data_split/label_map.json` do modelo arquivado.

## Verificação e limites

Executado com os pesos reais:

```bash
./.venv/Scripts/python.exe -m pytest tests/test_default_fusion25_startup.py tests/test_default_v2_websocket.py tests/test_config.py tests/test_web_video.py tests/test_collection_v2.py tests/test_live_fps_v2.py tests/test_training_v2.py tests/test_api.py tests/test_pipeline_v2.py -q
```

Resultado: **26 passaram**, dois avisos de depreciação de dependências.
Cobertura: inicialização e mapa reais, API, janela temporal WebSocket com
landmarks sintéticos, coleta, compatibilidade com a fusão v1, invariâncias e
recusa de geometria experimental. O teste sintético de treino usa dados
temporários e não retreina o modelo arquivado.

A suíte completa e a imagem Docker não foram executadas nesta ativação. Não
houve teste humano na webcam nem consulta aos 93 exemplos reservados de teste.
O modelo não contém uma classe negativa `_nenhum_sinal` treinada.

## Reproduzir o treino separadamente

Requer os landmarks e índices locais usados no experimento; os pesos não os
substituem. A configuração e os índices estão arquivados para conferência.

```bash
./.venv/Scripts/python.exe scripts/train.py --config configs/transformer_v2_grupo_expandido.yaml --data-dir data/landmarks/mixed_grupo_20261117 --processed-dir data/processed/mixed_grupo_20261117 --experiment transformer_v2_grupo41_seed42_repro --num-workers 0 --skip-test
```

Para voltar ao modelo anterior sem editar o padrão:

```bash
export SINALIZA_MODEL_PATH="Modelos salvos/transformer_v2_mixed23_full/best.pt"
unset SINALIZA_FUSION_MANIFEST_PATH
./.venv/Scripts/python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```
