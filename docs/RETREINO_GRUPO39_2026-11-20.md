# Retreino sem Cinco, Vacina e Amarelo

## Plano autorizado em 20/11/2026

1. Partir dos índices do modelo padrão de 41 classes, preservando pessoas e
   splits, e excluir Cinco, Vacina e Amarelo. Cinco já estava ausente: restam 39.
2. Criar índices separados (635 vídeos; 469/83/83), sem apagar vídeos ou landmarks.
3. Treinar um novo Transformer v2 do zero com a mesma arquitetura, parâmetros,
   augmentation e seed 42. Selecionar por F1 macro de validação, com early stopping.
4. Manter o teste reservado (`--skip-test`); arquivar checkpoint, índices,
   configurações, métricas, logs e hashes. Não inferir métricas de webcam.
5. Somente depois do treino concluído, definir o novo checkpoint como padrão,
   atualizar Docker e guias, e testar o carregamento real por API/WebSocket.

## Resultado

Índices preparados: 39 classes, 635 exemplos, splits 469/83/83. Todas as
sequências retidas existem e preservam os caminhos originais. A partição de
exemplos e ausência de pessoas em múltiplos splits foram verificadas.
Teste de remapeamento e regressão antes da ativação: 27 passaram (dois avisos de
depreciação de dependências). Retreino concluído e novo padrão ativado.

| Resultado registrado | Valor |
|---|---:|
| Melhor época (F1 macro) | 57 |
| Última época (early stopping) | 82 |
| Duração | 353,25 s |
| Validação top 1 | 52/83 — 62,65% |
| Validação top 3 | 71/83 — 85,54% |
| Validação top 5 | 77/83 — 92,77% |
| Validação F1 macro | 0,673681 |
| Validação MINDS | 30/55 — 54,55% |
| Validação grupo | 22/28 — 78,57% |

O treino teve `train_top1=0,9723` na época selecionada, com augmentation e sampler
balanceado. Os metadados não exigiram fallback de FPS ou dimensões.
O modelo anterior de 41 classes tinha 70/93 (75,27%) e F1 macro 0,7512 na
validação. O retreino ficou abaixo desse resultado; são vocabulários e conjuntos
de validação diferentes, e não foi executada comparação pareada nem várias
sementes para atribuir a diferença somente à exclusão. Não se presume melhora
de acurácia. O novo modelo foi ativado conforme o pedido de remover as palavras.

Os 83 vídeos reservados de teste **não foram avaliados**. Não houve teste humano
na webcam. O vocabulário não inclui `_nenhum_sinal` treinado.

Checkpoint padrão: `Modelos salvos/transformer_v2_grupo39_full/best.pt`.
SHA-256: `2782ff6651c8e6234742abdb2ea78358ecbcbf50194ae85a71a526f4a01777ab`.
O checkpoint de 41 classes e seus registros foram preservados.

## Verificação da ativação

`backend/config.py` aponta para o novo checkpoint; o manifesto de fusão permanece
vazio. Dockerfile inclui o modelo e o healthcheck do Compose espera 39 classes.
README e guias atuais foram atualizados.

```bash
./.venv/Scripts/python.exe -m pytest tests/test_grupo39_dataset.py tests/test_default_fusion25_startup.py tests/test_default_v2_websocket.py tests/test_config.py tests/test_web_video.py tests/test_collection_v2.py tests/test_live_fps_v2.py tests/test_training_v2.py tests/test_api.py tests/test_pipeline_v2.py -q
```

Após a troca, **27 testes passaram**, com dois avisos de depreciação.
Carregamento dos pesos reais, API, janela WebSocket, ausência das classes
excluídas e compatibilidade explícita com a fusão v1 foram verificados.
A suíte completa e a construção Docker não foram executadas nesta alteração.

Para iniciar o novo padrão no Git Bash, pare o backend anterior e rode:

```bash
unset SINALIZA_MODEL_PATH SINALIZA_FUSION_MANIFEST_PATH
./.venv/Scripts/python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

O `.env` local não contém overrides de modelo na verificação desta execução.
Se outra máquina tiver esses campos, remova-os para usar o padrão.
`/api/health` deve informar `model_loaded: true` e `num_classes: 39`.
`/api/model/info` deve informar `model_type: transformer_v2`, `pipeline_version: 2`
e a lista sem Cinco, Vacina e Amarelo.

## Reprodução no Git Bash (raiz do projeto)

```bash
./.venv/Scripts/python.exe scripts/prepare_grupo39_dataset.py
./.venv/Scripts/python.exe scripts/train.py --config configs/transformer_v2_grupo39.yaml --data-dir data/landmarks/mixed_grupo39_20261120 --processed-dir data/processed/mixed_grupo39_20261120 --experiment transformer_v2_grupo39_20261120_seed42_repro --num-workers 0 --skip-test
```

O preparo requer o manifesto e os splits originais em
`data/processed/mixed_grupo_20261117` e seus `.npz` nas pastas referenciadas.
Não reextrai nem apaga landmarks. Cinco já estava ausente, enquanto Vacina e
Amarelo somavam 80 vídeos (60 treino, 10 validação e 10 teste).
Os novos índices ficam em `data/landmarks/mixed_grupo39_20261120` e
`data/processed/mixed_grupo39_20261120`. O mapa é reindexado em ordem alfabética.
Não use o mapa anterior para interpretar os logits do modelo novo.

O retreino desta execução usa o nome sem sufixo `_repro`. Seu log persistente
fica em `logs/experiments/transformer_v2_grupo39_20261120_seed42/train.log`;
`run_record.json` registra a configuração, os hashes dos índices e o resultado.
O console também foi salvo em `logs/treino_grupo39_20261120_console.log`.

## Arquivamento e ativação

Depois de concluído o experimento original:

```bash
./.venv/Scripts/python.exe scripts/archive_transformer_v2_grupo39.py
```

O script verifica status concluído, época selecionada, hashes dos índices,
configuração e ausência das três palavras. Só então copia para
`Modelos salvos/transformer_v2_grupo39_full/`, com configurações, índices,
histórico, métricas, logs, README e checksums. O script é específico do nome do
experimento desta execução; para arquivar uma reprodução, ajuste a origem.
