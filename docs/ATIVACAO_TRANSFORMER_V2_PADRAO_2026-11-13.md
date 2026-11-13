# Transformer v2 como modelo principal

Data: 13/11/2026. Pedido do usuário: trocar o padrão antigo pelo novo `transformer_v2`. [Plano antes da alteração](PLANO_ATIVACAO_TRANSFORMER_V2_PADRAO_2026-11-13.md).

## Configuração ativa

`backend/config.py` define `model_path="Modelos salvos/transformer_v2_mixed23_full/best.pt"` e `fusion_manifest_path=None`. Ao iniciar sem variáveis de seleção, o backend carrega **um Transformer v2 completo de 23 classes**, com pose, face e mãos. Não há dupla de modelos nem BiLSTM nesse caminho padrão.

O checkpoint é o `best.pt` do experimento `transformer_v2_mixed23_seed42`, escolhido na época 69 por F1 macro da validação. Seu hash SHA-256 é `245728571a4336c48c7f751f56e23dec29320b1141e3a70bf06433a530bbdc7b`. O arquivamento verificou versão 2, 725 features, 23 classes, parâmetros temporais, mapa de rótulos e hashes dos manifestos usados no treino. Configurações, índices, histórico, métricas de validação e logs foram copiados junto ao modelo; `checksums.sha256` lista a integridade dos arquivos. `scripts/archive_transformer_v2_mixed23.py` reproduz esse arquivamento, recusando sobrescrever arquivos divergentes.

O treino não foi repetido e os 75 vídeos de teste não foram avaliados. A validação registrada permanece 60/75 (80%) e F1 macro 0,8176. Veja [resultado do treino](RESULTADO_TREINO_V2_MIXED23_2026-11-12.md).

O backend usa a configuração temporal salva no checkpoint: grade de 15 FPS, janela de 3,2 s e novas tentativas a cada 0,4 s, sujeitas à chegada de quadros e cooldown. O pré-processamento é o compartilhado da v2. A suavização EMA e o limiar de exibição continuam configuráveis. Este checkpoint não tem `_nenhum_sinal` treinado; pode produzir palpites fora do vocabulário ou em repouso.

`Dockerfile.backend` inclui o novo arquivo de modelos, e o healthcheck do Compose espera 23 classes. Os modelos históricos continuam arquivados e selecionáveis por manifesto, por exemplo `SINALIZA_FUSION_MANIFEST_PATH=configs/fusion25_frozen.json`.

## Como iniciar

Pare o backend anterior com `Ctrl+C`. Na raiz do projeto, no Git Bash:

```bash
# Apenas se esse terminal já tinha variáveis de seleção exportadas:
unset SINALIZA_FUSION_MANIFEST_PATH SINALIZA_MODEL_PATH
./.venv/Scripts/python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Em outro terminal, `npm --prefix frontend run dev`. Abra `http://localhost:5173`.

Confira `http://127.0.0.1:8000/api/health`: `model_loaded: true`, `num_classes: 23`. Em `http://127.0.0.1:8000/api/model/info`, confira `model_type: transformer_v2`, `pipeline_version: 2`, `feature_dim: 725`, `target_fps: 15` e `window_seconds: 3.2`. A lista `labels` identifica as palavras do checkpoint, sem Cinco nem Espelho.

Não havia `.env` local nesta ativação. Se for criado um `.env` ou houver variáveis exportadas, elas têm precedência sobre os padrões do código; remova seleções antigas caso a API anuncie outro modelo. A inferência não precisa dos vídeos nem dos landmarks de treinamento, apenas do checkpoint e do código.

## Verificação

O comando abaixo passou com **18 testes**, sem falhas:

```bash
./.venv/Scripts/python.exe -m pytest tests/test_default_fusion25_startup.py tests/test_default_v2_websocket.py tests/test_config.py tests/test_web_video.py tests/test_collection_v2.py tests/test_live_fps_v2.py tests/test_training_v2.py tests/test_api.py -q
```

A inicialização FastAPI foi exercitada com o arquivo v2 real. No WebSocket, 49 JPEGs em 15 FPS simulados cobriram 3,2 s antes da classificação, com overlay e resposta válidos. O teste usa extrator determinístico e os pesos reais; não mede acurácia da webcam. O caminho de fusão v1 foi testado explicitamente para preservar compatibilidade. Os testes também cobrem coleta, sampler, treino sintético, metadados e estado negativo opcional.

Dois avisos preexistentes de descontinuação (Pydantic e Starlette/httpx) permanecem. As seis comparações históricas de probabilidades que divergiram após reextração dos landmarks, documentadas no guia mixed23, não foram alteradas nem reexecutadas nesta ativação. Webcam com pessoa real e Docker em execução não foram verificados. A validação em 23 classes não permite afirmar superioridade frente aos testes históricos de 25 classes.
