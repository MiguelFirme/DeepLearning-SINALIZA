# Pipeline v2 do Sinaliza

Implementação em fases conforme [plano](PLANO_PIPELINE_V2.md). A extração continua com 346 valores e máscara `(T,4)`. Os arquivos brutos não são modificados.

## Features e ordem

O construtor compartilhado `build_features_v2` recebe amostras em grade temporal uniforme. Corrige x pelo aspecto largura/altura; mantém somente os primeiros 17 pontos de pose e remove visibilidade; preenche lacunas internas de mãos de até cinco amostras; centraliza e escala pelos ombros com referência mediana quando a pose falta; calcula forma local por pulso/base do dedo médio; acrescenta cinco distâncias e velocidades globais em unidades por segundo. Lacunas maiores e ausências nas extremidades ficam zeradas. Sem pose válida na janela, a referência é centro zero e escala um; o bloqueio de rastreamento da aplicação continua necessário.

Dimensão e offsets: coordenadas globais `0:297` (126 mãos + 51 pose + 120 face), forma local das mãos `297:423`, cinco distâncias `423:428`, velocidades globais `428:725`. **Total: 725**. As velocidades são diferenças multiplicadas pelo FPS alvo e zeradas se uma das duas detecções falta. O primeiro vetor de velocidade é zero. A forma local das mãos não recebe outro bloco de velocidades.

## Verificações por fase

- Fase 1: `pytest tests -q`: **122 passaram, 8 ignorados** por datasets locais opcionais ausentes. Testados aspecto, remoção de pernas/visibilidade, invariância local de mão e lacunas curtas/longas.
- Fase 2: `pytest tests -q`: **126 passaram, 8 ignorados**. Reamostragem física, bordas repetidas, validade por parte, janelas com cobertura de 80%, dataset determinístico e velocidade equivalente a 9/30 FPS verificados com dados sintéticos.
- Fase 3: `pytest tests -q`: **128 passaram, 8 ignorados**. Espelhamento duplo, troca das mãos, partes ausentes zeradas e máscara/timestamps sincronizados após descarte foram verificados.
- Fase 4: `pytest tests -q`: **130 passaram, 8 ignorados**. Um treino sintético de uma época verificou metadados do checkpoint, F1 como monitor, matrizes por origem, carregamento v2 e contrato `_nenhum_sinal`. O carregamento padrão v1 e WebSocket existente também passaram. Não é um experimento de acurácia.
- Fase 5: `pytest tests -q`: **132 passaram, 8 ignorados**; build TypeScript/Vite passou (1.593 módulos). Coleta por WebSocket, consentimento obrigatório, igualdade dos landmarks salvos, timestamps e splits por pessoa foram verificados. Câmera humana e contagem na interface ainda precisam de execução manual.

## Coleta na interface

Inicie backend e frontend normalmente e abra **Coleta** (`http://localhost:5173/collection`). A coleta funciona com v1 ou v2 carregada. Inicie a câmera, use IDs estáveis como `P01` e `sessao01`, escolha uma palavra do mapa do modelo ou `_nenhum_sinal`, informe a duração (padrão 4 s) e confirme consentimento. Clique **Gravar exemplo**, aguarde 3 s e execute o gesto completo. **Cancelar e descartar** não salva uma gravação parcial. Se a conexão cair, repita a tentativa. O backend pode finalizar pela duração ou por controle explícito; precisa de pelo menos dois quadros.

São salvos apenas landmarks, máscaras e horários — não vídeo nem áudio — em `data/coleta/<pessoa>/<palavra>/<timestamp>.npz`, no computador do backend. Metadados: largura/altura, pessoa, sessão, palavra, data UTC, versão do app, FPS observado, duração solicitada/observada e consentimento. As colunas são exatamente as mesmas da inferência, antes de qualquer aumento ou feature. O relógio é monotônico; empates de ticks são desambiguados pelo menor incremento representável. O diretório está no `.gitignore`.

Depois de coletar todas as palavras com pessoas separadas para os três conjuntos, gere o dataset (IDs abaixo são exemplos, substitua pelos reais):

```bash
./.venv/Scripts/python.exe scripts/build_coleta_manifest.py --input data/coleta --output data/landmarks/coleta_v2 --processed-dir data/processed/coleta_v2 --train-persons P01 P02 P03 --val-persons P04 --test-persons P05
./.venv/Scripts/python.exe scripts/train.py --config configs/transformer_v2_mixed25.yaml --data-dir data/landmarks/coleta_v2 --processed-dir data/processed/coleta_v2 --experiment transformer_v2_coleta_seed42 --num-workers 0 --skip-test
```

O script não copia nem altera `.npz`; cria caminhos relativos no manifesto e separa por identidade, inclusive entre sessões. Recusa pessoas em dois splits, pessoas não atribuídas, arquivos duplicados, ausência de consentimento/dimensões e classes sem treino. Saídas existentes exigem outro diretório ou `--force`. Para mesclar, adicione `--merge-manifest data/landmarks/mixed_25/manifest.json` e inclua **todas as pessoas** de ambas as origens nos três grupos. Use `--person-map caminho/identidades.json` para mapear IDs antigos e novos para a mesma identidade real; caso contrário, nomes diferentes da mesma pessoa não podem ser detectados automaticamente. Confirme essa identidade antes de mesclar para evitar vazamento.

## Treinar e avaliar por pessoa

Na raiz, com os `.npz` originais e os três arquivos `manifest.json`, `splits.json` e `label_map.json` em `data/processed/mixed_25` correspondentes ao manifesto de `data/landmarks/mixed_25`:

```bash
./.venv/Scripts/python.exe scripts/train.py --config configs/transformer_v2_mixed25.yaml --data-dir data/landmarks/mixed_25 --processed-dir data/processed/mixed_25 --experiment transformer_v2_mixed25_seed42 --num-workers 0 --skip-test
./.venv/Scripts/python.exe scripts/run_signer_cv.py --config configs/transformer_v2_mixed25.yaml --data-dir data/landmarks/mixed_25 --processed-root data/processed/signer_cv_v2 --run-name transformer_v2_cv_seed42 --epochs 150 --num-workers 0 --prepare-only
# Após conferir os splits por pessoa, execute sem --prepare-only para treinar todos os folds.
./.venv/Scripts/python.exe scripts/run_signer_cv.py --config configs/transformer_v2_mixed25.yaml --data-dir data/landmarks/mixed_25 --processed-root data/processed/signer_cv_v2 --run-name transformer_v2_cv_seed42 --epochs 150 --num-workers 0
```

A CV v2 descobre as pessoas no manifesto: uma no teste, a próxima na validação e as restantes no treino; nenhum peso é reutilizado entre folds. `--round 1` permite uma rodada; `--signers` permite informar as identidades exatas do manifesto. Os índices e arquivos brutos são preservados. Classes sem treino ou identidade desconhecida geram erro. Cada teste é avaliado após escolher o checkpoint pelo F1 macro da validação. `by_origin` e `confusion_matrix` ficam nos JSONs e no terminal. Origens sem amostras são registradas com contagem zero e métricas nulas.

Configuração: Transformer 128/4 cabeças/3 camadas/dropout 0,15, AdamW 1e-3/L2 0,01, cosine com warmup 5, 150 épocas, suavização de rótulo 0,1, amostragem balanceada com reposição calculada somente no treino e early stopping por F1 macro com paciência 25. O sampler substitui a ponderação adicional da loss. Logs ficam em `logs/`, pesos e métricas em `experiments/`. Checkpoints v2 incluem `pipeline_version`, `pipeline_config`, `feature_dim`, `label_map`, `target_fps`, `window_seconds` e os parâmetros da rede.

## Ativar a v2 explicitamente

```bash
SINALIZA_FUSION_MANIFEST_PATH='' SINALIZA_MODEL_PATH=experiments/transformer_v2_mixed25_seed42/best.pt ./.venv/Scripts/python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Sem essas variáveis, a aplicação usa a fusão v1 `configs/fusion25_frozen.json`, conforme o pedido desta tarefa. A dupla ConvBiLSTM anterior continua disponível em seu manifesto, sem apagar pesos. `/api/model/info` informa versão, dimensão e parâmetros temporais. A v2 exige timestamps e dimensões reais; o REST `/api/predict` aceita esses campos e máscara por parte. A webcam os fornece automaticamente. A classe opcional `_nenhum_sinal` pode estar no mapa e nos splits como qualquer outra; se vencer o top 1 suavizado, o backend responde `no_sign`, palavra nula e candidatos vazios, mesmo abaixo do limiar de exibição. Não são gerados negativos artificiais nesta fase.

## Aumentos somente no treino

`RawAugmentorV2` atua nos dados brutos. Configuração em `configs/augmentation_v2.yaml`: espelhamento 50%, fator de tempo 0,7–1,3 (multiplica os timestamps; fator maior deixa o movimento mais lento), escala 0,85–1,15, rotação ±10°, translação ±0,05, ruído 0,005, descarte de até 30% dos quadros e ocultação contígua de uma mão com probabilidade 0,2. A rotação usa aspecto físico e volta ao formato bruto. As bordas temporais são mantidas ao descartar quadros. Avaliação não aceita augmentor.

O mapa de espelhamento usa os pares de pose e face explicitados no código. Os pontos centrais da face permanecem em sua posição na lista. O **267 não tem par** nesta seleção e é zerado no exemplo espelhado; permanece zerado também após normalização e na velocidade. Espelhar duas vezes recupera os pontos com par, mas não recupera o 267 descartado. A probabilidade é configurável: sinais direcionais devem ser revisados pelo grupo antes de adotar espelhamento indiscriminado.

## Tempo e metadados

`prepare_features_v2` é o caminho comum: reamostra os dados brutos e chama o único construtor de features. A grade termina em `end_time` e contém 48 instantes com passo 1/15 s; o intervalo entre primeiro e último é 47/15 s. O buffer ao vivo aguarda 3,2 s completos e tenta classificar a cada 0,4 s. Guarda aproximadamente 6,4 s, preservando um vizinho anterior à borda. Mudança de resolução limpa a janela.

O dataset estima a atividade com a mesma janela de 40% de maior movimento dos punhos usada na auditoria. Essa é uma heurística, não anotação do sinal. Validação/teste centralizam uma janela nessa região; treino sorteia janelas que cobrem ao menos 80% dela ou, se ela exceder a janela, sorteia dentro dela. Vídeos curtos repetem bordas. Timestamps explícitos de coleta têm prioridade; arquivos antigos usam `arange(T)/fps`. FPS ausente ou fora de 10–60 gera aviso e usa 30. Dimensões ausentes geram aviso e usam aspecto neutro 1:1, que **não recupera o aspecto real**. `metadata_issues` conta arquivos afetados uma vez ao criar o dataset.
