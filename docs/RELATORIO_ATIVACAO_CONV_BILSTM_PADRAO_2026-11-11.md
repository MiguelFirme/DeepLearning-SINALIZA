# Fusão ConvBiLSTM como modelo principal

Data: 11/11/2026. Pedido: integrar a nova fusão Conv1D + BiLSTM + atenção ao backend como padrão. [Plano anterior à alteração](PLANO_ATIVACAO_CONV_BILSTM_PADRAO_2026-11-11.md).

Atualização posterior no mesmo dia: a tarefa [pipeline v2](PIPELINE_V2.md) determinou `configs/fusion25_frozen.json` como padrão v1 e ativação explícita da v2 por checkpoint. A dupla ConvBiLSTM e seu manifesto foram preservados; este relatório documenta a integração anterior.

## Modelo carregado ao iniciar

`backend/config.py` agora seleciona automaticamente `configs/fusion25_conv_bilstm_default.json`. O manifesto carrega dois checkpoints já treinados, com SHA-256 e mapa de classes verificados pelo `FusionBundle`:

- `Modelos salvos/conv_bilstm_attention_mixed25_pose_face/best.pt`: pose e face, sem os landmarks detalhados das mãos.
- `Modelos salvos/conv_bilstm_attention_mixed25_full/best.pt`: pose, face e mãos.

Cada rede tem 116.939 parâmetros e usa 40 pontos faciais. As duas recebem sequências de 48 quadros, com o mesmo pré-processamento e 697 features do treinamento. O modelo pose/face propõe o top 5; a fusão usa 22% de seu escore e 78% do escore do modelo completo. As temperaturas são 1,0. A suavização EMA da sessão continua em 0,6, com limiar de exibição 0,30 e cooldown de 1 segundo. O WebSocket mantém a janela inicial de 48 quadros e atualiza a classificação a cada 12 quadros analisados.

O manifesto de implantação foi criado a partir do manifesto experimental, preservando os hashes dos pesos e a regra de fusão. Ele usa o mapa de rótulos arquivado junto ao próprio ConvBiLSTM. O arquivo experimental `configs/fusion25_conv_bilstm_attention_fixed_rule.json` permanece como registro do estudo. Os checkpoints não foram retreinados nesta ativação.

O backend agora registra os tipos de modelo lidos dos checkpoints e informa `model_type: conv_bilstm_attention_fusion` em `/api/model/info`, em vez de anunciar um tipo fixo Transformer. A tela mostra o rótulo **Modelos** no campo do tempo de inferência. `Dockerfile.backend` inclui os dois diretórios ConvBiLSTM, que contêm os pesos e o mapa necessário à inferência; vídeos e landmarks de treinamento são opcionais para executar a aplicação.

## Evidências e limites

A escolha de tornar essa dupla principal foi solicitada pelo usuário. O experimento anterior registrou 47/85 acertos na validação e 65/85 no teste exploratório, enquanto a dupla Transformer original registrou 50/85 e 56/85. O teste já tinha sido consultado anteriormente e não comprova generalização para pessoas novas. Os detalhes do treinamento estão no [relatório ConvBiLSTM](RELATORIO_CONV_BILSTM_ATENCAO_MIXED25_2026-11-04.md).

Verificações desta integração:

- `python -m pytest tests -q`: **120 passaram e 8 foram ignorados**. Inicialização FastAPI, carregamento estrito dos pesos, verificação de hashes, endpoints e fluxo WebSocket com 48 JPEGs passaram. O teste WebSocket usa um extrator determinístico e as duas redes reais; ele valida transporte e inferência, sem medir a qualidade de detecção do MediaPipe na webcam.
- Os 8 testes ignorados dependem de landmarks ou manifestos locais ausentes, incluindo paridade com previsões offline de estudos anteriores e grid search escolar. Eles agora verificam a disponibilidade desses dados opcionais antes de executar. A paridade numérica com o vídeo antigo não foi repetida nesta máquina.
- `npm.cmd run build`: TypeScript e Vite passaram, com 1.591 módulos. A primeira execução foi bloqueada por leitura do sandbox; a execução autorizada concluiu normalmente.
- Docker não está disponível no ambiente de execução; o Dockerfile foi atualizado, mas a construção e execução do contêiner não foram verificadas.
- O reconhecimento com a webcam de uma pessoa real continua dependendo da rodada prática do grupo. O modelo ainda pode atribuir palavras a movimentos fora do vocabulário ou repouso, pois não possui rejeição treinada com negativos revisados.

O README, o guia de execução, o índice de modelos salvos e o roteiro prático foram atualizados para identificar o novo padrão.

## Iniciar no Git Bash do VS Code

Encerre o backend anterior com `Ctrl+C`. Na raiz do projeto:

```bash
# Remove uma seleção temporária feita anteriormente neste terminal, se houver.
unset SINALIZA_FUSION_MANIFEST_PATH
./.venv/Scripts/python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Se existir uma definição de `SINALIZA_FUSION_MANIFEST_PATH` em `.env`, remova-a ou atualize-a para `configs/fusion25_conv_bilstm_default.json`, pois variáveis de ambiente e `.env` podem substituir o padrão do código.

Em outro terminal:

```bash
cd frontend
npm run dev
```

Confira `http://127.0.0.1:8000/api/health`: `model_loaded: true` e `num_classes: 25`. Em `http://127.0.0.1:8000/api/model/info`, confira `model_type: conv_bilstm_attention_fusion`. Abra `http://localhost:5173` e siga o [roteiro prático](ROTEIRO_TESTE_PRATICO_INTERFACE_25_CLASSES_2026-11-04.md).

## Retornar temporariamente ao Transformer anterior

Na raiz, no terminal do backend:

```bash
SINALIZA_FUSION_MANIFEST_PATH=configs/fusion25_frozen.json ./.venv/Scripts/python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Essa seleção vale para o processo iniciado pelo comando. O padrão do código permanece ConvBiLSTM.
