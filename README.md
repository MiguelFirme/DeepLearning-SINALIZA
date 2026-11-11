# Sinaliza

Protótipo de um projeto de extensão para reconhecer **sinais isolados de Libras** pela webcam e apresentar a palavra identificada em português. O objetivo de longo prazo é apoiar a comunicação em escolas, casas e recepções. A versão atual ainda não traduz frases nem está validada para uso com pessoas novas.

## Estado do projeto — 11/11/2026

- A aplicação web usa **React/Vite** no frontend e **FastAPI** no backend.
- O backend carrega por padrão a [fusão de dois modelos Conv1D + BiLSTM + atenção, com 25 classes](docs/RELATORIO_ATIVACAO_CONV_BILSTM_PADRAO_2026-11-11.md), definida em `configs/fusion25_conv_bilstm_default.json`. Ambos usam 40 pontos faciais; o primeiro usa pose e face, o segundo usa pose, face e mãos.
- A webcam envia quadros JPEG pelo WebSocket. O backend extrai landmarks com MediaPipe Holistic, prepara sequências de 48 quadros e devolve o sinal, a confiança e os candidatos.
- Não há filtro de movimento para cenas paradas: uma janela com pose/face detectada pode receber uma das 25 palavras mesmo sem um sinal. O experimento anterior de rejeição por movimento foi revertido a pedido do usuário.
- O dicionário da interface é um catálogo separado e **não indica quais palavras o modelo reconhece**. No estado atual, a API do dicionário lê `data/processed/label_map.json` quando presente, que pertence ao histórico do V-Librasil.

### Palavras reconhecidas pelo modelo ativo

O modelo ativo reconhece as **25 classes** do mapa `Modelos salvos/conv_bilstm_attention_mixed25_full/data_split/label_map.json`: 15 sinais selecionados do MINDS-Libras e dez gravados pelo grupo. A API `/api/health` informa a quantidade de classes carregadas.

## Rodar localmente no Git Bash do VS Code

Execute os comandos a partir da **raiz do projeto** (`DeepLearning-SINALIZA`). Para abrir a aplicação, não é preciso baixar vídeos nem refazer o treinamento; os dois checkpoints e o manifesto de fusão precisam estar presentes em `Modelos salvos/` e `configs/`.

Pré-requisitos: **Python 3.11**, **Node.js 18+** com npm e acesso à webcam. A GPU não é necessária para usar a aplicação; o backend usa CPU por padrão.

Na primeira instalação, crie a `.venv` se ela ainda não existir e instale as dependências:

```bash
py -3.11 -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
cd frontend
npm install
cd ..
```

Se a `.venv` já existe e funciona, comece em `./.venv/Scripts/python.exe -m pip install -r requirements.txt`. Se `py -3.11` não encontrar o Python, instale a versão 3.11 e recrie a `.venv`. O terminal precisa estar na raiz: dentro de `backend/`, o caminho `./.venv` não existe.

Abra **dois terminais**. No primeiro, inicie o backend:

```bash
./.venv/Scripts/python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

No segundo, inicie o frontend:

```bash
cd frontend
npm run dev
```

Abra **http://localhost:5173**, clique em **Iniciar Câmera** e autorize a webcam. Confira **http://127.0.0.1:8000/api/health**: a resposta deve incluir `"model_loaded": true` e `"num_classes": 25`. Em **http://127.0.0.1:8000/api/model/info**, `model_type` deve ser `conv_bilstm_attention_fusion`. A primeira classificação exige 48 quadros processados, então pode levar alguns segundos, especialmente na CPU. Use `Ctrl+C` em cada terminal para parar.

Para testar o pipeline com um vídeo existente, sem webcam:

```bash
./.venv/Scripts/python.exe scripts/smoke_web_inference.py \
  --video data/raw/minds_libras/08BanheiroSinalizador11-1.mp4
```

Esse MP4 é opcional e pode não estar presente em outra instalação. O [guia da aplicação web](docs/COMO_RODAR_APLICACAO_WEB.md) traz solução para problemas comuns de instalação e inicialização.

## Como funciona

```text
Webcam → JPEG via WebSocket → MediaPipe Holistic no backend
       → 346 valores brutos por quadro → pose + face / pose + face + mãos
       → normalização pelos ombros
       → velocidades e distâncias: 697 features por quadro
       → janela de 48 quadros → verificação de presença de pose/face
       → dois modelos Conv1D + BiLSTM + atenção → top 5 + fusão → palavra e confiança → frontend
```

O modelo `pose_face` **zera os pontos detalhados das mãos**, mas preserva braços, cotovelos, pulsos e face; o modelo `full` recebe também as mãos. A primeira predição usa 48 quadros; depois, a janela avança 12 quadros. O frontend captura cada quadro novo entregue pelo navegador, solicita 30 FPS como preferência da câmera e controla a fila para acompanhar o backend. O indicador "FPS câmera" mede quadros recebidos pelo navegador, não quadros processados pelo modelo. Veja [PLANO_FPS_WEBCAM.md](docs/PLANO_FPS_WEBCAM.md).

Para reduzir o atraso dos pontos desenhados, o frontend mantém apenas um quadro pendente no WebSocket, codifica JPEG somente quando o backend pode receber um novo quadro e faz uma breve antecipação visual limitada entre respostas. Essa antecipação afeta apenas o canvas; os modelos continuam recebendo os landmarks reais do MediaPipe no backend. Veja [plano e verificação do acompanhamento](docs/PLANO_ACOMPANHAMENTO_LANDMARKS_2026-10-28.md).

Para investigar latência, ative `SINALIZA_PERF_LOGGING=true` no backend e abra o frontend com `?perf=1`. O Console e o terminal mostram um resumo simples a cada três segundos. Os detalhes por quadro são opcionais; veja [perfil de tempo real](docs/PLANO_PERFIL_TEMPO_REAL_2026-10-28.md).

O backend não classifica janelas com detecção insuficiente de pose/face. Para as demais, a confiança mínima para mostrar uma palavra é **0,30**, configurável por `SINALIZA_CONFIDENCE_THRESHOLD`. O modelo não possui uma classe treinada de “nenhum sinal”; portanto, pode atribuir palavras a uma pessoa parada. A [experiência anterior com filtro de movimento](docs/PLANO_REJEICAO_SEM_SINAL.md) foi documentada e revertida.

## Dados e modelos

### Fusão padrão de 25 classes

As 15 classes selecionadas do MINDS-Libras e dez palavras gravadas pelo grupo formam um conjunto de 670 vídeos (500 treino, 85 validação e 85 teste). Dois modelos Conv1D + BiLSTM + atenção arquivados, um `pose_face` e outro `full`, são carregados automaticamente pelo manifesto `configs/fusion25_conv_bilstm_default.json`. A fusão propõe as cinco classes mais prováveis do modelo `pose_face` e as reordena com 22% do escore desse modelo e 78% do `full`. A variável `SINALIZA_FUSION_MANIFEST_PATH` pode selecionar outro manifesto; o antigo par Transformer continua disponível em `configs/fusion25_frozen.json`.

Na validação, a nova fusão acertou 47/85, contra 50/85 do par Transformer anterior; no teste já consultado em experimentos anteriores, 65/85 contra 56/85. A escolha de ativá-la foi do grupo, não uma comprovação de superioridade em pessoas novas. A auditoria marcou 168/670 vídeos para revisão, principalmente por ausência da mão ativa. Não existem negativos reais revisados, portanto ainda não há rejeição treinada de “nenhum sinal”. O [relatório do experimento ConvBiLSTM](docs/RELATORIO_CONV_BILSTM_ATENCAO_MIXED25_2026-11-04.md) traz o protocolo e os limites.

Um [piloto de rejeição com início e fim dos vídeos](docs/RELATORIO_NEGATIVOS_BORDAS_25_CLASSES_2026-10-27.md) foi treinado separadamente com 100 trechos de repouso no treino e 11 na validação. Ele reconheceu esses repousos, mas piorou a tradução quando usado sozinho. Como bloqueio antes da fusão, produziu cinco rejeições indevidas entre 85 sinais no teste exploratório. Seus rótulos são provisórios, e o checkpoint **não está ativo na aplicação**.

Um [experimento com 16 pontos faciais](docs/RELATORIO_ABLACAO_FACE16_MIXED25_2026-11-04.md) retreinou os dois Transformers de 25 classes com os mesmos splits e parâmetros. A versão `pose_face` isolada melhorou, mas a fusão caiu de 50/85 para 45/85 na validação e de 56/85 para 42/85 no teste exploratório já consultado anteriormente. Os checkpoints e logs estão arquivados em `Modelos salvos/`; a dupla ConvBiLSTM ativa usa 40 pontos faciais.

A [fusão híbrida face16 + modelo completo de 40 pontos](docs/RELATORIO_FUSAO_HIBRIDA_FACE16_FULL40_2026-11-04.md) foi testada com checkpoints existentes e a regra original fixa. Acertou 46/85 na validação e 49/85 no teste exploratório, abaixo da dupla original (50/85 e 56/85). O manifesto experimental está em `configs/fusion25_hybrid_face16_full40.json`.

A [fusão híbrida invertida face40 + modelo completo de 16 pontos](docs/RELATORIO_FUSAO_HIBRIDA_FACE40_FULL16_2026-11-04.md) acertou 49/85 na validação e 48/85 no teste exploratório. Aproximou-se da dupla original na validação, mas manteve F1 macro inferior. Seu manifesto experimental está em `configs/fusion25_hybrid_face40_full16.json`.

A [simulação de fusão sequencial dos dois modelos de 15 classes](docs/EXPERIMENTO_FUSAO_SEQUENCIAL_15_CLASSES_2026-10-21.md) acertou 52/75 vídeos na validação e 44/75 no teste exploratório com top 5. A inversão da ordem não mudou as previsões top 5 neste conjunto. É uma avaliação offline; o backend ativo não foi alterado.

O MINDS-Libras original tem 20 classes, 800 vídeos, 8 sinalizadores e 40 vídeos por classe; o subconjunto histórico de 8 classes tem 320 vídeos. O modelo atual de 25 classes combina 15 classes selecionadas desse conjunto com dez gravadas pelo grupo. Nos vídeos MINDS, os splits separam pessoas: Sinalizadores 01, 02, 05, 06, 08 e 10 no treino; 11 na validação; 12 no teste. Não foram usadas variações sintéticas nesses experimentos.

| Checkpoint arquivado | Vídeos treino/validação/teste | Validação top-1 | Teste top-1 | Uso atual |
|---|---:|---:|---:|---|
| [Transformer `pose_face`, 8 classes](Modelos%20salvos/transformer_minds_8_classes_pose_face/README.md) | 240 / 40 / 40 | 40/40 (100%) | 37/40 (92,5%) | Experimento arquivado |
| [Transformer `pose_face`, 15 classes](Modelos%20salvos/transformer_minds_15_classes_pose_face/README.md) | 450 / 75 / 75 | 49/75 (65,3%) | 41/75 (54,7%), exploratório | Experimento arquivado |
| [Transformer `full`, 15 classes](Modelos%20salvos/transformer_minds_15_classes_full/README.md) | 450 / 75 / 75 | 45/75 (60%) | 34/75 (45,3%), exploratório | Experimento arquivado |
| [Transformer `pose_face`, 20 classes](Modelos%20salvos/transformer_minds_20_classes_pose_face/README.md) | 600 / 100 / 100 | 37/100 (37%) | Não executado | Experimento arquivado |

O checkpoint histórico de 8 classes foi selecionado pela menor perda de validação, na época 94. A arquitetura tem duas camadas Transformer, quatro cabeças de atenção, dimensão 64, dropout 0,45 e 121.018 parâmetros. O treino usou Adam com `weight_decay=1e-4`, augmentation apenas no treino, scheduler por perda de validação e early stopping. Os detalhes e comandos completos estão no [README do modelo](Modelos%20salvos/transformer_minds_8_classes_pose_face/README.md).

**Essas métricas são exploratórias.** Os vídeos de teste pertencem ao mesmo dataset, e o Sinalizador12 já havia sido consultado em experimentos anteriores. Os resultados não medem o desempenho com usuários, câmeras, iluminação ou frases novas. O modelo não aprendeu uma classe explícita de “nenhum sinal”; gravar exemplos negativos reais e avaliar falsos positivos por minuto é um próximo passo.

O projeto mantém dados e experimentos anteriores com **V-Librasil** e outras arquiteturas, incluindo código de Bi-LSTM, TCN e I3D. Eles não são o modelo carregado pela aplicação web atual. Os MP4 do MINDS-Libras ficam em `data/raw/minds_libras/`; os landmarks `.npz`, em `data/landmarks/`; os splits, em `data/processed/`. Modelos e métricas ficam em `Modelos salvos/` e `experiments/`; logs de treinamento, em `logs/`.

### Reproduzir o treino de 8 classes

Com os landmarks e splits arquivados disponíveis, execute na raiz do projeto:

```bash
./.venv/Scripts/python.exe scripts/train.py \
  --config "Modelos salvos/transformer_minds_8_classes_pose_face/config/transformer_minds_pose_face.yaml" \
  --data-dir data/landmarks/minds_libras_school8 \
  --processed-dir "Modelos salvos/transformer_minds_8_classes_pose_face/data_split" \
  --experiment transformer_minds_school8_pose_face_repro_seed42 \
  --num-workers 0 --skip-test
```

O nome do experimento evita sobrescrever o treino original. `--skip-test` preserva o teste durante a reprodução. O [registro completo do modelo](Modelos%20salvos/transformer_minds_8_classes_pose_face/README.md) explica os dados, os hiperparâmetros e a auditoria do checkpoint.

## API

| Método | Rota | Função |
|---|---|---|
| GET | `/api/health` | Estado da API e do modelo |
| GET | `/api/model/info` | Informações da rede carregada |
| POST | `/api/predict` | Inferência a partir de landmarks brutos `(T, 346)` |
| POST | `/api/predict/reset` | Limpa a suavização da inferência REST |
| GET | `/api/dictionary` | Catálogo de palavras, separado do checkpoint ativo |
| WebSocket | `/api/ws/predict` | Recebe JPEG binário; envia `ready`, `frame_ack` com landmarks para o overlay, `prediction` e `error` |

Documentação interativa da API: **http://127.0.0.1:8000/docs**. O [plano de integração web](docs/PLANO_INTEGRACAO_WEB_8_CLASSES.md) registra o contrato WebSocket e os testes realizados.

## Verificação

```bash
./.venv/Scripts/python.exe -m pytest tests -q
cd frontend
npm run build
```

Na ativação da fusão de 25 classes, **122 testes passaram**, incluindo inicialização da API e transmissão de 48 quadros por WebSocket. O reconhecimento ao vivo com pessoas novas ainda precisa ser medido. O teste antigo com o MP4 `Banheiro` e 79,2% de confiança pertencia ao modelo histórico de oito classes.

Há configuração de Docker Compose no repositório, mas a execução em contêiner não foi validada nesta atualização.

## Documentação

- [Plano e relatório do experimento de 25 classes](docs/RELATORIO_QUALIDADE_FUSAO_25_CLASSES_2026-10-27.md)
- [Ativação da fusão de 25 classes como padrão](docs/RELATORIO_ATIVACAO_FUSAO25_PADRAO_2026-11-04.md)
- [Revisão de três PDFs de um projeto semelhante e prioridades para o Sinaliza](docs/RELATORIO_REVISAO_PROJETOS_LIBRAS_2026-11-04.md)
- [Experimento Conv1D + BiLSTM + atenção nas 25 classes](docs/RELATORIO_CONV_BILSTM_ATENCAO_MIXED25_2026-11-04.md)
- [Ativação da fusão ConvBiLSTM como padrão](docs/RELATORIO_ATIVACAO_CONV_BILSTM_PADRAO_2026-11-11.md)
- [Experimento Conv1D + Transformer + atenção nas 25 classes](docs/RELATORIO_CONV_TRANSFORMER_ATENCAO_MIXED25_2026-11-04.md)
- [Piloto de rejeição com trechos de repouso dos vídeos](docs/RELATORIO_NEGATIVOS_BORDAS_25_CLASSES_2026-10-27.md)
- [Como rodar a aplicação web](docs/COMO_RODAR_APLICACAO_WEB.md)
- [Roteiro de teste prático da interface com 25 classes](docs/ROTEIRO_TESTE_PRATICO_INTERFACE_25_CLASSES_2026-11-04.md)
- [Integração do modelo com a webcam](docs/PLANO_INTEGRACAO_WEB_8_CLASSES.md)
- [Experimento revertido de filtro de cenas paradas](docs/PLANO_REJEICAO_SEM_SINAL.md)
- [Modelos arquivados e reprodução](Modelos%20salvos/README.md)
- [Treinamento e experimentos anteriores](docs/TRAINING.md)
- [Localização dos vídeos originais](docs/LOCALIZACAO_VIDEOS_ORIGINAIS.md)

## Licença

Este projeto usa a licença MIT; consulte [LICENSE](LICENSE).
