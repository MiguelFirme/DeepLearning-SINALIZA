# Sinaliza

Protótipo de um projeto de extensão para reconhecer **sinais isolados de Libras** pela webcam e apresentar a palavra identificada em português. O objetivo de longo prazo é apoiar a comunicação em escolas, casas e recepções. A versão atual ainda não traduz frases nem está validada para uso com pessoas novas.

## Estado do projeto — 07/10/2026

- A aplicação web usa **React/Vite** no frontend e **FastAPI** no backend.
- O backend carrega por padrão o [Transformer MINDS-Libras de 8 classes](Modelos%20salvos/transformer_minds_8_classes_pose_face/README.md), salvo em `Modelos salvos/transformer_minds_8_classes_pose_face/best.pt`.
- A webcam envia quadros JPEG pelo WebSocket. O backend extrai landmarks com MediaPipe Holistic, prepara sequências de 48 quadros e devolve o sinal, a confiança e os candidatos.
- Um filtro de atividade responde **“Nenhum sinal detectado”** quando a pessoa está parada. Ele reduz palpites em cenas estáticas, mas movimentos sem sinal ainda podem ser classificados incorretamente.
- O dicionário da interface é um catálogo separado e **não indica quais palavras o modelo reconhece**. No estado atual, a API do dicionário lê `data/processed/label_map.json` quando presente, que pertence ao histórico do V-Librasil.

### Palavras reconhecidas pelo modelo ativo

**Acontecer, Aluno, Banheiro, Barulho, Conhecer, Medo, Ruim e Vontade.** O modelo só pode escolher entre essas oito classes ou retornar sem sinal/incerto por meio dos filtros da aplicação.

## Rodar localmente no Git Bash do VS Code

Execute os comandos a partir da **raiz do projeto** (`DeepLearning-SINALIZA`). Para abrir a aplicação, não é preciso baixar vídeos nem refazer o treinamento; é preciso ter o checkpoint arquivado em `Modelos salvos/`.

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

Abra **http://localhost:5173**, clique em **Iniciar Câmera** e autorize a webcam. Confira **http://127.0.0.1:8000/api/health**: a resposta deve incluir `"model_loaded": true` e `"num_classes": 8`. A primeira classificação exige 48 quadros processados, então pode levar alguns segundos, especialmente na CPU. Use `Ctrl+C` em cada terminal para parar.

Para testar o pipeline com um vídeo existente, sem webcam:

```bash
./.venv/Scripts/python.exe scripts/smoke_web_inference.py \
  --video data/raw/minds_libras/08BanheiroSinalizador11-1.mp4
```

Esse MP4 é opcional e pode não estar presente em outra instalação. O [guia da aplicação web](docs/COMO_RODAR_APLICACAO_WEB.md) traz solução para problemas comuns de instalação e inicialização.

## Como funciona

```text
Webcam → JPEG via WebSocket → MediaPipe Holistic no backend
       → 346 valores brutos por quadro → pose e face → normalização pelos ombros
       → velocidades e distâncias: 697 features por quadro
       → janela de 48 quadros → filtro de presença e movimento
       → Transformer temporal → palavra e confiança → frontend
```

O modo `pose_face` **zera os pontos detalhados das mãos** antes do modelo, mas preserva braços, cotovelos, pulsos e face. A primeira predição usa 48 quadros; depois, a janela avança 24 quadros. O frontend tenta capturar até 15 JPEGs por segundo, com controle de fila para acompanhar a capacidade de processamento do backend.

O filtro rejeita janelas com pouca detecção ou movimento dos braços abaixo de **0,35 largura de ombro**. A confiança mínima para mostrar uma palavra é **0,50**. Esses valores podem ser configurados por `SINALIZA_MIN_ARM_MOTION` e `SINALIZA_CONFIDENCE_THRESHOLD`. Eles foram escolhidos com vídeos do dataset e cenas repetidas, não com um conjunto representativo de webcams. Veja a [medição e as limitações do filtro](docs/PLANO_REJEICAO_SEM_SINAL.md).

## Dados e modelos

O experimento ativo usa o **MINDS-Libras**: 20 classes, 800 vídeos, 8 sinalizadores e 40 vídeos por classe no conjunto completo. O subconjunto de 8 classes tem 320 vídeos. Os splits separam pessoas: Sinalizadores 01, 02, 05, 06, 08 e 10 no treino; 11 na validação; 12 no teste. Não foram usadas variações sintéticas nesses experimentos.

| Checkpoint arquivado | Vídeos treino/validação/teste | Validação top-1 | Teste top-1 | Uso atual |
|---|---:|---:|---:|---|
| [Transformer `pose_face`, 8 classes](Modelos%20salvos/transformer_minds_8_classes_pose_face/README.md) | 240 / 40 / 40 | 40/40 (100%) | 37/40 (92,5%) | Aplicação web |
| [Transformer `pose_face`, 20 classes](Modelos%20salvos/transformer_minds_20_classes_pose_face/README.md) | 600 / 100 / 100 | 37/100 (37%) | Não executado | Experimento arquivado |

O checkpoint de 8 classes foi selecionado pela menor perda de validação, na época 94. A arquitetura tem duas camadas Transformer, quatro cabeças de atenção, dimensão 64, dropout 0,45 e 121.018 parâmetros. O treino usou Adam com `weight_decay=1e-4`, augmentation apenas no treino, scheduler por perda de validação e early stopping. Os detalhes e comandos completos estão no [README do modelo](Modelos%20salvos/transformer_minds_8_classes_pose_face/README.md).

**Essas métricas são exploratórias.** Os vídeos de teste pertencem ao mesmo dataset, e o Sinalizador12 já havia sido consultado em experimentos anteriores. Os resultados não medem o desempenho com usuários, câmeras, iluminação ou frases novas. O modelo também não aprendeu uma classe explícita de “nenhum sinal”; o filtro de movimento é uma proteção provisória. Gravar exemplos negativos reais e avaliar falsos positivos por minuto é um próximo passo.

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
| WebSocket | `/api/ws/predict` | Recebe JPEG binário; envia `ready`, `frame_ack`, `prediction` e `error` |

Documentação interativa da API: **http://127.0.0.1:8000/docs**. O [plano de integração web](docs/PLANO_INTEGRACAO_WEB_8_CLASSES.md) registra o contrato WebSocket e os testes realizados.

## Verificação

```bash
./.venv/Scripts/python.exe -m pytest tests/test_web_video.py tests/test_api.py tests/test_feature_mode.py -q
cd frontend
npm run build
```

Na última verificação desses componentes, **8 testes passaram** e o build do frontend concluiu. Um MP4 de validação da classe `Banheiro`, recodificado como JPEG, foi reconhecido como **Banheiro** com cerca de **79,2%** de confiança. O uso ao vivo com a webcam de outra pessoa ainda precisa ser medido.

Há configuração de Docker Compose no repositório, mas a execução em contêiner não foi validada nesta atualização.

## Documentação

- [Como rodar a aplicação web](docs/COMO_RODAR_APLICACAO_WEB.md)
- [Integração do modelo com a webcam](docs/PLANO_INTEGRACAO_WEB_8_CLASSES.md)
- [Filtro de cenas paradas](docs/PLANO_REJEICAO_SEM_SINAL.md)
- [Modelos arquivados e reprodução](Modelos%20salvos/README.md)
- [Treinamento e experimentos anteriores](docs/TRAINING.md)
- [Localização dos vídeos originais](docs/LOCALIZACAO_VIDEOS_ORIGINAIS.md)

## Licença

Este projeto usa a licença MIT; consulte [LICENSE](LICENSE).
