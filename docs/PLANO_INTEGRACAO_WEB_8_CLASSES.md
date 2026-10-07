# Integração web do Transformer MINDS-Libras de 8 classes

## Plano

1. Carregar `Modelos salvos/transformer_minds_8_classes_pose_face/best.pt` com a arquitetura e o `feature_mode` registrados no checkpoint, e o mapa de classes arquivado ao lado dele. Falhar claramente se houver incompatibilidade entre pesos, arquitetura e rótulos.
2. Enviar quadros JPEG da webcam pelo WebSocket `/api/ws/predict`. O backend extrai landmarks com `ml/features/landmarks.py`, igual ao treinamento, aplica o modo `pose_face`, normaliza e amostra 48 frames por inferência.
3. Devolver `sign`, `confidence`, `top_k` e tempo de processamento pelo WebSocket; exibir a resposta no frontend. Manter o estado de cada câmera separado por conexão.
4. Testar carregamento do checkpoint, pipeline de 48 frames e a interface; registrar comandos e limitações observadas.

## Contrato previsto

- Cliente -> servidor: quadros JPEG como mensagens WebSocket binárias. Mensagens de controle JSON continuam disponíveis (`reset` e `ping`).
- Servidor -> cliente: `{ "type": "ready", "data": { ... } }`, `{ "type": "frame_ack", "data": {} }`, `{ "type": "prediction", "data": { ... } }` ou `{ "type": "error", "data": { "message": "..." } }`.
- Cada conexão mantém seu extrator MediaPipe, buffer temporal e suavização. A primeira predição ocorre após 48 quadros; as seguintes usam janelas com avanço de 24 quadros. O cliente limita a dois quadros sem confirmação, para evitar fila crescente.

## Limites de interpretação

O modelo classifica apenas oito sinais isolados: Acontecer, Aluno, Banheiro, Barulho, Conhecer, Medo, Ruim e Vontade. Não traduz frases nem foi validado em pessoas e webcams novas. A métrica do conjunto MINDS-Libras não mede a qualidade da demonstração ao vivo; predições fora dessas oito classes devem ser tratadas como desconhecidas pelo usuário.

Para cenas paradas, a aplicação agora aplica um filtro de atividade antes do Transformer. Medição, testes e limites estão em [PLANO_REJEICAO_SEM_SINAL.md](PLANO_REJEICAO_SEM_SINAL.md).

## Execução local no Git Bash

Na raiz do projeto, em dois terminais:

```bash
./.venv/Scripts/python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

```bash
cd frontend
npm install
npm run dev
```

Abra `http://localhost:5173`, clique em **Iniciar Câmera** e permita o acesso à webcam. `http://127.0.0.1:8000/api/health` deve mostrar `model_loaded: true` e `num_classes: 8`. A câmera envia até 15 JPEGs por segundo, mas a capacidade de processamento do MediaPipe pode reduzir a taxa efetiva. A primeira predição demanda 48 quadros efetivamente processados. O frontend usa o proxy WebSocket `/api/ws/predict` do Vite.

Para testar o mesmo pipeline sem webcam:

```bash
./.venv/Scripts/python.exe scripts/smoke_web_inference.py --video data/raw/minds_libras/08BanheiroSinalizador11-1.mp4
```

## Verificação realizada em 07/10/2026

- Checkpoint arquivado carregado com `strict=True`: Transformer `pose_face`, 697 features, 8 classes, 121.018 parâmetros.
- `python -m pytest tests/test_web_video.py tests/test_api.py tests/test_feature_mode.py -q`: **8 testes passaram**. O teste WebSocket enviou 48 JPEGs e recebeu `ready`, confirmações de quadros e uma predição real do modelo.
- `npm run build`: compilação TypeScript e Vite concluída.
- WebSocket real através do proxy do Vite em `ws://127.0.0.1:5173/api/ws/predict`: recebeu `ready` com 8 classes, 48 frames e `pose_face`.
- Vídeo real `08BanheiroSinalizador11-1.mp4`, amostrado a aproximadamente 15 FPS e recodificado em JPEG: **Banheiro**, confiança aproximada de **79,2%**, em uma janela de 48 quadros. É um vídeo de validação já conhecido; não é uma medição de generalização para webcams/pessoas novas.
- O navegador com a webcam do usuário não pôde ser testado neste ambiente. Confirmar permissões da câmera, conexão WebSocket e tempo até a primeira resposta na máquina do usuário.
