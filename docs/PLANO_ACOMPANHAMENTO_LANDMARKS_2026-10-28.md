# Plano: reduzir atraso dos pontos na webcam

## Problema observado

O frontend envia JPEGs ao backend e só desenha os landmarks após o `frame_ack` do MediaPipe. A fila atual permite dois quadros pendentes; em análise mais lenta que a câmera, o desenho representa um quadro anterior. O MediaPipe é executado de forma sequencial no backend, então a taxa real de análise pode ser menor que o FPS da webcam.

## Alteração planejada

1. Permitir apenas um quadro aguardando confirmação para evitar acúmulo de imagens antigas.
2. Identificar o instante de captura do quadro e descartar o desenho de respostas antigas ou fora de ordem.
3. Animar apenas o overlay visual entre duas respostas recentes com uma previsão curta, limitada e desativada quando o rastreamento some. Manter os landmarks brutos enviados ao classificador e os pesos dos modelos intactos.
4. Verificar compilação do frontend e um teste de ida e volta com o backend; registrar o resultado e as limitações neste documento.

## Critérios de aceitação

- Pontos acompanham a webcam com menor atraso percebido, sem fila crescente.
- Overlay desaparece quando a câmera para, o WebSocket desconecta ou a detecção deixa de existir.
- O classificador continua recebendo apenas os landmarks extraídos pelo backend, sem dados extrapolados do desenho.

## Implementação e verificação

- `frontend/src/hooks/useCameraFrames.ts`: codifica JPEG apenas quando o WebSocket tem capacidade; registra o horário real de captura.
- `frontend/src/hooks/useWebSocket.ts`: mantém no máximo um quadro pendente, mede latência desde a captura e ignora mensagens de uma conexão antiga.
- `frontend/src/components/WebcamView.tsx` e `frontend/src/utils/predictTrackingOverlay.ts`: redesenham os pontos entre respostas com avanço limitado a 80 ms e 4% da largura/altura do quadro por coordenada. O avanço para após 250 ms sem resposta; pontos são limpos ao parar, trocar câmera, perder detecção ou desconectar.
- O classificador continua usando os 346 valores extraídos pelo MediaPipe no backend. A previsão visual é restrita ao canvas do navegador.

Verificações em 28/10/2026: `tsc --noEmit` passou; `npm.cmd run build` passou (Vite 5.4.21, 1.587 módulos); `pytest tests/test_web_video.py -q` passou (1 teste). A compilação inicialmente falhou dentro do sandbox por bloqueio de leitura da configuração do Vite; a execução autorizada fora do sandbox concluiu normalmente. Ainda falta medir o atraso visual com webcam real e comparar com o comportamento anterior no computador do grupo. O processamento MediaPipe no backend continua sendo o limite da taxa de detecção.
