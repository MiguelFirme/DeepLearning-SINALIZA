# Plano: localizar o gargalo do vídeo em tempo real

## Ajuste de legibilidade solicitado em 28/10/2026

O primeiro formato produziu uma linha extensa para cada quadro, além de linhas separadas para câmera e overlay. Isso dificultou identificar a etapa dominante. O plano é mostrar por padrão **um resumo a cada três segundos** com FPS da câmera, FPS analisado, tempos médios de JPEG, MediaPipe, modelo, comunicação/outros e idade dos pontos. O resumo terá uma frase em português indicando a maior etapa medida. Os registros por quadro ficarão disponíveis somente no modo detalhado (`?perf=full` e `SINALIZA_PERF_DETAIL=true`). As medições e o protocolo de inferência permanecem os mesmos.

## Objetivo

Registrar no console do navegador e no terminal do backend quanto tempo cada etapa leva, sem alterar os pixels, landmarks, janelas de 48 quadros nem os pesos dos modelos.

## Instrumentação planejada

1. Numerar os quadros por conexão. Como o frontend mantém no máximo um quadro pendente, a ordem dos números em cada ponta é a mesma.
2. No navegador, medir intervalo da câmera (FPS), desenho no canvas e codificação JPEG, envio até confirmação (`round trip`) e tempo entre captura e overlay. Registrar timestamp local, número do quadro e tamanho do JPEG.
3. No backend, medir decodificação JPEG, MediaPipe, montagem da janela, inferência e envio do `frame_ack`; registrar timestamp, número do quadro, detecções e estado da janela.
4. Incluir os tempos de backend no `frame_ack` de diagnóstico para que o navegador estime o tempo residual de comunicação/agendamento. Não interpretar esse residual como latência de rede de ida: relógios diferentes e serialização impedem essa conclusão.
5. Deixar a telemetria opcional por `SINALIZA_PERF_LOGGING=true` no backend e `?perf=1` no frontend. O modo `?perf=full` e `SINALIZA_PERF_DETAIL=true` libera as linhas por quadro. Isso evita que a escrita frequente no console interfira no uso normal.
6. Validar contrato do WebSocket, compilação do frontend e documentar como coletar e interpretar os logs.

## Cuidados

- Usar relógios monotônicos (`performance.now` e `time.perf_counter`) para durações; timestamps de parede são apenas para localizar eventos.
- Não registrar imagens, landmarks, nomes de pessoas ou probabilidades.
- O teste automatizado verifica o formato e coerência dos tempos; a medida do gargalo real depende da câmera e máquina de quem executar.

## Implementação

- Com `?perf=1`, o navegador mostra **um resumo a cada três segundos**: FPS da câmera, FPS de análise, atraso dos pontos, médias das etapas e a maior etapa medida. Com `?perf=full`, também mostra câmera, quadro e desenho individual.
- Com `SINALIZA_PERF_LOGGING=true`, o backend mostra **um `[PERF RESUMO]` a cada três segundos**. Com `SINALIZA_PERF_DETAIL=true`, também mostra cada quadro. O `frame_id` aparece em ambas as pontas no modo detalhado.
- O perfil do backend é enviado no `frame_ack` apenas quando `SINALIZA_PERF_LOGGING=true`. Caso só `?perf=1` esteja ativado, o navegador mostra `null` nos tempos internos do backend.
- `inferencia_ms` mede a chamada completa ao preditor, incluindo normalização e modelo. Na fusão, inclui os dois modelos. A primeira chamada ocorre após 48 quadros; depois, a cada 12 quadros processados.

## Como executar no Git Bash do VS Code

Na raiz do projeto, no terminal do backend:

```bash
SINALIZA_PERF_LOGGING=true \
/c/Users/migue/AppData/Local/Temp/sinaliza_python311_embed/python.exe \
-m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Em outro terminal:

```bash
cd frontend
npm run dev
```

Abra `http://localhost:5173/?perf=1`, pressione F12 e veja a aba **Console**. Inicie a câmera e mova os braços por pelo menos seis segundos. O backend imprime uma linha `[PERF RESUMO]` a cada três segundos. Para testar a fusão de 25 classes, acrescente `SINALIZA_FUSION_MANIFEST_PATH=configs/fusion25_frozen.json` antes do executável Python no comando do backend.

Exemplo ilustrativo de leitura: `câmera: 16 FPS | análise: 8 FPS | atraso dos pontos: 150 ms` indica que a câmera entrega 16 quadros por segundo, o backend analisa 8 e os pontos representam em média uma imagem de 150 ms atrás. Se a próxima linha mostrar `pontos/MediaPipe 100 ms` e `modelo 8 ms`, o MediaPipe é a etapa medida mais lenta. Esses números são apenas exemplo, não resultado do computador do grupo.

A indicação de **etapa mais lenta** considera o impacto médio por quadro. O tempo do modelo é exibido por predição, mas uma predição ocorre apenas depois de 48 quadros e, em seguida, a cada 12 quadros analisados. Por isso, uma predição isolada pode demorar mais que um quadro de MediaPipe sem ser o maior custo contínuo.

**Se o resumo não bastar**, use `http://localhost:5173/?perf=full` e adicione `SINALIZA_PERF_DETAIL=true` ao comando do backend. Nessa modalidade, compare o `quadro` do navegador com o mesmo número no terminal.

## Campos disponíveis no modo detalhado

| Campo | Interpretação |
| --- | --- |
| `camera.fps` e `intervalo_medio_ms` | Taxa de quadros entregue pelo navegador; não mede exposição nem driver da câmera. |
| `captura_jpeg_ms` | Desenho do vídeo no canvas e codificação JPEG. |
| `envio_ate_ack_ms` | Tempo total desde `ws.send` até a resposta no navegador. |
| `decode_ms` | Decodificação JPEG no backend. |
| `mediapipe_ms` | Extração de landmarks do quadro. |
| `preparo_ms` | Montagem dos dados da janela no backend. |
| `inferencia_ms` | Normalização e predição quando `classificou=true`. |
| `ack` no backend | Tempo para serializar/enviar a resposta, incluindo os pontos para o overlay. |
| `comunicacao_e_outros_ms` | Resíduo entre ida e volta e processamento medido; inclui transporte, espera, serialização e agendamento. Não equivale a latência de rede de ida. |
| `captura_ate_tela_ms` | Idade aproximada do quadro quando os pontos aparecem pela primeira vez. |

Compare quadros com o mesmo número. Um `mediapipe_ms` repetidamente maior que os outros tempos aponta para o extrator; `captura_jpeg_ms` alto aponta para o navegador; FPS baixo com etapas rápidas aponta para a câmera ou sua entrega ao navegador. Não compare `performance.now()` diretamente com `time.perf_counter()`; os relógios têm origens diferentes.

## Verificação

Em 28/10/2026, após a simplificação: `tsc --noEmit` passou; `pytest tests/test_web_video.py -q` passou (1 teste com 48 quadros e verificação dos IDs e tempos); `compileall` do backend passou; `npm.cmd run build` passou (Vite 5.4.21, 1.588 módulos); `git diff --check` não apontou erros de whitespace. A compilação Vite precisou ser executada fora do sandbox porque o esbuild não conseguiu ler a configuração ali. Ainda não foi medido o gargalo na webcam real do grupo.
