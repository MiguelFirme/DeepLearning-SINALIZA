# FPS da webcam no fluxo web

## Plano

1. Remover o temporizador de 15 FPS do frontend e capturar cada quadro novo disponibilizado pelo navegador.
2. Solicitar 30 FPS como preferencia ao abrir a camera, sem impor um maximo. Medir no HUD a taxa real de quadros novos do video.
3. Manter o limite de dois JPEGs em transito no WebSocket para que o backend nao acumule uma fila de quadros atrasados.
4. Compilar o frontend e registrar o resultado e os limites do teste.

## Interpretacao

O Transformer recebe janelas de 48 quadros. Esse numero e o comprimento da sequencia, nao uma taxa minima ou maxima de FPS. A primeira predicao ocorre apos 48 quadros processados; depois, a cada 24. A taxa efetiva depende da camera, do navegador, da codificacao JPEG e do MediaPipe no backend. O FPS exibido pela camera Canon pode diferir do stream de webcam entregue ao navegador.

## Implementacao e verificacao em 12/10/2026

- `useCameraFrames.ts` passou a usar `requestVideoFrameCallback`, com fallback para `requestAnimationFrame` em navegadores sem essa API. Nao ha mais temporizador de 15 FPS nesse fluxo.
- `getUserMedia` pede `frameRate.ideal = 30`; o navegador pode entregar outra taxa. O HUD `FPS camera` conta quadros novos recebidos pelo elemento de video, mesmo quando a codificacao JPEG ou o envio ficam ocupados.
- O WebSocket ainda permite no maximo dois quadros sem confirmacao. Assim, o FPS de quadros processados pelo backend pode ser menor que o valor no HUD.
- `npm run build` concluiu, incluindo TypeScript e Vite. Ainda e necessario medir o stream da Canon R100 no navegador do usuario; nao ha teste de hardware automatizado neste ambiente.
