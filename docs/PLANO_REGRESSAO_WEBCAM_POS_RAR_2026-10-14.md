# Regressao percebida na webcam apos os arquivos do RAR

## Plano

1. Comparar o fluxo atual de camera, WebSocket e inferencia com o commit anterior aos arquivos do RAR.
2. Remover trabalho novo por quadro que possa atrasar a captura no navegador, sem alterar checkpoint, landmarks usados no classificador, janela ou limiar de confianca.
3. Verificar compilacao do frontend e testes de inferencia por WebSocket. Registrar o que depende de teste na webcam real.

## Evidencia antes da correcao

- O hook de camera atual voltou ao fluxo anterior ao RAR: solicita 30 FPS e envia JPEG. A unica diferenca em relacao ao commit e a limpeza do canvas ao parar.
- O backend passou a incluir os landmarks brutos em cada `frame_ack`; a inferencia continua usando os mesmos landmarks, normalizacao, checkpoint e janela.
- O frontend novo executa `setOverlay` em cada `frame_ack`. Isso renderiza novamente `HomePage`, `WebcamView` e o painel de traducao a cada quadro processado. A taxa efetiva e o impacto na Canon nao foram medidos.
- O texto `Sinal incerto` sobre a frase ja existia no commit anterior ao RAR. Portanto nao se pode atribuir esse comportamento a copia dos arquivos.

## Correcao aplicada

O WebSocket agora entrega os pontos por callback para a referencia de `WebcamView`. O componente desenha diretamente no canvas e atualiza o indicador de rastreamento somente quando o valor booleano muda. `HomePage` deixa de receber um novo estado `overlay` a cada `frame_ack`. A captura JPEG, os resultados das predicoes e os parametros do backend permanecem iguais.

## Verificacao

`npm.cmd run build` passou: TypeScript e Vite concluiram sem erro. `git diff --check` nao encontrou erro de whitespace. Os testes Python nao puderam ser executados neste ambiente: a `.venv` referencia um Python 3.11 removido, e `py -0p` nao lista instalacoes disponiveis. O teste automatizado nao substitui uma comparacao ao vivo com a Canon, no mesmo enquadramento e com o mesmo sinal. Ainda nao ha medicao que prove que o redesenho era a causa principal da piora percebida.
