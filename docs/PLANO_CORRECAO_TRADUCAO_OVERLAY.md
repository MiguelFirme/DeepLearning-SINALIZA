# Correcao da traducao e dos pontos da webcam

## Sintomas relatados em 14/10/2026

- O site deixou de apresentar traducoes depois da copia dos arquivos do RAR.
- As marcas de bracos, maos, rosto e corpo nao aparecem sobre a camera.

## Hipotese baseada no codigo

O hook copiado do RAR cria um segundo MediaPipe Holistic no navegador antes de iniciar a captura de JPEGs. Ele busca os arquivos do modelo numa CDN externa e chama `send()` sem aguardar a conclusao anterior. Falha no carregamento pode impedir a abertura do WebSocket; falha no `send()` pode interromper o callback antes da codificacao JPEG. Erros assincronos do overlay sao descartados silenciosamente. O backend ja executa MediaPipe Holistic para cada quadro, entao existe processamento duplicado.

## Plano

1. Manter a captura JPEG independente do rastreamento visual local.
2. Reaproveitar os landmarks brutos e as mascaras que o backend ja extrai de cada quadro e inclui-los na confirmacao `frame_ack`.
3. Desenhar no canvas do frontend maos, bracos/tronco e pontos faciais com os dados da confirmacao. Limpar o desenho ao parar a camera ou desconectar.
4. Verificar no teste WebSocket que os landmarks enviados correspondem ao extrator e que a predicao continua chegando apos 48 quadros. Compilar o frontend e registrar os limites da verificacao.

Esta correcao altera arquivos que tinham acabado de ser igualados ao RAR. A igualdade de hashes documentada em `PLANO_CONFERENCIA_FRONTEND_RAR.md` descreve o estado anterior a esta correcao.

## Implementacao e verificacao

- `useCameraFrames.ts` voltou a cuidar somente da camera e do JPEG. Nao carrega mais o MediaPipe no navegador, nao acessa a CDN e nao roda `send()` dentro do callback de video. A preferencia de 30 FPS e o controle de quadro novo permanecem.
- `VideoStreamSession` guarda os 346 landmarks brutos e a mascara de quatro partes extraidos no ultimo quadro. O WebSocket os envia em `frame_ack.data.overlay`, preservando o formato das predicoes. O frontend desenha maos, bracos/tronco e 40 pontos faciais no canvas sobre o video. As mascaras evitam desenhar partes nao detectadas.
- O texto do HUD indica `Pontos detectados` somente quando alguma parte foi detectada. O overlay e limpo ao parar ou desconectar.
- `npm run build` passou. `tests/test_web_video.py` passou: 48 confirmacoes carregaram os 346 pontos e a mascara esperada, e a predicao ainda chegou apos 48 quadros.
- `scripts/smoke_web_inference.py --video data/raw/minds_libras/08BanheiroSinalizador11-1.mp4` reconheceu Banheiro com confianca aproximada de 79,2% em 71 quadros amostrados. Deteccoes: mao esquerda 71/71, mao direita 69/71, pose 71/71 e face 71/71. Esse clipe pertence ao dataset e nao prova funcionamento com a webcam do usuario.

## Como validar no computador do usuario

Reiniciar backend e frontend para carregar o novo contrato WebSocket. Conferir `/api/health` (`model_loaded: true`), abrir a camera, observar se o indicador passa para `Pontos detectados` e se aparecem pontos no rosto, corpo e maos. A primeira predicao exige 48 quadros efetivamente processados; FPS de camera e FPS do backend podem diferir. Se ainda houver falha, registrar o estado de conexao na tela e as mensagens do terminal do backend; isso distingue ausencia de quadros, falha no rastreamento e falta de confianca do classificador.
