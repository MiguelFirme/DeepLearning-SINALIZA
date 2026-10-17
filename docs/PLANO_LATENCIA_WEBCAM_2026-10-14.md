# Latencia percebida apos recuperar a acuracia da webcam

## Plano

1. Conferir o numero de quadros ate a primeira e as proximas predicoes, o controle de fila e o custo por quadro.
2. Reduzir o intervalo entre predicoes posteriores a primeira, mantendo os 48 quadros de contexto do modelo e sem alterar detector, resolucao ou checkpoint.
3. Exibir o FPS efetivamente processado pelo backend, separado do FPS da camera, para identificar o gargalo na Canon.
4. Compilar o frontend e verificar o contrato WebSocket; testar ao vivo para avaliar tempo e acuracia.

## Estado encontrado

- O primeiro resultado exige 48 quadros processados. Depois a sessao esperava mais 24 quadros para cada resultado.
- O navegador pode entregar 30 FPS, mas quadros sao descartados quando ha dois envios pendentes. A taxa efetiva depende do MediaPipe no backend.
- O campo `processing_time_ms` da predicao mede normalizacao + Transformer, sem incluir decodificacao JPEG e MediaPipe. Logo nao representa a latencia completa de um quadro.
- O MediaPipe Holistic opera com `model_complexity=2`, igual ao extrator usado no dataset. Trocar essa configuracao agora poderia alterar a qualidade dos pontos e os acertos recuperados.

## Implementacao e verificacao

- A sessao usa janelas de 48 quadros com avancos de 12, em vez de 24, depois da primeira predicao. Isso diminui pela metade a espera maxima entre tentativas posteriores, medida em quadros processados. Nao reduz a espera pela primeira janela nem o custo do MediaPipe.
- `frame_ack` agora informa o tempo total de processamento do quadro no backend (JPEG, MediaPipe e, em alguns quadros, Transformer). O frontend conta as confirmacoes por segundo como `FPS analise` e mede a ida e volta do quadro correspondente. O painel separa esse numero do tempo informado pelo Transformer. A media e atualizada uma vez por segundo, sem renderizar a pagina a cada quadro.
- `npm.cmd run build` passou com TypeScript e Vite. `git diff --check` passou. O teste WebSocket foi ampliado para conferir o tempo de processamento informado, mas nao pode ser executado aqui porque a `.venv` referencia Python 3.11 ausente e `py -0p` nao encontra Python instalado.
- Falta medir na Canon R100 o `FPS analise`, `Ida e volta/quadro` e `Backend/quadro`, alem do tempo entre concluir Banheiro e receber a palavra. Se `FPS analise` for baixo, o principal limite pode ser a extracao MediaPipe; a mudanca do intervalo de predicao nao resolve isso sozinha.
