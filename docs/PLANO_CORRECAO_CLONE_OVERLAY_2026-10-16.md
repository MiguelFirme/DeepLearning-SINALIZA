# Pontos nao aparecem em clones do repositorio

## Plano

1. Comparar o frontend e o backend versionados no commit atual com o diretorio local que funciona.
2. Versionar o contrato WebSocket completo para os pontos, junto com um teste que verifique a resposta do backend.
3. Validar o que for possivel neste ambiente e publicar a correcao no repositorio remoto para que novos clones a recebam.
4. Documentar como as colegas podem atualizar um clone existente e confirmar que frontend e backend usam a mesma revisao.

## Diagnostico

No commit `0fce899`, o frontend espera `frame_ack.data.overlay` e desenha os 346 valores de landmarks no canvas. O backend desse mesmo commit envia `frame_ack.data = {}`. Os arquivos `backend/routers/websocket.py` e `backend/services/video_stream.py` que geram esse payload estao alterados apenas no diretorio local. Assim, o clone tem um contrato WebSocket incompatível entre frontend e backend. O frontend pode conectar e traduzir, mas nao recebe coordenadas para as marcas.

O remoto `origin/main` esta em `dcda8c4`, dois commits a frente da copia local, com alteracoes de dicionario. Os arquivos do contrato WebSocket nao mudaram nesses commits. A correcao sera aplicada a partir dessa versao remota em uma area isolada para preservar os commits e as alteracoes locais.

## Implementacao e verificacao

- O backend envia `frame_ack.data.overlay` com 346 coordenadas e quatro mascaras, alem de `frame_processing_ms`. O `ready` anuncia `overlay: true`.
- O frontend mostra uma mensagem explicita quando se conecta a uma versao antiga do backend sem esse anuncio. Isso evita que a ausencia de pontos pareca uma falha silenciosa da webcam.
- O teste `tests/test_web_video.py` verifica o anuncio e o payload de pontos no WebSocket. O build do frontend passou sobre `origin/main` em `dcda8c4`; `git diff --check` passou.
- O teste Python nao pode ser executado nesta maquina: a `.venv` aponta para um Python 3.11 ausente e o launcher `py -0p` nao encontra instalacoes. A camera das colegas tambem nao pode ser testada daqui. Se o contrato estiver correto mas as mascaras vierem `false`, sera necessario investigar a deteccao MediaPipe na webcam delas.

## Atualizacao de um clone existente

Na raiz do repositorio, em Git Bash, depois que a correcao estiver no remoto:

```bash
git pull --ff-only origin main
git rev-parse --short HEAD
```

Reiniciar **backend e frontend** apos atualizar. No navegador, abrir o console ou a aba Network e confirmar que a mensagem WebSocket `ready` traz `overlay: true` e que `frame_ack.data.overlay` contem `landmarks` e `mask`. Se a pagina mostrar "Backend sem dados dos pontos", o backend ainda esta rodando uma revisao antiga ou em outra pasta/porta. O desenho depende tambem de o MediaPipe detectar partes do corpo naquele quadro.

## Publicacao

O commit `0e69de4` foi publicado em `origin/main` em 16/10/2026. Ele inclui o backend, o aviso de incompatibilidade no frontend, o teste e esta documentacao. O build passou; a validacao com a webcam das colegas ainda depende de elas atualizarem e reiniciarem os processos.
