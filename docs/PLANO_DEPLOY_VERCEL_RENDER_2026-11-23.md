# Plano de correção do deploy Vercel + Render

1. Centralizar URLs HTTP, WebSocket e mídia a partir de VITE_API_URL, mantendo
   `/api` e o proxy Vite no desenvolvimento. Testar os casos com e sem `/api` e
   barra final, HTTPS/WSS, URLs relativas e mídia externa.
2. Publicar a logo via public/ e configurar fallback SPA da Vercel.
3. Resolver vídeos, capas e imagens do sinalário contra a API do Render.
   Auditoria local: 185 arquivos em backend/static/dictionary versionados,
   incluindo 74 MP4 e 74 capas, cerca de 4,3 MB; COPY backend/ já os inclui.
4. Permitir CORS por lista de origens e regex opcionais configuradas no Render,
   com testes de preflight permitido e origem recusada.
5. Inicializar Uvicorn em 0.0.0.0:${PORT:-8000} e excluir dados/segredos/cache
   do contexto Docker sem excluir os modelos e as mídias publicados.
6. Validar URLs com Node, build TypeScript/Vite em modo produção, testes FastAPI
   e carregamento do modelo atual de 39 classes. Documentar variáveis, redeploy
   e limites de verificação. Não retreinar nem trocar o modelo.

## Conclusão

Implementado e documentado em `docs/DEPLOY_VERCEL_RENDER.md`. Nove testes de
URLs e 13 testes FastAPI passaram; build de produção passou e a logo foi
conferida no dist. Bundle contém a origem Render e não contém localhost:8000
nem a antiga mensagem fixa. Corrigidos também os marcadores de conflito Git
preexistentes em App.tsx. Health e vídeo público do Render foram conferidos
por leitura. Modelo e pesos de 39 classes preservados. Docker CLI indisponível;
painéis e câmera no site publicado não foram operados.

## Integração com o GitHub

O rebase sobre `b6ea868` apresentou conflito em App.tsx somente na linha em
branco junto à rota histórica `/collection`. Marcadores removidos preservando
as três páginas e o redirecionamento. Build de produção TypeScript/Vite passou
novamente antes de concluir o rebase. O script `iniciar_sinaliza.bat` recebido
do remoto foi preservado.
