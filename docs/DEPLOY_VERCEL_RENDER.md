# Deploy do Sinaliza: Vercel + Render

Correções verificadas em 23/11/2026. O modelo padrão continua sendo o Transformer
v2 de **39 classes**. Não houve retreino ou modificação dos pesos.

## Diagnóstico e alterações

- HTTP já lia `VITE_API_URL`, mas o WebSocket usava `window.location.host`,
  conectando à Vercel. A mensagem fixa de erro mencionava 8000 mesmo em HTTPS.
  Ambos agora usam `apiUrls`, com normalização de `/api`, barras e HTTP/HTTPS
  para WS/WSS. O único localhost:8000 de configuração permanece no proxy Vite,
  utilizado apenas em desenvolvimento. Os testes também usam exemplos locais.
- Os vídeos estavam presentes, versionados e incluídos por `COPY backend/`:
  74 MP4, 74 capas e 36 imagens, cerca de 4,3 MB. A API retorna caminhos como
  `/api/media/dictionary/...`; o frontend tentava buscá-los no próprio domínio.
  Agora o cliente resolve vídeos, capas e imagens no Render. URLs HTTP externas
  preservam sua origem. Nenhuma cópia do dataset de treino é necessária.
- `frontend/logos/Logo Maos.jpeg` não era publicado automaticamente pelo Vite.
  Uma cópia idêntica está em `frontend/public/logos/logo-maos.jpeg`; a Navbar
  usa `BASE_URL` para compor o caminho. A imagem foi conferida no `dist` por hash.
- Foram removidos marcadores de conflito Git já presentes no `HEAD` de
  `frontend/src/App.tsx`, que impediam qualquer build TypeScript. O endereço
  histórico `/collection` continua redirecionando para Início.
- `frontend/vercel.json` fornece fallback de SPA para acessos diretos às páginas.
- CORS aceita lista de origens por `SINALIZA_CORS_ORIGINS` (já disponível) e regex
  opcional por `SINALIZA_CORS_ORIGIN_REGEX`. Sem configuração, mantém as origens
  locais. Não foi liberada indiscriminadamente toda a Vercel.
- O CMD do Docker inicia Uvicorn em `0.0.0.0:${PORT:-8000}`, com `exec` para
  encaminhar sinais. `EXPOSE 8000` é metadado e não fixa a porta no Render.
  `.dockerignore` exclui `.env`, datasets, caches e experimentos do contexto;
  preserva `backend/static/dictionary` e `Modelos salvos/`.

## Após o push: Render

1. Use o repositório como contexto (raiz vazia ou `.`), Dockerfile
   `Dockerfile.backend`, e health check `/api/health`.
2. Deixe **Docker Command** sem override antigo para usar o CMD da imagem.
3. Em Environment, configure a origem real do frontend, sem caminho ou barra
   final. Exemplo, substituindo o domínio abaixo:

```text
SINALIZA_CORS_ORIGINS=["https://SEU-PROJETO.vercel.app"]
```

Se usar domínio próprio, inclua-o no mesmo array. Para também usar o backend
público com Vite local, inclua `http://localhost:5173` explicitamente.
O domínio real da Vercel não foi fornecido nesta tarefa; ele precisa ser
informado no painel. O arquivo `.env.render.example` é referência, não é lido
automaticamente pelo Render.

Para previews, a opção preferível é acrescentar suas origens exatas ao array.
Opcionalmente use `SINALIZA_CORS_ORIGIN_REGEX` com uma expressão restrita ao seu
projeto/equipe. Exemplo a adaptar ao formato real dos seus previews:
`https://SEU-PROJETO-[a-z0-9-]+-SUA-EQUIPE\.vercel\.app`.
Não copie a regex sem conferir os domínios produzidos pelo seu projeto.
O CORS regula as chamadas HTTP do navegador; não é autenticação nem habilita
WebSocket na Vercel. A conexão de câmera vai diretamente ao WebSocket do Render.

4. Não é preciso definir `PORT` manualmente. O serviço usa a variável fornecida
   pelo Render, ou 8000 quando executado localmente sem ela. Isso segue o
   [contrato de porta de web services do Render](https://render.com/docs/web-services#port-binding).
5. Faça redeploy do backend após o push e a configuração das variáveis.

## Após o push: Vercel

1. Root Directory: **frontend**; framework: **Vite**; Build Command:
   `npm run build`; Output Directory: **dist**.
2. Em Environment Variables, mantenha em Production e, se utilizado, Preview:

```text
VITE_API_URL=https://deeplearning-sinaliza.onrender.com/api
```

O código também aceita barra final ou a origem sem `/api`, mas prefira o valor
acima. Não use localhost, `:8000` nem `wss://` nessa variável.
3. Faça **Redeploy** para gerar um novo bundle com a variável. Variáveis alteradas
   não atualizam bundles existentes; veja a
   [documentação de variáveis da Vercel](https://vercel.com/docs/environment-variables/managing-environment-variables).
4. Não é necessário cadastrar um proxy para o WebSocket na Vercel. O navegador
   deve conectar a `wss://deeplearning-sinaliza.onrender.com/api/ws/predict`.
   O fallback SPA segue o [guia Vite da Vercel](https://vercel.com/docs/frameworks/frontend/vite#using-vite-to-make-spas).

## Conferência depois do deploy

- `/api/health` no Render: `model_loaded: true`, `num_classes: 39`.
- DevTools → Network no frontend: HTTP em `https://...onrender.com/api/...` e
  WebSocket em `wss://...onrender.com/api/ws/predict` com handshake 101.
- Navbar: a logo deve abrir em `https://SEU-PROJETO.vercel.app/logos/logo-maos.jpeg`.
- Sinalário: vídeos, capas e imagens devem usar a origem `onrender.com`,
  conservando os nomes e acentos codificados. Não deve haver chamadas de API para
  a origem da Vercel nem mensagens mencionando porta 8000.
- Acesso direto a `/dictionary` e `/settings` deve abrir a página normalmente.

Para conferir o CORS pelo Git Bash, usando a origem real:

```bash
curl -i -X OPTIONS 'https://deeplearning-sinaliza.onrender.com/api/health' \
  -H 'Origin: https://SEU-PROJETO.vercel.app' \
  -H 'Access-Control-Request-Method: GET' \
  -H 'Access-Control-Request-Headers: content-type'
```

Espere `access-control-allow-origin` com a origem informada. Se a origem não
estiver liberada, o preflight retorna 400. Um GET no terminal sozinho não prova
que o CORS está correto no navegador.

## Validação executada e limites

```bash
cd frontend
npm run test:urls
VITE_API_URL=https://deeplearning-sinaliza.onrender.com/api npm run build
cd ..
./.venv/Scripts/python.exe -m pytest tests/test_production_deploy.py tests/test_api.py tests/test_default_fusion25_startup.py tests/test_default_v2_websocket.py tests/test_config.py -q
```

**9 testes de URLs passaram**; build TypeScript/Vite com a variável de produção
passou (1.595 módulos). **13 testes de backend passaram**, com dois avisos de
depreciação de dependências: origens CORS permitidas/recusadas, mídia com
requisição Range (206), imagem/capa, API e checkpoint real padrão de 39 classes.

Antes de qualquer redeploy desta tarefa, consultas somente de leitura ao Render
confirmaram health OK com 39 classes, catálogo com caminhos relativos e um vídeo
de Educação com status 200, Content-Type video/mp4 e Accept-Ranges: bytes.
Logo original e versão em dist têm o mesmo SHA-256.

Não foi possível construir uma imagem Docker neste ambiente (CLI indisponível).
O painel da Vercel/Render não foi alterado, o domínio do frontend não foi
informado e não houve teste humano de câmera no site publicado. A validação
fim a fim no seu domínio ocorre depois do push, variáveis e redeploy acima.

## Arquivos desta alteração

Modificados:

- `Dockerfile.backend`: porta dinâmica.
- `backend/config.py`, `backend/main.py`: regex CORS opcional.
- `frontend/package.json`: comando de testes de URLs.
- `frontend/src/App.tsx`: remoção de marcadores de conflito existentes.
- `frontend/src/components/Navbar.tsx`: logo pública.
- `frontend/src/hooks/useWebSocket.ts`: base compartilhada e mensagem de erro.
- `frontend/src/services/api.ts`: base compartilhada e resolução de mídia.
- `README.md`: acesso a este guia de produção.

Criados:

- `.dockerignore`, `.env.render.example`.
- `frontend/.env.example`, `frontend/vercel.json`.
- `frontend/public/logos/logo-maos.jpeg`.
- `frontend/src/services/urls.ts`, `frontend/src/services/apiUrls.ts`.
- `frontend/tests/urls.test.mjs`, `tests/test_production_deploy.py`.
- `docs/PLANO_DEPLOY_VERCEL_RENDER_2026-11-23.md` e este guia.
