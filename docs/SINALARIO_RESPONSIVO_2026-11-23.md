# Sinalário responsivo

## Plano

Corrigir a sobreposição reproduzida nos prints: a sidebar continua sticky após
o grid virar uma coluna no celular. No mobile, categorias devem seguir o fluxo
normal, em uma grade compacta. No desktop, preservar a sidebar sticky na coluna
própria. Ajustar cabeçalho e espaço reservado para ele em telas estreitas, evitar
overflow de títulos/cards e manter o vídeo contido na miniatura sem cortar mãos.
Validar build e geometria antes/depois da rolagem em celular e desktop.
Preservar as alterações de estabilização de tradução já presentes no workspace.

## Resultado

Implementado:

- Até 768 px: categorias em duas colunas compactas, `position: static`; o painel
  rola normalmente antes dos vídeos. Acima disso: sidebar sticky em coluna própria.
- Colunas e cards usam limites que permitem encolher sem overflow. Nomes longos
  quebram linha, e as miniaturas contêm o vídeo com `object-fit: contain`.
- Efeitos de deslocamento por hover ficam restritos a ponteiro fino com hover.
- Menu superior usa duas linhas até 768 px (marca e navegação), com altura
  compartilhada por variável CSS e espaço correspondente no conteúdo. Isso
  evita menu cortado e título escondido. Botões mobile têm ao menos 44 px de altura.
- Fonte da busca em 16 px no mobile; espaçamento lateral ajustado a 16 px.

## Validação

Build TypeScript/Vite de produção passou (1.596 módulos). O build real foi
aberto no Edge headless, com catálogo local e arquivos de mídia existentes,
simulando larguras 320, 390, 768, 1024 e 1440 px. Em todas:

- sem overflow horizontal;
- título inicial abaixo do cabeçalho;
- todos os links da navegação dentro da tela;
- categorias e vídeos separados antes e depois de rolar 600 px;
- layout mobile estático e desktop sticky, sem exceções JavaScript.

Também passaram os oito testes de estabilização de tradução e os nove testes
de URLs, preservando o comportamento das alterações anteriores.

Capturas em `logs/sinalario_responsivo/<largura>-inicio.png` e
`<largura>-rolagem.png`; métricas em `resultado.json`. As capturas de 390 px
(início/rolagem) e 1440 px foram inspecionadas visualmente. A verificação usa
interceptação somente das respostas de catálogo, mantendo componentes e CSS do
build real; não altera API/modelo ou dados.

O teste cobre Chromium em emulação de viewport, não um celular físico ou Safari.
O tema escuro forçado pelo navegador dos prints não foi emulado.

## Reproduzir no Git Bash

Requer Node 22+ para o teste de navegador, Edge instalado no Windows (ou
`BROWSER_PATH` com caminho completo para outro Chromium) e dependências Python
do projeto para ler o catálogo. O build normal continua usando o requisito de
Node do projeto.

```bash
mkdir -p logs/sinalario_responsivo
./.venv/Scripts/python.exe -c "import json; from pathlib import Path; from backend.routers.dictionary import _load_from_folder; Path('logs/sinalario_responsivo/catalogo.json').write_text(json.dumps([e.model_dump() for e in _load_from_folder()], ensure_ascii=False), encoding='utf-8')"
cd frontend
VITE_API_URL=https://deeplearning-sinaliza.onrender.com/api npm run build
npm run test:responsive
```

O teste não acessa o navegador pessoal: cria perfil temporário isolado e um
servidor local temporário, encerrados ao final. Capturas e catálogo ficam em logs,
ignorados pelo Git.

## Entrega

Alterados: `frontend/src/pages/DictionaryPage.module.css`,
`frontend/src/components/Navbar.module.css`, `frontend/src/components/SignCard.module.css`,
`frontend/src/styles/global.css`, `frontend/src/App.tsx`, `frontend/package.json`
e `README.md`. Criados: este registro e `frontend/tests/check-responsive.mjs`.
As alterações anteriores de estabilização da tradução foram preservadas.
Após commit/push, fazer novo deploy do frontend na Vercel. Render e modelos
não precisam de alteração para esta correção.
