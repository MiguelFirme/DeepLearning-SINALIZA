# Integração da `main` com GitHub — 21/10/2026

## Objetivo e plano

Resolver a rejeição `non-fast-forward` do `git push origin main` sem descartar o commit local do modelo de 15 classes nem os commits remotos. Procedimento: conferir árvore local limpa; buscar `origin/main`; inspecionar divergência; fazer merge sem commit automático; resolver conflitos se houver; testar backend e frontend; registrar resultado; criar commit de merge; enviar `main` e verificar a referência remota.

## Estado encontrado

- `main` local limpa antes da integração, com um commit à frente: `723c42b`, que registra o modelo de 15 classes e trabalhos anteriores.
- `origin/main` com quatro commits ainda ausentes localmente: `d300549`, `dcda8c4`, `0e69de4` e `f7a7199`. Eles incluem mídia do dicionário e correções do overlay da webcam.
- Nove caminhos modificados em ambos os lados. `git merge --no-ff --no-commit origin/main` terminou automaticamente, sem conflitos e sem sobrescrever o commit local.

## Verificações antes do commit

- `python -m pytest tests -q`: **98 testes passaram**. A coleta irrestrita da raiz encontrou um script externo em `artifacts/wlasl_source/` que executa `argparse` no import; a suíte correta do projeto fica em `tests/`.
- `npm.cmd run build` em `frontend/`: compilação TypeScript e build Vite concluídos. O primeiro build sob sandbox não conseguia ler `vite.config.ts`; a execução com acesso ao workspace confirmou a compilação.
- Dois espaços finais em linhas do frontend adicionadas no remoto foram removidos. Conferir `git diff --cached --check` após incluir esse ajuste.
- O merge preserva os dois pais, mantendo visíveis o commit local e os quatro commits remotos.

## Publicação

O resultado do `push` e a igualdade entre `main` local e `origin/main` são verificados após o commit de merge, antes de considerar a integração concluída.
