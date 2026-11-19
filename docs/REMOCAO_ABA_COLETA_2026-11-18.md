# Remoção da aba de captura/coleta

## Plano

Por solicitação da equipe, remover o item Coleta da navegação e a página de
gravação do frontend. Redirecionar o endereço antigo `/collection` para Início,
evitando uma tela vazia em links salvos. Excluir o componente e seu CSS exclusivos.
Preservar o reconhecimento pela câmera, o modelo padrão e os dados já gravados.
O protocolo de coleta do backend permanece disponível para ferramentas técnicas.
Atualizar os guias atuais e verificar a compilação TypeScript/Vite.

## Resultado

Item Coleta removido da navegação, página e CSS exclusivos excluídos, e rota
antiga redirecionada para Início com substituição do histórico. README e guia
do pipeline atualizados. O backend e o modelo padrão não foram alterados.

Verificação: `npm.cmd run build` em `frontend/` passou (TypeScript + Vite,
1.593 módulos); `git diff --check` sem erros de whitespace. A primeira tentativa
de build foi bloqueada por acesso do esbuild fora do sandbox; a repetição com
permissão ampliada passou. Não foi realizado teste manual no navegador.
