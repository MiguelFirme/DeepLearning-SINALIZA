# Plano: ativar por padrão a fusão de 25 classes

Data: 04/11/2026.

## Objetivo

Ao iniciar a aplicação sem selecionar modelo nem definir variável de ambiente, carregar a dupla original de 40 pontos faciais: `pose_face` e `full` (pose, face e mãos), ambos com 25 classes. Manter `configs/fusion25_frozen.json` como fonte dos caminhos, hashes, temperaturas e regra top 5 já avaliados.

## Alterações planejadas

1. Definir esse manifesto como valor padrão da configuração do backend. Preservar a opção de sobrescrevê-lo por variável de ambiente para experimentos e a opção de carregar explicitamente um checkpoint único em código.
2. Garantir que a imagem Docker inclua os dois checkpoints e seus mapas de classes. Conferir os caminhos do manifesto no contêiner.
3. Corrigir os textos do README e do guia de inicialização para que descrevam o modelo ativo de 25 classes e o resultado esperado em `/api/health`.
4. Testar inicialização real dos dois modelos, contrato de resposta, e regressão da configuração. Registrar o que foi verificado e os limites.

## Critérios de conclusão

Sem `SINALIZA_FUSION_MANIFEST_PATH` no ambiente, a configuração aponta para `configs/fusion25_frozen.json`; o backend carrega `pose_face` + `full`, ambos de 40 pontos faciais, e anuncia 25 classes. O manifesto mantém validação de hashes e vocabulário. O modo padrão não requer novos dados nem treinamento.
