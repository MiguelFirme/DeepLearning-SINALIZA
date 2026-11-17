# Plano: retreino v2 com o grupo ampliado

Data: 17/11/2026. Pedido: fornecer comandos para retreinar a arquitetura Transformer v2 padrão com os novos vídeos em `data/raw/Videos do grupo`.

## Evidências antes da alteração

- Modelo padrão: `Modelos salvos/transformer_v2_mixed23_full/best.pt`, pipeline v2 original, sem segmentação experimental.
- MINDS local: 13 classes, 520 landmarks.
- Grupo ampliado: 28 pastas de classes, 195 MP4. Os números 01–07 são usados como identidade de pessoa, como no treino anterior.
- `EscolaSinalizador04.mp4` está ausente. Isso reduz em uma amostra o treino, sem impedir que Escola esteja nos três splits.
- O usuário confirmou que `Errado/ERRADO.mp4` pertence à pessoa 07 e que os dois arquivos `DiretorSinalizador06/07.mp4` dentro de Diretoria devem ter o rótulo dos demais arquivos dessa pasta. Serão renomeados para o padrão, sem mudar conteúdo.

## Ações

1. Validar os três caminhos e nomes de destino, registrar hashes e renomear somente os três arquivos confirmados.
2. Criar um montador de manifesto misto que descubra classes do grupo sem limitar a dez palavras ou exigir cinco vídeos por classe no treino.
3. Manter holdout MINDS 01/02/05/06/08/10, 11, 12 e grupo 01–05, 06, 07, em treino/validação/teste. Exigir identidade válida, arquivos únicos e exemplos de cada classe nos três splits.
4. Fornecer configuração da mesma arquitetura/pré-processamento do modelo padrão. Usar novos diretórios para landmarks, splits e experimento, preservando os anteriores.
5. Testar o montador com classes novas, Escola sem a pessoa 04 e rejeição de vazamento/duplicação. Documentar os comandos; o usuário executará a extração e o treino.

Contagens esperadas para os arquivos presentes: 41 classes, 715 vídeos, 529 treino, 93 validação, 93 teste. O retreino é completo, pois o classificador passa de 23 para 41 saídas. O modelo servido não será trocado automaticamente ao terminar esse experimento.
