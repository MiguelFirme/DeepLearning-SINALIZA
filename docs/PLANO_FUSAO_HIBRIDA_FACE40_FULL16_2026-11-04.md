# Plano: fusão pose + face40 com modelo completo face16

Data: 04/11/2026.

## Objetivo

Avaliar a inversão da dupla híbrida anterior usando os checkpoints existentes: `transformer_mixed25_pose_face` com 40 pontos faciais e sem mãos, seguido por `transformer_mixed25_full_face16` com 16 pontos faciais e ambas as mãos. Não haverá reextração nem treino.

## Protocolo fixado

Usar as mesmas 25 classes, 85 vídeos de validação e 85 de teste, conferindo caminhos e rótulos. O modelo `pose_face40` propõe as cinco classes candidatas; o escore final mantém peso 0,22 para ele e 0,78 para `full_face16`. Temperaturas herdadas da dupla original: 1,028734578 e 0,792446596. Não ajustar esses números após conferir os resultados. Comparar acertos, F1 macro, classes novas e exemplos pareados com a dupla original 40/40 e a híbrida 16/40.

Salvar o CSV por vídeo, relatório JSON, manifesto experimental com hashes e teste que confirme equivalência entre avaliação offline e inferência do backend. O teste já foi consultado em trabalhos anteriores e será interpretado como exploratório.
