# Plano: fusão pose + face16 com pose + face40 + mãos

Data: 04/11/2026.

## Objetivo

Testar, nas mesmas 25 classes e nos mesmos vídeos, a dupla híbrida formada pelo checkpoint `transformer_mixed25_pose_face16` e pelo checkpoint `transformer_mixed25_full` (40 pontos faciais e mãos). Os modelos já estão treinados; esta etapa não requer nova extração nem treino.

## Protocolo fixado antes da avaliação

1. Usar o split congelado de 500/85/85 vídeos e o mesmo mapa de 25 classes.
2. Usar o `pose_face16` para propor as cinco classes candidatas. Combinar os escores calibrados com peso 0,22 para pose + face16 e 0,78 para o modelo completo de 40 pontos.
3. Herdar as temperaturas da fusão original: 1,028734578 para o modelo de pose e 0,792446596 para o completo. Não ajustar pesos, temperaturas ou top-k após ver os resultados.
4. Calcular acurácia, F1 macro e acertos nas dez classes novas, com comparação pareada aos pares 40/40 e 16/16. Verificar que todos os 85 caminhos e rótulos correspondem em cada split.
5. Salvar predições por vídeo, métricas, manifesto experimental com hashes e uma checagem de que a inferência no backend reproduz a avaliação offline.

## Limites

O teste de 85 vídeos já foi consultado em experiências anteriores e serve apenas como análise exploratória. Os parâmetros de calibração foram escolhidos para os modelos de 40 pontos, podendo ser subótimos para a dupla híbrida. A combinação não substitui automaticamente o modelo usado pela aplicação.
