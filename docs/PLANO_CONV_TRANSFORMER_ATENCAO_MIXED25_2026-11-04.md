# Plano: Conv1D + Transformer + atenção temporal

Data: 04/11/2026.

## Objetivo

Testar a substituição da BiLSTM pela codificação Transformer no experimento de 25 classes: projeção e Conv1D temporal, duas camadas Transformer com autoatenção, atenção mascarada para resumir os quadros e classificador. Comparar com a dupla Transformer original e a dupla Conv1D + BiLSTM + atenção já treinada.

## Protocolo congelado antes do treino

1. Preservar os 670 vídeos, mapa de 25 classes, splits 500/85/85, 48 quadros, 697 features, normalização e seleção de 40 pontos faciais. Treinar `pose_face` e `full` separadamente, seed 42, sem alterar os dados.
2. Preservar a configuração de treino anterior: augmentation apenas em treino, pesos de classe `sqrt_inverse`, Adam `3e-4`, L2 `1e-4`, dropout 0,45, scheduler por `val_loss`, early stopping 15. Selecionar checkpoint somente por menor `val_loss` e não avaliar teste durante o treino.
3. Usar projeção 64, Conv1D com 64 canais/kernel 3, duas camadas Transformer de dimensão 64, quatro cabeças, feedforward 128, e atenção temporal final. Validar máscaras e carregamento estrito do checkpoint.
4. Avaliar ambos isolados e a fusão fixa: top 5 do `pose_face`, peso 0,22 para pose e 0,78 para completo, temperaturas 1,0. Não otimizar essa regra depois de ver o teste.
5. Comparar acertos, F1 macro, classes novas e latência de CPU na mesma janela com os dois pares anteriores. Arquivar pesos, configurações, logs, hashes e manifesto experimental. Não mudar o padrão da aplicação com base no teste já consultado.

## Limites

O teste antigo de 85 vídeos já foi consultado e permanece exploratório. A atenção temporal adicional resume saídas do Transformer; ela não é automaticamente melhor que o token CLS do modelo atual. Uma decisão de adoção exige vídeos de pessoas/sessões inéditas e rejeição de não sinais.
