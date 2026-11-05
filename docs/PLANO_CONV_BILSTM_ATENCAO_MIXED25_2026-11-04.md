# Plano: Conv1D + BiLSTM + atenção nas 25 classes

Data: 04/11/2026.

## Pergunta experimental

Uma rede pequena com convolução temporal antes de duas camadas BiLSTM e atenção temporal generaliza melhor que os Transformers atuais no mesmo conjunto de 25 sinais? O teste deve manter os 670 vídeos, os mesmos splits 500/85/85, 48 quadros, 697 features, normalização, augmentation apenas no treino, loss, otimizador, seed e seleção por menor `val_loss` usados na dupla original de 40 pontos faciais.

## Protocolo fixado antes do treino

1. Adicionar `conv_bilstm_attention` ao registro de modelos. Entrada LayerNorm → projeção 64 → Conv1D temporal 64 canais/kernel 3 → BatchNorm/GELU/dropout → BiLSTM de duas camadas com 32 unidades por direção → atenção temporal mascarada → classificador. Preservar máscara dos quadros em todas as etapas. Registrar total de parâmetros.
2. Treinar do zero, seed 42, dois checkpoints: `pose_face` (40 pontos faciais, sem detalhes das mãos) e `full` (mesmos pontos mais as mãos). Não alterar dados, split nem checkpoints atuais.
3. Selecionar cada melhor época apenas por `val_loss`, com early stopping. Salvar logs, configuração efetiva, hashes e checkpoints. Não usar o teste para decisões de configuração.
4. Avaliar os modelos isolados e uma fusão experimental fixa em validação e teste: top 5 do `pose_face`, peso 0,22 para ele e 0,78 para `full`, temperatura 1,0 em ambos. A fusão é apenas uma comparação exploratória; as temperaturas da dupla Transformer não são transferíveis automaticamente.
5. Comparar acurácia/F1 macro e acertos nas dez classes novas contra a fusão Transformer congelada. O conjunto de teste já foi consultado em experimentos anteriores e permanece exploratório. Não ativar a nova rede no backend sem evidência adicional.

## Critérios técnicos

Teste de formato de saída e máscara, carregamento estrito do checkpoint via `create_model`, validação do split e execução offline reproduzível. Caso o treino não caiba no ambiente, preservar o código e registrar claramente a execução pendente.
