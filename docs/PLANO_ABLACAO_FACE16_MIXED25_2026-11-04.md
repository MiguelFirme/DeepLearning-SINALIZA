# Plano: ablação facial de 16 pontos nos dois Transformers de 25 classes

Data: 04/11/2026.

## Pergunta

Selecionar só 16 pontos da face, concentrados em olhos, sobrancelhas e boca, melhora a classificação em relação aos modelos atuais de 40 pontos faciais?

## Protocolo fixado antes do treino

1. Usar os mesmos 670 vídeos, manifesto, mapa de 25 classes e splits `500/85/85` em `data/landmarks/mixed_25` e `data/processed/mixed_25`. Não reextrair os MP4: todos os 16 pontos já estão nos arquivos `.npz`.
2. Manter os 346 valores brutos e os 697 atributos normalizados por quadro. Zerar as coordenadas dos outros 24 pontos faciais antes **e depois** da normalização, inclusive velocidades e possíveis alterações pela augmentation no treino. A normalização centralizada nos ombros transforma zeros em valores não nulos; a segunda máscara é indispensável. Assim, a arquitetura, quantidade de parâmetros e demais modalidades ficam iguais às dos baselines; apenas a informação facial muda.
3. Preservar exatamente os parâmetros dos dois treinos anteriores, seed 42, 48 quadros, augmentation, loss, otimizador e early stopping. Treinar um modelo `pose_face16` e um `full_face16` em experimentos novos, sem sobrescrever os checkpoints de 40 pontos.
4. Comparar individualmente os modelos e a fusão sequencial. Fixar antes de olhar os resultados a mesma regra da fusão antiga: proposta top 5 do `pose_face`, escore `0,22 × pose_face + 0,78 × full`, temperaturas `1,028734578` e `0,792446596` herdadas do protocolo de 40 pontos. Isso isola a mudança dos pontos. Não ajustar parâmetros no teste.
5. Registrar acurácia top 1, F1 macro, acertos nas 10 palavras novas e diferença por classe na validação. O teste de 85 vídeos já havia sido consultado em experimentos anteriores; qualquer comparação nele será rotulada **exploratória**. Não afirmar generalização a novas pessoas.
6. Salvar checkpoints, configurações, hashes, logs e relatório reproduzível em caminhos distintos. Manter o backend padrão atual até o resultado ser avaliado.

## Seleção facial exata (índices MediaPipe)

- Sobrancelhas: `46, 65, 276, 295` (4).
- Olhos: `33, 133, 160, 144, 362, 263, 387, 373` (8).
- Boca: `13, 14, 78, 308` (4).

Total: 16 pontos, todos presentes entre os 40 já armazenados. A ordem física do tensor segue o extrator original; esta lista só define quais posições continuam ativas.

## Estado do ambiente

No ambiente isolado, `.venv/Scripts/python.exe` parecia apontar para um Python 3.11 removido. A checagem fora desse isolamento confirmou que a instalação existe: Python 3.11.9, PyTorch 2.14.0+cu130, CUDA disponível na NVIDIA GeForce RTX 4050 Laptop GPU. Os comandos de treino precisam acessar essa instalação.

## Diário de execução

- Seleção implementada nos modos `pose_face16` e `full_face16`, aplicada no dataset e nos caminhos de inferência. A normalização recriava valores em pontos brutos zerados; a máscara posterior resolveu esse detalhe.
- Treinos concluídos na RTX 4050 com 500 amostras de treino e 85 de validação. `pose_face16`: melhor época 67, parada na 82. `full_face16`: melhor época 63, parada na 78.
- Avaliação pareada com a regra fixa: fusão 40 → 16 pontos, validação 50/85 → 45/85; teste exploratório 56/85 → 42/85. Ver relatório detalhado em `RELATORIO_ABLACAO_FACE16_MIXED25_2026-11-04.md`.
- Checkpoints, configuração, splits, histórico, logs, hashes e manifesto experimental arquivados em `Modelos salvos/` e `configs/fusion25_face16_fixed_rule.json`.
- Testes selecionados: 18 passaram. Inferência online da fusão nova conferida contra as probabilidades offline; a fusão antiga permaneceu compatível.
