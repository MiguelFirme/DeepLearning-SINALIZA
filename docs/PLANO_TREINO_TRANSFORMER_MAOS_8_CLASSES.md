# Treino do Transformer de mãos — MINDS-Libras, 8 classes

## Objetivo

Treinar e salvar um segundo modelo das mesmas oito classes, com a mesma arquitetura temporal, splits, seed e técnicas de treino do Transformer `pose_face`, mas usando somente os landmarks das mãos e derivadas calculadas a partir deles. Este experimento não altera o checkpoint `pose_face` nem ativa fusão no backend.

## Plano de execução

1. Adicionar `feature_mode=hands_only` compartilhado por treino e inferência. Zerar pose e face, inclusive suas máscaras, antes de augmentation e normalização. Sem referência aos ombros, o normalizador mantém as coordenadas MediaPipe das mãos e calcula velocidades e distância entre mãos; as demais features ficam zero. A entrada permanece com 697 posições, permitindo a mesma arquitetura e número de parâmetros do modelo atual.
2. Verificar por teste automatizado que mudar pose/face não altera a entrada `hands_only`, que mudar mãos altera, e que o modo `pose_face` existente não sofre regressão.
3. Criar configuração derivada do mesmo YAML do modelo de oito classes, alterando somente `data.feature_mode`.
4. Treinar com os mesmos 240 vídeos de treino e 40 de validação; escolher `best.pt` exclusivamente pela menor `val_loss`. Registrar histórico, logs, hashes dos splits, métricas e ambiente.
5. Avaliar o checkpoint escolhido sem usar o resultado do teste para escolher configurações. O sinalizador 12 já foi consultado antes e qualquer número nele é exploratório.
6. Arquivar checkpoint, configuração, splits, métricas e README em `Modelos salvos/transformer_minds_8_classes_hands_only/`. Verificar que o checkpoint arquivado carrega com pesos estritos e anuncia `feature_mode=hands_only`.

## Riscos conhecidos

A auditoria de 18/10/2026 encontrou detecção de ao menos uma mão em 48,77% dos quadros de treino, contra 99,75% de validação. A contagem inclui repouso e não mede a qualidade da mão ativa durante o sinal. O modelo pode aprender padrões de falha de detecção e não generalizar; por isso será mantido separado do backend até comparação com o `pose_face` em pessoas novas. A interface de entrada de 697 posições contém muitas posições zeradas, escolhida para controlar arquitetura e evitar informações de pose/face.

## Execução e resultados — 18/10/2026

- Implementado `feature_mode=hands_only` em `ml/features/feature_modes.py`, usado pelo dataset e pela inferência. A configuração `configs/transformer_minds_hands_only.yaml` herda a do Transformer `pose_face` e muda somente essa modalidade.
- Testes de isolamento das partes do corpo, normalização e backend: 11 passaram. O checkpoint arquivado carrega no `PredictionService` com pesos estritos e contém `feature_mode=hands_only`, 48 frames e a mesma arquitetura de 121.018 parâmetros. O SHA-256 do arquivo salvo é igual ao do experimento original; 23 checksums do pacote foram conferidos.
- Treino CUDA: 240 vídeos de treino, 40 de validação; early stopping na época 16; melhor época 1 por `val_loss=2,0803`.
- Validação: 5/40 (12,5%), F1 macro 0,0278. O modelo previu `Acontecer` para todos os 40 vídeos.
- Teste exploratório: 6/40 (15%), F1 macro 0,0648; 38 previsões `Acontecer` e 2 `Conhecer`. O teste não orientou escolha de checkpoint ou hiperparâmetros.
- Modelo, cadeia de configuração, splits, logs e previsões arquivados em `Modelos salvos/transformer_minds_8_classes_hands_only/`; instruções de reprodução no README da pasta.

**Decisão:** manter o checkpoint separado. A validação não sustenta ativá-lo no backend ou fundir seus escores com o modelo `pose_face`. O próximo experimento deve investigar qualidade, continuidade e distribuição dos landmarks de mãos entre sinalizadores, sem escolher ajustes pela métrica do sinalizador de teste já consultado.
