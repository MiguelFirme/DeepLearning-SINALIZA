# Experimento Conv1D + BiLSTM + atenção em 25 classes

Data: 04/11/2026. [Plano e protocolo antes do treino](PLANO_CONV_BILSTM_ATENCAO_MIXED25_2026-11-04.md).

Atualização em 11/11/2026: por solicitação do usuário, a dupla foi ativada como padrão por meio de um manifesto de implantação separado. Veja o [relatório da integração](RELATORIO_ATIVACAO_CONV_BILSTM_PADRAO_2026-11-11.md). O texto abaixo registra o estado e os resultados do experimento original.

## Implementação e dados

Foi adicionada a arquitetura `conv_bilstm_attention`: LayerNorm de entrada → projeção 697→64 → Conv1D temporal de 64 canais e kernel 3 → BatchNorm/GELU/dropout → duas BiLSTMs de 32 unidades por direção → atenção temporal mascarada → classificador. Cada modelo tem **116.939 parâmetros**, contra aproximadamente 122 mil de cada Transformer anterior. A implementação preserva a máscara de padding antes/depois da convolução e na atenção.

Foram treinados **dois modelos do zero**, `pose_face` e `full`, com os mesmos 40 pontos faciais da dupla ativa. Usaram os mesmos 670 vídeos e mapa de 25 classes, splits 500/85/85, 48 quadros, 697 features, normalização, augmentation apenas no treino, pesos de classe `sqrt_inverse`, Adam `lr=3e-4`, `weight_decay=1e-4`, scheduler por `val_loss`, dropout 0,45, seed 42 e early stopping com paciência 15. Os hashes do manifesto, split e mapa de classes coincidem com os da fusão Transformer congelada. O conjunto de teste não foi usado para selecionar épocas: ambos os treinos rodaram com `--skip-test`.

| Modelo | Melhor época por `val_loss` | Parada | Tempo GPU |
| --- | ---: | ---: | ---: |
| ConvBiLSTM `pose_face` | 30 | 45 | 172 s |
| ConvBiLSTM `full` | 45 | 60 | 251 s |

Os logs completos, configuração efetiva, histórico e checkpoints estão em `logs/experiments/conv_bilstm_attention_mixed25_*_seed42/`, `experiments/conv_bilstm_attention_mixed25_*_seed42/` e `Modelos salvos/conv_bilstm_attention_mixed25_*/`.

## Resultados pareados

A fusão experimental aplicou a regra antiga **sem ajuste**: top 5 proposto por `pose_face`, `0,22 × pose_face + 0,78 × full`. As temperaturas dos novos modelos ficaram em 1,0; as temperaturas da dupla Transformer pertencem aos checkpoints antigos e não foram transferidas.

| Split e modelo | Pose + face | Completo com mãos | Fusão top 5 |
| --- | ---: | ---: | ---: |
| Validação (85), Transformer anterior | 37/85 (43,5%) | 46/85 (54,1%) | **50/85 (58,8%)** |
| Validação (85), ConvBiLSTM novo | 33/85 (38,8%) | **49/85 (57,6%)** | 47/85 (55,3%) |
| Teste exploratório (85), Transformer anterior | 27/85 (31,8%) | 48/85 (56,5%) | 56/85 (65,9%) |
| Teste exploratório (85), ConvBiLSTM novo | 49/85 (57,6%) | 63/85 (74,1%) | **65/85 (76,5%)** |

F1 macro da fusão: **0,452 → 0,385** na validação e **0,480 → 0,682** no teste exploratório (Transformer → ConvBiLSTM). O `full` novo, isolado, teve F1 macro 0,462 na validação e 0,649 no teste; nas dez classes gravadas pelo grupo, acertou 5/10 e 7/10. A fusão nova acertou 3/10 e 7/10 nessas classes. Em comparação pareada com a fusão Transformer, a nova corrigiu 14 vídeos e perdeu 17 na validação; no teste corrigiu 15 e perdeu seis. O primeiro estágio incluiu a classe correta no top 5 em 72/85 vídeos de validação e 82/85 de teste. A [comparação por classe](../reports/fusion_mixed25_conv_bilstm_attention/comparison.json) mostra deslocamentos grandes em classes com apenas cinco vídeos por split.

**Interpretação:** a arquitetura mostrou potencial, especialmente no modelo completo e no teste exploratório. Porém, a fusão atual ainda vence na validação por três vídeos e por F1 macro. O teste já tinha sido consultado antes no projeto, e o resultado de 65/85 não é uma confirmação independente nem justificativa para trocar automaticamente o backend. A diferença entre validação e teste pode refletir diferenças entre pessoas/sessões ou a variância de conjuntos pequenos. A regra de fusão e as temperaturas não foram calibradas para a nova dupla; reajustá-las agora depois de ver o teste criaria mais risco de seleção otimista.

## Latência e integração

Na mesma janela de landmarks e CPU, 30 repetições com três aquecimentos, o tempo p50 de pré-processamento + duas inferências + fusão foi **6,12 ms** na dupla Transformer e **6,84 ms** na dupla ConvBiLSTM. O novo `full` isolado teve 3,49 ms p50 contra 3,01 ms do Transformer `full`. Os arquivos `reports/fusion_mixed25_conv_bilstm_attention/latency_*_cpu.json` guardam p50, p95 e média. O ensaio não inclui JPEG, MediaPipe, webcam ou transporte; não mede resposta de ponta a ponta.

O manifesto experimental `configs/fusion25_conv_bilstm_attention_fixed_rule.json` carrega os dois checkpoints arquivados no `FusionBundle` existente, com hashes verificados. Um teste de paridade confirmou probabilidades, candidatos e classe final do backend contra o CSV offline de validação, tolerando diferença numérica máxima de 5e-5 entre kernels CPU/GPU. A configuração padrão da aplicação continua em `configs/fusion25_frozen.json` (Transformers).

## Reproduzir no Git Bash do VS Code

Na raiz do repositório, com os landmarks e splits locais:

```bash
./.venv/Scripts/python.exe scripts/train.py --config configs/conv_bilstm_attention_mixed25_pose_face.yaml --data-dir data/landmarks/mixed_25 --processed-dir data/processed/mixed_25 --experiment conv_bilstm_attention_mixed25_pose_face_seed42_repro --num-workers 0 --skip-test
./.venv/Scripts/python.exe scripts/train.py --config configs/conv_bilstm_attention_mixed25_full.yaml --data-dir data/landmarks/mixed_25 --processed-dir data/processed/mixed_25 --experiment conv_bilstm_attention_mixed25_full_seed42_repro --num-workers 0 --skip-test
```

Para avaliar os checkpoints **já arquivados**, repita o comando abaixo para `val` e `test`, alterando `--split` e usando um `--output-dir` novo se quiser preservar os resultados originais:

```bash
./.venv/Scripts/python.exe scripts/evaluate_two_stage_fusion.py \
  --pose-checkpoint 'Modelos salvos/conv_bilstm_attention_mixed25_pose_face/best.pt' \
  --full-checkpoint 'Modelos salvos/conv_bilstm_attention_mixed25_full/best.pt' \
  --pose-split-dir data/processed/mixed_25 --full-split-dir data/processed/mixed_25 \
  --data-dir data/landmarks/mixed_25 --split val --first pose_face \
  --top-k 5 --pose-weight 0.22 --pose-temperature 1.0 --full-temperature 1.0 \
  --output-dir experiments/fusion_mixed25_conv_bilstm_attention_repro
```

As configurações, mapas e logs do treino original estão nos dois diretórios em `Modelos salvos/`. `scripts/archive_conv_bilstm_attention_mixed25.py` recria esses diretórios e o manifesto a partir dos experimentos com nomes originais. `scripts/compare_conv_bilstm_attention_mixed25.py` gera a comparação por vídeo/classe. Para verificar o código: `./.venv/Scripts/python.exe -m pytest tests/test_conv_bilstm_attention.py tests/test_conv_bilstm_attention_runtime.py tests/test_fusion25_protocol.py -q`.

## Próximo teste decisivo

Gravar vídeos de **novas pessoas e sessões**, incluindo sinais e movimentos fora do vocabulário, e congelar a avaliação antes de olhar seus resultados. Comparar ambas as duplas e o novo `full` isolado em acurácia, F1 macro, recusas indevidas, falsos positivos por minuto e latência ao vivo. Só então decidir se alguma variante substitui o padrão.
