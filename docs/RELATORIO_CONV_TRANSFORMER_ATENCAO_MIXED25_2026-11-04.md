# Experimento Conv1D + Transformer + atenção temporal

Data: 04/11/2026. [Plano registrado antes do treino](PLANO_CONV_TRANSFORMER_ATENCAO_MIXED25_2026-11-04.md).

## O que foi feito

Substituímos a BiLSTM do experimento Conv1D + BiLSTM + atenção por duas camadas Transformer. A rede `conv_transformer_attention` usa LayerNorm na entrada de 697 features, projeção para 64 dimensões, Conv1D temporal com 64 canais/kernel 3, BatchNorm, GELU, dropout, codificação posicional aprendida, duas camadas Transformer (quatro cabeças, feedforward 128), atenção temporal mascarada e classificador. Cada rede tem **136.907 parâmetros**. A máscara de quadros inválidos é respeitada na convolução, no encoder e no resumo temporal.

Treinamos dois checkpoints do zero: `pose_face` e `full` (pose, face e mãos). Mantivemos os 670 vídeos, 25 classes, 40 pontos faciais, splits por pessoa 500/85/85, 48 quadros, normalização e augmentation dinâmica apenas no treino. Os hashes de manifesto, splits e rótulos coincidem com o protocolo original. Configuração: seed 42, Adam, `lr=3e-4`, `weight_decay=1e-4`, pesos de classe `sqrt_inverse`, dropout 0,45, scheduler por `val_loss` e early stopping 15. Ambos os treinos usaram `--skip-test`; o checkpoint foi escolhido exclusivamente pela menor `val_loss`.

| Rede | Melhor época | Parada antecipada | Tempo GPU |
| --- | ---: | ---: | ---: |
| Pose + face | 16 | 31 | 119 s |
| Completa | 34 | 49 | 179 s |

## Comparação com os mesmos vídeos

Aplicamos a regra de fusão definida antes do treino: top 5 do primeiro modelo `pose_face`; escore `0,22 × pose_face + 0,78 × full`. As temperaturas novas ficaram em 1,0. A dupla Transformer original conserva suas próprias temperaturas calibradas na validação. Nenhuma regra foi reajustada a partir do teste.

| Split / arquitetura | Pose + face | Completa | Fusão top 5 | F1 macro da fusão |
| --- | ---: | ---: | ---: | ---: |
| Validação: Transformer original | 37/85 | 46/85 | **50/85 (58,8%)** | **0,452** |
| Validação: Conv1D + BiLSTM + atenção | 33/85 | 49/85 | 47/85 (55,3%) | 0,385 |
| Validação: Conv1D + Transformer + atenção | **42/85** | 37/85 | 40/85 (47,1%) | 0,318 |
| Teste exploratório: Transformer original | 27/85 | 48/85 | 56/85 (65,9%) | 0,480 |
| Teste exploratório: Conv1D + BiLSTM + atenção | **49/85** | **63/85** | **65/85 (76,5%)** | **0,682** |
| Teste exploratório: Conv1D + Transformer + atenção | 42/85 | 46/85 | 46/85 (54,1%) | 0,512 |

Na validação, o novo `pose_face` isolado melhorou, mas a rede `full` caiu e a fusão teve dez acertos a menos que a dupla original. O primeiro estágio incluiu a resposta correta no top 5 em 79/85 vídeos de validação e 71/85 do teste. A fusão nova acertou 2/10 vídeos de classes gravadas pelo grupo na validação e 7/10 no teste. Contra a fusão ConvBiLSTM, ela corrigiu 13 vídeos e perdeu 20 na validação; no teste, corrigiu dois e perdeu 21. [Comparação pareada e por classe](../reports/fusion_mixed25_conv_transformer_attention/comparison.json).

Os 85 vídeos de teste já tinham sido consultados em experimentos anteriores; os números de teste são **exploratórios**. A diferença entre validação e teste também indica que uma única divisão pequena é instável. A nova arquitetura **não superou** o padrão na validação e não foi ativada no backend.

## Latência e reprodução

Em 30 repetições na mesma janela de landmarks e CPU, a latência p50 de pré-processamento, duas inferências e fusão foi 6,12 ms no Transformer original, 6,84 ms no ConvBiLSTM e **6,96 ms** no ConvTransformer. O benchmark não inclui câmera, JPEG, MediaPipe nem comunicação. [Medição nova](../reports/fusion_mixed25_conv_transformer_attention/latency_conv_transformer_cpu.json).

Os pesos e hashes estão em `Modelos salvos/conv_transformer_attention_mixed25_pose_face/` e `Modelos salvos/conv_transformer_attention_mixed25_full/`; os logs em `logs/experiments/conv_transformer_attention_mixed25_*_seed42/`. O [manifesto experimental](../configs/fusion25_conv_transformer_attention_fixed_rule.json) permite carregar a dupla no `FusionBundle`. O backend continua usando `configs/fusion25_frozen.json` por padrão. Um teste de integração confirmou que os pesos arquivados reproduzem a probabilidade, os candidatos e a decisão registrados no CSV offline.

Na raiz do repositório, no Git Bash do VS Code, com os `.npz` e splits locais:

```bash
./.venv/Scripts/python.exe scripts/train.py --config configs/conv_transformer_attention_mixed25_pose_face.yaml --data-dir data/landmarks/mixed_25 --processed-dir data/processed/mixed_25 --experiment conv_transformer_attention_mixed25_pose_face_seed42_repro --num-workers 0 --skip-test
./.venv/Scripts/python.exe scripts/train.py --config configs/conv_transformer_attention_mixed25_full.yaml --data-dir data/landmarks/mixed_25 --processed-dir data/processed/mixed_25 --experiment conv_transformer_attention_mixed25_full_seed42_repro --num-workers 0 --skip-test
```

Para avaliar os checkpoints **já arquivados**, execute para `val` e depois `test`, mudando `--split` e preservando um diretório de saída distinto caso queira guardar os resultados originais:

```bash
./.venv/Scripts/python.exe scripts/evaluate_two_stage_fusion.py \
  --pose-checkpoint 'Modelos salvos/conv_transformer_attention_mixed25_pose_face/best.pt' \
  --full-checkpoint 'Modelos salvos/conv_transformer_attention_mixed25_full/best.pt' \
  --pose-split-dir data/processed/mixed_25 --full-split-dir data/processed/mixed_25 \
  --data-dir data/landmarks/mixed_25 --split val --first pose_face \
  --top-k 5 --pose-weight 0.22 --pose-temperature 1.0 --full-temperature 1.0 \
  --output-dir experiments/fusion_mixed25_conv_transformer_attention_repro
```

`scripts/compare_conv_transformer_attention_mixed25.py` recria o relatório pareado. `scripts/archive_conv_transformer_attention_mixed25.py` recria os dois arquivos de modelo e o manifesto a partir dos experimentos originais. Uma decisão de implantação exige vídeos de pessoas e sessões novas, sinais fora do vocabulário e medição de latência da aplicação completa.
