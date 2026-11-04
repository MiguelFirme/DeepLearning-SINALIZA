# Experimento: 16 versus 40 pontos faciais em 25 classes

Data: 04/11/2026. Plano e seleção de pontos: [PLANO_ABLACAO_FACE16_MIXED25_2026-11-04.md](PLANO_ABLACAO_FACE16_MIXED25_2026-11-04.md).

## O que foi treinado

Dois Transformers novos, inicializados do zero: `pose_face16` e `full_face16`. Cada um usa os mesmos 25 rótulos, 670 vídeos, split de 500/85/85, 48 quadros por vídeo, seed 42, normalização, augmentation, arquitetura de 122.123 parâmetros, loss ponderada, Adam, scheduler e early stopping dos dois baselines de 40 pontos. Os arquivos `.npz` originais não foram modificados nem reextraídos. A fusão usa a regra antiga sem reajuste: top 5 do `pose_face`, `0,22 × pose_face + 0,78 × full`, temperaturas `1,028734578` e `0,792446596`. Dessa forma, a variável principal do teste é a informação facial disponível aos modelos.

Os 16 índices MediaPipe escolhidos foram `46, 65, 276, 295` (sobrancelhas), `33, 133, 160, 144, 362, 263, 387, 373` (olhos) e `13, 14, 78, 308` (boca). Os outros 24 pontos faciais são zerados antes e depois da normalização. A segunda máscara impede que a centralização pelos ombros recrie valores nos campos removidos. No treino, a seleção é reaplicada depois da augmentation.

Os hashes de manifesto, splits e mapa de classes nos registros de treino coincidem com os antigos; o ambiente também coincide: Python 3.11.9, PyTorch 2.14.0+cu130, NumPy 1.26.4, CUDA 13.0 e RTX 4050 Laptop GPU. O `pose_face16` escolheu a época 67 por menor perda de validação e parou na 82. O `full_face16` escolheu a época 63 e parou na 78. O teste não participou da escolha dos checkpoints.

## Resultado na validação (85 vídeos)

| Arquitetura | 40 pontos | 16 pontos | Diferença de acertos | F1 macro: 40 → 16 |
| --- | ---: | ---: | ---: | ---: |
| Pose + face | 37/85 (43,5%) | 41/85 (48,2%) | +4 | 0,267 → 0,289 |
| Pose + face + mãos | 46/85 (54,1%) | 46/85 (54,1%) | 0 | 0,422 → 0,338 |
| Fusão dos dois | **50/85 (58,8%)** | **45/85 (52,9%)** | **−5** | **0,452 → 0,382** |

Nas dez classes novas, a fusão foi de 4/10 para 3/10. Na fusão, as maiores diferenças por classe foram `Barulho −5`, `Medo −5`, `Vontade +5` e `Filho +2`. Cada classe antiga tem apenas cinco vídeos de validação, então esses deslocamentos precisam ser interpretados com cautela.

## Teste exploratório (85 vídeos já consultados antes no projeto)

| Arquitetura | 40 pontos | 16 pontos | Diferença de acertos | F1 macro: 40 → 16 |
| --- | ---: | ---: | ---: | ---: |
| Pose + face | 27/85 (31,8%) | 37/85 (43,5%) | +10 | 0,188 → 0,321 |
| Pose + face + mãos | 48/85 (56,5%) | 40/85 (47,1%) | −8 | 0,438 → 0,366 |
| Fusão dos dois | **56/85 (65,9%)** | **42/85 (49,4%)** | **−14** | **0,480 → 0,359** |

Esse conjunto de teste já havia sido consultado em experimentos anteriores. É uma comparação pareada útil para análise, mas **não constitui confirmação independente** em novas pessoas. Nas dez classes novas, a fusão permaneceu em 4/10. As maiores perdas foram `Barulho −5`, `Conhecer −4` e `Aluno −4`; `Vacina` ganhou três acertos.

## Conclusão e limites

Reduzir a face de 40 para 16 pontos **ajudou o modelo pose + face isolado**, mas **prejudicou a fusão com o modelo completo** sob a mesma regra fixa. O modelo completo manteve a quantidade de acertos na validação, porém perdeu F1 macro e oito acertos no teste exploratório. Nesta execução, a fusão de 40 pontos continua sendo a escolha com melhor resultado para as 25 classes. Não há evidência para substituir os checkpoints ativos pelos novos.

Este foi um único treino por variante e seed. A regra e as temperaturas vieram da versão de 40 pontos para isolar a ablação; elas podem não ser ideais para a dupla de 16 pontos, e reajustá-las agora sobre o teste já conhecido causaria vazamento. Reduzir somente os pontos selecionados após o MediaPipe tampouco acelera substancialmente a extração da face: ele continua processando a imagem, e os dois Transformers mantêm 697 entradas e 122.123 parâmetros.

## Artefatos e reprodução

- Checkpoints e documentação: `Modelos salvos/transformer_mixed25_pose_face16/` e `Modelos salvos/transformer_mixed25_full_face16/`.
- Configurações: `configs/transformer_mixed25_pose_face16.yaml`, `configs/transformer_mixed25_full_face16.yaml`.
- Regra experimental fixa e hashes: `configs/fusion25_face16_fixed_rule.json`.
- Logs de treino: `logs/experiments/transformer_mixed25_pose_face16_seed42/` e `logs/experiments/transformer_mixed25_full_face16_seed42/`.
- Métricas e CSV por vídeo: `experiments/fusion_mixed25_face16/`.
- Comparação por classe: `reports/fusion_mixed25_face16/comparison.json`.

No Git Bash do VS Code, a partir da raiz, com Python 3.11 e dependências instaladas:

```bash
./.venv/Scripts/python.exe scripts/train.py --config configs/transformer_mixed25_pose_face16.yaml --data-dir data/landmarks/mixed_25 --processed-dir data/processed/mixed_25 --experiment transformer_mixed25_pose_face16_seed42_repro --num-workers 0 --skip-test
./.venv/Scripts/python.exe scripts/train.py --config configs/transformer_mixed25_full_face16.yaml --data-dir data/landmarks/mixed_25 --processed-dir data/processed/mixed_25 --experiment transformer_mixed25_full_face16_seed42_repro --num-workers 0 --skip-test
```

Para avaliar os checkpoints já salvos sem retreinar, execute no Git Bash:

```bash
for split in val test; do
  ./.venv/Scripts/python.exe scripts/evaluate_two_stage_fusion.py \
    --pose-checkpoint 'Modelos salvos/transformer_mixed25_pose_face16/best.pt' \
    --full-checkpoint 'Modelos salvos/transformer_mixed25_full_face16/best.pt' \
    --pose-split-dir data/processed/mixed_25 \
    --full-split-dir data/processed/mixed_25 \
    --data-dir data/landmarks/mixed_25 \
    --split "$split" --first pose_face --top-k 5 --pose-weight 0.22 \
    --pose-feature-mode pose_face16 --full-feature-mode full_face16 \
    --pose-temperature 1.0287345780239197 \
    --full-temperature 0.7924465962305567 \
    --output-dir experiments/fusion_mixed25_face16
done
./.venv/Scripts/python.exe scripts/compare_face16_mixed25.py
```

O arquivamento original foi feito com `./.venv/Scripts/python.exe scripts/archive_face16_mixed25.py`, que lê os experimentos com nomes sem `_repro`. Os comandos de treino acima usam `_repro` para preservar os checkpoints originais durante uma reprodução.

Verificações: 18 testes de seleção de features, fusão antiga e inferência da nova dupla passaram; o runtime `FusionBundle` reproduziu as probabilidades offline de um vídeo de validação. O backend padrão não foi trocado.
