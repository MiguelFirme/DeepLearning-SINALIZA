# Plano: Transformer 1-D com landmarks das mãos no MINDS-Libras

Data: 2026-10-05.

## Pergunta

Adicionar os 42 pontos detalhados das mãos (21 por mão) ao Transformer `pose_face` melhora a classificação dos cinco sinais?

## Protocolo

1. Usar os mesmos 200 vídeos e o mesmo split por pessoa: 150 treino, 25 validação (Sinalizador11) e 25 teste (Sinalizador12).
2. Alterar somente `data.feature_mode` de `pose_face` para `full`. Preservar arquitetura do Transformer, seed 42, 48 frames, augmentation apenas no treino, otimizador, loss, early stopping e seleção do checkpoint pela menor `val_loss`.
3. Registrar histórico, configuração resolvida, checkpoint e log persistente em um experimento novo. Comparar a validação com o Transformer `pose_face` (19/25, F1 macro 0,726).
4. Como o usuário pediu treinamento **e teste**, avaliar uma vez o melhor checkpoint nos 25 vídeos do Sinalizador12 e salvar matriz de confusão e previsões por vídeo. Este teste já foi consultado antes; tratá-lo como exploratório e não ajustar hiperparâmetros a partir dele.
5. Conferir erros por classe, especialmente `Vontade`, que teve 0/5 no teste `pose_face`. Uma pessoa e cinco vídeos por classe não bastam para afirmar desempenho em produção.

## Comando de treino no Git Bash

```bash
./.venv/Scripts/python.exe scripts/train.py \
  --config configs/transformer_minds_full.yaml \
  --data-dir data/landmarks/minds_libras_school5 \
  --processed-dir data/processed/minds_libras_school5 \
  --experiment transformer_minds_school5_full_seed42 \
  --num-workers 0 --skip-test
```

Resultados previstos: `experiments/transformer_minds_school5_full_seed42/` e `logs/experiments/transformer_minds_school5_full_seed42/`. O teste será executado após a seleção pela validação com `scripts/audit_checkpoint_val.py --split test`.

## Execução e resultados (2026-10-05)

O treino usou CUDA, 120.823 parâmetros e os mesmos 150 vídeos de treino. O early stopping encerrou na época 16; a menor `val_loss` foi na época 1. O checkpoint selecionado foi avaliado nos 25 vídeos de validação e nos 25 de teste, sem augmentation.

| Entrada do Transformer | Validação top-1 | Validação F1 macro | Teste top-1 | Teste F1 macro |
|---|---:|---:|---:|---:|
| `pose_face` (mãos detalhadas zeradas) | 19/25 (76%) | 0,726 | 20/25 (80%) | 0,733 |
| `full` (mãos detalhadas incluídas) | 5/25 (20%) | 0,067 | 5/25 (20%) | 0,067 |

O Transformer `full` classificou **todos** os vídeos de validação e teste como `Banheiro`; acertou somente os cinco dessa classe em cada split. A perda de validação do melhor checkpoint foi 1,611, próxima da referência aleatória `ln(5)=1,609`. Durante o treino, o top-1 chegou a 68% na época 14, enquanto a validação permaneceu em 20% e a perda piorou. Isso é compatível com memorização ou com uma diferença na distribuição/qualidade dos landmarks das mãos entre pessoas, mas o piloto não isola qual mecanismo causou a queda.

Arquivos: `experiments/transformer_minds_school5_full_seed42/{history.csv,best.pt,val_metrics.json,val_predictions.csv,test_metrics.json,test_predictions.csv}`; registro de configuração e ambiente em `logs/experiments/transformer_minds_school5_full_seed42/run_record.json`, com `train.log` na mesma pasta. Os arquivos `.npz`, os splits e os checkpoints anteriores não foram modificados.

Comandos de reprodução das avaliações no Git Bash:

```bash
./.venv/Scripts/python.exe scripts/audit_checkpoint_val.py \
  --checkpoint experiments/transformer_minds_school5_full_seed42/best.pt \
  --data-dir data/landmarks/minds_libras_school5 \
  --processed-dir data/processed/minds_libras_school5 --split val

./.venv/Scripts/python.exe scripts/audit_checkpoint_val.py \
  --checkpoint experiments/transformer_minds_school5_full_seed42/best.pt \
  --data-dir data/landmarks/minds_libras_school5 \
  --processed-dir data/processed/minds_libras_school5 --split test
```

**Decisão:** manter `pose_face` como a variante superior neste piloto. A próxima investigação útil é auditar, por pessoa e por classe, a presença e qualidade das mãos e o alinhamento dos canais antes de propor outra configuração; não selecionar hiperparâmetros pelo teste Sinalizador12, que já foi consultado.
