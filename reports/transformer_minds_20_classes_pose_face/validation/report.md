# Transformer MINDS-Libras — 20 classes — Validação

Fonte: C:\Users\migue\DeepLearning-SINALIZA\Modelos salvos\transformer_minds_20_classes_pose_face\records\val_metrics.json

Total: **100 videos**, **37 acertos** (37.0%).
Linhas da matriz = classe real; colunas = classe prevista.

## Erros por classe

| Classe real | Acertos | Erros | Acuracia |
|---|---:|---:|---:|
| Acontecer | 0/5 | 5 | 0% |
| Amarelo | 0/5 | 5 | 0% |
| Aproveitar | 0/5 | 5 | 0% |
| Bala | 0/5 | 5 | 0% |
| Banheiro | 0/5 | 5 | 0% |
| Conhecer | 0/5 | 5 | 0% |
| Espelho | 0/5 | 5 | 0% |
| Esquina | 0/5 | 5 | 0% |
| Filho | 0/5 | 5 | 0% |
| Medo | 0/5 | 5 | 0% |
| Sapo | 0/5 | 5 | 0% |
| Aluno | 1/5 | 4 | 20% |
| Cinco | 2/5 | 3 | 40% |
| Barulho | 4/5 | 1 | 80% |
| America | 5/5 | 0 | 100% |
| Banco | 5/5 | 0 | 100% |
| Maca | 5/5 | 0 | 100% |
| Ruim | 5/5 | 0 | 100% |
| Vacina | 5/5 | 0 | 100% |
| Vontade | 5/5 | 0 | 100% |

## Pares de confusao mais frequentes

| Real | Previsto | Videos |
|---|---|---:|
| Acontecer | America | 5 |
| Amarelo | Maca | 5 |
| Banheiro | America | 5 |
| Espelho | Barulho | 5 |
| Esquina | Sapo | 5 |
| Filho | Medo | 5 |
| Medo | Filho | 5 |
| Sapo | America | 5 |
| Aluno | Vacina | 4 |
| Bala | Cinco | 4 |
| Cinco | Espelho | 3 |
| Conhecer | Vontade | 3 |
| Aproveitar | Bala | 2 |
| Conhecer | Filho | 2 |
| Aproveitar | Conhecer | 1 |

Cada classe tem cinco vídeos neste split; uma diferença de um acerto equivale a 20 pontos percentuais. As métricas descrevem este split e este checkpoint.
