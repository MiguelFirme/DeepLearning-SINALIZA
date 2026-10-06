# Plano: Transformer 1-D no piloto MINDS-Libras

Data: 2026-10-05.

## Pergunta

Uma arquitetura temporal diferente do BiLSTM consegue aproveitar melhor os landmarks no piloto de cinco classes, sem consultar novamente o sinalizador de teste?

## Protocolo

1. Reutilizar exatamente `data/landmarks/minds_libras_school5` e `data/processed/minds_libras_school5`: 150 vídeos de seis sinalizadores no treino, 25 vídeos do Sinalizador11 na validação e 25 do Sinalizador12 reservados.
2. Reutilizar `feature_mode=pose_face`: 42 pontos detalhados das mãos zerados, pose e face preservados. O experimento isola arquitetura temporal, não qualidade da detecção de mãos.
3. Usar Transformer 1-D pequeno (`d_model=64`, duas camadas, quatro cabeças, feedforward 128, dropout 0,45) para evitar comparar um modelo muito maior com o BiLSTM regularizado de 130.039 parâmetros. Reutilizar a mesma configuração de treino, augmentation dinâmica apenas em treino, early stopping por `val_loss` e seed 42.
4. Executar `scripts/train.py` com `--skip-test`. Registrar número de parâmetros, melhor época, `val_loss`, top-1, F1 macro e distribuição de previsões. Comparar com o BiLSTM `pose_face` (44% top-1, F1 macro 0,350 no checkpoint de menor perda) e com o I3D congelado (40%, F1 macro 0,327). Diferenças de um vídeo equivalem a quatro pontos percentuais nesta validação pequena.
5. Não inferir superioridade geral a partir de um único sinalizador. Se houver melhora, repetir avaliação por mais sinalizadores antes de qualquer afirmação de generalização.

## Arquivos

- Modelo existente: `ml/models/transformer.py`.
- Configuração nova: `configs/transformer_minds_pose_face.yaml`.
- Resultados previstos: `experiments/transformer_minds_school5_pose_face_seed42/` e `logs/experiments/transformer_minds_school5_pose_face_seed42/`.

Comando no Git Bash:

```bash
./.venv/Scripts/python.exe scripts/train.py \
  --config configs/transformer_minds_pose_face.yaml \
  --data-dir data/landmarks/minds_libras_school5 \
  --processed-dir data/processed/minds_libras_school5 \
  --experiment transformer_minds_school5_pose_face_seed42 \
  --num-workers 0 --skip-test
```

## Execução e resultado em 2026-10-05

O Transformer teve **120.823 parâmetros**, contra 130.039 no BiLSTM regularizado. Treinou por 80 épocas até o early stopping; o checkpoint escolhido pela menor `val_loss` foi o da época 65. A validação foi recalculada diretamente de `best.pt` com `scripts/audit_checkpoint_val.py`, sem consultar o teste:

| Modelo | Val top-1 | Val F1 macro | Val loss | Acertos |
|---|---:|---:|---:|---:|
| BiLSTM `pose_face` | 44% | 0,350 | 1,043 | 11/25 |
| I3D RGB/WLASL2000 congelado | 40% | 0,327 | 3,081 | 10/25 |
| Transformer 1-D `pose_face` | **76%** | **0,726** | **0,483** | **19/25** |

Acertos do Transformer por classe: `Aluno` 3/5, `Banheiro` 5/5, `Conhecer` 5/5, `Medo` 5/5, `Vontade` 1/5. Quatro `Vontade` foram classificados como `Conhecer`; dois `Aluno` como `Medo`. Matriz e previsões por vídeo: `experiments/transformer_minds_school5_pose_face_seed42/val_metrics.json` e `val_predictions.csv`. Histórico e checkpoint na mesma pasta; log persistente em `logs/experiments/transformer_minds_school5_pose_face_seed42/`.

Durante o treino, o top-1 de validação oscilou muito, inclusive com 100% na época 13, mas o checkpoint **não foi escolhido pelo pico de acurácia**; foi escolhido pela menor perda. Isso e o fato de a validação conter somente uma pessoa impedem concluir que 76% se repetirá em novos usuários. O sinalizador 12 já havia sido consultado em um experimento anterior e não foi usado aqui. Próxima verificação com valor científico: repetir por diferentes sinalizadores reservados e sementes, sem usar o sinalizador 12 para selecionar arquitetura ou hiperparâmetros. Não substituir o modelo em produção com base somente nesse piloto.

Comando para auditar a validação do checkpoint no Git Bash:

```bash
./.venv/Scripts/python.exe scripts/audit_checkpoint_val.py \
  --checkpoint experiments/transformer_minds_school5_pose_face_seed42/best.pt \
  --data-dir data/landmarks/minds_libras_school5 \
  --processed-dir data/processed/minds_libras_school5
```

## Comparação no mesmo sinalizador de teste solicitada pelo usuário

O usuário pediu explicitamente avaliar o Transformer pelo mesmo protocolo aplicado ao BiLSTM. O treino dos dois modelos `pose_face` já usou os mesmos 150 vídeos de treino e 25 de validação; não é necessário treiná-los novamente. Comparar os checkpoints `best.pt` escolhidos por menor `val_loss`, uma única vez nos mesmos 25 vídeos do Sinalizador12, registrando acurácia, F1, matriz de confusão e previsões por vídeo. O Sinalizador12 já havia sido consultado no baseline completo em rodada anterior; por isso esta comparação é **exploratória**, não uma estimativa final intocada para escolher novas arquiteturas. Não ajustar hiperparâmetros com base no resultado.

### Resultado do teste (2026-10-05)

Os dois checkpoints foram avaliados nos mesmos 25 vídeos limpos do Sinalizador12, cinco por classe, sem augmentation. Cada checkpoint foi selecionado pela menor perda de validação antes deste teste.

| Modelo `pose_face` | Melhor época | Validação top-1 | Teste top-1 | Teste F1 macro | Teste loss |
|---|---:|---:|---:|---:|---:|
| BiLSTM | 15 | 11/25 (44%) | 15/25 (60%) | 0,478 | 0,985 |
| Transformer 1-D | 65 | 19/25 (76%) | **20/25 (80%)** | **0,733** | **0,362** |

No teste, o Transformer acertou todos os cinco vídeos de `Aluno`, `Banheiro`, `Conhecer` e `Medo`, mas classificou os cinco vídeos de `Vontade` como `Conhecer`. O BiLSTM acertou `Aluno`, `Banheiro` e `Conhecer` (5/5 cada) e errou todos os vídeos de `Medo` e `Vontade`. Portanto, a melhora de cinco acertos veio inteiramente de `Medo`; **nenhum dos dois modelos reconheceu `Vontade` nessa pessoa**. A matriz de confusão e cada previsão estão em `test_metrics.json` e `test_predictions.csv` nas respectivas pastas de experimento.

Comandos de reprodução no Git Bash:

```bash
./.venv/Scripts/python.exe scripts/audit_checkpoint_val.py \
  --checkpoint experiments/transformer_minds_school5_pose_face_seed42/best.pt \
  --data-dir data/landmarks/minds_libras_school5 \
  --processed-dir data/processed/minds_libras_school5 --split test

./.venv/Scripts/python.exe scripts/audit_checkpoint_val.py \
  --checkpoint experiments/bilstm_minds_school5_pose_face_seed42/best.pt \
  --data-dir data/landmarks/minds_libras_school5 \
  --processed-dir data/processed/minds_libras_school5 --split test
```

A diferença de 20 pontos percentuais corresponde a apenas cinco vídeos de uma pessoa. Além disso, o teste Sinalizador12 já tinha sido consultado no baseline completo. A conclusão sustentada é que o Transformer foi melhor **neste piloto**, com estes cinco sinais e este sinalizador. Para medir generalização de forma confiável, repetir a avaliação por sinalizador com pessoas ainda não usadas para escolhas de arquitetura e conferir a qualidade dos vídeos de `Vontade`. Não usar este resultado para ajustar o modelo no Sinalizador12.
