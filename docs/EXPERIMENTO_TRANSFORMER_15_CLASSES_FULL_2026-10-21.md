# Experimento — Transformer de 15 classes com pose, face e mãos

## Pergunta

Incluir os pontos detalhados das mãos melhora o checkpoint `pose_face` de 15 classes do MINDS-Libras?

## Plano anterior ao treino

1. Reutilizar os mesmos arquivos `.npz` de `data/landmarks/minds_libras_all20/`, as mesmas 15 classes e o mesmo split de `data/processed/minds_libras_15_sem_cinco/`: 450 vídeos de treino, 75 de validação e 75 de teste, separados por sinalizador. Não reextrair nem sintetizar dados.
2. Usar `configs/transformer_minds_full.yaml`. Confirmar que sua configuração efetiva é idêntica à `configs/transformer_minds_pose_face.yaml`, exceto `data.feature_mode: full`. A entrada terá mãos, pose e face; arquitetura, augmentation, otimizador, seed 42, 48 frames e seleção por menor `val_loss` permanecem iguais.
3. Treinar do zero em `experiments/transformer_minds_15_full_seed42/`, sem sobrescrever o checkpoint de 15 classes `pose_face`. Auditar o `best.pt` na validação e fazer uma avaliação exploratória única no teste após fixá-lo. O sinalizador de teste já foi usado em experimentos anteriores; não orientar escolhas por ele.
4. Comparar os dois checkpoints nos **mesmos 75 vídeos** de validação, inclusive erros por classe. Salvar checkpoint, configurações, split, logs, métricas, previsões e README em `Modelos salvos/transformer_minds_15_classes_full/`. Conferir carregamento estrito e checksums.

## Risco conhecido

No subconjunto de oito classes, a auditoria encontrou mão detectada em pelo menos um lado em 48,77% dos quadros de treino e 99,75% da validação. O Transformer completo de oito classes piorou em relação a `pose_face`. Essa auditoria inclui repouso, não cobre todas as 15 classes e não estabelece causa. O experimento atual verificará empiricamente se a inclusão direta das mãos ajuda neste vocabulário.

## Execução e resultados — 21/10/2026

- Configurações efetivas verificadas: apenas `data.feature_mode` difere entre o controle e este experimento (`pose_face` → `full`). Split idêntico 450/75/75; treino em CUDA com 121.473 parâmetros.
- Treino em `experiments/transformer_minds_15_full_seed42/`: melhor época **30** por `val_loss=1,6027`; early stopping na época 45. Na época final, `train_top1=81,3%`, `val_top1=53,3%`.
- Auditoria do `best.pt`: validação **45/75 = 60%**, F1 macro **0,550**; controle `pose_face` **49/75 = 65,3%**, F1 macro **0,581**. Teste exploratório **34/75 = 45,3%**, F1 macro **0,358**; controle **41/75 = 54,7%**, F1 macro **0,480**. O teste não foi usado para escolher o checkpoint.
- Na validação pareada: ambos corretos 29 vídeos; somente `full` correto 16; somente `pose_face` correto 20; ambos errados 10. `full` ganhou em `Acontecer` (5/5 contra 1/5) e `Conhecer` (5/5 contra 0/5), mas perdeu em `Aluno`, `Cinco`, `Espelho` e `Medo` (todos 0/5 no `full`).
- Arquivo em `Modelos salvos/transformer_minds_15_classes_full/`: checkpoint, cadeia YAML, splits, logs, histórico, métricas, previsões e comandos de reprodução.
- Verificação final: o checkpoint arquivado tem SHA-256 `7877a600af9ee6c00b18558d88416adf1b905091ed6e0e346ee4d30e757dccd6`, igual ao arquivo original, e carrega no `PredictionService` com pesos estritos, 15 classes, 48 frames e `feature_mode=full`. Configuração YAML arquivada igual à original; 23 checksums válidos no pacote; 11 testes de modalidades, normalização e backend passaram.

## Interpretação e decisão

Adicionar os pontos das mãos diretamente ao mesmo Transformer piorou a acurácia de validação e a de teste exploratório. As previsões complementares sugerem informação útil em algumas classes, mas não demonstram que uma fusão seja segura ou melhor para pessoas novas. A possível influência da qualidade e cobertura das mãos precisa de uma auditoria no intervalo ativo do gesto. **Manter o modelo `pose_face` como checkpoint ativo no backend.**
