# Resultado do retreino v2 com o grupo ampliado

Verificado em 17/11/2026 nos arquivos locais de `transformer_v2_grupo_expandido_20261117_seed42`: `run_record.json`, `history.json` e `val_metrics.json`. Status: `completed`.

- Vocabulário: 41 classes; 715 vídeos, separados em 529 treino, 93 validação e 93 teste por identidade/origem.
- Melhor checkpoint: época **80**, escolhido pelo F1 macro da validação.
- Early stopping: época 105. Duração registrada: 465,51 s (aproximadamente 7 min 46 s).
- Acertos top 1 na validação: **70/93 — 75,27%**.
- Top 3: 85/93 — 91,40%; top 5: 88/93 — 94,62%.
- F1 macro da validação: **0,7512**.
- MINDS na validação: **46/65 — 70,77%**; F1 macro 0,7094.
- Grupo na validação: **24/28 — 85,71%**; F1 macro 0,8333. Há somente um vídeo por classe do grupo nesse split.
- Treino na época selecionada: `train_top1=0,981096` (98,11%), sob sampler balanceado e augmentation, não diretamente comparável à distribuição limpa da validação.
- Metadados: nenhum arquivo precisou de fallback de FPS ou dimensões nos três splits.

O treino foi feito com `--skip-test`; **não há resultado dos 93 vídeos de teste**. Esta consulta não executou teste nem simulação da webcam. A validação anterior teve 80% em 23 classes; a nova tem 75,27% em 41 classes e 93 vídeos, portanto as porcentagens não são comparação direta da mesma tarefa. A diferença treino/validação continua relevante.

Checkpoint selecionado: `experiments/transformer_v2_grupo_expandido_20261117_seed42/best.pt`. Log: `logs/experiments/transformer_v2_grupo_expandido_20261117_seed42/train.log`. Os modelos antigos foram preservados; esta análise não alterou o checkpoint padrão do backend.
