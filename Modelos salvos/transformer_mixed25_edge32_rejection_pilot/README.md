# Transformer de 26 classes — piloto de rejeição por bordas

**Status: experimental. Não integrado ao backend.** Os exemplos `__nenhum_sinal__` vieram de trechos de 32 quadros no início/fim dos vídeos positivos das 25 classes, com rótulos fracos. A IA inspecionou prévias de quatro quadros por trecho; os clipes completos não passaram por revisão humana.

- Checkpoint: `best.pt`, SHA-256 `74b0545eddefebf92a26bcc239373f15c29fc42108bbdcc8ded663d3faa16394`.
- Arquitetura: Transformer `full`, 697 atributos por quadro, 26 saídas, 122.188 parâmetros; 48 quadros de entrada após remostrar os 32 quadros dos negativos.
- Dados: 500 sinais + 100 repousos no treino; 85 sinais + 11 repousos na validação. Os negativos preservam o split da pessoa/vídeo original. Nenhum negativo entrou no teste durante o treino.
- Treinamento: seed 42, Adam, `weight_decay=1e-4`, pesos de classe `sqrt_inverse`, augmentation dinâmica somente no treino, scheduler e early stopping. Melhor época 23/35; seleção por `val_loss=1,7154`. GPU NVIDIA RTX 4050 Laptop; cerca de 129 s.
- Resultado em validação: 11/11 repousos classificados como `__nenhum_sinal__`, mas apenas 29/85 sinais positivos corretos. O modelo `full` anterior acertava 46/85 sinais. Assim, este checkpoint **não substitui** o classificador de 25 classes.
- Uso como bloqueio antes da fusão: validação 50/85 sinais corretos e 11/11 repousos rejeitados; teste exploratório anterior 55/85 sinais corretos versus 56/85 sem o bloqueio, com 20/21 trechos de repouso de 32 quadros e 6/6 de 48 quadros rejeitados. Os trechos de teste pertencem a uma única pessoa do mesmo dataset.

`data_split/` preserva o manifesto, o mapa de classes e os índices do piloto; `config/resolved_config.json` é a configuração completa gravada no checkpoint. `history.csv` registra cada época. A análise por vídeo está em `reports/idle_edges25/pilot_evaluation/` e `reports/idle_edges25/pilot_evaluation_48/` na raiz do projeto.

Para reproduzir a partir dos dados originais, execute na raiz do projeto:

```bash
./.venv/Scripts/python.exe scripts/mine_idle_edges.py --length 32 --threshold 0.2 --preview-count 111
./.venv/Scripts/python.exe scripts/build_idle_edge_pilot.py --allow-weak-labels
./.venv/Scripts/python.exe scripts/train.py --config configs/transformer_mixed25_edge32_pilot.yaml --data-dir data/landmarks/mixed25_edge32_pilot --processed-dir data/processed/mixed25_edge32_pilot --experiment transformer_mixed25_edge32_rejection_pilot_seed42_repro --num-workers 0 --skip-test
```

Use um nome novo para não sobrescrever o treino registrado. Leia o [relatório do piloto](../../docs/RELATORIO_NEGATIVOS_BORDAS_25_CLASSES_2026-10-27.md) antes de interpretar os números.
