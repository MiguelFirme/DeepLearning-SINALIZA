# Transformer das cinco palavras restantes — pose e face

Checkpoint: `best.pt`, SHA-256 em `checksums.sha256`. Classes: **Brincar, Cansado, Certo, Comer e Diretor**. Entrada de 48 quadros com `feature_mode: pose_face`; landmarks das mãos não entram no modelo.

Foram usados 35 vídeos de `data/raw/<classe>/`: sinalizadores 01–05 no treino (25), 06 na validação (5) e 07 no teste (5). `data_split/` contém o manifesto, o mapa de classes, o CSV e os índices exatos. `config/` guarda os YAMLs e a lista fechada de classes. `records/run_record.json` guarda o comando, o ambiente e a configuração efetiva; os demais arquivos de `records/` guardam histórico, métricas e log.

Arquitetura e treino: Transformer com 2 camadas, dimensão 64, 4 cabeças e 120.823 parâmetros; dropout 0,45; Adam, `weight_decay=1e-4`; augmentation leve somente no treino; scheduler por platô e early stopping com paciência 15, monitorando `val_loss`. Semente 42. Melhor época: **96/111**. Validação: **4/5**. Teste: **4/5**, F1 macro **0,733**. O erro de validação e teste foi Brincar → Cansado. Com um vídeo por classe em cada split, esses números são exploratórios.

Reproduzir na raiz do projeto após configurar Python 3.11 e as dependências:

```bash
./.venv/Scripts/python.exe scripts/train.py --config configs/transformer_minds_pose_face.yaml --data-dir data/landmarks/novas_5_restantes --processed-dir data/processed/novas_5_restantes --experiment transformer_novas_5_restantes_pose_face_seed42 --num-workers 0
```

Preparação de dados, fusão e limitações: `docs/EXPERIMENTO_NOVAS_5_RESTANTES_2026-10-22.md`. O backend não usa este checkpoint.
