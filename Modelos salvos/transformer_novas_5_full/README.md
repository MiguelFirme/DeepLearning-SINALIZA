# Transformer de cinco palavras novas — pose, face e mãos

Checkpoint treinado: `best.pt` (SHA-256 em `checksums.sha256`). Classes: **Água, Ajudar, Amigo, Aprender, Beber**. Entrada: sequência de 48 quadros de landmarks MediaPipe Holistic, transformada pelo `feature_mode: full`, que inclui pose, face e mãos. O mapa de classes e os splits são os mesmos do modelo `pose_face`.

Foram usados 35 vídeos de `data/raw/<classe>/`: 25 de `Sinalizador01` a `05` no treino, 5 de `06` na validação e 5 de `07` no teste. O manifesto e os índices exatos estão em `data_split/`. A configuração efetiva e o comando original estão em `records/run_record.json`; `config/` guarda os YAML de origem. O treino usa a mesma arquitetura e estratégia do modelo `pose_face`: Transformer de 2 camadas, dimensão 64, 4 cabeças, dropout 0,45, Adam com `weight_decay=1e-4`, augmentation leve apenas no treino, scheduler por platô e early stopping por `val_loss` com paciência 15. Só muda o modo de entrada.

Melhor época: 118/120. Validação: **5/5**. Teste: **2/5**; F1 macro **0,233**. A detecção das mãos diferiu muito entre pessoas; detalhes estão no relatório do experimento. A amostra de teste tem apenas um vídeo por classe, portanto o resultado é exploratório. O modelo ativo no backend não foi alterado.

Para reproduzir o treino a partir dos `.npz` originais, na raiz do projeto e com as dependências instaladas no Python 3.11:

```bash
./.venv/Scripts/python.exe scripts/train.py --config configs/transformer_minds_full.yaml --data-dir data/landmarks/novas_5_palavras --processed-dir data/processed/novas_5_palavras --experiment transformer_novas_5_full_seed42 --num-workers 0
```

Consulte `docs/EXPERIMENTO_NOVAS_5_PALAVRAS_2026-10-22.md` para o preparo completo dos dados e as limitações.
