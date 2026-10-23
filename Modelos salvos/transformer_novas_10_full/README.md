# Transformer das dez palavras novas

Checkpoint treinado **do zero** somente com Água, Ajudar, Amigo, Aprender, Beber, Brincar, Cansado, Certo, Comer e Diretor. Use `best.pt` para carregar os pesos e `data_split/label_map.json` para interpretar os índices. Os YAML em `config/` e `records/run_record.json` registram a configuração efetiva; `records/` também contém o histórico, log e previsões por vídeo. O SHA-256 de `best.pt` está em `checksums.sha256`.

Entrada: sequência de 48 quadros de landmarks MediaPipe Holistic com pose, face e mãos (`feature_mode: full`). Transformer de 2 camadas, dimensão 64, 4 cabeças, dropout 0,45, 121.148 parâmetros. Treino: 50 vídeos de cinco sinalizadores, CrossEntropy sem pesos, Adam com `weight_decay=1e-4`, augmentation leve só no treino, scheduler por platô, early stopping por menor `val_loss`. Validação e teste têm dez vídeos cada, um por classe e por sinalizador distinto.

Melhor época: 101/116. Validação: **6/10 (60%)**, F1 macro 0,533. Teste exploratório: **6/10 (60%)**, F1 macro 0,550. Com somente um vídeo de teste por palavra, o resultado não estima com precisão o desempenho em pessoas novas. O checkpoint ativo no backend não foi alterado.

Os `.npz` usados pelo manifesto arquivado precisam estar disponíveis em `data/landmarks/novas_5_palavras/` e `data/landmarks/novas_5_restantes/`. O preparo, os comandos e os erros por palavra estão em `docs/EXPERIMENTO_NOVAS_10_CLASSES_2026-10-23.md`.
