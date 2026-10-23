# Transformer de 25 classes — pose e face

Checkpoint `best.pt` para as 15 classes anteriores do MINDS-Libras mais Água, Ajudar, Amigo, Aprender, Beber, Brincar, Cansado, Certo, Comer e Diretor. O mapa completo está em `data_split/label_map.json`. Entrada: 48 quadros de landmarks MediaPipe Holistic com `feature_mode: pose_face`. O checkpoint foi treinado **do zero**; a cabeça anterior de 15 classes não foi reutilizada.

Foram 500 vídeos de treino, 85 de validação e 85 de teste, com isolamento de pessoas por origem. Há 30/5/5 vídeos por classe antiga e 5/1/1 por classe nova. A rede tem 122.123 parâmetros, Transformer de 2 camadas, dimensão 64, 4 cabeças, dropout 0,45, Adam com `weight_decay=1e-4`, augmentation leve apenas no treino e `CrossEntropyLoss` com pesos `sqrt_inverse` calculados somente no treino. O melhor checkpoint foi escolhido pela menor `val_loss` na época 56; early stopping encerrou na época 71.

Validação: **37/85 (43,5%)**, F1 macro **0,267**. Teste exploratório: **27/85 (31,8%)**, F1 macro **0,188**. Nas dez classes novas: 1/10 na validação e 2/10 no teste. A saída inclui apenas sinais isolados; não é tradução de frases. O backend ativo não foi alterado.

O pacote contém `best.pt`, `checksums.sha256`, YAML em `config/`, manifesto/splits em `data_split/` e histórico/log/configuração efetiva em `records/`. Os `.npz` ficam nas três pastas de origem sob `data/landmarks/` e não são duplicados. Comandos e análise em `docs/EXPERIMENTO_25_CLASSES_2026-10-22.md`.
