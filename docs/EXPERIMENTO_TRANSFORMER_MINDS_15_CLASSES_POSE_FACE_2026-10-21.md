# Experimento — Transformer MINDS-Libras de 15 classes, pose e face

## Pedido e hipótese

Retreinar do zero o Transformer `pose_face` de 20 classes, retirando exatamente `America`, `Esquina`, `Sapo`, `Bala` e `Maca`. A grafia é a do dataset; `America` e `Maca` não têm acento nos rótulos originais. As 15 classes restantes estão em `configs/minds_libras_15_sem_cinco.json`.

A remoção reduz a dificuldade do problema e o número de saídas, mas não garante maior generalização. Na validação anterior, `America` e `Maca` tiveram 5/5 acertos, enquanto `Esquina`, `Sapo` e `Bala` tiveram 0/5. A seleção foi definida pelo usuário antes deste treino.

## Plano anterior à execução

1. Reutilizar os 800 `.npz` já extraídos em `data/landmarks/minds_libras_all20/`. Selecionar 15 classes em `scripts/build_dataset.py --class-list`, com manifesto original preservado e índices dos splits filtrados. Não copiar nem reextrair vídeos.
2. Manter os sinalizadores do experimento de 20 classes: treino 01/02/05/06/08/10; validação 11; teste 12. Esperado: **450/75/75 vídeos**, 30/5/5 por classe, zero sintéticos. Verificar que, para cada split, os índices são exatamente os antigos sem as cinco classes removidas e que não há vazamento entre pessoas.
3. Treinar **do zero** com `configs/transformer_minds_pose_face.yaml`, seed 42, mesma arquitetura, augmentation e critério `val_loss`; apenas a camada final muda de 20 para 15 saídas. Salvar em um nome novo, sem tocar no modelo de 20 classes. Selecionar checkpoint somente pela validação.
4. Auditar o `best.pt` na validação e registrar matriz de confusão e previsões por vídeo. Se avaliar o sinalizador de teste, fazê-lo uma única vez depois de fixado o checkpoint; seu resultado será exploratório, pois esse sinalizador já foi consultado em outros experimentos.
5. Arquivar checkpoint, lista de classes, cadeia YAML, splits, logs, histórico, métricas, previsões e instruções de reprodução em `Modelos salvos/transformer_minds_15_classes_pose_face/`; verificar hash e carregamento estrito. Não ativar automaticamente no backend.

## Execução e resultados — 21/10/2026

- `scripts/build_dataset.py` selecionou 15 classes do CSV de 20 e criou `data/processed/minds_libras_15_sem_cinco/`: treino 450, validação 75, teste 75, 30/5/5 por classe, zero vazamento. Verificação programática confirmou manifesto idêntico ao antigo e cada split exatamente igual ao anterior após filtrar as cinco classes.
- Treino CUDA `experiments/transformer_minds_15_pose_face_seed42/`: 121.473 parâmetros, seed 42, early stopping na época 79. Melhor checkpoint na **época 64** por `val_loss=1,3728`; sua validação top-1 foi **49/75 = 65,3%**, F1 macro **0,581**. Houve época com acurácia maior (53/75 na época 58), mas ela não foi selecionada pelo critério fixado.
- Nas mesmas 75 amostras de validação, o checkpoint de 20 classes acertou **27/75 = 36%**. Entre as removidas no antigo, `America` e `Maca` acertavam 5/5 cada; `Bala`, `Esquina` e `Sapo`, 0/5. A comparação tem vocabulários de saída diferentes e não isola o efeito de retirar cada palavra.
- Teste exploratório do checkpoint fixado: **41/75 = 54,7%**, F1 macro **0,480**. Nenhum ajuste foi escolhido por esse resultado.
- Erros da validação ainda importantes: `Aproveitar`, `Conhecer` e `Filho` com 0/5; `Acontecer` com 1/5. Métricas completas e previsões em `experiments/transformer_minds_15_pose_face_seed42/`.
- Checkpoint, cadeia de configuração, splits, logs e resultados arquivados em `Modelos salvos/transformer_minds_15_classes_pose_face/`, com README de reprodução. O backend continua apontando para o modelo que já estava ativo.
- Verificação final: o checkpoint arquivado tem SHA-256 `3a3e1c2be5885cfcc30028ade9cf81f496492506b77bca92a0b55127408b5799`, igual ao arquivo original, e carrega com pesos estritos no `PredictionService` com 15 classes, 48 frames e `feature_mode=pose_face`. A cadeia YAML arquivada tem a mesma configuração efetiva; 49 testes de split, vocabulário, MINDS-Libras e pré-processamento passaram. O log foi arquivado como `records/train.txt` para que o Git não o ignore.

## Interpretação

Reduzir o vocabulário e retreinar aumentou a acurácia de validação nas 15 classes mantidas, mas o resultado de teste foi menor e algumas classes seguem sem acerto. O número não justifica apresentar o modelo como pronto para tradução de uso real. A escolha de cinco classes foi feita antes do treino; a próxima análise deve tratar as classes difíceis ainda presentes e testar generalização em sinalizadores e câmeras novos, sem otimizar pelo teste já consultado.
