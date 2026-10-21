# Transformer MINDS-Libras — 15 classes, pose e face

## Identidade e resultado

- Experimento: `experiments/transformer_minds_15_pose_face_seed42/`.
- Modelo treinado **do zero**: `best.pt`, selecionado pela menor perda de validação na **época 64** (`epoch=63` dentro do checkpoint, índice iniciado em zero). `val_loss=1,3728` no treino; early stopping na época 79.
- Validação (Sinalizador11): **49/75 = 65,3% top-1**, F1 macro **0,581**.
- Teste (Sinalizador12): **41/75 = 54,7% top-1**, F1 macro **0,480**, avaliação única e exploratória após a seleção do checkpoint.
- O checkpoint anterior de 20 classes acertou **27/75 = 36%** nos *mesmos 75 vídeos de validação das 15 classes mantidas*; o novo acertou 49/75. O checkpoint de 20 classes acertou 37/100 = 37% em sua validação completa. As tarefas têm quantidades distintas de saídas, então essa comparação mostra o efeito prático da restrição de vocabulário e do retreino, sem isolar qual parte do ganho veio de cada um.

As cinco classes removidas por solicitação do grupo foram `America`, `Esquina`, `Sapo`, `Bala` e `Maca`, com essa grafia exata no dataset. No checkpoint anterior, `America` e `Maca` tinham **5/5 acertos** na validação; as outras três tinham **0/5**. As classes finais e os índices estão em `data_split/label_map.json`.

Persistem erros relevantes: na validação, `Aproveitar`, `Conhecer` e `Filho` tiveram **0/5 acertos** e `Acontecer` teve **1/5**. Consulte `records/val_metrics.json` e `records/val_predictions.csv` para a matriz completa e os vídeos individuais. Este resultado é de sinais isolados em um dataset pequeno; não mede tradução de frases nem desempenho em uma pessoa nova fora do conjunto. O Sinalizador12 já foi consultado em outros experimentos, por isso sua métrica aqui é exploratória. **Este modelo não foi ativado no backend.**

## Dados e pré-processamento

- Mesmos landmarks do experimento de 20 classes, em `data/landmarks/minds_libras_all20/`. O manifesto contém os 800 vídeos originais; `data_split/splits.json` aponta somente os **600 vídeos das 15 classes**.
- Split por sinalizador, sem vídeos sintéticos: treino Sinalizador01, 02, 05, 06, 08 e 10 (**450 vídeos, 30 por classe**); validação Sinalizador11 (**75, 5 por classe**); teste Sinalizador12 (**75, 5 por classe**). Os índices são exatamente os do experimento antigo filtrados pelas cinco classes excluídas.
- Sequências amostradas em **48 frames**, com **697 features por frame**. `feature_mode=pose_face` zera os pontos detalhados das duas mãos; pose, inclusive braços/pulsos, e face permanecem. A normalização usa ombros quando disponíveis, seguida por features derivadas.
- Augmentation dinâmica somente no treino: ruído gaussiano (`std=0,002`), escala de 0,98 a 1,02 e rotação de até 2°, cada uma com probabilidade 0,7. Validação e teste usam landmarks limpos.

## Rede e otimização

- Mesmo Transformer temporal 1-D do experimento de 20 classes: dimensão 64, **2 camadas**, **4 cabeças**, feedforward 128, token CLS, LayerNorm, dropout **0,45** e classificador oculto 64. A camada de saída tem 15 unidades; total **121.473 parâmetros**.
- CrossEntropy padrão sem pesos de classe; Adam com `learning_rate=3e-4`, `weight_decay=1e-4`; batch 16, clipping 1,0 e mixed precision no treino CUDA.
- `ReduceLROnPlateau` pela `val_loss` (paciência 5, fator 0,5); early stopping pela `val_loss` (paciência 15); máximo 120 épocas; seed 42.

`config/` preserva a cadeia YAML e a lista fechada das 15 classes. `data_split/` preserva manifesto, splits, mapa, CSV de anotações e metadados. `records/` contém configuração efetiva, ambiente, hashes, log em `train.txt`, histórico, métricas e previsões por vídeo. `checksums.sha256` verifica a integridade do pacote. Os `.npz` de entrada não estão duplicados nesta pasta.

## Reproduzir no Git Bash, a partir da raiz do projeto

Com `data/landmarks/minds_libras_all20/` disponível, use um nome novo para não sobrescrever o experimento:

```bash
./.venv/Scripts/python.exe scripts/train.py \
  --config "Modelos salvos/transformer_minds_15_classes_pose_face/config/transformer_minds_pose_face.yaml" \
  --data-dir data/landmarks/minds_libras_all20 \
  --processed-dir "Modelos salvos/transformer_minds_15_classes_pose_face/data_split" \
  --experiment transformer_minds_15_pose_face_repro_seed42 \
  --num-workers 0 --skip-test
```

Auditar **o checkpoint arquivado** na validação:

```bash
./.venv/Scripts/python.exe scripts/audit_checkpoint_val.py \
  --checkpoint "Modelos salvos/transformer_minds_15_classes_pose_face/best.pt" \
  --data-dir data/landmarks/minds_libras_all20 \
  --processed-dir "Modelos salvos/transformer_minds_15_classes_pose_face/data_split" \
  --split val --output-dir experiments/audit_modelo_15_salvo
```

Treinos CUDA podem produzir pequenas diferenças entre execuções; compare métricas, previsões e splits, não apenas o hash de pesos novos.
