# Transformer MINDS-Libras — 15 classes, pose, face e mãos

## Resultado

Este é o checkpoint **`best.pt`** do experimento `transformer_minds_15_full_seed42`, selecionado pela menor perda de validação na **época 30** (`epoch=29` no checkpoint, índice iniciado em zero). `val_loss=1,6027` no treino; early stopping na época 45.

| Modelo | Validação: Sinalizador11 | Teste exploratório: Sinalizador12 |
|---|---:|---:|
| Pose + face + mãos (este) | **45/75 = 60,0%**, F1 macro 0,550 | **34/75 = 45,3%**, F1 macro 0,358 |
| Pose + face (controle) | 49/75 = 65,3%, F1 macro 0,581 | 41/75 = 54,7%, F1 macro 0,480 |

Os dois modelos foram avaliados nos mesmos vídeos. Na validação, ambos acertaram 29; somente este acertou 16; somente `pose_face` acertou 20; ambos erraram 10. Incluir as mãos **não melhorou a acurácia total** nesta execução. Houve troca de erros por classe: este modelo acertou `Acontecer` e `Conhecer` 5/5 na validação, ante 1/5 e 0/5 do controle, mas acertou `Aluno`, `Cinco`, `Espelho` e `Medo` 0/5, ante 5/5, 4/5, 5/5 e 4/5 do controle. As matrizes e previsões individuais estão em `records/`.

O Sinalizador12 já foi consultado em outros experimentos; sua métrica é **exploratória** e não guiou checkpoint ou hiperparâmetros. A cobertura das mãos diferia muito entre sinalizadores na auditoria das oito classes (`docs/AUDITORIA_MAOS_MINDS_8_CLASSES.md`), uma possível explicação que **não foi isolada** para as 15 classes. O modelo foi salvo para análise e **não substituiu o checkpoint ativo no backend**. São sinais isolados de um dataset pequeno; as métricas não medem tradução de frases nem pessoas e câmeras novas.

## Comparação controlada

- Mesmos landmarks `.npz` de `data/landmarks/minds_libras_all20/`, lista de 15 classes e split do modelo `pose_face` arquivado. O manifesto contém 800 vídeos; `data_split/splits.json` aponta apenas os **600 vídeos elegíveis**. Treino: sinalizadores 01, 02, 05, 06, 08 e 10 (**450 vídeos, 30 por classe**). Validação: 11 (**75, 5 por classe**). Teste: 12 (**75, 5 por classe**). Zero sintéticos.
- `config/transformer_minds_full.yaml` herda toda a configuração do controle e altera somente `data.feature_mode` para `full`. As duas mãos, pose e face entram no mesmo pipeline de normalização e derivação, com **48 frames × 697 features** por vídeo.
- Mesmo Transformer temporal 1-D, **121.473 parâmetros**: dimensão 64, duas camadas, quatro cabeças, feedforward 128, token CLS, LayerNorm, dropout 0,45, classificador oculto 64.
- Mesma augmentation dinâmica somente no treino: ruído gaussiano (`std=0,002`), escala 0,98–1,02 e rotação até 2°, cada operação com probabilidade 0,7. Adam (`lr=3e-4`, `weight_decay=1e-4`), CrossEntropy sem pesos de classe, batch 16, seed 42, scheduler `ReduceLROnPlateau` e early stopping por `val_loss` com paciência 15.

`config/` guarda a cadeia YAML e a lista de classes; `data_split/`, o manifesto e os splits; `records/`, configuração efetiva, histórico, log em `train.txt`, métricas e previsões. `checksums.sha256` verifica a integridade desta cópia. Os `.npz` de entrada não são duplicados aqui.

## Reproduzir no Git Bash, na raiz do projeto

```bash
./.venv/Scripts/python.exe scripts/train.py \
  --config "Modelos salvos/transformer_minds_15_classes_full/config/transformer_minds_full.yaml" \
  --data-dir data/landmarks/minds_libras_all20 \
  --processed-dir "Modelos salvos/transformer_minds_15_classes_full/data_split" \
  --experiment transformer_minds_15_full_repro_seed42 \
  --num-workers 0 --skip-test
```

Para auditar **o checkpoint salvo** na validação sem alterar o arquivo:

```bash
./.venv/Scripts/python.exe scripts/audit_checkpoint_val.py \
  --checkpoint "Modelos salvos/transformer_minds_15_classes_full/best.pt" \
  --data-dir data/landmarks/minds_libras_all20 \
  --processed-dir "Modelos salvos/transformer_minds_15_classes_full/data_split" \
  --split val --output-dir experiments/audit_modelo_15_full_salvo
```

Um novo treino CUDA pode variar ligeiramente; compare splits, métricas e previsões, não apenas o hash dos pesos.
