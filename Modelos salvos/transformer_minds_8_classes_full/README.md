# Transformer MINDS-Libras — 8 classes, pose, face e mãos

## Resultado e decisão

Este é o checkpoint **`best.pt`**, selecionado na época **34** pela menor perda de validação (`val_loss=0,9659`). O treino terminou por early stopping na época 49. O número salvo no checkpoint é `epoch=33`, pois começa em zero.

| Modelo | Validação: Sinalizador11 | Teste exploratório: Sinalizador12 |
|---|---:|---:|
| Pose + face + mãos (este) | **23/40 = 57,5%**, F1 macro 0,495 | **35/40 = 87,5%**, F1 macro 0,856 |
| Pose + face (controle arquivado) | 40/40 = 100% | 37/40 = 92,5% |

O modelo completo **não melhorou** o controle na validação. Os cinco vídeos de `Aluno` foram classificados como `Medo`; os cinco de `Ruim` e os cinco de `Vontade` foram classificados como `Conhecer`. Dos cinco vídeos de `Medo`, três foram corretos e dois viraram `Vontade`. Os demais cinco exemplos por classe foram corretos. As previsões por vídeo e matrizes completas estão em `records/`.

O sinalizador de teste já havia sido consultado em experimentos anteriores. Seu resultado é exploratório e não deve orientar seleção de arquitetura ou hiperparâmetros. O checkpoint foi salvo para análise, **sem alterar o modelo ativo no backend**.

## Comparação controlada

- Mesmos 320 vídeos MINDS-Libras, 8 classes, 48 frames amostrados por vídeo e split por pessoa do modelo `pose_face`: treino Sinalizador01, 02, 05, 06, 08 e 10 (**240 vídeos**); validação Sinalizador11 (**40**); teste Sinalizador12 (**40**). Não há vídeos sintéticos.
- `config/transformer_minds_full.yaml` herda a configuração completa do controle e muda apenas `data.feature_mode` de `pose_face` para `full`. O pipeline mantém os pontos e máscaras das duas mãos, a pose e a face. O normalizador deriva velocidades e distâncias. Entrada final: **697 features por frame**.
- Mesmo Transformer temporal 1-D de **121.018 parâmetros**: dimensão 64, duas camadas, quatro cabeças, feedforward 128, dropout 0,45 e classificador oculto 64.
- Mesma augmentation dinâmica somente no treino: ruído gaussiano (`std=0,002`), escala 0,98–1,02 e rotação até 2°, cada operação com probabilidade 0,7. Mesmo Adam (`lr=3e-4`, `weight_decay=1e-4`), CrossEntropy sem pesos de classe, batch 16, seed 42, scheduler `ReduceLROnPlateau` e early stopping pela `val_loss` com paciência 15.

A auditoria anterior em `docs/AUDITORIA_MAOS_MINDS_8_CLASSES.md` encontrou ao menos uma mão detectada em 48,77% dos quadros de treino contra 99,75% da validação. Essa diferença é uma **hipótese** para a pior generalização, não uma causa comprovada: a auditoria inclui repouso e não mede a qualidade dos pontos durante o gesto. O desempenho muito melhor no teste que na validação também mostra que o resultado depende do sinalizador.

## Arquivos

- `best.pt`: pesos, arquitetura e modalidade `full`.
- `config/`: cadeia YAML e lista das oito classes.
- `data_split/`: manifesto, splits, mapa de classes e metadados.
- `records/`: configuração efetiva, ambiente, hashes, log do treino, histórico, métricas e previsões individuais.
- `checksums.sha256`: hashes de todos os arquivos arquivados, exceto ele próprio.

Os `.npz` de entrada permanecem em `data/landmarks/minds_libras_school8/` e são necessários para reproduzir treino e avaliação. `records/code_hashes.json` registra os hashes dos arquivos de código relevantes.

## Reproduzir no Git Bash, a partir da raiz do projeto

```bash
./.venv/Scripts/python.exe scripts/train.py \
  --config "Modelos salvos/transformer_minds_8_classes_full/config/transformer_minds_full.yaml" \
  --data-dir data/landmarks/minds_libras_school8 \
  --processed-dir "Modelos salvos/transformer_minds_8_classes_full/data_split" \
  --experiment transformer_minds_school8_full_repro_seed42 \
  --num-workers 0 --skip-test
```

Auditar **o checkpoint salvo** na validação:

```bash
./.venv/Scripts/python.exe scripts/audit_checkpoint_val.py \
  --checkpoint "Modelos salvos/transformer_minds_8_classes_full/best.pt" \
  --data-dir data/landmarks/minds_libras_school8 \
  --processed-dir "Modelos salvos/transformer_minds_8_classes_full/data_split" \
  --split val --output-dir experiments/audit_modelo_full_salvo
```

Um novo treino CUDA pode não gerar pesos bit a bit iguais. Compare métricas, splits e previsões; não presuma hash idêntico.
