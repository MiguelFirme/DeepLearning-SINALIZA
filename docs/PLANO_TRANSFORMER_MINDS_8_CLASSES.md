# Plano: Transformer pose e face com 8 classes MINDS-Libras

Data: 2026-10-05.

## Pergunta

Como se comporta o mesmo Transformer 1-D `pose_face` ao ampliar de cinco para oito classes no MINDS-Libras?

## Vocabulário e dados

Manter `Aluno`, `Banheiro`, `Conhecer`, `Medo`, `Vontade`; acrescentar `Acontecer`, `Barulho`, `Ruim`. Lista fechada em `configs/minds_libras_school8.json`. O índice Kaggle local registra 40 vídeos por classe (oito sinalizadores, cinco tomadas), total previsto de 320 vídeos. Antes deste experimento, somente as cinco classes antigas (200 MP4 e 200 `.npz`) estavam disponíveis. As três novas classes exigem aproximadamente 6,48 GiB de download; todos os arquivos devem ter tamanho conferido contra o índice.

## Protocolo antes da execução

1. Baixar apenas as três classes novas em `data/raw/minds_libras/`, sem alterar vídeos antigos. Reusar os 200 landmarks existentes; extrair os 120 novos com a mesma configuração do MediaPipe em uma pasta nova `data/landmarks/minds_libras_school8/`. Não misturar pontos de outro extrator.
2. Construir `data/processed/minds_libras_school8/` com os mesmos sinalizadores: treino 01, 02, 05, 06, 08, 10; validação 11; teste 12. Verificar 240/40/40 vídeos, 30/5/5 por classe, sem vazamento de pessoa ou vídeo e sem amostras sintéticas.
3. Treinar o Transformer de `configs/transformer_minds_pose_face.yaml` com `feature_mode=pose_face`, seed 42 e os mesmos hiperparâmetros do piloto de cinco classes. Selecionar `best.pt` somente pela menor `val_loss`; augmentation dinâmica apenas no treino. Salvar histórico, checkpoint e log persistente.
4. Auditar o checkpoint na validação e, a pedido do usuário, no teste. Salvar matriz de confusão e previsões por vídeo. Reportar top-1, F1 macro e acertos por classe, comparando com a referência aleatória de 12,5% para oito classes equilibradas.
5. Interpretar o teste como exploratório: Sinalizador12 já foi consultado em pilotos anteriores. Acurácia de oito classes não é diretamente comparável à de cinco classes; acompanhar as cinco classes comuns e a matriz de confusão. Não ajustar configuração com base no teste.

## Comandos planejados no Git Bash

```bash
./.venv/Scripts/python.exe scripts/download_minds_libras.py --labels Acontecer Barulho Ruim --jobs 2
./.venv/Scripts/python.exe scripts/extract_landmarks.py --input data/raw/minds_libras --output data/landmarks/minds_libras_school8 --dataset-format minds_libras --workers 2
./.venv/Scripts/python.exe scripts/build_dataset.py --landmarks data/landmarks/minds_libras_school8 --annotations data/landmarks/minds_libras_school8/annotations.csv --output data/processed/minds_libras_school8 --strategy signer_holdout --class-list configs/minds_libras_school8.json --train-signers Sinalizador01 Sinalizador02 Sinalizador05 Sinalizador06 Sinalizador08 Sinalizador10 --val-signer Sinalizador11 --test-signer Sinalizador12
./.venv/Scripts/python.exe scripts/train.py --config configs/transformer_minds_pose_face.yaml --data-dir data/landmarks/minds_libras_school8 --processed-dir data/processed/minds_libras_school8 --experiment transformer_minds_school8_pose_face_seed42 --num-workers 0 --skip-test
```

O segundo comando pressupõe copiar os 200 `.npz` antigos para a nova pasta antes da extração; o extrator pula arquivos já presentes. Os comandos de avaliação serão registrados após o treino.

## Execução em andamento

O `--dry-run` confirmou 120 arquivos pendentes, 6,48 GiB. Foram copiados 200 `.npz` da pasta do piloto de cinco classes para `data/landmarks/minds_libras_school8/` (aproximadamente 27,6 MiB), sem modificar os originais. O download sequencial foi interrompido após dois arquivos completos para habilitar `--jobs 8` no downloader; os arquivos parciais são retomáveis. A variante paralela baixa arquivos diferentes em threads e só promove `.part` a `.mp4` depois da verificação de tamanho. O download das três classes foi reiniciado com `--jobs 8`.

## Resultado (2026-10-05)

O download foi concluído e os tamanhos dos 120 MP4 novos foram conferidos. Como a taxa caiu com oito conexões, o downloader foi retomado com `--jobs 2`; os arquivos parciais foram preservados. A extração foi feita em lotes e terminou com **320 `.npz`, oito classes, oito sinalizadores e zero erros**. O manifesto contém exatamente 40 vídeos por classe e cinco tomadas por sinalizador. `build_dataset.py` confirmou ausência de vazamento e gerou 240 amostras de treino, 40 de validação e 40 de teste, com 30/5/5 por classe e nenhum arquivo sintético.

O Transformer `pose_face` usou 121.018 parâmetros e CUDA. O early stopping encerrou na época 109; a menor `val_loss=0,1349` foi na época **94**, que definiu `best.pt`. Esse mesmo checkpoint foi auditado em validação e teste:

| Conjunto | Top-1 | F1 macro | Loss | Amostras |
|---|---:|---:|---:|---:|
| Validação, Sinalizador11 | **40/40 (100%)** | 1,000 | 0,135 | 40 |
| Teste, Sinalizador12 | **37/40 (92,5%)** | 0,923 | 0,203 | 40 |

No teste: `Acontecer` 5/5, `Aluno` 5/5, `Banheiro` 5/5, `Barulho` 5/5, `Conhecer` 5/5, `Medo` 4/5, `Ruim` 5/5, `Vontade` 3/5. Os erros foram `Medo` → `Aluno` (1) e `Vontade` → `Conhecer` (2). Nas cinco classes comuns, este modelo acertou **22/25** vídeos do Sinalizador12; o Transformer de cinco classes acertara **20/25**. As três classes novas somaram 15/15. Essa diferença de dois acertos nas classes comuns é observacional: não prova que incluir classes adicionais causou a melhora, pois o treino e o espaço de rótulos mudaram.

Artefatos do experimento: `experiments/transformer_minds_school8_pose_face_seed42/` contém `best.pt`, histórico, matrizes em `val_metrics.json` e `test_metrics.json`, além de previsões em `val_predictions.csv` e `test_predictions.csv`. Cópias das métricas/previsões, `train.log`, histórico e `run_record.json` ficam em `logs/experiments/transformer_minds_school8_pose_face_seed42/`. O registro contém configuração efetiva, hashes do manifesto e splits, ambiente, seed e grupos por split.

Comandos de auditoria no Git Bash:

```bash
./.venv/Scripts/python.exe scripts/audit_checkpoint_val.py \
  --checkpoint experiments/transformer_minds_school8_pose_face_seed42/best.pt \
  --data-dir data/landmarks/minds_libras_school8 \
  --processed-dir data/processed/minds_libras_school8 --split val

./.venv/Scripts/python.exe scripts/audit_checkpoint_val.py \
  --checkpoint experiments/transformer_minds_school8_pose_face_seed42/best.pt \
  --data-dir data/landmarks/minds_libras_school8 \
  --processed-dir data/processed/minds_libras_school8 --split test
```

**Limite da conclusão:** o modelo classificou bem estes sinais isolados nos dois sinalizadores retidos deste dataset. O Sinalizador12 já tinha sido consultado em experimentos anteriores, e cinco tomadas por classe de uma pessoa são uma amostra pequena. Não inferir que a acurácia se repetirá com pessoas novas, câmeras diferentes ou frases completas. Para a próxima verificação, reservar novos participantes ou repetir uma avaliação por sinalizador com protocolo previamente fixado, sem escolher hiperparâmetros pelo teste atual.
