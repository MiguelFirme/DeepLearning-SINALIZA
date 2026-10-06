# Plano: Transformer pose e face com as 20 classes MINDS-Libras

Data: 2026-10-05. Execução pelo usuário no Git Bash do VS Code.

## Objetivo

Repetir o experimento de 8 classes com todas as 20 classes do MINDS-Libras, mantendo o Transformer 1-D e `feature_mode=pose_face`. Esta é uma tarefa de classificação de sinais isolados; o resultado não mede tradução de frases.

## Inventário conferido antes da execução

- Índice local do Kaggle: 20 classes, 800 MP4, 40 vídeos por classe, 8 sinalizadores e 5 tomadas por sinalizador.
- Já disponíveis: 320 MP4 e 320 landmarks `.npz` das 8 classes anteriores.
- Pendentes: 480 MP4, aproximadamente 26,66 GiB. O downloader confere o tamanho informado no índice antes de promover cada `.part` a `.mp4`.
- Classes exatas: `configs/minds_libras_all20.json`. Inclui inclusive palavras fora do foco escolar por escolha explícita de testar todas as classes.

## Protocolo

1. Baixar os 480 MP4 restantes na pasta já usada. O script reaproveita arquivos completos e retoma `.part`.
2. Copiar os 320 `.npz` existentes para uma pasta nova e extrair somente os 480 novos com a mesma configuração MediaPipe. Conferir 800 amostras e 20 classes no manifesto.
3. Construir split por sinalizador: treino 01/02/05/06/08/10 (600 vídeos), validação 11 (100), teste 12 (100), 30/5/5 por classe. Exigir ausência de vazamento e de amostras sintéticas.
4. Treinar com `configs/transformer_minds_pose_face.yaml`, seed 42 e os hiperparâmetros do piloto de 8 classes. Selecionar o checkpoint pela menor `val_loss`; augmentation somente no treino. Registrar configuração, histórico e log persistente.
5. Auditar o `best.pt` na validação e depois no teste, uma vez. Salvar matriz de confusão e previsões por vídeo; examinar top-1, F1 macro e desempenho por classe. A referência aleatória uniforme para 20 classes equilibradas é 5% de top-1.

O Sinalizador12 já foi consultado nos pilotos anteriores. Este teste permanece exploratório e não deve guiar ajustes subsequentes de hiperparâmetros. Acurácias de 8 e 20 classes não são diretamente comparáveis; avaliar as oito classes comuns separadamente se necessário.

## Comandos no Git Bash

Execute na raiz `~/DeepLearning-SINALIZA`, em ordem. O download e a extração podem ser repetidos após interrupção; use um nome novo de experimento se decidir refazer o treino.

```bash
./.venv/Scripts/python.exe scripts/download_minds_libras.py --dry-run --jobs 2
./.venv/Scripts/python.exe scripts/download_minds_libras.py --jobs 2
mkdir -p data/landmarks/minds_libras_all20
cp -n data/landmarks/minds_libras_school8/*.npz data/landmarks/minds_libras_all20/
./.venv/Scripts/python.exe scripts/extract_landmarks.py \
  --input data/raw/minds_libras \
  --output data/landmarks/minds_libras_all20 \
  --dataset-format minds_libras --workers 2
./.venv/Scripts/python.exe scripts/build_dataset.py \
  --landmarks data/landmarks/minds_libras_all20 \
  --annotations data/landmarks/minds_libras_all20/annotations.csv \
  --output data/processed/minds_libras_all20 \
  --strategy signer_holdout \
  --class-list configs/minds_libras_all20.json \
  --train-signers Sinalizador01 Sinalizador02 Sinalizador05 Sinalizador06 Sinalizador08 Sinalizador10 \
  --val-signer Sinalizador11 --test-signer Sinalizador12
./.venv/Scripts/python.exe scripts/train.py \
  --config configs/transformer_minds_pose_face.yaml \
  --data-dir data/landmarks/minds_libras_all20 \
  --processed-dir data/processed/minds_libras_all20 \
  --experiment transformer_minds_all20_pose_face_seed42 \
  --num-workers 0 --skip-test
./.venv/Scripts/python.exe scripts/audit_checkpoint_val.py \
  --checkpoint experiments/transformer_minds_all20_pose_face_seed42/best.pt \
  --data-dir data/landmarks/minds_libras_all20 \
  --processed-dir data/processed/minds_libras_all20 --split val
./.venv/Scripts/python.exe scripts/audit_checkpoint_val.py \
  --checkpoint experiments/transformer_minds_all20_pose_face_seed42/best.pt \
  --data-dir data/landmarks/minds_libras_all20 \
  --processed-dir data/processed/minds_libras_all20 --split test
cp experiments/transformer_minds_all20_pose_face_seed42/*metrics.json \
   experiments/transformer_minds_all20_pose_face_seed42/*predictions.csv \
   logs/experiments/transformer_minds_all20_pose_face_seed42/
```

Conferir nos logs da extração `800 amostras, 20 classes, 8 sinalizadores` e nos logs do build `train=600, val=100, test=100`. Se aparecer outro total ou erro de cobertura, interromper antes do treino. Os resultados ficarão em `experiments/transformer_minds_all20_pose_face_seed42/`; log, configuração efetiva e cópias das métricas em `logs/experiments/transformer_minds_all20_pose_face_seed42/`.

## Treino executado pelo usuário e auditoria de validação (2026-10-05)

O usuário baixou e extraiu as 20 classes e treinou `transformer_minds_all20_pose_face_seed42` com `--skip-test`. O registro do experimento confirma 800 amostras, 600/100/100 por split, 30/5/5 por classe, zero sintéticas e sinalizadores disjuntos. O modelo usou CUDA, 121.798 parâmetros, `pose_face` e seed 42.

O early stopping encerrou na época 50. O `best.pt` é da época **35**, selecionada pela menor `val_loss=1,5430`; naquela época, treino top-1=56%, validação top-1=37/100 e F1 macro=0,286. A auditoria de `best.pt` com `scripts/audit_checkpoint_val.py --split val` reproduziu 37/100, F1 macro 0,286 e loss 1,5429. Validação top-3 foi 88/100 e top-5 96/100 na época 35: muitas classes corretas ainda aparecem entre as cinco maiores pontuações, mas a primeira escolha é fraca.

O histórico mostra top-1 de validação de **50/100 na época 47**, porém com `val_loss=1,7422`; esse checkpoint não foi salvo como `best.pt`. O `last.pt` da época 50 terminou com treino 66,5%, validação 47% e `val_loss=1,6209`. Não confundir esses picos com o resultado do checkpoint escolhido. A diferença treino/validação no fim é de 19,5 pontos percentuais, e a validação oscila bastante.

No `best.pt`, **11 classes tiveram 0/5 acertos** na validação: `Acontecer`, `Amarelo`, `Aproveitar`, `Bala`, `Banheiro`, `Conhecer`, `Espelho`, `Esquina`, `Filho`, `Medo` e `Sapo`. `America`, `Banco`, `Maca`, `Ruim`, `Vacina` e `Vontade` tiveram 5/5; `Barulho` 4/5, `Cinco` 2/5 e `Aluno` 1/5. Nas oito classes do piloto anterior, o novo modelo acertou **15/40** na validação, contra 40/40 do modelo treinado só com oito classes. A comparação indica que o vocabulário ampliado tornou esta configuração menos eficaz, mas não isola se o limite é da representação pose/face, da otimização, dos vídeos ou de outras diferenças entre classes.

As classes são equilibradas (30 exemplos de treino por classe), portanto desbalanceamento não é a explicação principal. A ausência dos pontos detalhados das mãos pode dificultar distinções entre sinais com movimentos parecidos de pose/face; isso permanece hipótese, especialmente porque um teste anterior com mãos completas teve desempenho pior nas cinco classes. Ajustar perda, dropout ou arquitetura sem auditar as confusões primeiro não está justificado.

Arquivos: `experiments/transformer_minds_all20_pose_face_seed42/{best.pt,last.pt,history.csv,val_metrics.json,val_predictions.csv}`; cópias da validação, `train.log`, `history.json` e `run_record.json` em `logs/experiments/transformer_minds_all20_pose_face_seed42/`.

**Teste:** nenhum `test_metrics.json` foi produzido para este experimento. O Sinalizador12 não foi consultado neste treino. O próximo passo, se quisermos reportar o desempenho deste checkpoint, é avaliar `best.pt` uma única vez com `--split test`, sem usar esse resultado para ajustar hiperparâmetros. Também é útil auditar as confusões na validação antes de decidir outro experimento.
