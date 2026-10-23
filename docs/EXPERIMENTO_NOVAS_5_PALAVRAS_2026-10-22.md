# Experimento: cinco palavras novas com dois Transformers

## Plano registrado antes da execução

1. Usar as cinco primeiras pastas novas pela ordem de criação em `data/raw`: Água, Ajudar, Amigo, Aprender e Beber. Cada uma contém sete vídeos, um por `Sinalizador01` a `Sinalizador07`. Os arquivos originais permanecem intactos.
2. Extrair landmarks MediaPipe Holistic apenas desses 35 vídeos. Como os nomes não seguem o padrão MINDS-Libras, registrar classe pela pasta e pessoa pelo sufixo `SinalizadorNN`; gerar manifesto e CSV compatíveis com `build_dataset.py`.
3. Reservar pessoas distintas: `01` a `05` para treino (25 vídeos), `06` para validação (5) e `07` para teste (5). Conferir leitura dos vídeos, qualidade da extração e ausência de vazamento antes de treinar.
4. Treinar dois Transformers com as mesmas configurações do experimento anterior, variando somente `feature_mode`: `pose_face` e `full` (pose, face e mãos). Salvar pesos, configurações, logs e métricas. Selecionar checkpoints pela perda de validação.
5. Repetir a fusão sequencial pré-definida: top 3 e top 5, peso 0,65 para `pose_face` e 0,35 para `full`, nas duas ordens. Comparar primeiro na validação; consultar o teste uma única vez com regras fixas. Salvar as predições por vídeo e documentar limitações.

## Limitações previstas

Há apenas um vídeo por classe no conjunto de validação e um no teste. Cada acerto muda a acurácia em 20 pontos percentuais. Assim, os resultados servem como diagnóstico inicial da arquitetura, não como estimativa confiável de desempenho em produção. Não se ajustam pesos da fusão com base nesses cinco vídeos de teste.

## Execução e resultados

Executado em 22/10/2026. Os 35 arquivos MP4 abriram no OpenCV; a extração Holistic terminou com **35 OK, 0 erros**. O manifesto identifica classe pela pasta e sinalizador pelo nome do arquivo. A verificação do `build_dataset.py` não encontrou vazamento entre treino, validação e teste. O `Sinalizador01` tem vídeos a aproximadamente 12 fps, enquanto os demais ficaram perto de 30 fps; o pré-processamento amostra 48 quadros por vídeo.

A máscara de detecção do Holistic mostrou diferença relevante entre pessoas. A média de presença das duas mãos por sinalizador, calculada nos cinco vídeos de cada um, foi: 01 (27%/26%), 02 (23%/23%), 03 (46%/46%), 04 (22%/22%), 05 (46%/66%), 06 (32%/42%) e 07 (96%/93%). Pose e face tiveram presença de 100% nesse resumo. A diferença entre os sinalizadores de treino e teste é um possível fator para o resultado do modelo `full`; a auditoria sozinha não prova causalidade.

Os dois treinamentos usaram os YAML `configs/transformer_minds_pose_face.yaml` e `configs/transformer_minds_full.yaml`, sem alterar arquitetura ou hiperparâmetros entre eles. Cada modelo possui **120.823 parâmetros**, 2 camadas Transformer, dimensão interna 64, 4 cabeças, dropout 0,45, Adam com L2 `1e-4`, augmentation leve apenas no treino, `ReduceLROnPlateau` e early stopping com paciência 15. O segundo modelo só troca `feature_mode` de `pose_face` para `full`. Os logs completos e a configuração efetiva estão em `logs/experiments/transformer_novas_5_{pose_face,full}_seed42/`.

| Método | Validação: 5 vídeos | Teste: 5 vídeos | F1 macro teste |
|---|---:|---:|---:|
| `pose_face` | 5/5 (100%) | **3/5 (60%)** | 0,467 |
| `full` | 5/5 (100%) | 2/5 (40%) | 0,233 |
| `pose_face` → `full`, top 3 | 5/5 (100%) | **3/5 (60%)** | 0,467 |
| `pose_face` → `full`, top 5 | 5/5 (100%) | **3/5 (60%)** | 0,467 |
| `full` → `pose_face`, top 3 | 5/5 (100%) | 2/5 (40%) | 0,233 |
| `full` → `pose_face`, top 5 | 5/5 (100%) | **3/5 (60%)** | 0,467 |

Na fusão foi mantido o peso previamente escolhido de **0,65 × P(pose_face) + 0,35 × P(full)**. Com cinco classes, top 5 inclui todas; logo as duas ordens top 5 devem ter a mesma previsão. O top 3 iniciado por `full` excluiu a classe verdadeira em um dos cinco vídeos de teste. A fusão não melhorou o melhor modelo individual neste piloto.

No teste, `pose_face` e a fusão top 5 acertaram **Ajudar, Amigo e Água**. Erraram **Aprender** (previram Amigo) e **Beber** (previram Água). `full` também errou Amigo (previu Ajudar). São exatamente cinco amostras: diferenças de um acerto produzem saltos de 20 pontos percentuais. A validação 5/5 não representa garantia de generalização nem justifica substituir o modelo ativo na aplicação. O conjunto de teste só foi examinado após os checkpoints e pesos da fusão estarem fixados; os resultados individuais do teste foram gerados automaticamente pelo script de treino antes da fusão, sem ajuste posterior.

### Arquivos salvos

- Pesos e pacote de reprodução: `Modelos salvos/transformer_novas_5_pose_face/` e `Modelos salvos/transformer_novas_5_full/`, cada um com `best.pt`, YAML, split, métricas, histórico, log, configuração efetiva e SHA-256.
- Landmarks e CSV: `data/landmarks/novas_5_palavras/`; divisões: `data/processed/novas_5_palavras/`.
- Métricas, matriz de confusão e predições por vídeo das quatro avaliações: `experiments/fusion_novas_5_palavras/`.
- Código da nova entrada de dados: `ml/data/folder_signer.py`, opção `--dataset-format folder_signer` em `scripts/extract_landmarks.py`; `tests/test_folder_signer.py` passou (2 testes).

### Reproduzir no Git Bash do VS Code

Na raiz do repositório, com Python 3.11, dependências instaladas e os vídeos em `data/raw/<classe>/`:

**Ambiente desta execução:** o `.venv/Scripts/python.exe` local não inicia porque `pyvenv.cfg` aponta para um Python 3.11 removido em `C:\Users\migue\AppData\Local\Programs\Python\Python311`. Para concluir este experimento, foi usado o Python 3.11 portátil em `C:\Users\migue\AppData\Local\Temp\sinaliza_python311_embed\python.exe` com os pacotes da `.venv` existente; PyTorch 2.14.0+cu130 executou em CUDA. Os comandos abaixo são os de reprodução após reparar ou recriar a `.venv` com Python 3.11. O Python temporário não integra o projeto e pode desaparecer.

```bash
./.venv/Scripts/python.exe scripts/extract_landmarks.py \
  --input data/raw --output data/landmarks/novas_5_palavras \
  --dataset-format folder_signer --include-classes Água Ajudar Amigo Aprender Beber --workers 2

./.venv/Scripts/python.exe scripts/build_dataset.py \
  --landmarks data/landmarks/novas_5_palavras \
  --annotations data/landmarks/novas_5_palavras/annotations.csv \
  --output data/processed/novas_5_palavras --strategy signer_holdout \
  --class-list configs/novas_5_palavras.json \
  --train-signers Sinalizador01 Sinalizador02 Sinalizador03 Sinalizador04 Sinalizador05 \
  --val-signer Sinalizador06 --test-signer Sinalizador07

./.venv/Scripts/python.exe scripts/train.py \
  --config configs/transformer_minds_pose_face.yaml \
  --data-dir data/landmarks/novas_5_palavras \
  --processed-dir data/processed/novas_5_palavras \
  --experiment transformer_novas_5_pose_face_seed42 --num-workers 0

./.venv/Scripts/python.exe scripts/train.py \
  --config configs/transformer_minds_full.yaml \
  --data-dir data/landmarks/novas_5_palavras \
  --processed-dir data/processed/novas_5_palavras \
  --experiment transformer_novas_5_full_seed42 --num-workers 0

./.venv/Scripts/python.exe scripts/evaluate_two_stage_fusion.py \
  --pose-checkpoint experiments/transformer_novas_5_pose_face_seed42/best.pt \
  --full-checkpoint experiments/transformer_novas_5_full_seed42/best.pt \
  --pose-split-dir data/processed/novas_5_palavras \
  --full-split-dir data/processed/novas_5_palavras \
  --data-dir data/landmarks/novas_5_palavras \
  --split test --first pose_face --top-k 3 5 --pose-weight 0.65 \
  --output-dir experiments/fusion_novas_5_palavras
```

Para repetir validação ou ordem inversa, mude `--split val` ou `--first full`.
