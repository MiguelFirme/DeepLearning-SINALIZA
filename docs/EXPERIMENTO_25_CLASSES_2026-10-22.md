# Experimento: 15 classes anteriores + 10 palavras novas

## Plano antes da execução

1. Preservar as 15 classes e os índices de treino/validação/teste do experimento `minds_libras_15_sem_cinco`. Acrescentar as 10 pastas de `data/raw/`: Água, Ajudar, Amigo, Aprender, Beber, Brincar, Cansado, Certo, Comer e Diretor. Usar os `.npz` já extraídos das primeiras cinco e extrair apenas as cinco restantes.
2. Criar um manifesto misto sem alterar os vídeos nem os landmarks de origem. Prefixar caminhos e IDs de pessoas com a origem (`minds:` e `novas:`) para que `Sinalizador06` dos dois datasets não seja tratado como a mesma pessoa. Nas classes novas, `01` a `05` serão treino, `06` validação e `07` teste. O conjunto esperado tem 670 vídeos: 500 de treino, 85 de validação e 85 de teste. Verificar 25 classes nos três conjuntos, duplicações e vazamento entre splits.
3. Treinar dois Transformers **do zero**, com os mesmos 48 quadros, arquitetura, augmentation, otimizador, scheduler, early stopping e semente da comparação anterior. A cabeça de classificação passa de 15 para 25 classes. Comparar `pose_face` e `full`. Para compensar o desbalanceamento de 30 contra 5 vídeos de treino por classe, usar `CrossEntropyLoss` com pesos `sqrt_inverse` calculados apenas no treino; isso é uma mudança explícita em relação ao treino anterior.
4. Selecionar os checkpoints pela menor `val_loss`, medir acurácia e F1 macro gerais e separadamente para as 15 antigas e as 10 novas. Repetir a fusão com regra fixa de 0,65 para `pose_face` e 0,35 para `full`, top 3 e top 5, nas duas ordens. A validação orienta a interpretação; o teste é exploratório e não ajustará pesos, arquitetura ou ordem.
5. Salvar os dois `best.pt`, hashes, YAML, manifesto, splits, histórico, logs, resultados por vídeo e comandos de reprodução em `Modelos salvos/` e neste documento. O backend da aplicação permanece no checkpoint ativo até haver evidência para trocá-lo.

## Riscos previstos

As classes novas têm um único vídeo por classe na validação e no teste, enquanto as antigas têm cinco. Um resultado agregado pode esconder desempenho ruim nas dez novas. Os sinalizadores de teste já foram consultados em experimentos anteriores; números de teste são exploratórios. A detecção das mãos varia muito entre pessoas nos vídeos novos.

## Resultados

Executado em 22/10/2026. Os landmarks das cinco palavras restantes já estavam extraídos ao iniciar esta rodada; o extrator confirmou **35 arquivos aproveitados, 0 erros**. O script `scripts/build_mixed_25_dataset.py` gerou 670 entradas e verificou 500/85/85, 25 classes nos três splits, contagens exatas por classe e isolamento dos IDs de pessoas com prefixo de origem. O manifesto em `data/landmarks/mixed_25/manifest.json` aponta para os `.npz` das três fontes por caminhos relativos `../`; não há segunda cópia dos landmarks.

Ambos os modelos tiveram **122.123 parâmetros** e usaram CUDA. O `pose_face` parou na época 71, com melhor `val_loss=1,8533` na época 56. O `full` parou na época 51, com melhor `val_loss=1,7469` na época 36. `--skip-test` preservou o teste até a fusão. As demais configurações foram iguais, inclusive `class_weight_method: sqrt_inverse` e semente 42; somente `feature_mode` mudou. Logs completos estão em `logs/experiments/transformer_mixed25_{pose_face,full}_seed42/`.

| Método | Validação: 85 vídeos | F1 macro val | Teste exploratório: 85 vídeos | F1 macro teste |
|---|---:|---:|---:|---:|
| `pose_face` | 37/85 (43,5%) | 0,267 | 27/85 (31,8%) | 0,188 |
| **`full`** | 46/85 (54,1%) | **0,422** | **48/85 (56,5%)** | 0,438 |
| `pose_face` → `full`, top 3 | 43/85 (50,6%) | 0,362 | 39/85 (45,9%) | 0,373 |
| `pose_face` → `full`, top 5 | 47/85 (55,3%) | 0,396 | 42/85 (49,4%) | 0,399 |
| `full` → `pose_face`, top 3 | **48/85 (56,5%)** | 0,397 | 45/85 (52,9%) | **0,439** |
| `full` → `pose_face`, top 5 | 47/85 (55,3%) | 0,397 | 45/85 (52,9%) | 0,432 |

Todos os resultados da fusão usam o peso pré-definido de **0,65 para `pose_face` e 0,35 para `full`**. A ordem `full` → `pose_face` top 3 deu mais acertos na validação, mas o F1 macro do `full` isolado foi maior. No teste, `full` isolado acertou mais vídeos que qualquer fusão. A fusão anterior de 15 classes não se transferiu como ganho para este vocabulário de 25.

### Desempenho por origem

| Método | Validação antigas (75) | Validação novas (10) | Teste antigas (75) | Teste novas (10) |
|---|---:|---:|---:|---:|
| `pose_face` | 36 | 1 | 25 | 2 |
| **`full`** | 42 | **4** | **44** | 4 |
| `pose_face` → `full`, top 5 | 44 | 3 | 37 | **5** |
| `full` → `pose_face`, top 3 | **45** | 3 | 40 | **5** |

No teste das dez palavras novas, o `full` reconheceu Ajudar, Brincar, Certo e Comer. A fusão `full` → `pose_face` top 3 acertou Ajudar, Aprender, Certo, Comer e Diretor. Amigo, Beber, Cansado e Água ficaram erradas nessa fusão. O arquivo `experiments/fusion_mixed25/subgroup_summary.json` contém as previsões de cada uma das dez palavras e todas as combinações. Cada acerto nesse subconjunto vale 10 pontos percentuais.

Comparado ao checkpoint anterior de 15 classes, o novo `pose_face` perdeu acertos nas 15 antigas (49/75 → 36/75 na validação; 41/75 → 25/75 no teste). O novo `full` teve 42/75 contra 45/75 na validação e 44/75 contra 34/75 no teste. Esses números pertencem a treinos diferentes e a validação é pequena. **O experimento não justifica trocar automaticamente o backend para 25 classes**: a generalização das dez novas ainda foi de 4/10 no melhor modelo isolado e 5/10 na fusão, em um único sinalizador por conjunto. Há indícios de dependência da qualidade de landmarks e de escassez de vídeos novos. Mais pessoas e tomadas por palavra são o próximo dado necessário para uma avaliação confiável.

### Artefatos e reprodução

- Checkpoints arquivados em `Modelos salvos/transformer_mixed25_pose_face/` e `Modelos salvos/transformer_mixed25_full/`, com YAML, split, histórico, `run_record.json`, logs e hashes SHA-256.
- Predições e métricas das duas ordens em `experiments/fusion_mixed25/`; sumário por origem em `subgroup_summary.json`.
- `.npz` originais em `data/landmarks/minds_libras_all20/`, `novas_5_palavras/` e `novas_5_restantes/`. Eles são necessários para reproduzir inferência ou treino e são ignorados pelo Git.

Com Python 3.11 e dependências instaladas, na raiz do projeto:

```bash
./.venv/Scripts/python.exe scripts/build_mixed_25_dataset.py

./.venv/Scripts/python.exe scripts/train.py \
  --config configs/transformer_mixed25_pose_face.yaml \
  --data-dir data/landmarks/mixed_25 --processed-dir data/processed/mixed_25 \
  --experiment transformer_mixed25_pose_face_seed42 --num-workers 0 --skip-test

./.venv/Scripts/python.exe scripts/train.py \
  --config configs/transformer_mixed25_full.yaml \
  --data-dir data/landmarks/mixed_25 --processed-dir data/processed/mixed_25 \
  --experiment transformer_mixed25_full_seed42 --num-workers 0 --skip-test

for split in val test; do
  for first in pose_face full; do
    ./.venv/Scripts/python.exe scripts/evaluate_two_stage_fusion.py \
      --pose-checkpoint experiments/transformer_mixed25_pose_face_seed42/best.pt \
      --full-checkpoint experiments/transformer_mixed25_full_seed42/best.pt \
      --pose-split-dir data/processed/mixed_25 --full-split-dir data/processed/mixed_25 \
      --data-dir data/landmarks/mixed_25 --split "$split" --first "$first" \
      --top-k 3 5 --pose-weight 0.65 --output-dir experiments/fusion_mixed25
  done
done
./.venv/Scripts/python.exe scripts/summarize_mixed25_results.py
```

O `.venv` local ainda aponta para uma instalação removida de Python 3.11.9. Nesta máquina, a execução usou o Python 3.11 portátil temporário em `C:\Users\migue\AppData\Local\Temp\sinaliza_python311_embed\python.exe` com os pacotes da `.venv`; para repetir pelos comandos acima, recrie ou repare a `.venv`. O backend ativo não foi modificado.
