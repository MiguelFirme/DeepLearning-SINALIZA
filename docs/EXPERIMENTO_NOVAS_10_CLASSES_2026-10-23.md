# Experimento: um Transformer para as dez palavras novas

## Plano antes do treino

1. Unir os 35 `.npz` de `novas_5_palavras` aos 35 de `novas_5_restantes` em um manifesto de dez classes, com referências relativas aos arquivos originais para não duplicar dados. Conferir classe, sinalizador, caminho e contagem de 70 vídeos.
2. Preservar o protocolo por pessoa: Sinalizador01–05 no treino (50 vídeos), 06 na validação (10) e 07 no teste (10). Cada classe terá 5/1/1 vídeos. Verificar ausência de vazamento.
3. Treinar **um** Transformer do zero com `feature_mode: full` (pose, face e mãos), 48 quadros, mesma arquitetura e técnicas dos modelos anteriores; como as classes estão equilibradas, usar CrossEntropy sem pesos. Selecionar `best.pt` pela menor `val_loss`, com early stopping e scheduler. Não consultar o teste durante a escolha do checkpoint.
4. Avaliar o checkpoint fixado nos dez vídeos de teste, salvar previsões por classe, F1 macro, histórico, logs, configuração efetiva e checkpoint em `Modelos salvos/`. Documentar comandos e limites. Manter o backend atual até haver avaliação mais robusta.

## Limite previsto

Há apenas um vídeo de validação e um de teste por palavra. Um acerto altera a acurácia em dez pontos percentuais. Os sinalizadores já apareceram em experimentos exploratórios anteriores, então o teste não é uma confirmação independente de desempenho em produção.

## Resultados

Executado em 23/10/2026. `scripts/build_new10_manifest.py` reuniu **70 vídeos de 10 classes e 7 sinalizadores** sem duplicar `.npz`. O `build_dataset.py` confirmou ausência de vazamento e a divisão 50/10/10. Os arquivos de origem permanecem em `data/landmarks/novas_5_palavras/` e `data/landmarks/novas_5_restantes/`; o manifesto combinado em `data/landmarks/novas_10/` usa caminhos relativos `../` para acessá-los.

O Transformer `full` teve **121.148 parâmetros**, foi treinado com CUDA e a configuração `configs/transformer_minds_full.yaml`: 2 camadas, dimensão 64, 4 cabeças, dropout 0,45, Adam com `weight_decay=1e-4`, CrossEntropy sem pesos, augmentation leve apenas no treino, `ReduceLROnPlateau` e early stopping por `val_loss` com paciência 15. Treinou 116 épocas; o melhor checkpoint foi o da época **101**, com `val_loss=0,8430`. O teste foi avaliado somente depois de fixar o checkpoint (`--skip-test` no treino).

| Split | Acertos | Acurácia | F1 macro |
|---|---:|---:|---:|
| Validação — Sinalizador06 | 6/10 | 60% | 0,533 |
| Teste exploratório — Sinalizador07 | 6/10 | 60% | 0,550 |

No teste, acertou **Ajudar, Amigo, Brincar, Cansado, Certo e Diretor**. Confundiu Aprender → Comer, Comer → Aprender, Beber → Diretor e Água → Diretor. Os arquivos `test_predictions.csv` e `val_predictions.csv` registram a previsão e o escore de cada vídeo. O valor softmax indicado como confiança não é uma probabilidade calibrada de acerto.

O resultado de 6/10 é superior aos 4/10 do Transformer `full` de 25 classes nos mesmos dez vídeos novos do Sinalizador07, mas os modelos classificam vocabulários diferentes e essa comparação é exploratória. Dez vídeos de teste e nenhum participante externo ao conjunto não sustentam uma alegação de desempenho em produção. O backend atual não foi alterado.

### Arquivos

- Checkpoint e pacote: `Modelos salvos/transformer_novas_10_full/` (`best.pt`, SHA-256, configuração, split, métricas, histórico, previsões e log).
- Execução original: `experiments/transformer_novas_10_full_seed42/` e `logs/experiments/transformer_novas_10_full_seed42/`.
- Manifesto combinado: `data/landmarks/novas_10/`; splits e rótulos: `data/processed/novas_10/`.

### Reproduzir no Git Bash do VS Code

Na raiz do projeto, com Python 3.11 e dependências instaladas:

```bash
./.venv/Scripts/python.exe scripts/build_new10_manifest.py

./.venv/Scripts/python.exe scripts/build_dataset.py \
  --landmarks data/landmarks/novas_10 \
  --annotations data/landmarks/novas_10/annotations.csv \
  --output data/processed/novas_10 --strategy signer_holdout \
  --class-list configs/novas_10_palavras.json \
  --train-signers Sinalizador01 Sinalizador02 Sinalizador03 Sinalizador04 Sinalizador05 \
  --val-signer Sinalizador06 --test-signer Sinalizador07

./.venv/Scripts/python.exe scripts/train.py \
  --config configs/transformer_minds_full.yaml \
  --data-dir data/landmarks/novas_10 --processed-dir data/processed/novas_10 \
  --experiment transformer_novas_10_full_seed42 --num-workers 0 --skip-test

for split in val test; do
  ./.venv/Scripts/python.exe scripts/audit_checkpoint_val.py \
    --checkpoint experiments/transformer_novas_10_full_seed42/best.pt \
    --data-dir data/landmarks/novas_10 --processed-dir data/processed/novas_10 \
    --split "$split" --output-dir experiments/transformer_novas_10_full_seed42
done
```

Nesta máquina, `.venv/Scripts/python.exe` ainda aponta para uma instalação removida de Python 3.11.9. Esta execução usou o Python 3.11 portátil temporário em `C:\Users\migue\AppData\Local\Temp\sinaliza_python311_embed\python.exe` com os pacotes da `.venv`. Repare ou recrie a `.venv` para usar os comandos acima.
