# Matrizes de confusao para compartilhar

## Plano

1. Reusar as metricas ja salvas do Transformer `pose_face` de 8 classes: validacao (Sinalizador11) e teste (Sinalizador12). Conferir cada matriz com o CSV de previsoes por video.
2. Atualizar o gerador para mostrar no titulo o dataset, o numero de classes e o split em cada matriz. Regenerar a matriz de validacao de 20 classes com titulo explicito.
3. Exportar JPG em resolucao legivel para WhatsApp, preservando os SVG e os CSVs para inspecao detalhada.
4. Registrar os comandos e os resultados sem retrainar ou alterar os checkpoints.

## Criterio de leitura

Linhas = sinal verdadeiro; colunas = previsao do modelo. As celulas na diagonal sao acertos. Cada classe possui cinco videos em cada split usado aqui. As matrizes de 8 e 20 classes pertencem a experimentos diferentes; a comparacao de porcentagens isoladas nao mede so o efeito da quantidade de classes.

## Execucao em 13/10/2026

Na raiz do projeto, no Git Bash do VS Code:

```bash
node scripts/plot_confusion_matrix.mjs
node scripts/plot_confusion_matrix.mjs \
  --metrics "Modelos salvos/transformer_minds_8_classes_pose_face/records/val_metrics.json" \
  --output-dir reports/transformer_minds_8_classes_pose_face/validation
node scripts/plot_confusion_matrix.mjs \
  --metrics "Modelos salvos/transformer_minds_8_classes_pose_face/records/test_metrics.json" \
  --output-dir reports/transformer_minds_8_classes_pose_face/test

powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/export_confusion_jpg.ps1 \
  -SvgPath reports/transformer_minds_20_classes_pose_face/validation/confusion_matrix.svg
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/export_confusion_jpg.ps1 \
  -SvgPath reports/transformer_minds_8_classes_pose_face/validation/confusion_matrix.svg
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/export_confusion_jpg.ps1 \
  -SvgPath reports/transformer_minds_8_classes_pose_face/test/confusion_matrix.svg
```

O exportador JPG usa Microsoft Edge em modo headless e o codificador JPEG do Windows. O script Node continua independente do Python; lê somente os registros de avaliação e verifica todas as previsões por vídeo contra a matriz correspondente.

| Matriz | Vídeos | Acertos | JPG |
|---|---:|---:|---|
| 20 classes, validação | 100 | 37 (37%) | `reports/transformer_minds_20_classes_pose_face/validation/confusion_matrix.jpg` |
| 8 classes, validação | 40 | 40 (100%) | `reports/transformer_minds_8_classes_pose_face/validation/confusion_matrix.jpg` |
| 8 classes, teste | 40 | 37 (92,5%) | `reports/transformer_minds_8_classes_pose_face/test/confusion_matrix.jpg` |

As três imagens foram abertas para inspeção visual: títulos, eixos, valores e layout estão legíveis. O teste de 8 classes já existia nos registros, não foi reexecutado. Nele os três erros foram Medo → Aluno (1) e Vontade → Conhecer (2). O teste de 20 classes continua não avaliado. Os JPGs têm aproximadamente 1600 a 1900 pixels de lado e preservam os títulos que distinguem classe e split.
