# Matriz de confusao do Transformer MINDS-Libras de 20 classes

## Plano

1. Usar `val_metrics.json` do checkpoint `best.pt` arquivado, com 100 videos do Sinalizador11 e cinco exemplos por classe.
2. Gerar uma matriz visual com linhas = classe real e colunas = classe prevista, uma tabela de acertos/erros por classe e os pares de confusao mais frequentes.
3. Conferir as somas da matriz e comparar a acuracia calculada com `top1` do arquivo original.
4. Manter o teste Sinalizador12 reservado. O resultado da validacao serve para diagnostico, nao mede o desempenho em pessoas novas.

## Fonte

`Modelos salvos/transformer_minds_20_classes_pose_face/records/val_metrics.json`. O checkpoint foi selecionado pela menor perda de validacao na epoca 35. A matriz ja foi calculada por `scripts/audit_checkpoint_val.py`; este trabalho apenas a visualiza e resume. Os nomes e indices das classes estao no campo `classes` do proprio JSON.

## Execucao e resultados em 13/10/2026

Da raiz do projeto, no Git Bash do VS Code:

```bash
node scripts/plot_confusion_matrix.mjs
```

O script usa apenas Node.js e escreve em `reports/transformer_minds_20_classes_pose_face/validation/`: `confusion_matrix.svg` (visual), `confusion_matrix.csv` (valores), `per_class.csv` (acertos por classe) e `report.md` (resumo). Aceita `--metrics` e `--output-dir` para outros experimentos produzidos por `audit_checkpoint_val.py`.

A matriz tem 100 videos e 37 acertos, igual ao `top1=0,37` do JSON de origem. Onze classes tiveram zero acertos em cinco videos. Confusoes com cinco casos: Acontecer → America, Amarelo → Maca, Banheiro → America, Espelho → Barulho, Esquina → Sapo, Filho → Medo, Medo → Filho e Sapo → America. O relatorio lista todas as classes e os demais pares. Cada caso isolado altera a acuracia da classe em 20 pontos percentuais, portanto os percentuais por classe sao instaveis.

O Python da `.venv` nao estava disponivel nesta sessao; por isso nao houve nova inferencia. O script valida dimensoes, total e acuracia da matriz salva antes de criar os arquivos. O split de teste permaneceu reservado.

## Conferencia adicional em 13/10/2026

Depois de uma duvida sobre o grafico, o script passou a reconstruir a matriz a partir das 100 linhas de `val_predictions.csv` e exigir igualdade exata com `val_metrics.json`. A conferencia passou: 100 previsoes, 37 acertos e todas as 400 celulas iguais. O calculo original em `ml/training/metrics.py` soma `cm[true, pred]`, portanto linhas representam classes reais e colunas representam previsoes. Exemplos conferidos: os cinco videos reais de Banheiro foram previstos como America; os cinco de Filho como Medo; os cinco de Medo como Filho. Isso confirma a leitura dos artefatos salvos, mas nao constitui uma nova inferencia nem valida a qualidade dos rotulos originais.

Em 13/10/2026, a matriz foi regenerada com o titulo `Transformer MINDS-Libras — 20 classes — Validação` e exportada em JPG para compartilhamento. Os comandos e as matrizes equivalentes do modelo de 8 classes estao em [PLANO_MATRIZES_COMPARTILHAVEIS.md](PLANO_MATRIZES_COMPARTILHAVEIS.md).
