# Experimento: 100 sinais de necessidades na escola

## Plano — 2026-09-30

Objetivo: testar um vocabulário menor e útil para pedidos, localização,
saúde, urgência e recepção escolar. O usuário priorizou necessidades do dia a
dia, sem concentrar o recorte em conteúdos de ensino.

1. Manter um arquivo de 100 classes **exatas** do `annotations.csv`, em cinco
   grupos de 20. Verificar presença no CSV, cobertura dos três articuladores,
   duplicatas e contagem antes de gerar splits.
2. Acrescentar ao construtor de dataset a opção `--class-list`, isolada da
   política geral de 398 classes. Neste experimento específico, o limite de
   100 classes tem prioridade sobre a regra geral de manter todos os verbos;
   a política padrão permanece intacta.
3. Treinar o Bi-LSTM de `configs/bilstm_focus.yaml`, sem aumentação, com
   Articulador1, 2 e 3 reservados alternadamente. Usar a mesma lista nas três
   rodadas, sem dados do articulador de teste no treino.
4. Registrar top-1, top-3, F1 macro, distribuições por classe e eventuais
   erros de execução. Comparar com os resultados de 398 classes apenas como
   contexto, pois a dificuldade da tarefa mudou.

## Lacunas do CSV

`Preciso`, `Quero`, `Vou` e `Diretoria` não aparecem como rótulos exatos.
`Necessário`, `Quer`, `Vai`/`Ir` são rótulos existentes, mas não equivalem
gramaticalmente às frases pedidas. Não serão renomeados; novos vídeos
anotados seriam necessários para acrescentar os sinais ausentes.

## Resultado

Lista: `configs/vocabulary_school_needs_100.json` (cinco grupos de 20 classes).
São 100 rótulos distintos, presentes no CSV e nos landmarks, com um vídeo de
cada articulador por classe. O construtor aceita `--class-list` como seleção
explícita; sem essa opção, a política geral de verbos e termos essenciais
continua ativa. O teste automatizado completo passou: 79 testes.

Comando para repetir no Git Bash do VS Code:

```bash
./.venv/Scripts/python.exe scripts/run_signer_cv.py \
  --config configs/bilstm_focus.yaml \
  --class-list configs/vocabulary_school_needs_100.json \
  --max-classes 100 --epochs 120 --num-workers 0
```

O comando cria três holdouts. Em cada um: 200 amostras de treino dos outros
dois articuladores; 100 amostras de teste do articulador reservado; zero de
validação. O checkpoint é escolhido por perda de treino, sem consultar o teste.
Métricas e pesos ficam em
`experiments/bilstm_focus_signer_cv_100classes_120epochs_seed42/`.

## Métricas finais — 2026-09-30

| Articulador de teste | Top-1 | Top-3 | F1 macro |
| --- | ---: | ---: | ---: |
| Articulador1 | 7% | 15% | 3,80% |
| Articulador2 | 12% | 22% | 8,57% |
| Articulador3 | 7% | 15% | 5,57% |
| Média | **8,67%** | **17,33%** | **5,98%** |

Cada avaliação usou exatamente 100 vídeos, um por classe, todos de uma pessoa
ausente do treino. Acerto aleatório top-1 em 100 classes equilibradas: 1%.
O treino chegou a aproximadamente 95–99% de acerto nas épocas finais, enquanto
o teste ficou entre 7% e 12%: há forte diferença entre memorizar os exemplos
disponíveis e generalizar a outro articulador. Reduzir para 100 classes, por si
só, não alcançou desempenho de produção. A seleção da melhor época pela perda
de treino decorre da ausência de um terceiro conjunto independente de
validação. A tarefa classifica sinais isolados; ainda não mede tradução de
frases para texto natural.

`summary.json` registra as médias; cada subpasta `articuladorN` contém
`test_metrics.json`, `history.json` e `best.pt`. Os três processos de treino
retornaram código nativo do Windows `3221226505` após gravar esses artefatos;
o executor verificou sua presença, preservou as métricas e concluiu a rodada.

Próximo experimento recomendado: coletar mais pessoas e mais de uma tomada
por classe, reservar pessoas também para validação e medir uma linha de base
com normalização dos landmarks relativa ao corpo. Só depois comparar aumento
de dados, loss e arquitetura no mesmo protocolo de holdout.
