# Experimento: sete variações de landmarks por classe da categoria `escola`

## Plano — 2026-09-30

1. Usar as 23 classes de `ESSENTIAL_BY_CATEGORY["escola"]` com três arquivos
   originais por classe. Criar uma saída isolada com cópias intactas dos 69
   arquivos originais e 161 arquivos sintéticos: 10 por classe, 230 no total.
2. Distribuir as sete variações por classe entre os três articuladores e
   registrar o arquivo de origem de cada uma. Aplicar apenas deslocamento,
   escala e rotação muito pequenos às coordenadas X/Y da sequência inteira.
   Preservar exatamente ordem/quantidade de frames, configuração das mãos,
   valores Z, visibilidade e máscaras. Não espelhar, cortar, descartar frames
   nem combinar sinais de pessoas diferentes.
3. Rejeitar variações que excedam o limite de deslocamento por coordenada.
   Gerar com seed fixa para reprodutibilidade; validar os arquivos e contagens.
4. Adaptar o split por articulador para usar sintéticos somente no treino.
   No teste, manter exclusivamente o arquivo real do articulador reservado.
   Impedir que origem e derivação atravessem treino e teste.
5. Documentar os parâmetros exatos e a inspeção. Uma perturbação numérica
   pequena reduz o risco de mudar o sinal, mas não comprova equivalência
   linguística; revisar visualmente reconstruções/amostras antes de produção.

## Resultado

Gerador: `scripts/generate_landmark_variants.py`, com transformação em
`ml/data/offline_variants.py`. Saída:
`data/augmented/escola_10/` (ignorada pelo Git, pois contém o dataset).
`augmentation_report.json` registra seed, classes, contagens e deslocamento
máximo; `manifest.json` marca cada sintético com `is_synthetic: true` e
`source_path`.

Comando usado no Git Bash do VS Code:

```bash
./.venv/Scripts/python.exe scripts/generate_landmark_variants.py \
  --source data/landmarks \
  --output data/augmented/escola_10 \
  --annotations data/annotations.csv \
  --categories escola --seed 42
```

Resultado: 23 classes, 69 arquivos originais copiados sem alterações e 161
sintéticos, totalizando **230 NPZ, exatamente 10 por classe**. Em cada classe,
o gerador aloca as sete variações como 3+2+2 entre os três originais; o
articulador que recebe a terceira variação alterna conforme a classe.

Parâmetros da transformação (`subtle_spatial_v1`): translação X/Y uniforme
em ±0,004; escala uniforme de 0,997 a 1,003; rotação uniforme em ±0,75°;
viés fixo por ponto em ±0,0005. O limite absoluto de deslocamento de qualquer
coordenada X/Y é 0,015; candidatos que o excedem são rejeitados. O máximo
observado nos 161 arquivos foi **0,014871**. Não há espelhamento, inversão
de mão, corte temporal, alteração de velocidade nem combinação de pessoas.
Frames, coordenadas Z, visibilidade e máscaras permanecem iguais às origens.

Auditoria de todos os arquivos confirmou: 23 classes com 10 arquivos cada,
69 cópias idênticas byte a byte, 161 arquivos sintéticos legíveis, dimensões
e máscaras preservadas e limite espacial respeitado. A suíte completa passou
com **82 testes**. Houve uma primeira falha de leitura porque o extrator salva
`metadata` como texto JSON; o gerador foi corrigido e a saída parcial isolada
foi removida antes da geração final.

O split de verificação foi preparado sob
`data/processed/signer_cv/bilstm_focus_signer_cv_escola_23classes_10files_seed42/`.
Seus três holdouts têm respectivamente 153, 153 e 154 amostras de treino,
e **23 vídeos reais de teste cada**. As variações do articulador reservado
não entram em nenhum split da rodada. O construtor valida que `source_path`
aponta para um original da mesma classe e pessoa; o split separa por `signer`.

Para treinar e testar depois, sem misturar o resultado anterior de 23 classes:

```bash
./.venv/Scripts/python.exe scripts/run_signer_cv.py \
  --config configs/bilstm_focus.yaml \
  --data-dir data/augmented/escola_10 \
  --essential-categories escola --max-classes 23 \
  --epochs 120 --num-workers 0 \
  --run-name bilstm_focus_signer_cv_escola_23classes_10files_seed42
```

Este passo gerou e validou os dados e splits; **não treinou novos modelos**.
As sete variações são derivadas de apenas três execuções reais por classe,
portanto não equivalem a sete novos vídeos nem a novos articuladores.
As mudanças numéricas pequenas preservam a trajetória e a configuração
observada, mas a equivalência linguística final requer revisão humana de
visualizações ou vídeos correspondentes. A normalização do pipeline pode
remover parte das variações globais; só um novo treino holdout medirá o efeito
na generalização.

## Treino posterior e comparação — 2026-09-30

O usuário iniciou o comando de treino acima no Git Bash. As três rodadas
terminaram, cada uma com 120 épocas. Métricas em
`experiments/bilstm_focus_signer_cv_escola_23classes_10files_seed42/summary.json`.
Cada teste continuou com 23 vídeos reais da pessoa reservada.

| Articulador de teste | Antes: 3 arquivos/classe | Agora: 10 arquivos/classe | Top-3 agora | F1 macro agora |
| --- | ---: | ---: | ---: | ---: |
| Articulador1 | 1/23 (4,35%) | 2/23 (8,70%) | 17,39% | 4,35% |
| Articulador2 | 4/23 (17,39%) | 5/23 (21,74%) | 30,43% | 14,86% |
| Articulador3 | 1/23 (4,35%) | 3/23 (13,04%) | 26,09% | 8,99% |
| Média | 6/69 (8,70%) | **10/69 (14,49%)** | **24,64%** | **9,40%** |

Ganho top-1: quatro acertos em 69 testes, ou +5,80 pontos percentuais.
O top-3 médio permaneceu em 24,64%; o F1 macro médio passou de 5,00% para
9,40%. A última época registrou 100% de acerto no treino em todas as rodadas,
portanto o desnível entre treino e pessoa não vista ainda é grande. Com um
vídeo real por classe e articulador no teste, cada acerto muda a taxa de uma
rodada em 4,35 pontos percentuais. O resultado sugere benefício neste recorte,
mas ainda não demonstra robustez para produção ou para outras pessoas.

Os três subprocessos retornaram o código nativo do Windows `3221226505`
depois de salvar `best.pt`, `history.json` e `test_metrics.json`; o executor
verificou esses artefatos e concluiu as três rodadas. Nenhum teste sintético
entrou nas métricas finais.
