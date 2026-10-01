# Experimento: categoria `escola`

## Plano — 2026-09-30

1. Contar os itens definidos em `ESSENTIAL_BY_CATEGORY["escola"]` e cruzá-los
   com os rótulos reais de `data/annotations.csv` e o manifesto de landmarks.
2. Permitir a seleção de uma ou mais categorias no construtor do dataset,
   sempre por rótulos presentes no CSV. Manter o modo padrão e a lista curada
   de 100 classes independentes desse modo.
3. Rodar três holdouts por articulador com as classes disponíveis da categoria
   `escola`, `configs/bilstm_focus.yaml`, 120 épocas e seed 42. Não usar o
   articulador de teste na seleção do checkpoint.
4. Registrar top-1, top-3 e F1 macro por articulador, além de média e
   limitações. Comparar apenas com cautela ao experimento de 100 classes.

## Inventário prévio

- 34 entradas literais no código; 33 distintas após normalizar acentos.
- 23 classes presentes no CSV, todas com vídeo dos três articuladores.
- 10 termos normalizados sem vídeo: `caderno`, `caneta`, `carteira`,
  `colega`, `diretoria`, `estudante`, `professor`, `professora`, `quadro`, `sala`.
- `secretaria` e `secretária` coincidem após normalização; há no CSV o rótulo
  `Secretária`.

Inventário das outras categorias para futuros recortes (entradas do código /
classes encontradas no CSV): `casa` 23/20,
`necessidades_saude_emergencia` 30/23, `interacao` 26/19 e `recepcao` 11/10.
As categorias podem compartilhar rótulos; suas contagens não devem ser somadas
para prever o tamanho de uma união.

## Resultado

O modo `--essential-categories escola` selecionou 23 rótulos do CSV em tempo
de execução. Cada holdout teve 46 vídeos de treino (dois articuladores) e 23
de teste (articulador reservado), um vídeo por classe e pessoa. Não houve
vazamento entre os splits. Foram 120 épocas por holdout, Bi-LSTM sem aumento,
seed 42. A suíte do projeto passou com 81 testes.

Rótulos usados: `Aluno do Segundo Ano do Ensino Médio`, `Atividade`,
`Biblioteca`, `Borracha`, `Calculadora`, `Campainha`, `Classe`, `Colégio`,
`Computador`, `Dever de casa`, `Dicionário`, `Escola`, `Estudo`, `Faculdade`,
`Giz`, `Livro`, `Lápis`, `Matemática`, `Mochila`, `Papel`, `Secretária`,
`Secretário`, `Teste`.

| Articulador de teste | Acertos/23 | Top-1 | Top-3 | F1 macro |
| --- | ---: | ---: | ---: | ---: |
| Articulador1 | 1 | 4,35% | 17,39% | 1,09% |
| Articulador2 | 4 | 17,39% | 34,78% | 11,74% |
| Articulador3 | 1 | 4,35% | 21,74% | 2,17% |
| Média | 2 | **8,70%** | **24,64%** | **5,00%** |

A referência de acerto aleatório top-1 entre 23 classes é 4,35%. Acurácia de
treino próxima de 100% e acurácia de teste baixa indicam generalização fraca
entre pessoas. Com apenas dois vídeos de treino por classe e nenhuma pessoa
extra para validação, o checkpoint foi selecionado pela perda de treino.
O recorte de 23 classes difere do anterior de 100, então porcentagens entre
eles não medem uma melhoria direta. `Banheiro` está em `casa`, não em `escola`.
Para testar necessidades escolares, uma união de `escola`, `casa`,
`necessidades_saude_emergencia`, `interacao` e `recepcao` é mais próxima do
vocabulário desejado, ainda condicionada aos vídeos existentes.

Comando para repetir no Git Bash do VS Code:

```bash
./.venv/Scripts/python.exe scripts/run_signer_cv.py \
  --config configs/bilstm_focus.yaml \
  --essential-categories escola --max-classes 23 \
  --epochs 120 --num-workers 0
```

Os artefatos estão em
`experiments/bilstm_focus_signer_cv_escola_23classes_120epochs_seed42/`.
`summary.json` contém o agregado. Cada subpasta do articulador contém
`test_metrics.json`, `history.json` e `best.pt`. Os três processos de treino
retornaram código nativo do Windows `3221226505` após gravar os artefatos;
o executor verificou os arquivos, preservou os resultados e concluiu.
