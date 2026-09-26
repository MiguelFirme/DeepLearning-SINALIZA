# Guia de treinamento — Sinaliza

## Estratégia atual

O V-LIBRASIL possui aproximadamente três vídeos por classe, um de cada
articulador. O experimento recomendado treina com dois articuladores e testa no
terceiro, repetindo o processo três vezes:

| Rodada | Treino | Teste |
|---|---|---|
| 1 | Articulador2 + Articulador3 | Articulador1 |
| 2 | Articulador1 + Articulador3 | Articulador2 |
| 3 | Articulador1 + Articulador2 | Articulador3 |

Cada rodada começa com pesos novos. O articulador de teste não participa do
treino nem da escolha do checkpoint.

## Pré-requisitos

Execute os comandos a partir da raiz do projeto:

```text
C:\Users\migue\DeepLearning-SINALIZA
```

No Git Bash do VS Code, invoque o Python do projeto diretamente:

```bash
./.venv/Scripts/python.exe --version
./.venv/Scripts/python.exe -m pytest tests -q
```

O prompt `(.venv)` não prova que `python` aponta para esse ambiente. Confirme
com `which python` ou use sempre `./.venv/Scripts/python.exe`. Neste projeto,
`mediapipe==0.10.21` exige uma versão de Python com wheel compatível; o
ambiente verificado usa Python 3.11.9. A instalação global Python 3.14 não
possui as dependências deste repositório.

Os landmarks devem existir em `data/landmarks`, incluindo `manifest.json`.
Não é necessário executar o MediaPipe novamente.

## Filtrar o vocabulário e conferir os splits sem treinar

O arquivo `data/annotations.csv` é lido pelo código em cada preparação. A política
editável está em `ml/data/vocabulary.py`: infinitivos e locuções verbais,
exceções verbais e termos essenciais. `data/processed/.../vocabulary_report.json`
lista todas as classes incluídas e excluídas. Revise esse relatório antes de
um treino longo; identificar a função gramatical só pela grafia é uma
aproximação, e expressões ambíguas podem exigir inclusão manual.

Classes que não existem no CSV (por exemplo, alguns cômodos citados como
exemplo de escopo) não podem entrar no modelo sem novos vídeos.

```bash
./.venv/Scripts/python.exe scripts/run_signer_cv.py \
  --config configs/bilstm_focus.yaml \
  --annotations data/annotations.csv \
  --epochs 120 \
  --prepare-only
```

Isso cria três splits. Em cada rodada haverá:

- em geral, duas amostras de treino por classe;
- nenhuma validação independente;
- em geral, uma amostra de teste por classe;
- zero articuladores compartilhados entre treino e teste.

`Congelar` e `Solicitar` têm apenas dois articuladores: permanecem no
vocabulário, mas seu teste não estará completo em todas as rodadas. O relatório
`dataset_meta.json` registra isso e a distribuição por classe e split.

`--max-classes` é somente um limite de segurança. Se for menor que o número de
classes protegidas, o programa falha em vez de excluir verbos ou `Banheiro`.
Use `--extra-keep "Nome Exato"` para acrescentar classes do CSV à política.

## Primeiro treinamento recomendado

```bash
./.venv/Scripts/python.exe scripts/run_signer_cv.py \
  --config configs/bilstm_focus.yaml \
  --epochs 120 \
  --batch-size 32 \
  --num-workers 0
```

O dispositivo é escolhido automaticamente. Para escolher explicitamente:

```bash
# GPU NVIDIA
python scripts/run_signer_cv.py --config configs/bilstm_focus.yaml --epochs 120 --device cuda

# CPU
python scripts/run_signer_cv.py --config configs/bilstm_focus.yaml --epochs 120 --device cpu
```

No Windows, `--num-workers 0` é a opção inicial mais previsível para o treino.
Ela não controla a extração do MediaPipe; controla somente o carregamento dos
arquivos durante o treinamento.

## Retomada após interrupção

O executor grava continuamente:

```text
experiments/<execução>/run_state.json
```

Se houver interrupção, execute novamente o mesmo comando. Rodadas que já possuem
`test_metrics.json` serão reutilizadas. Uma rodada interrompida antes desse
arquivo começa novamente do início, sem reutilizar pesos incompletos.

Para refazer deliberadamente rodadas concluídas:

```bash
python scripts/run_signer_cv.py --config configs/bilstm_focus.yaml --epochs 120 --force
```

## Executar uma rodada isolada

```bash
python scripts/run_signer_cv.py \
  --config configs/bilstm_focus.yaml \
  --epochs 120 \
  --round 3
```

Nesse exemplo, Articuladores 1 e 2 treinam e o Articulador 3 é testado.

## Arquivos gerados

Para uma execução do vocabulário filtrado:

```text
experiments/bilstm_focus_signer_cv_allclasses_120epochs_seed42/
├── run_state.json
├── summary.json
├── summary.csv
├── articulador1/
│   ├── best.pt
│   ├── last.pt
│   ├── history.json
│   ├── history.csv
│   └── test_metrics.json
├── articulador2/
└── articulador3/
```

`summary.json` contém os resultados individuais e média, desvio padrão, mínimo e
máximo entre articuladores.

## Protocolo de diagnóstico

1. Revise `vocabulary_report.json` e `class_counts_by_split` no metadado.
2. Compare `train_top1` e `train_loss` com as métricas do articulador retido.
3. Se nem o treino aprende, investigue a qualidade dos landmarks, duração dos
   sinais, normalização, taxa de aprendizado e capacidade da rede antes de
   aplicar Focal Loss ou aumento. Treine um subconjunto pequeno e equilibrado
   somente como diagnóstico, sem tratá-lo como modelo final.
4. Se o treino aprende mas o teste segue baixo, colete repetições de mais
   articuladores e varie câmeras, iluminação e contexto. A validação por
   articulador deve permanecer separada.
5. Só compare Bi-LSTM, TCN e Transformer 1-D com mesmo vocabulário, splits e
   orçamento; o conjunto atual não sustenta afirmar acurácia de produção.

## Configuração diagnóstica

O diagnóstico inicial utiliza:

- Cross Entropy padrão; Focal Loss e pesos (`sqrt_inverse`, `inverse`,
  `effective`) já podem ser configurados, mas não ajudam um split equilibrado;
- `label_smoothing: 0.0`;
- sem pesos de classe;
- sem aumentação;
- número fixo de épocas;
- checkpoint escolhido pela menor loss de treino;
- teste executado somente após o fim das épocas.

Depois de comprovar aprendizado, a aumentação pode ser habilitada em
`configs/bilstm_focus.yaml`. O scheduler cosseno agora decai ao longo das
épocas posteriores ao warmup, sem reinícios periódicos.

## Conferir acurácia depois do treino

`train_top1` em `history.json` mede as amostras de treino. `top1` em
`test_metrics.json` mede o articulador reservado. Os 28% da execução antiga
foram `train_top1` na época 50; seu teste foi 12%.

```bash
./.venv/Scripts/python.exe -c 'import json; from pathlib import Path; p=Path("experiments/bilstm_focus_signer_cv_allclasses_120epochs_seed42/articulador2"); h=json.loads((p/"history.json").read_text(encoding="utf-8")); t=json.loads((p/"test_metrics.json").read_text(encoding="utf-8")); print("treino:", format(h[-1]["train_top1"], ".1%"), "teste:", format(t["top1"], ".1%"))'
```

Uma execução curta de uma época foi concluída para verificar o pipeline, com
top-1 de teste de 0,50% em 398 classes; esse número não é uma avaliação do
modelo treinado. No Windows, o processo retornou código `0xC0000409` ao sair
depois de gravar os artefatos. O executor de CV preserva os resultados se
`best.pt`, `history.json` e `test_metrics.json` já estiverem completos.

## Features

Cada frame contém 346 landmarks brutos:

- mãos: 126 valores;
- pose: 100 valores;
- face: 120 valores.

O carregamento usa a máscara de detecção `(T, 4)`, normaliza tudo com referência
nos ombros e acrescenta:

- 346 velocidades;
- 5 distâncias relativas.

Entrada final do modelo: 697 features por frame e 48 frames por sequência.

## Testes de desenvolvimento

```bash
./.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
./.venv/Scripts/python.exe -m pytest tests/ -v
```

O histórico anterior de validação por articulador fica em
[`SIGNER_CV_EXECUTION.md`](SIGNER_CV_EXECUTION.md). O plano e as mudanças desta
etapa ficam em [`PLANO_FILTRAGEM_TREINO.md`](PLANO_FILTRAGEM_TREINO.md).
O experimento de variações sintéticas está documentado em
[`PLANO_AUMENTACAO_LANDMARKS.md`](PLANO_AUMENTACAO_LANDMARKS.md). Na rodada
com Articulador2 reservado, a configuração testada reduziu top-1 de 3,78%
para 2,52%; ela não foi adotada como padrão.
