# Registro de treinos e busca de hiperparâmetros — categoria `escola`

## Plano — 2026-09-30

1. Criar `logs/` e passar a registrar cada treino em uma subpasta identificada
   por experimento e rodada: configuração efetiva, versões, seed, dispositivo,
   resumo dos dados e splits, classes, arquitetura, otimizador, loss,
   métricas por época, métricas finais, avisos e caminhos dos checkpoints.
   Pesos permanecem em `experiments/`; os logs registram suas referências.
2. Preparar uma separação por pessoa para ajuste: Articulador1 no treino
   (real + derivados), Articulador2 na validação (somente real) e
   Articulador3 no teste final (somente real). Validar que não haja derivados
   de validação/teste no treino e registrar o manifesto/split exatos.
3. Rodar uma grade pequena de Bi-LSTM variando largura, camadas e taxa de
   aprendizado, com dropout inicial fixo. Comparar pela validação; guardar
   os resultados de todas as combinações, inclusive falhas.
4. Sobre a melhor configuração da grade, testar dropout separado (incluindo
   o valor atual) na mesma validação. Escolher configuração e época apenas
   pela validação. Reajustar com Articuladores1+2 e medir Articulador3 uma
   única vez como teste final deste protocolo.
5. Registrar tabela, critério de escolha e limites da evidência. Esses três
   articuladores já apareceram em experimentos anteriores, então o resultado
   continua exploratório; um novo articulador será necessário para estimativa
   independente de desempenho de produção.

## Estado anterior

O pipeline já salvava `history.csv/json`, `test_metrics.json`, `best.pt`,
`last.pt`, `run_state.json` e `summary.json` em `experiments/`. Faltavam logs
persistentes do terminal e um registro consolidado da configuração efetiva,
ambiente e proveniência dos dados. O `scripts/tune_hyperparams.py` existente
presume validação não vazia e, nos holdouts atuais, `val=[]`; portanto não é
adequado para este teste sem uma separação explícita por articulador.

## Resultados

A primeira grade (8 arquiteturas e 2 testes adicionais de dropout) foi
executada. O script inicial escolheu o checkpoint pela menor `val_loss` e
comparou o `val_top1` naquele checkpoint. Com apenas 23 vídeos de validação,
a menor perda frequentemente ocorreu nas primeiras épocas. A configuração
escolhida por esse critério foi 2 camadas Bi-LSTM, 128 unidades por direção,
taxa 0,00015 e dropout 0,2, com 3/23 (13,04%) na validação. Seu refit de 10
épocas acertou 0/23 no Articulador3. Esse resultado está registrado em
`logs/grid_escola_23_synthetic_seed42/final_result.json`.

Revisão planejada **após esse resultado**: usar os históricos já salvos para
escolher arquitetura pela melhor acurácia de validação ao longo das 120 épocas,
com F1 macro e perda como desempate. Repetir a etapa de dropout para a nova
arquitetura, mantendo o mesmo split. Reajustar por 120 épocas fixas (mesma
duração da linha de base), pois transferir a época ótima de um treino com só
Articulador1 para um novo treino com Articuladores1+2 subtreinou o modelo.
Registrar essa segunda análise como **exploratória**: o Articulador3 já foi
avaliado na primeira rodada e em experimentos anteriores, logo não é um teste
independente para novas decisões.

## Resultado da revisão por acurácia

Treino para ajuste: 77 amostras do Articulador1, incluindo derivados;
validação: 23 vídeos reais do Articulador2. O Articulador3 ficou fora da grade.
Todas as tentativas rodaram 120 épocas, seed 42, com a mesma loss CE, batch 32
e mesmo pré-processamento. A tabela mostra o **melhor top-1 de validação ao
longo das épocas**, com F1 e perda como desempate. Essa análise foi feita após
o teste inicial de 0/23 descrito acima e deve ser interpretada como revisão
exploratória do protocolo.

| Largura por direção | Camadas Bi-LSTM | Taxa de aprendizado | Melhor top-1 validação |
| ---: | ---: | ---: | ---: |
| 64 | 1 | 0,00015 | 4/23 (17,39%) |
| **64** | **1** | **0,0003** | **6/23 (26,09%)** |
| 64 | 2 | 0,00015 | 4/23 (17,39%) |
| 64 | 2 | 0,0003 | 4/23 (17,39%) |
| 128 | 1 | 0,00015 | 3/23 (13,04%) |
| 128 | 1 | 0,0003 | 4/23 (17,39%) |
| 128 | 2 | 0,00015 | 4/23 (17,39%) |
| 128 | 2 | 0,0003 | 3/23 (13,04%) |

Na arquitetura vencedora, dropout 0,0 e 0,4 chegaram ambos a 4/23 (17,39%)
na validação; dropout **0,2** chegou a 6/23 (26,09%). A configuração escolhida
foi 1 camada Bi-LSTM, 64 unidades por direção, taxa 0,0003 e dropout 0,2.
Esse modelo tem **237.577 parâmetros**, contra 855.433 do Bi-LSTM anterior
de 2 camadas e 128 unidades por direção.

Reajuste por 120 épocas com Articuladores1+2: 154 amostras de treino. No
Articulador3, usando apenas 23 vídeos reais, o modelo acertou **3/23 (13,04%)
top-1**, **9/23 (39,13%) top-3** e F1 macro **10,14%**. O modelo maior
anterior com o mesmo recorte e dados aumentados acertou 3/23 top-1,
6/23 top-3 e F1 macro 8,99%. A arquitetura menor igualou o top-1 com menos
parâmetros; há poucos exemplos para afirmar uma melhora confiável de top-3.

Como o Articulador3 já foi visto em análises anteriores, esse resultado não
é uma estimativa independente para produção. O próximo protocolo precisa de
mais articuladores e um teste final de pessoa ainda não usada no projeto.

## Arquivos de log e reprodução

`logs/grid_escola_23_synthetic_seed42/` contém:

- `identity.json`: hashes do spec e do manifesto e tamanhos dos splits.
- `grid_results.csv/json`: primeira comparação pelos checkpoints de menor
  perda de validação; `selection.json` e `final_result.json` dessa etapa.
- `accuracy_review/grid_results.csv/json`: revisão por pico de acurácia em
  cada histórico; `selection.json`, `final_result.json` e `review.log`.
- `trials/<tentativa>/`, `accuracy_trials/<tentativa>/`, `final/selected_refit/`
  e `accuracy_final/selected_refit/`: `run_record.json`, `train.log`,
  `history.json` e, nos testes finais, `test_metrics.json`.
- `subprocess/`: saída capturada dos subprocessos. O código agora separa
  saídas finais pelo nome da etapa para evitar colisões entre tentativas.

Pesos (`best.pt` e `last.pt`), configs YAML específicas e históricos completos
também ficam em `experiments/grid_escola_23_synthetic_seed42/`. A pasta
`logs/` contém dados locais de execução e é ignorada pelo Git; este documento
e o código são versionáveis. A suíte completa passou com **85 testes**.

Comandos usados no Git Bash do VS Code:

```bash
./.venv/Scripts/python.exe scripts/grid_search_escola.py
./.venv/Scripts/python.exe scripts/reselect_grid_accuracy.py
```

O primeiro comando pode reutilizar tentativas completas quando executado de
novo com o mesmo spec e manifesto; a revisão também reutiliza suas tentativas
completas. Para uma nova grade com parâmetros diferentes, altere `run_name`
em `configs/grid_escola_23.json` antes de executar, preservando os resultados
anteriores.
