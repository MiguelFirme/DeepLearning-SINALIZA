# Plano de filtragem e treino — 2026-09-26

## Objetivo

Selecionar automaticamente, a partir de `annotations.csv`, classes úteis para
escola, casa e recepção, preservando os verbos e registrando a seleção em cada
execução. Melhorar o diagnóstico e a estabilidade do treino de landmarks.

## Evidência inicial

- CSV recebido: 4.086 linhas, 1.364 classes; 1.358 classes com três vídeos e
  seis com dois.
- Execução anterior: 50 classes escolhidas aleatoriamente, dois vídeos de
  treino por classe e um vídeo de teste de outro articulador.
- Top-1 de teste: 12%, 12% e 8% nas três rodadas; média 10,7%. Na rodada 2,
  o top-1 de treino na época 50 foi 28%. É um sinal de aprendizado insuficiente
  no treino e de dificuldade adicional de generalização entre articuladores.
- O equilíbrio por classe das rodadas completas é praticamente uniforme;
  pesos de classe não são a primeira intervenção indicada nesse protocolo.

## Implementação planejada

1. Criar uma política de vocabulário explícita e editável. Detectar infinitivos
   portugueses na primeira palavra, com exceções documentadas; adicionar
   expressões verbais e termos essenciais manualmente. Produzir relatório de
   incluídos e excluídos para revisão humana.
2. Aplicar a política ao construir splits a partir do CSV, antes de calcular
   cobertura de articuladores. Exigir que o CSV e o manifesto concordem sobre
   classes; não remover silenciosamente verbos sem cobertura.
3. Fazer o treino conferir a integridade de índices e labels dos splits,
   calcular pesos somente nas amostras de treino quando houver desbalanceamento
   e expor parâmetros de loss e otimizador já existentes.
4. Corrigir a interação entre warmup e scheduler e registrar as escolhas
   efetivas da execução. Verificar filtragem e proteções com testes.

## Critério de avaliação

Nenhuma mudança garante acurácia de produção com dois exemplos de treino por
classe. Comparar top-1, top-3, macro-F1 e confusão por classe em rodadas por
articulador, mantendo o mesmo vocabulário e o teste reservado. Antes de falar
em produção, coletar mais pessoas, contextos e repetições, criar validação
independente e medir latência e erros em ambiente real.

## Implementação realizada

- `data/annotations.csv` foi incorporado ao projeto. `ml/data/vocabulary.py`
  mantém regras e categorias editáveis. `scripts/build_dataset.py` lê o CSV a
  cada execução, cruza classes e contagens por articulador com o manifesto,
  gera splits filtrados, `label_map.json`, `vocabulary_report.json` e contagens
  por split. `--extra-keep` permite adicionar classes presentes no CSV.
- `--max-classes` agora é um teto de proteção: uma execução falha se o teto
  remover qualquer classe selecionada. O executor de validação por articulador
  usa todo o vocabulário protegido por padrão.
- `LibrasDataset` elimina amostras fora do `label_map`, impedindo labels `-1`
  quando é usado diretamente com um mapa filtrado.
- `train.py` rejeita splits com índices inválidos, repetidos ou com classes sem
  treino, e rejeita um manifesto alterado após a preparação. Focal Loss,
  smoothing, pesos por classe e acumulação são configuráveis; pesos são
  calculados apenas no treino. O cálculo de pesos falha se faltar uma classe.
- O scheduler cosseno passou a decair ao longo das épocas após o warmup. O
  perfil `configs/bilstm_focus.yaml` usa um Bi-LSTM menor, AdamW, CE sem pesos
  e taxa inicial de 3e-4 para diagnóstico do vocabulário filtrado. Esse perfil
  agora é o padrão do executor de CV (120 épocas), com `num_workers=0` no
  Windows.
- O último grupo de acumulação de gradientes agora usa seu tamanho real; antes,
  um grupo incompleto reduzia o gradiente por um fator indevido. Um teste
  compara a atualização do modelo com passos manuais do otimizador.
- Testes de filtro, CSV, split e Dataset foram acrescentados.

## Verificação e limitações

- Inspeção de CSV e manifesto com PowerShell confirmou 4.086 registros em
  ambos. Há nove pares classe/articulador com grafias divergentes entre os
  arquivos; nenhum deles está entre os termos protegidos pela política atual.
- A partir do ambiente com acesso à instalação Python do usuário,
  `./.venv/Scripts/python.exe` executou Python 3.11.9, PyTorch e pytest.
  `python` no Git Bash do usuário resolveu para Python 3.14 global, sem pytest;
  por isso, os comandos do guia usam o executável explícito do `.venv`.
- A regra de infinitivos é uma heurística, complementada por exceções. Revise
  `vocabulary_report.json` para possíveis verbos em formas não canônicas e
  substantivos ambíguos antes do experimento longo.
- `Congelar`, `Criança` e `Solicitar` só têm dois articuladores. Ficam no
  vocabulário, mas suas métricas por três articuladores têm cobertura incompleta.
- Revisão dos 1.364 rótulos encontrou formas verbais fora do infinitivo (`Ir`,
  `Faz`, `Vai`, `Espere`, `Eu vejo`, entre outras). Elas foram acrescentadas à
  lista explícita, junto com termos de escola, casa, emergência e recepção.
- Os três splits reais foram preparados com 398 classes, 794 amostras de
  treino e 397 de teste na rodada do Articulador2. O treino contém 1–2
  amostras por classe. `vocabulary_report.json` registra 398 inclusões e 966
  exclusões.
- A suíte completa passou: 75 testes. Um smoke test de uma época na GPU
  gravou `best.pt`, `last.pt`, histórico e métricas de teste para 398 classes.
  O top-1 de teste foi 0,50%; isso apenas confirma o funcionamento do
  pipeline após uma época. O processo Windows retornou `0xC0000409` ao sair,
  mesmo com os artefatos completos; o executor de CV já preserva resultados
  completos nessa situação. A causa nativa ainda precisa ser isolada.
- A rodada completa de 120 épocas do Articulador2 foi executada depois desta
  verificação inicial; seus resultados estão registrados abaixo.

## Próxima decisão experimental

Revisar `vocabulary_report.json` e executar uma rodada completa com o
Articulador2 reservado. Medir separadamente treino e teste. Se o treino
continuar baixo, inspecionar sinais individuais e normalização e comparar um
baseline simples com o Bi-LSTM. Se o treino subir mas o teste não, coletar
mais articulações por classe de pessoas diferentes. O corpus atual contém
somente sinais isolados; tradução contínua para texto natural requer dados de
frases e um modelo de segmentação/linguagem além deste classificador.

## Resultado da rodada completa — Articulador2, 2026-09-26

- Comando: `./.venv/Scripts/python.exe scripts/run_signer_cv.py --round 2 --num-workers 0`.
- Vocabulário: 398 classes; treino com 794 vídeos dos Articuladores 1 e 3;
  teste com 397 vídeos do Articulador 2. Sem validação independente.
- 120 épocas, Bi-LSTM com 951.808 parâmetros, CE sem pesos e sem aumento.
  O melhor checkpoint por loss de treino ocorreu na época 117.
- Última época: `train_top1=97,86%`, `train_loss=0,2319`. No checkpoint de
  menor loss, `train_top1=97,98%`.
- Teste reservado: `top1=3,78%` (15/397), `top3=8,82%`, `top5=11,59%`,
  `f1_macro=2,35%`, `loss=8,4739`.
- Interpretação: o modelo discrimina os exemplos de treino, mas generaliza mal
  para uma pessoa nova. Não é subajuste do treino nesta configuração. O
  conjunto tem em geral dois vídeos de treino por classe; pesos de classe não
  atacam o principal problema, pois as classes são quase equilibradas.
- O teste antigo de 12% usava somente 50 classes. A diferença percentual não
  mede melhora ou piora isolada da arquitetura: o número e a composição de
  classes mudaram. A probabilidade uniforme de acerto casual com 398 classes
  é aproximadamente 0,25%.
- O processo Python retornou `3221226505` após salvar os artefatos. O
  executor marcou a rodada como concluída e registrou `process_warning`.
  O mesmo padrão já apareceu em smoke tests; a causa do encerramento nativo
  não foi identificada. Os arquivos de métricas e checkpoints existem.

Próximos experimentos devem manter um articulador retido e medir dados de
várias pessoas. Antes de ampliar a arquitetura, comparar com baselines
simples sobre os mesmos landmarks e inspecionar a normalização e classes com
erro. Com os dados atuais não há evidência de prontidão para produção.
