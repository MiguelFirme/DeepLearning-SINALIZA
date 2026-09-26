# Plano de execução — validação cruzada por articulador

Este documento é o ponto de retomada do trabalho para melhorar a acurácia do
Sinaliza. Ele deve ser atualizado à medida que cada etapa for executada.

## Objetivo

Treinar modelos independentes usando dois articuladores e avaliar no terceiro,
sem vazamento de dados:

| Rodada | Articuladores de treino | Articulador de teste |
|---|---|---|
| 1 | Articulador2 + Articulador3 | Articulador1 |
| 2 | Articulador1 + Articulador3 | Articulador2 |
| 3 | Articulador1 + Articulador2 | Articulador3 |

As rodadas não compartilham pesos. Depois de comparar as três, os hiperparâmetros
serão congelados e um modelo final poderá ser treinado com os três articuladores.

## Diagnóstico que motivou o plano

- Dataset: 4.086 amostras e 1.364 classes, aproximadamente três vídeos por classe.
- Split anterior: aproximadamente uma amostra por classe no treino, uma na
  validação e uma no teste.
- Top-1 observado: 0,000735 (0,0735%).
- Loss estabilizada em aproximadamente 7,218, que equivale a `ln(1364)`.
- O classificador colapsou e passou a prever majoritariamente uma única classe.
- Aumentações espaciais estavam sendo aplicadas depois da criação de features
  derivadas, podendo corromper velocidades, visibilidade e distâncias.

## Regras experimentais

1. O articulador de teste nunca participa do treino, escolha de checkpoint ou
   ajuste de hiperparâmetros daquela rodada.
2. Como existem somente três articuladores, inicialmente não haverá conjunto de
   validação independente. O treino usará épocas fixas e o teste será executado
   uma única vez ao final.
3. Cada rodada começa com pesos aleatórios novos e seed registrada.
4. O primeiro experimento usará poucas classes para validar o pipeline antes do
   treinamento completo.
5. Os landmarks `.npz` existentes serão reutilizados; não é necessário executar
   o MediaPipe novamente para estas correções.

## Etapas e estado

### Etapa 1 — Adaptar e validar a nova raiz do projeto

Estado: **concluída**

- [x] Confirmar nova raiz: `C:\Users\migue\DeepLearning-SINALIZA`.
- [x] Confirmar que `scripts/`, `ml/`, `configs/`, `data/` e demais diretórios
  estão diretamente na raiz.
- [x] Verificar que os scripts principais usam `parents[1]` como raiz.
- [x] Procurar referências à estrutura antiga `Sinaliza/Sinaliza`.
- [x] Remover caminho absoluto do cache Kaggle em `extract_landmarks.py`.
- [x] Corrigir exemplos de comandos do README para a interface real dos scripts.
- [x] Executar verificação de sintaxe/CLI sem iniciar treinamento.

Observação Git: após a movimentação manual, o índice ainda mostra os caminhos
antigos como removidos e os arquivos na raiz como novos. Isso será preservado;
nenhum reset, commit ou push faz parte desta implementação sem solicitação.

### Etapa 2 — Splits explícitos por articulador

Estado: **concluída**

- [x] Implementar split `leave-one-signer-out`.
- [x] Permitir escolher o articulador de teste pela CLI.
- [x] Permitir limitar o vocabulário de forma determinística (`--max-classes`).
- [x] Validar ausência de vazamento e cobertura das classes.
- [x] Salvar metadados completos de cada rodada.

### Etapa 3 — Corrigir processamento e aumentação

Estado: **concluída**

- [x] Usar a máscara de detecção presente nos `.npz`.
- [x] Preservar partes ausentes como zero após a normalização.
- [x] Aplicar aumentação apenas nos landmarks brutos e válidos.
- [x] Calcular velocidade e distâncias somente após transformações espaciais.
- [x] Normalizar mãos, pose e face de forma coerente.
- [x] Corrigir as cinco distâncias derivadas documentadas.
- [x] Tornar a aumentação configurável e desativá-la no diagnóstico inicial.

### Etapa 4 — Treino sem validação contaminada

Estado: **concluída**

- [x] Permitir `val` vazio no dataset e no treinador.
- [x] Usar número fixo de épocas nessa modalidade.
- [x] Usar Cross Entropy sem smoothing e sem pesos no teste inicial.
- [x] Registrar loss e Top-1/3/5 de treino.
- [x] Avaliar o articulador retido somente após o treino.

### Etapa 5 — Executor das três rodadas

Estado: **concluída**

- [x] Criar `scripts/run_signer_cv.py`.
- [x] Criar diretórios de experimento independentes.
- [x] Permitir uma rodada isolada ou as três sequencialmente.
- [x] Gerar resumo consolidado com média e desvio padrão.
- [x] Impedir reutilização acidental de pesos entre rodadas.

### Etapa 6 — Testes e documentação operacional

Estado: **concluída**

- [x] Testar splits e invariantes das features.
- [x] Testar ausência de NaN/inf e dimensões.
- [x] Executar smoke test pequeno, sem treinamento demorado.
- [x] Atualizar README com comandos de execução e retomada.
- [x] Registrar arquivos alterados e resultados das verificações neste documento.

## Sequência de experimentos após a implementação

1. 50 classes, três rodadas.
2. Se o treino aprender corretamente, 100 classes, três rodadas.
3. Depois, 200 classes, três rodadas.
4. Avaliar uma rede prototípica antes de tentar novamente as 1.364 classes.
5. Escolher configuração e treinar o modelo final com os três articuladores.

## Registro cronológico

### 2026-09-23 — Início

- Plano aprovado.
- Estrutura atual inspecionada sem alterações destrutivas.
- Manifesto confirmado com campos `label` e `signer`, incluindo
  `Articulador1`, `Articulador2` e `Articulador3`.
- Identificado caminho absoluto remanescente no extrator.
- Identificadas opções divergentes no README (`--input`) em relação a
  `build_dataset.py` (`--landmarks`) e `validate_landmarks.py` (`--dir`).

### 2026-09-23 — Etapa 1 concluída

- `extract_landmarks.py` não depende mais de um caminho absoluto do computador.
- Entrada padrão alterada para `<raiz>/data/raw`.
- A variável `SINALIZA_VIDEO_DIR` pode definir outro diretório padrão.
- `--input` e `--output` são convertidos em caminhos absolutos antes do uso.
- Diretório de entrada inexistente agora gera erro explícito antes de criar
  workers.
- README corrigido para usar `validate_landmarks.py --dir` e
  `build_dataset.py --landmarks`.
- Busca global confirmou que não há referências executáveis restantes a
  `Sinaliza/Sinaliza` ou a caminhos absolutos de usuário.
- `py_compile` concluiu sem erros para os scripts principais.
- As CLIs de extração, construção do dataset e treino foram carregadas com
  sucesso. Nenhum vídeo foi processado e nenhum treinamento foi iniciado.
- A `.venv` movida funciona com Python 3.11.9. Dentro do ambiente isolado da
  ferramenta foi necessário executar a validação com permissão ampliada; isso
  não exige alteração na máquina do usuário.

### 2026-09-23 — Etapa 2 iniciada

- Contagem confirmada: Articulador1=1.364, Articulador2=1.360 e
  Articulador3=1.362 amostras.
- Há 1.358 classes completas, presentes nos três articuladores.
- Seis classes incompletas foram identificadas: `Através`, `Congelar`,
  `Criança`, `Poesia`, `Solicitar` e `Terça-feira`.
- Decisão: o modo por articulador exigirá cobertura em todos os articuladores
  usados na rodada. As classes incompletas serão excluídas para que as métricas
  entre rodadas permaneçam comparáveis.

### 2026-09-23 — Etapa 2 concluída

- `DataSplitter.split_by_signer` implementado com treino, validação vazia e
  teste explicitamente separados por articulador.
- O método rejeita articulador desconhecido, treino vazio e qualquer tentativa
  de incluir o articulador de teste no treino.
- `labels_with_signer_coverage` seleciona somente classes presentes em todos os
  articuladores necessários.
- `build_dataset.py` agora aceita:
  - `--strategy signer_holdout`;
  - `--test-signer ArticuladorN`;
  - `--train-signers ArticuladorX ArticuladorY` (opcional);
  - `--max-classes N` para seleção reproduzível com `--seed`.
- `dataset_meta.json` registra classes selecionadas, articuladores disponíveis,
  articuladores de treino, articulador de teste, seed e tamanhos dos splits.
- Adicionado `tests/test_split.py` com testes de isolamento e validação de
  argumentos. A `.venv` atual não possui `pytest`, então as mesmas invariantes
  foram executadas diretamente em Python nesta etapa.
- Validação temporária das três rodadas com 50 classes:
  - 100 amostras de treino;
  - 0 de validação;
  - 50 de teste;
  - zero sobreposição de articuladores.
- Validação em memória do conjunto completo:
  - 1.358 classes completas;
  - 2.716 amostras de treino;
  - 1.358 amostras de teste;
  - validação vazia e zero vazamento.
- Os arquivos temporários de verificação foram escritos apenas no diretório
  temporário do Windows. `data/processed` não foi alterado.

### 2026-09-23 — Etapa 3 concluída

- `LibrasDataset` agora carrega a máscara `(T, 4)` de cada `.npz`.
- Aumentação foi movida para antes da normalização e das features derivadas.
- Cortes, descarte, mudança de velocidade e jitter temporal mantêm a máscara
  sincronizada com os frames.
- Ruído, escala, translação e rotação agem somente sobre coordenadas x/y/z;
  visibilidade da pose e features derivadas não são mais tratadas como pontos.
- Mãos, pose ou face ausentes são zeradas novamente após cada transformação.
- O normalizador centraliza e escala mãos, pose e face com a mesma referência
  dos ombros, usando referência mediana nos frames sem pose válida.
- Velocidade é anulada quando uma parte não existe no frame atual ou anterior.
- As cinco distâncias agora são efetivamente calculadas: mão-mão, cada mão ao
  nariz e cada mão ao centro do tronco.
- A API mantém compatibilidade com sua máscara temporal antiga e só usa máscara
  de partes quando recebe `(T, 4)`.
- `AugmentationConfig.from_mapping` lê a seção `augmentation` do YAML.
- `configs/default.yaml` deixa aumentação desativada para o diagnóstico inicial
  e documenta transformações compatíveis.
- Novos testes foram adicionados em `tests/test_normalization.py` e
  `tests/test_augmentation.py`.
- Verificação real concluída sem NaN/inf:
  - landmarks brutos: `(172, 346)`;
  - exemplo aumentado: `(153, 346)`, com máscara de mesmo comprimento;
  - entrada final do modelo: `(48, 697)`.
- Nenhum `.npz` foi regravado e nenhum treinamento foi iniciado.

### 2026-09-23 — Etapa 4 concluída

- `Trainer` aceita `val_loader=None` e detecta automaticamente ausência de
  validação independente.
- Nesse modo, early stopping é desativado mesmo se solicitado e todas as épocas
  configuradas são executadas.
- O melhor checkpoint é escolhido pela menor `train_loss`; o checkpoint registra
  `selection_source=train_loss` para deixar essa decisão auditável.
- Scheduler do tipo plateau usa `train_loss` somente quando não há validação.
- O histórico agora registra `train_loss`, `train_top1`, `train_top3` e
  `train_top5` em todas as épocas.
- Corrigido o passo final do otimizador quando o total de batches não é múltiplo
  de `gradient_accumulation_steps`.
- `train.py` não cria `DataLoader` de validação quando `splits["val"]` está vazio.
- O loader de treino não descarta mais o último batch (`drop_last=False`), o que
  é importante no diagnóstico com poucas amostras.
- Configuração diagnóstica alterada para `ce_standard`, smoothing `0.0`, pesos de
  classe desativados, aumentação desativada e early stopping desativado.
- O dataset de teste só é construído depois de `trainer.train()` terminar. O
  checkpoint escolhido é carregado e o teste é avaliado uma única vez.
- Métricas finais são salvas em `<experimento>/test_metrics.json`.
- Corrigido o gráfico de acurácia para usar `train_top1` em vez de `train_loss`.
- Adicionado `tests/test_trainer_no_validation.py`.
- Smoke test CPU com dados artificiais concluído:
  - duas épocas executadas integralmente;
  - histórico sem chaves falsas de validação;
  - checkpoint selecionado por `train_loss`;
  - carregamento do melhor checkpoint e avaliação pós-treino concluídos.
- Nenhum landmark real foi treinado nesta etapa.

### 2026-09-23 — Etapa 5 concluída

- Criado `scripts/run_signer_cv.py`, que executa as três rodadas independentes:
  - teste no Articulador1, treino nos Articuladores2 e 3;
  - teste no Articulador2, treino nos Articuladores1 e 3;
  - teste no Articulador3, treino nos Articuladores1 e 2.
- Cada rodada recebe seu próprio dataset processado, diretório de experimento e
  inicialização aleatória. Nenhum checkpoint é compartilhado entre rodadas.
- O executor aceita `--round all`, `--round 1`, `--round 2` ou `--round 3`, além
  de sobrescritas de épocas, seed, batch size, workers e dispositivo.
- `--prepare-only` constrói e valida os splits sem iniciar treinamento.
- `--dry-run` mostra os comandos que seriam executados sem escrever datasets ou
  treinar modelos.
- `run_state.json` é gravado atomicamente e registra o estado de cada rodada.
  Ao repetir exatamente o mesmo comando, rodadas concluídas são reutilizadas e
  uma rodada interrompida é reiniciada de forma isolada.
- O resumo consolidado é salvo em `summary.json` e `summary.csv`, com resultados
  por articulador e média, desvio padrão, mínimo e máximo das métricas.
- O nome padrão da execução registra configuração, quantidade de classes,
  épocas e seed, evitando misturar experimentos incompatíveis.
- Foi tratado um encerramento anormal observado no TensorFlow/MediaPipe no
  Windows com CUDA. A recuperação só é aceita quando métricas, histórico e
  checkpoint já existem; caso contrário, o executor mantém o erro como falha.

### 2026-09-23 — Etapa 6 concluída

- Criado `requirements-dev.txt` com as dependências do projeto e `pytest`.
- README atualizado com preparação, execução das três rodadas, retomada e
  estrutura dos resultados.
- `docs/TRAINING.md` atualizado com o procedimento completo e a progressão
  recomendada de 50 para 100 e 200 classes.
- `scripts/train.py` passou a aceitar `--save-dir` e `--seed`, permitindo que o
  orquestrador isole e reproduza cada rodada.
- Corrigida a leitura UTF-8 do dicionário no backend.
- Testes antigos foram alinhados às interfaces atuais e foram adicionados testes
  para split, normalização, aumento, treino sem validação e orquestração.
- Suíte completa executada: **47 testes aprovados**, com duas advertências de
  depreciação não bloqueantes do Pydantic/Starlette.
- `compileall` concluiu sem erros.
- Integração `--prepare-only` validada nas três rodadas com duas classes.
- Smoke test CPU executou as três rodadas completas, uma época por rodada,
  gerando checkpoints, métricas e resumo consolidado.
- Smoke test CUDA de uma rodada também concluiu pelo orquestrador e confirmou a
  recuperação protegida do encerramento anormal descrito acima.
- Todas as execuções de validação usaram diretórios temporários do Windows; os
  experimentos reais do usuário não foram iniciados nem sobrescritos.

## Próximo ponto de retomada

A implementação está concluída. O próximo passo é o usuário executar o primeiro
experimento real com 50 classes. Se houver interrupção, repetir o mesmo comando:
as rodadas já concluídas serão preservadas e o processo retomará na pendente.
