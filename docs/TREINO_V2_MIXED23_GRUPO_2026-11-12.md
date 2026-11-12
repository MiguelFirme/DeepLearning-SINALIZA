# Treino v2: 13 classes MINDS e dez palavras do grupo

Data: 12/11/2026. Pedido: iniciar o novo treino incluindo `data/raw/Videos do grupo` após remover sete classes MINDS.

Atualização: o usuário concluiu o primeiro treino real; a validação do melhor checkpoint atingiu 60/75 (80%) e F1 macro 0,8176. O teste permanece não avaliado. Veja o [relatório do resultado](RESULTADO_TREINO_V2_MIXED23_2026-11-12.md). As seções abaixo registram a preparação anterior à execução.

## Plano e dados conferidos

A origem MINDS tem 520 landmarks de 13 classes. A pasta do grupo tem dez pastas (Água, Ajudar, Amigo, Aprender, Beber, Brincar, Cansado, Certo, Comer e Diretor), sete MP4 por classe, no padrão `ClasseSinalizadorNN.mp4`. O total esperado após extração é **590 amostras, 23 classes**. O montador novo não reutiliza os índices de 25 classes e não altera nem copia os `.npz`.

Protocolo por identidade/origem:

- MINDS: treino Sinalizadores 01/02/05/06/08/10, validação 11, teste 12.
- Grupo: treino 01–05, validação 06, teste 07.
- Contagens totais: **440 treino, 75 validação, 75 teste**.

Os IDs do grupo devem representar a mesma pessoa real em todas as palavras, e as pessoas MINDS são distintas das do grupo. O script prefixa as origens para evitar colisões dos IDs. Se os números dos arquivos representarem apenas a ordem das gravações, corrija os IDs antes de treinar: o protocolo depende da identidade real.

## Comandos no Git Bash

Na raiz `DeepLearning-SINALIZA`, execute o bloco completo para encadear as etapas. Aspas no caminho com espaços são obrigatórias.

```bash
./.venv/Scripts/python.exe scripts/extract_landmarks.py \
  --input "data/raw/Videos do grupo" \
  --output data/landmarks/videos_do_grupo \
  --dataset-format folder_signer \
  --workers 4 \
&& ./.venv/Scripts/python.exe scripts/build_mixed_23_dataset.py \
  --minds-landmarks data/landmarks/minds_libras_all20 \
  --group-landmarks data/landmarks/videos_do_grupo \
  --output data/landmarks/mixed_23 \
  --processed-dir data/processed/mixed_23 \
&& ./.venv/Scripts/python.exe scripts/train.py \
  --config configs/transformer_v2_mixed23.yaml \
  --data-dir data/landmarks/mixed_23 \
  --processed-dir data/processed/mixed_23 \
  --experiment transformer_v2_mixed23_seed42 \
  --num-workers 0 \
  --skip-test
```

O construtor exige exatamente as classes e contagens previstas, pessoas identificadas e arquivos existentes. Se a extração falhar ou estiver incompleta, o conjunto não é montado. Permite repetir a montagem somente quando os arquivos de protocolo existentes forem idênticos. Os parâmetros de treino são os da v2 já documentada (150 épocas máximas, early stopping F1 macro, sampler balanceado, augmentation apenas no treino).

Saídas: landmarks do grupo em `data/landmarks/videos_do_grupo`; manifesto misto em `data/landmarks/mixed_23`; mapas e splits em `data/processed/mixed_23`; checkpoint em `experiments/transformer_v2_mixed23_seed42/best.pt`; log persistente em `logs/experiments/transformer_v2_mixed23_seed42/train.log`. `--skip-test` mantém as 75 amostras de teste fora desta execução; a validação seleciona o checkpoint. Um novo treino não troca o modelo principal da aplicação automaticamente.

Esta preparação não executou extração nem treino reais: os comandos são para execução pelo usuário. O montador foi testado com manifestos sintéticos, incluindo 590 arquivos referenciados, splits sem sobreposição, exclusão de classes e detecção de arquivo ausente. Consulte [PIPELINE_V2.md](PIPELINE_V2.md) para ativação do checkpoint e simulação de FPS, usando os caminhos `mixed_23` deste experimento.

## Verificação desta preparação

Os **13 testes específicos** do montador mixed23 e pipeline v2 passaram. A suíte completa registrou **137 passando, dois ignorados e seis falhando**: as seis falhas são comparações de probabilidades offline dos checkpoints históricos com o vídeo `Acontecer` cujos landmarks acabaram de ser reextraídos. Os valores diferem dos CSVs arquivados, portanto a paridade histórica exata não foi confirmada com a nova extração. As referências e tolerâncias antigas foram preservadas. Este resultado não é uma métrica de acurácia da v2, que ainda será treinada pelo usuário.
