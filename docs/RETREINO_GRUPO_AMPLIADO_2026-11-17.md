# Retreino Transformer v2 com o grupo ampliado

Data: 17/11/2026. [Plano](PLANO_RETREINO_GRUPO_2026-11-17.md).

Atualização: o treino real foi concluído pelo usuário; o checkpoint da época 80 obteve 70/93 acertos (75,27%) e F1 macro 0,7512 na validação. O teste permanece não avaliado. Veja o [relatório do resultado](RESULTADO_RETREINO_GRUPO_2026-11-17.md).

## Dados e correções confirmadas

A pasta `data/raw/Videos do grupo` contém **28 classes e 195 MP4**. Com as 13 classes MINDS restantes (520 exemplos), o conjunto terá **41 classes e 715 exemplos**, em **529 treino / 93 validação / 93 teste**.

O usuário confirmou que `Errado/ERRADO.mp4` pertence à pessoa 07 e que os dois arquivos com prefixo Diretor dentro de Diretoria devem ter o rótulo dos demais vídeos daquela pasta. Foram corrigidos somente estes nomes:

- `Errado/ERRADO.mp4` → `Errado/ErradoSinalizador07.mp4`.
- `Diretoria/DiretorSinalizador06.mp4` → `Diretoria/DiretoriaSinalizador06.mp4`.
- `Diretoria/DiretorSinalizador07.mp4` → `Diretoria/DiretoriaSinalizador07.mp4`.

Os caminhos e hashes SHA-256 estão em `reports/preparacao_grupo_2026-11-17/renomeacoes.json`. Os conteúdos foram conferidos como iguais antes/depois das renomeações. Todos os 195 vídeos foram verificados como compatíveis com `Classe/ClasseSinalizadorNN.mp4`.

`EscolaSinalizador04.mp4` está ausente: Escola terá quatro exemplos de treino, um de validação e um de teste. Não foi criado nem duplicado um exemplo para completar a classe. O sampler balanceado já existente continua sendo calculado somente no treino.

## Comandos no Git Bash

Na raiz do projeto, cole o bloco completo:

```bash
./.venv/Scripts/python.exe scripts/extract_landmarks.py \
  --input "data/raw/Videos do grupo" \
  --output data/landmarks/grupo_20261117 \
  --dataset-format folder_signer \
  --workers 4 \
&& ./.venv/Scripts/python.exe scripts/build_mixed_group_dataset.py \
  --minds-landmarks data/landmarks/minds_libras_all20 \
  --group-landmarks data/landmarks/grupo_20261117 \
  --group-videos "data/raw/Videos do grupo" \
  --output data/landmarks/mixed_grupo_20261117 \
  --processed-dir data/processed/mixed_grupo_20261117 \
&& ./.venv/Scripts/python.exe scripts/train.py \
  --config configs/transformer_v2_grupo_expandido.yaml \
  --data-dir data/landmarks/mixed_grupo_20261117 \
  --processed-dir data/processed/mixed_grupo_20261117 \
  --experiment transformer_v2_grupo_expandido_20261117_seed42 \
  --num-workers 0 \
  --skip-test
```

A extração usa um diretório novo para refletir os vídeos atuais, inclusive arquivos que tenham sido substituídos sob um nome existente. Os 520 landmarks MINDS são reutilizados. O montador descobre as classes do grupo e confirma correspondência entre MP4s e manifesto de landmarks, arquivos únicos e presença das classes nos três splits. Não usa os índices antigos mixed23/mixed25 nem presume sete amostras em todas as classes.

Protocolo por pessoa: MINDS 01/02/05/06/08/10 no treino, 11 na validação e 12 no teste; grupo 01–05 no treino, 06 na validação e 07 no teste. Os IDs de origem recebem prefixo. Os números precisam continuar identificando a mesma pessoa real entre todas as palavras; não representam apenas número de ordem do vídeo.

O treino cria pesos novos com **a mesma arquitetura do modelo padrão**: Transformer 128, quatro cabeças, três camadas, dropout 0,15, 725 features e janela de 3,2 s a 15 FPS. A saída passa a 41 classes, lidas do novo mapa. `transformer_v2_grupo_expandido.yaml` preserva explicitamente os parâmetros de geometria/segmentação do modelo padrão, em vez de ativar o experimento `*_webcam.yaml`. Não é ajuste fino dos pesos antigos de 23 classes.

Seleção por F1 macro da validação, até 150 épocas, paciência 25, sampler balanceado e augmentation somente no treino. `--skip-test` deixa os 93 vídeos de teste fora desta execução. Não compare diretamente uma porcentagem no novo vocabulário com a antiga validação de 23 classes.

## Artefatos

- Pesos selecionados: `experiments/transformer_v2_grupo_expandido_20261117_seed42/best.pt`.
- Histórico e métricas de validação: no mesmo diretório do checkpoint.
- Log: `logs/experiments/transformer_v2_grupo_expandido_20261117_seed42/train.log`.
- Mapas/splits: `data/processed/mixed_grupo_20261117/`.

O backend continua com o modelo de 23 classes durante esse treino. O novo checkpoint deve ser conferido antes de ser escolhido para a aplicação. Para repetir um treino concluído, use outro nome de experimento para preservar os resultados anteriores.

## Verificação desta preparação

Passaram **sete testes** em `test_mixed_group_dataset.py`, `test_mixed23_dataset.py`, `test_folder_signer.py` e `test_training_v2.py`. O teste do novo montador cobre 715 exemplos/41 classes, falta de Escola04, isolamento de identidade, reexecução idêntica, vídeo sem extração, arquivo duplicado e pessoa fora do protocolo. Um treino sintético valida a integração da v2. Não foram executados extração nem treinamento real nesta preparação e não há novas métricas de acurácia.

O Git ainda indicava sessão `am` pendente antes deste trabalho; ela não foi modificada. Os comandos de extração e treino não dependem de concluir a aplicação de patches. Os novos arquivos do montador/configuração/documentação precisam ser incluídos em um commit quando essa sessão do Git estiver resolvida.
