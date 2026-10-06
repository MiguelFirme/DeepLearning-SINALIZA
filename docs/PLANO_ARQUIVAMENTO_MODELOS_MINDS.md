# Plano de arquivamento dos Transformers MINDS-Libras

Data: 2026-10-05.

## Objetivo

Guardar cópias independentes dos checkpoints selecionados por menor `val_loss` dos experimentos de 8 e 20 classes na pasta `Modelos salvos`, com informações suficientes para identificar classes, reproduzir a divisão dos dados e repetir o treino no projeto.

## Procedimento

1. Confirmar os experimentos, o estado do teste e o conteúdo da pasta de destino. Não alterar os checkpoints originais.
2. Criar uma subpasta por experimento. Copiar `best.pt`, `label_map.json`, `splits.json`, `manifest.json`, `dataset_meta.json`, `annotations.csv`, histórico, métricas disponíveis e `run_record.json`.
3. Copiar a cadeia de YAMLs de configuração e a lista fechada de classes para que o arquivo de configuração funcione dentro da subpasta e o protocolo não dependa de edições futuras nos arquivos originais.
4. Escrever um README por modelo com dados, arquitetura, normalização/augmentation, loss, otimizador, scheduler, seleção do checkpoint, resultados confirmados, limitações e comandos de reprodução. Criar um índice na raiz.
5. Conferir hashes SHA-256 dos arquivos copiados em relação às origens e testar a leitura do checkpoint com as classes esperadas.

Os MP4 e os `.npz` não serão duplicados no arquivo de modelos: são datasets, não pesos. Reproduzir exatamente o treino requer manter ou recuperar os landmarks correspondentes ao manifesto e a versão do código. Uma exceção no `.gitignore` permitirá versionar somente os dois `best.pt` arquivados; a pasta local sozinha não é um backup externo.

## Resultado

Foram criadas as subpastas `Modelos salvos/transformer_minds_8_classes_pose_face/` e `Modelos salvos/transformer_minds_20_classes_pose_face/`, com README próprio, checkpoint, mapa de classes, manifesto, split, configurações, histórico e métricas disponíveis. O modelo de 20 classes permanece sem métrica de teste.

Foram conferidos por SHA-256 **36 arquivos copiados** (19 do modelo de 8 classes e 17 do de 20 classes), todos idênticos às origens. `checksums.sha256` em cada subpasta cobre também seu README. Os checkpoints abriram com `torch.load`, seus mapas de rótulos têm 8 e 20 classes, e os pesos carregaram integralmente com `strict=True`. Uma avaliação de validação dos pacotes arquivados reproduziu **40/40, F1=1,000** (8 classes) e **37/100, F1=0,286** (20 classes).
