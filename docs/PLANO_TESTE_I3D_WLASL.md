# Plano do teste I3D RGB/WLASL2000 no MINDS-Libras

Data: 2026-10-05. Projeto acadêmico de extensão, conforme esclarecimento do usuário.

## Objetivo

Comparar características visuais de um I3D pré-treinado em sinais isolados do WLASL2000 com o baseline de landmarks `pose_face` no piloto MINDS-Libras de cinco classes. A métrica de seleção é a validação no Sinalizador11; o Sinalizador12 não será usado para seleção.

## Protocolo

1. Obter a arquitetura e o checkpoint I3D RGB do repositório oficial WLASL; registrar URLs, hash, termos de uso, versão das bibliotecas e incompatibilidades encontradas. Não baixar os vídeos WLASL, pois só os pesos são necessários para transferência.
2. Confirmar que o checkpoint carrega integralmente na arquitetura esperada. Remover apenas a camada final de 2.000 classes, mantendo o encoder congelado.
3. Ler os 200 MP4s locais do MINDS-Libras usando `manifest.json` e os índices de `splits.json`. Amostrar frames de forma determinística e aplicar a mesma transformação RGB em treino e validação. Extrair embeddings sem gradiente; preservar rótulos e IDs de vídeo.
4. Treinar uma cabeça linear ou MLP pequeno só com os 150 vídeos de treino, escolher por perda de validação nos 25 vídeos do Sinalizador11 e registrar acurácia, F1, matriz de confusão e previsões por vídeo. Comparar com os 44% top-1 de validação do baseline `pose_face`. Não rodar teste no Sinalizador12 durante seleção.
5. Caso os pesos oficiais ou a execução estejam indisponíveis, deixar o código e o procedimento reproduzíveis com estado explícito; não apresentar números de um encoder aleatório como teste de transferência.

## Saídas previstas

- Código de extração RGB/I3D e treino da cabeça.
- Cache de embeddings e logs com proveniência em pastas ignoradas pelo Git.
- Resultado de validação e conclusão sobre a hipótese de transferência.

## Fontes oficiais

- https://github.com/dxli94/WLASL
- https://github.com/dxli94/WLASL/blob/master/code/I3D/pytorch_i3d.py
- https://drive.google.com/file/d/1jALimVOB69ifYkeT0Pe297S1z4U3jC48/view?usp=sharing

## Execução e resultado em 2026-10-05

O repositório oficial foi clonado em `artifacts/wlasl_source` no commit `ac00e6be631c1a2a486621b65f202219f3964d6b`. O ZIP oficial `archived.zip` (197 MB) foi baixado pelo link acima, e apenas `archived/asl2000/FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.pt` foi extraído. O checkpoint tem SHA256 `243a19e6deef3becffbfc5b7dd8adb32916c8bee482565ca243c082584732620`. O código oficial `pytorch_i3d.py` tem SHA256 `7ba3afc48e4bb7f27063bf67ad0578dacdace5fc8c29d30ed152c8721859f0dd`. Todos os 344 tensores carregaram com `strict=True` no `InceptionI3d(2000, in_channels=3)`; não foram usados pesos aleatórios.

Implementação local: `scripts/train_i3d_wlasl_probe.py`. A leitura usa os 16 frames uniformemente distribuídos pelos frames **efetivamente decodificados**, recorte central quadrado em 224×224, RGB em `[-1,1]`. O encoder permanece congelado. Uma `StandardScaler` ajustada **apenas no treino** alimenta uma `LogisticRegression(C=1)`. Não houve ajuste de hiperparâmetros pela validação. Todos os 175 embeddings de treino e validação foram salvos em `artifacts/i3d_wlasl_school5_embeddings`; o teste não foi extraído. A GPU foi RTX 4050 Laptop, PyTorch 2.14.0+cu130; execução completa em 301,9 segundos.

| Conjunto | Top-1 | F1 macro | Loss | Amostras |
|---|---:|---:|---:|---:|
| Treino | 100% | 1,000 | 0,005 | 150 |
| Validação (Sinalizador11) | 40% (10/25) | 0,327 | 3,081 | 25 |

Na validação, por classe: `Aluno` 5/5, `Banheiro` 0/5 (todos previstos como `Aluno`), `Conhecer` 4/5, `Medo` 0/5 (todos previstos como `Aluno`), `Vontade` 1/5. A matriz completa e os hashes estão em `experiments/i3d_wlasl_school5_seed42/run_record.json`; as previsões por vídeo, em `val_predictions.csv`; log persistente em `logs/experiments/i3d_wlasl_school5_seed42/train.log`.

Comparação: BiLSTM `pose_face` obteve 44% (11/25) na **mesma** validação e checkpoint escolhido por menor `val_loss`; o I3D congelado com cabeça linear obteve 40% (10/25). A diferença é de um vídeo e este piloto não sustenta superioridade de nenhum dos dois. O acerto perfeito no treino junto com perda de validação alta indica que a cabeça ainda aprende particularidades dos sinalizadores de treino. O teste do Sinalizador12 continua reservado.

Comandos para reproduzir no Git Bash, a partir da raiz do projeto:

```bash
./.venv/Scripts/python.exe -m pip install -r requirements.txt
git clone --depth 1 https://github.com/dxli94/WLASL.git artifacts/wlasl_source
./.venv/Scripts/python.exe -m gdown 1jALimVOB69ifYkeT0Pe297S1z4U3jC48 -O artifacts/wlasl_archived.zip
./.venv/Scripts/python.exe -c "import zipfile,pathlib,shutil; z=zipfile.ZipFile('artifacts/wlasl_archived.zip'); n='archived/asl2000/FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.pt'; p=pathlib.Path('artifacts/wlasl_asl2000.pt'); p.parent.mkdir(parents=True,exist_ok=True); shutil.copyfileobj(z.open(n),p.open('wb'))"
./.venv/Scripts/python.exe scripts/train_i3d_wlasl_probe.py
```

Para uma reprodução exata, usar o commit e os hashes acima. Caso o cache de embeddings seja reutilizado, o script confere os hashes de checkpoint, código, manifesto, splits e mapa de classes. O I3D é um baseline **RGB congelado**; este resultado não avalia ajuste fino do encoder nem tradução de frases.
