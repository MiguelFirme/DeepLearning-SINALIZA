# Plano de migração para MINDS-Libras

## Objetivo

Retreinar o classificador de sinais isolados do Sinaliza com o dataset [MINDS-Libras no Kaggle](https://www.kaggle.com/datasets/j0aopsantos/minds-libras/data), preservando separação por vídeo e, quando houver identidade confiável, por pessoa ou sessão. Registrar o mapeamento de classes, a proveniência e todas as métricas.

## Etapas

1. Inspecionar metadados, licença, tamanho, formatos e organização real do download. Não inferir classes a partir do nome do dataset.
2. Baixar em diretório isolado, sem sobrescrever `data/raw`, `data/landmarks`, `data/annotations.csv` ou experimentos anteriores. Inventariar vídeos, rótulos, pessoas, sessões e repetições por classe.
3. Adaptar a extração de landmarks e o manifesto à organização real do MINDS-Libras; rejeitar rótulos ou grupos ambíguos. Preservar referência ao vídeo original.
4. Definir vocabulário compatível com o objetivo escolar/doméstico e splits por pessoa/sessão, se possível. Conferir cobertura de classes em treino/validação/teste e ausência de vídeos repetidos.
5. Executar smoke test, treinamento com configuração registrada e avaliação uma única vez no teste reservado. Comparar com o baseline com ressalvas sobre diferenças de vocabulário e protocolo.

## Estado inicial

Em 2026-10-04, a API pública do Kaggle informou versão 3, licença MIT e 47.844.288.783 bytes (aproximadamente 44,6 GiB). O índice completo foi salvo em `data/raw/minds_libras/kaggle_file_index.json` (ignorado pelo Git): 800 vídeos MP4, 20 palavras, 8 sinalizadores (01, 02, 05, 06, 08, 10, 11, 12), 5 tomadas por pessoa e palavra. A nomenclatura é `NNPalavraSinalizadorXX-Y.mp4`; todos os 800 nomes seguem esse formato. Cada classe tem 40 vídeos.

O código anterior de `scripts/extract_landmarks.py` interpreta rótulos pelo padrão `..._ArticuladorN` ou pela pasta pai e, portanto, classificaria erroneamente os vídeos do MINDS-Libras se usado sem adaptação. O índice foi obtido apenas como metadados; ainda não há vídeos completos no projeto. Um download de teste de `08BanheiroSinalizador01-1.mp4` recebeu 2.353.837 de 36.919.436 bytes em 120 segundos e expirou. O arquivo parcial foi guardado como `.part`, fora da varredura de vídeos. Nesse ritmo, baixar todo o dataset neste ambiente é impraticável; a ferramenta de download precisa permitir retomada e validação de tamanho.

## Resultado e comandos

Foi criado um piloto com cinco classes úteis ao contexto escolar/doméstico: `Aluno`, `Banheiro`, `Conhecer`, `Medo` e `Vontade`. São 200 vídeos no Kaggle, aproximadamente 11,41 GiB. Após download e extração, o protocolo previsto contém 150 vídeos de seis pessoas para treino, 25 vídeos de uma pessoa para validação e 25 de outra para teste. O conjunto completo de 20 classes permanece disponível pelo índice; o piloto reduz o custo de baixar e depurar.

Na raiz do projeto, pelo Git Bash do VS Code:

```bash
./.venv/Scripts/python.exe scripts/download_minds_libras.py --dry-run --labels Aluno Banheiro Conhecer Medo Vontade
./.venv/Scripts/python.exe scripts/download_minds_libras.py --labels Aluno Banheiro Conhecer Medo Vontade
./.venv/Scripts/python.exe scripts/extract_landmarks.py --input data/raw/minds_libras --output data/landmarks/minds_libras_school5 --dataset-format minds_libras --workers 2
./.venv/Scripts/python.exe scripts/build_dataset.py --landmarks data/landmarks/minds_libras_school5 --annotations data/landmarks/minds_libras_school5/annotations.csv --output data/processed/minds_libras_school5 --strategy signer_holdout --class-list configs/minds_libras_school5.json --train-signers Sinalizador01 Sinalizador02 Sinalizador05 Sinalizador06 Sinalizador08 Sinalizador10 --val-signer Sinalizador11 --test-signer Sinalizador12
./.venv/Scripts/python.exe scripts/train.py --config configs/bilstm_regularized_school.yaml --data-dir data/landmarks/minds_libras_school5 --processed-dir data/processed/minds_libras_school5 --experiment bilstm_minds_libras_school5_seed42 --num-workers 0
```

`download_minds_libras.py` usa o índice da API do Kaggle, retoma arquivos `.part` e só libera um `.mp4` depois de conferir o tamanho informado pela API. `extract_landmarks.py --dataset-format minds_libras` gera `manifest.json` e `annotations.csv` com rótulo, pessoa, tomada e ID do vídeo; rejeita nomes fora do padrão. A extração pode ser repetida após interrupção e pula `.npz` existentes. O índice, os MP4 e os landmarks ficam em diretórios separados dos dados antigos e são ignorados pelo Git.

O download inicial neste ambiente não pôde ser concluído: o primeiro vídeo de 36,9 MB recebeu apenas 2,35 MB em 120 segundos. O usuário concluiu o download e a extração localmente em 2026-10-05; os 200 landmarks e vídeos estão agora disponíveis para análise.

Uma segunda requisição confirmou que o endpoint aceita `Range` (HTTP 206) e a retomada ampliou o `.part` para 6.321.524 bytes; o vídeo continua incompleto e fora do pipeline. A taxa variou, mas permanece baixa para 200 arquivos.

## Resultado do primeiro treinamento (2026-10-05)

O usuário executou `build_dataset.py` e `train.py` com os comandos acima. O manifesto tem 200 vídeos, 40 por classe. Os splits são disjuntos por pessoa: seis sinalizadores e 150 vídeos no treino; Sinalizador11 e 25 vídeos na validação; Sinalizador12 e 25 vídeos no teste. Cada classe tem 30/5/5 exemplos nos três splits. Não há arquivos sintéticos.

O treino usou BiLSTM regularizada de 130.039 parâmetros, Adam com `weight_decay=1e-4`, dropout 0,45, augmentation espacial no treino e seleção por menor `val_loss`. O early stopping parou na época 16. A menor `val_loss=1,8523` foi na época 1, que se tornou o checkpoint testado. A `train_top1` da época 16 era 70,67%, enquanto `val_top1` permaneceu 20% em todas as 16 épocas.

O teste desse checkpoint acertou **3/25 = 12%** top-1, top-3 **12/25 = 48%**, F1 macro **0,0632**. Para cinco classes equilibradas, 20% top-1 seria o valor esperado de um palpite uniforme; top-5 de 100% não informa nada porque abrange todas as classes. O `test_metrics.json` está em `experiments/bilstm_minds_libras_school5_seed42/`, e o histórico/configuração efetiva estão também em `logs/experiments/bilstm_minds_libras_school5_seed42/`.

Diagnóstico de leitura, sem novo treino:

- No checkpoint da época 1, a validação teve 20 previsões de `Conhecer` e 5 de `Medo`; só os 5 vídeos de `Conhecer` foram classificados corretamente. O teste teve 14 previsões de `Conhecer`, 6 de `Banheiro` e 5 de `Medo`; acertou apenas 3 vídeos de `Conhecer`. As outras quatro classes tiveram zero acertos.
- Todos os 200 arquivos `.npz` contêm frames não vazios e números finitos. Os vídeos têm mediana de 99 a 167 frames por sinalizador; o `SequenceProcessor` os amostra em 48 frames.
- A detecção média da mão esquerda por sinalizador de treino variou de 6,7% a 23,5% dos frames; a direita, de 35,7% a 54,7%. Em Sinalizador11 (validação), as taxas foram 100% e 96,7%; em Sinalizador12 (teste), 88,6% e 81,0%. Pose e face foram detectadas em quase 100% dos frames de todos os grupos.
- Três quadros centrais de `Banheiro` foram guardados em `logs/minds_libras_school5_diagnostics/`. Mostram diferença visível de roupa e oclusão das mãos. Na própria classe `Banheiro`, a detecção da mão esquerda foi 40,1% em Sinalizador01, 100% em Sinalizador11 e 89,1% em Sinalizador12. Isso sustenta uma hipótese de mudança de domínio na extração, mas não prova que seja a única causa do erro.

Conclusão: o pipeline de dados está completo e sem mistura de sinalizadores, mas o modelo selecionado ficou abaixo do acaso simples nas pessoas retidas. O aumento de acurácia no treino com a validação estagnada indica adaptação aos sinalizadores de treino. Antes de outro teste, analisar **somente o treino e a validação**: visualização de máscaras por vídeo, distribuição de features normalizadas e desempenho por sinalizador/classe na validação. Uma experiência controlada possível é treinar com features de pose/face versus mãos e comparar apenas em validação, ou melhorar a detecção de mãos e reextrair todos os grupos com o mesmo procedimento. O Sinalizador12 já foi consultado; não o usar para escolher novas variantes como se fosse teste virgem.

### Alternativa de landmarks pré-extraídos

A publicação [minds_mediapipe no Hugging Face](https://huggingface.co/datasets/danielelvs/minds_mediapipe) oferece um CSV de aproximadamente 3,46 GB com landmarks dos 800 vídeos; a [documentação da coleção](https://huggingface.co/datasets/danielelvs/multilingual-islr-mediapipe/blob/main/README.md) descreve as 543 posições MediaPipe por frame, os identificadores de vídeo/pessoa e recomenda separação por sinalizador. A largura e a ordem das features diferem do formato de 346 dimensões do Sinaliza; portanto, não se deve colocar esse CSV diretamente em `data/landmarks`. Uma conversão validada será necessária se essa rota for escolhida. Um teste de transferência de 1 MiB alcançou cerca de 59 KiB/s neste ambiente, ainda lento para 3,46 GB. O experimento acima permanece baseado nos vídeos específicos da versão Kaggle indicada pelo usuário.
