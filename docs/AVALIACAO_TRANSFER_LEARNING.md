# Avaliação da sugestão de transfer learning para o Sinaliza

Data: 2026-10-05.

## Contexto

O piloto MINDS-Libras tem cinco sinais isolados, 200 vídeos e divisão por sinalizador (150 treino, 25 validação, 25 teste). O BiLSTM de landmarks completos obteve 20% na validação; a ablação pose+face chegou a 44% no checkpoint de menor perda, ainda com colapso em algumas classes. O teste do sinalizador 12 já foi consultado e não deve orientar novas escolhas. Os experimentos de MediaPipe Hands em frame completo e em recortes guiados pela pose não recuperaram consistentemente as mãos ausentes.

## Análise das propostas

| Proposta | Adequação ao piloto | Motivo |
|---|---|---|
| Modelo pré-treinado em sinais isolados, como WLASL | Primeira hipótese de transferência | Pode ter aprendido movimento e formato de mãos; ASL e Libras têm vocabulários e variantes diferentes, então transferir representações visuais, não rótulos diretamente. |
| Backbone de vídeo pré-treinado em Kinetics-400 | Segunda hipótese, como controle | Aprende movimento humano geral, mas pode dar pouca ênfase a detalhes finos de dedos e expressões. |
| PHOENIX-2014T | Etapa futura de pesquisa | Corpus de sinais contínuos em língua de sinais alemã, domínio de previsão do tempo e tradução para alemão; desalinhado com cinco sinais isolados de Libras. |
| Tradução fim a fim de frases | Meta posterior | Os vídeos atuais têm rótulo de sinal isolado, sem pares vídeo-frase em português; é preciso outro conjunto de dados e avaliação. |
| Atenção espacial/temporal | Opção arquitetural posterior | Pode ponderar regiões e momentos, mas mapas de atenção não comprovam por si só que o modelo aprendeu o gesto correto. |

## Plano experimental recomendado

1. Auditar manualmente os trechos ativos e a visibilidade das mãos em alguns vídeos de treino e validação. Sinalizar vídeos em que o gesto está cortado ou borrado.
2. Selecionar um checkpoint visual de vídeo com pesos e licença verificáveis. O pipeline atual de landmarks (346 números por frame) não aceita diretamente pesos de uma rede treinada com pixels RGB: será necessário um fluxo de leitura dos MP4s, amostragem temporal e pré-processamento compatíveis com o checkpoint.
3. Primeiro congelar o encoder e treinar um classificador pequeno nas 150 amostras de treino. Depois, se houver sinal de generalização, descongelar parte das camadas com taxa de aprendizado menor. Comparar com o BiLSTM pose+face mantendo os mesmos cinco rótulos, pessoas e critério de seleção por validação.
4. Registrar versão do checkpoint, origem dos pesos, licença, amostragem de frames, tamanho de entrada, parâmetros treináveis, memória/tempo, seed, métricas por classe e matriz de confusão. Não usar o sinalizador 12 para escolher arquitetura ou hiperparâmetros.
5. Só avançar para frase contínua com vídeos de frases em Libras e transcrição em português, avaliados com separação por pessoa e sessão.

## Decisão atual

Vale investigar transferência de um encoder visual de sinais isolados. Não há garantia de superar o baseline: diferenças de língua, câmera, roupa e qualidade do vídeo continuam. Priorizar um ensaio pequeno com encoder congelado antes de qualquer treinamento grande. Não alterar o extrator atual ou os splits por causa apenas da sugestão.

### Escolha concreta para o primeiro teste

`WLASL` é o nome do **dataset**, não de uma arquitetura. O repositório oficial disponibiliza pesos de um **I3D RGB treinado em WLASL2000** (além de pesos I3D pré-treinados em Kinetics). Isso o torna uma comparação cientificamente relevante para sinais isolados. Porém, o repositório informa licença C-UDA e ausência de uso comercial para seus dados; qualquer uso dos recursos WLASL no produto precisa de revisão dos termos aplicáveis aos pesos. O checkpoint também depende de código e pré-processamento específicos do projeto.

O usuário esclareceu que o Sinaliza é **projeto acadêmico de extensão**. Portanto, para o **primeiro experimento científico**, escolher o I3D RGB treinado no WLASL2000, usando os pesos oficiais disponíveis no repositório. Ler e registrar o acordo C-UDA antes de usar os recursos. Transferir o encoder visual, remover o classificador de 2.000 rótulos ASL e treinar uma cabeça de cinco classes Libras com as 150 amostras de treino. Rodar a extração de embeddings em lotes pequenos e sem gradiente na RTX 4050 Laptop de 6 GB. A condição acadêmica informada pelo usuário elimina a hipótese de uso comercial que motivou a recomendação anterior; uma publicação ou uso futuro com outras condições deve ser reavaliado.

`torchvision.models.video.r3d_18` com `R3D_18_Weights.KINETICS400_V1` fica como **controle** posterior, com 16 frames e pré-processamento oficial. A documentação informa checkpoint de 127,4 MB, cerca de 33,4 milhões de parâmetros e entrada de 112×112 após transformação. Manter sempre o Sinalizador12 fora da seleção. A escolha WLASL ainda exige um pipeline RGB novo e compatibilidade exata com a arquitetura e os pesos oficiais; não se deve carregá-los diretamente no BiLSTM de landmarks.

## Fontes primárias

- WLASL, artigo: https://openaccess.thecvf.com/content_WACV_2020/papers/Li_Word-level_Deep_Sign_Language_Recognition_from_Video_A_New_Large-scale_WACV_2020_paper.pdf
- PHOENIX-2014T, página do corpus: https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/
- Kinetics-400 e transferência para tarefas menores: https://openaccess.thecvf.com/content_cvpr_2017/papers/Carreira_Quo_Vadis_Action_CVPR_2017_paper.pdf
- Repositório oficial WLASL, pesos I3D e condições de uso: https://github.com/dxli94/WLASL
- Torchvision R3D-18, pesos e pré-processamento: https://docs.pytorch.org/vision/0.26/models/generated/torchvision.models.video.r3d_18.html
- Licença do Torchvision: https://github.com/pytorch/vision/blob/main/LICENSE
