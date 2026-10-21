# Proposta: Transformer de maos e fusao com pose-face (8 classes)

## Objetivo e estado

Avaliar se os landmarks detalhados das maos acrescentam informacao ao Transformer `pose_face` de 8 classes. Este documento planeja o experimento; nenhum segundo modelo foi treinado nem integrado ao backend ate agora.

## Evidencia do projeto

- O modelo atual foi treinado em 320 videos, com 240/40/40 em treino/validacao/teste por sinalizador. O teste registrado foi 37/40 (92,5%), mas o sinalizador de teste ja foi consultado em experimentos anteriores; o numero e exploratorio.
- O `.npz` de cada video contem 126 valores de maos por frame (21 pontos x 3 coordenadas x 2 maos) e uma mascara de deteccao para cada mao. Assim, nao e preciso baixar nem extrair videos novamente para um primeiro teste.
- Uma auditoria anterior, restrita a cinco classes, encontrou forte variacao na deteccao de ambas as maos entre classes e sinalizadores: por exemplo, 41,1% dos frames de treino de `Banheiro` contra 0% de `Medo`. Esses numeros nao substituem uma auditoria especifica das oito classes.
- `LandmarkNormalizer` atualmente usa centro/distancia dos ombros. Zerar pose antes da normalizacao mudaria essa referencia; manter ombros para normalizar e depois ocultar pose do classificador seria um fluxo de maos com referencia corporal, nao estritamente maos apenas.

## Plano de experimento

1. Auditar as oito classes por video e por sinalizador: proporcao de quadros com mao esquerda, direita, ambas e nenhuma; inspecionar a qualidade visual dos casos sem deteccao. Nao preencher uma mao ausente com coordenadas inventadas.
2. Treinar um Transformer de maos com as mesmas oito classes, mesmos videos e splits, 48 quadros, arquitetura temporal e hiperparametros comparaveis (seed, augmentation apenas no treino, loss, otimizador, scheduler e early stopping). A projecao de entrada tera outra dimensao se forem usadas somente features das maos; por isso a contagem de pesos nao sera identica.
3. Definir explicitamente as features de maos: coordenadas relativas ao pulso/palma para a forma dos dedos, posicao das maos no quadro se desejada, velocidade e mascaras de deteccao. A mesma transformacao deve ser usada no treino e na inferencia. Registrar se alguma referencia de pose e usada na normalizacao.
4. Avaliar primeiro o modelo de maos sozinho, com acuracia, F1 macro, matriz de confusao e qualidade por classe/sinalizador. Verificar se seus acertos complementam os erros do pose-face nos mesmos videos, sem escolher a estrategia pelo teste ja consultado.
5. Comparar uma fusao de probabilidades por classe, `p = (1 - w) * p_pose_face + w * p_maos`, com `w` limitado pela fracao de quadros com maos detectadas. Escolher o peso e qualquer calibracao somente em validacao adequada, preferencialmente validacao por grupos entre os sinalizadores de treino, porque a validacao atual de 40 videos ja esta no teto de 100%.
6. Congelar a regra de fusao e avaliar em pessoas/gravacoes novas ainda nao usadas para escolher modelos ou pesos. Medir tambem latencia por quadro e tempo ate a traducao na webcam.

## Regra de decisao proposta

**Estado da etapa 1:** a contagem quantitativa dos 320 videos foi concluida em [AUDITORIA_MAOS_MINDS_8_CLASSES.md](AUDITORIA_MAOS_MINDS_8_CLASSES.md). Para sinais predominantemente de uma mao, como `Medo`, a cobertura de ambas as maos nao e criterio apropriado. No treino de `Medo`, pelo menos uma mao foi detectada em 44,26% dos quadros do video inteiro, contra 100% na validacao. Inspecao visual e recorte do intervalo ativo ainda faltam. O treino do ramo de maos pode ser feito como diagnostico, mas a integracao automatica e a definicao de peso dependem de validar a cobertura da mao ativa e o desempenho em pessoas novas.

O encontro dos top 3 pode ser mostrado como sinal de concordancia, mas nao decide sozinho: com oito classes, top 3 de cada modelo e uma lista ampla, e a intersecao pode conter mais de uma palavra ou estar vazia. Escolher o maior percentual bruto entre os seis palpites tambem e arriscado porque confiancas de modelos diferentes nao sao necessariamente comparaveis. A fusao por classe permite testar pesos e reduzir o peso do modelo de maos quando a deteccao falha. Se o modelo de maos nao trouxer ganho em videos novos, manter apenas o pose-face.

## Limites

Um modelo estritamente de maos perde informacoes sobre localizacao do sinal em relacao ao corpo, expressao facial e movimento dos bracos. A extracao MediaPipe atual continua sendo o custo principal por quadro; reutilizar os mesmos landmarks nos dois modelos evita extracao duplicada, mas o segundo Transformer acrescenta alguma latencia. O teste atual do pose-face ja erra apenas tres de 40 videos do dataset, entao ganhos nesse conjunto sao pequenos e instaveis; o foco deve incluir pessoas e cameras novas.

## Referencias tecnicas

- [SignBERT (ICCV 2021)](https://arxiv.org/abs/2110.05382): modelagem de sequencias de pontos das maos e comparacao com fusao de outras modalidades.
- [Skeleton Aware Multi-Modal Sign Language Recognition (CVPR Workshops 2021)](https://arxiv.org/abs/2103.08833): reconhecimento com esqueleto e combinacao de modalidades.
