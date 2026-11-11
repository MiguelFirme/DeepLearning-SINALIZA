# Pipeline v2 do Sinaliza

Implementação em fases conforme [plano](PLANO_PIPELINE_V2.md). A extração continua com 346 valores e máscara `(T,4)`. Os arquivos brutos não são modificados.

## Features e ordem

O construtor compartilhado `build_features_v2` recebe amostras em grade temporal uniforme. Corrige x pelo aspecto largura/altura; mantém somente os primeiros 17 pontos de pose e remove visibilidade; preenche lacunas internas de mãos de até cinco amostras; centraliza e escala pelos ombros com referência mediana quando a pose falta; calcula forma local por pulso/base do dedo médio; acrescenta cinco distâncias e velocidades globais em unidades por segundo. Lacunas maiores e ausências nas extremidades ficam zeradas. Sem pose válida na janela, a referência é centro zero e escala um; o bloqueio de rastreamento da aplicação continua necessário.

Dimensão e offsets: coordenadas globais `0:297` (126 mãos + 51 pose + 120 face), forma local das mãos `297:423`, cinco distâncias `423:428`, velocidades globais `428:725`. **Total: 725**. As velocidades são diferenças multiplicadas pelo FPS alvo e zeradas se uma das duas detecções falta. O primeiro vetor de velocidade é zero. A forma local das mãos não recebe outro bloco de velocidades.

## Verificações por fase

- Fase 1: `pytest tests -q`: **122 passaram, 8 ignorados** por datasets locais opcionais ausentes. Testados aspecto, remoção de pernas/visibilidade, invariância local de mão e lacunas curtas/longas.
