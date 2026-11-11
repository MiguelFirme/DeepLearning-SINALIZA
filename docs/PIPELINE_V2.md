# Pipeline v2 do Sinaliza

Implementação em fases conforme [plano](PLANO_PIPELINE_V2.md). A extração continua com 346 valores e máscara `(T,4)`. Os arquivos brutos não são modificados.

## Features e ordem

O construtor compartilhado `build_features_v2` recebe amostras em grade temporal uniforme. Corrige x pelo aspecto largura/altura; mantém somente os primeiros 17 pontos de pose e remove visibilidade; preenche lacunas internas de mãos de até cinco amostras; centraliza e escala pelos ombros com referência mediana quando a pose falta; calcula forma local por pulso/base do dedo médio; acrescenta cinco distâncias e velocidades globais em unidades por segundo. Lacunas maiores e ausências nas extremidades ficam zeradas. Sem pose válida na janela, a referência é centro zero e escala um; o bloqueio de rastreamento da aplicação continua necessário.

Dimensão e offsets: coordenadas globais `0:297` (126 mãos + 51 pose + 120 face), forma local das mãos `297:423`, cinco distâncias `423:428`, velocidades globais `428:725`. **Total: 725**. As velocidades são diferenças multiplicadas pelo FPS alvo e zeradas se uma das duas detecções falta. O primeiro vetor de velocidade é zero. A forma local das mãos não recebe outro bloco de velocidades.

## Verificações por fase

- Fase 1: `pytest tests -q`: **122 passaram, 8 ignorados** por datasets locais opcionais ausentes. Testados aspecto, remoção de pernas/visibilidade, invariância local de mão e lacunas curtas/longas.
- Fase 2: `pytest tests -q`: **126 passaram, 8 ignorados**. Reamostragem física, bordas repetidas, validade por parte, janelas com cobertura de 80%, dataset determinístico e velocidade equivalente a 9/30 FPS verificados com dados sintéticos.

## Tempo e metadados

`prepare_features_v2` é o caminho comum: reamostra os dados brutos e chama o único construtor de features. A grade termina em `end_time` e contém 48 instantes com passo 1/15 s; o intervalo entre primeiro e último é 47/15 s. O buffer ao vivo aguarda 3,2 s completos e tenta classificar a cada 0,4 s. Guarda aproximadamente 6,4 s, preservando um vizinho anterior à borda. Mudança de resolução limpa a janela.

O dataset estima a atividade com a mesma janela de 40% de maior movimento dos punhos usada na auditoria. Essa é uma heurística, não anotação do sinal. Validação/teste centralizam uma janela nessa região; treino sorteia janelas que cobrem ao menos 80% dela ou, se ela exceder a janela, sorteia dentro dela. Vídeos curtos repetem bordas. Timestamps explícitos de coleta têm prioridade; arquivos antigos usam `arange(T)/fps`. FPS ausente ou fora de 10–60 gera aviso e usa 30. Dimensões ausentes geram aviso e usam aspecto neutro 1:1, que **não recupera o aspecto real**. `metadata_issues` conta arquivos afetados uma vez ao criar o dataset.
