# Plano — repouso no início e fim dos vídeos de 25 classes

## Hipótese

Os vídeos positivos podem conter trechos sem sinal nas bordas. Esses trechos são candidatos úteis para uma classe provisória `__nenhum_sinal__`, desde que não contenham preparação, finalização ou parte do gesto. Baixo movimento sozinho não prova ausência de sinal: um sinal pode incluir uma configuração de mão estática ou expressão facial. Por isso, gerar candidatos automaticamente, revisar visualmente e treinar somente os aprovados.

## Protocolo antes da análise

1. Usar apenas os 670 vídeos das 25 classes aprovadas e os landmarks já extraídos. Preservar os originais.
2. Medir comprimento, movimento dos dois punhos relativo aos ombros, presença de pose/mãos/face e localizar janelas contíguas de 48 quadros nas bordas. Uma janela candidata deve estar inteiramente no primeiro ou último terço do vídeo e ser separada da atividade principal por uma margem. Não permitir janela que atravesse o pico de movimento. Se 48 quadros forem raros, registrar a limitação antes de considerar trechos menores.
3. Salvar CSV com vídeo de origem, pessoa, split herdado, intervalo exato, motivo de seleção e métricas. Criar prévias visuais dos candidatos e exigir decisão humana `aprovado/rejeitado` antes de rotular como `__nenhum_sinal__`.
4. Todas as janelas derivadas de um vídeo, sessão ou pessoa permanecem no split desse vídeo. Não usar o teste previamente consultado para selecionar limiares. Não extrair candidatos de teste para ajustar modelo ou limiares; usar teste apenas após congelar a regra e reportá-lo como exploratório.
5. Se a quantidade e diversidade de negativos revisados forem suficientes, treinar um Transformer de 26 classes no mesmo pipeline das 25 positivas; medir acertos positivos, recall de rejeição, falsos positivos por tipo de repouso e efeito em pessoas distintas. Manter o modelo de 25 classes como referência.

## Critério de decisão

Se as bordas não produzirem janelas longas e inequivocamente sem sinal, ou se a revisão mostrar preparação/finalização do gesto, não treinar com elas como negativos. Mesmo com bons trechos, será necessário coletar repouso, conversa e gestos fora do vocabulário em gravações separadas para avaliar rejeição no uso real.

## Registro de execução

Análise inicial: a base tem mediana de 132 quadros por vídeo. Com janelas de 48 quadros, exigência de pico de movimento fora das bordas e razão de movimento menor que 0,3, surgiram só 16 candidatos de treino e nenhum de validação. Portanto, 48 quadros estritos não sustentam um treino/validação de rejeição neste recorte.

Com janelas de 32 quadros e razão menor que 0,2, surgiram 100 candidatos de treino e 11 de validação; 107/111 pertencem ao MINDS-Libras e quatro às gravações novas. Foram geradas prévias de quatro frames por candidato em `reports/idle_edges25/previews/` e 14 folhas de contato. As folhas foram examinadas pela IA: as pessoas aparecem com braços abaixados nos frames amostrados, mas isso não equivale à revisão humana de cada clipe inteiro.

Decisão experimental: preparar um **piloto com rótulos fracos** a partir desses 111 candidatos, claramente separado dos dados aprovados. Não chamar os exemplos de revisados e não ativar o modelo resultante no backend. O treino usa apenas candidatos do split de treino, valida nos 11 candidatos de validação e preserva o teste sem negativos até congelar a regra. O processamento atual repete alguns frames ao converter 32 para 48; isso pode criar um atalho artificial de classificação e será registrado como limitação. Se o piloto indicar ganho, solicitar revisão humana dos clipes completos e coletar negativos independentes antes de uma versão utilizável.

Piloto executado e arquivado em `Modelos salvos/transformer_mixed25_edge32_rejection_pilot/`. O modelo de 26 classes sozinho acertou 29/85 sinais positivos e 11/11 repousos de validação. A fusão de 25 classes preservou 50/85 sinais quando o piloto foi usado apenas como bloqueio; no teste exploratório a cascata caiu de 56/85 para 55/85 sinais e rejeitou 20/21 trechos de 32 quadros e 6/6 janelas originais de 48 quadros. Como houve cinco rejeições indevidas de sinais no teste e nenhum negativo independente, **o backend não foi alterado para carregar esse checkpoint**. Procedimento, métricas e limitações: `docs/RELATORIO_NEGATIVOS_BORDAS_25_CLASSES_2026-10-27.md`.
