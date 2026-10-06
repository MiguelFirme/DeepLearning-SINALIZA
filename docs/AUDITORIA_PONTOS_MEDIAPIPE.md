# Auditoria da seleção de pontos MediaPipe

Data: 2026-10-05. Comparação da sugestão recebida com `ml/features/landmarks.py`.

| Região | Sugestão | Extração atual | Observação |
|---|---|---|---|
| Mãos | 21 pontos por mão | 21 por mão, 42 no total | Todos os pontos detalhados de cada mão são salvos quando detectados. |
| Ombros, cotovelos, pulsos | 11–16 da pose, seis pontos | Os seis estão incluídos entre os 25 da pose | A sugestão principal de pose já está coberta. |
| Outros pontos de pose | Tronco superior prioritário | Mais 19 pontos: cabeça, dedos aproximados, quadris e pernas | Os oito pontos de quadris/pernas podem ser testados por ablação sem reextrair os vídeos. |
| Face expressiva | Boca, sobrancelhas, olhos/pálpebras | 40 selecionados: 10 sobrancelhas, 8 olhos, 10 boca, 4 nariz, 6 contorno, 2 orientação | A seleção já prioriza as regiões expressivas. |

O MediaPipe Holistic calcula um conjunto maior internamente (33 pose, 468 face e 21 por mão), mas o arquivo `.npz` salva os subconjuntos acima. A configuração atual não habilita refinamento da face/íris; 478 não é o número usado neste extrator. Salvar só 40 pontos de face reduz o tamanho da entrada da rede, mas não faz o Holistic deixar de executar a detecção facial completa.

O BiLSTM `full` recebe 42+25+40=107 pontos ativos, convertidos em 346 valores brutos por frame (3 coordenadas por ponto, mais visibilidade nos 25 pontos de pose). O modo `pose_face` zera os 42 pontos detalhados das mãos, mantendo 25+40=65 pontos ativos. A pose ainda contém seis pontos grosseiros de dedos/mãos.

## Conclusão e próximo teste possível

A seleção **já inclui** todas as regiões citadas na sugestão. Não há evidência de que a baixa acurácia resulte de ter poucos pontos de sobrancelhas, boca, ombros, cotovelos ou pulsos. Mais pontos de face não corrigiriam mãos ausentes ou diferenças entre sinalizadores. Uma próxima ablação de baixo custo seria zerar os oito pontos de quadris/pernas nos `.npz` durante o carregamento, sem reexecutar o MediaPipe, e comparar na mesma validação por sinalizador. Não escolher pelo sinalizador 12 já consultado.

Fonte oficial para a contagem total do Holistic: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/holistic.md
