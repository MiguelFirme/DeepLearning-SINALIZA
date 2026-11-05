# Revisão de três PDFs sobre outro protótipo de Libras

Data: 04/11/2026. [Plano da revisão](PLANO_REVISAO_PROJETOS_LIBRAS_2026-11-04.md).

## Material examinado

Foram lidas integralmente as 4 páginas de `C:/Users/migue/Downloads/explicacao_completa_projeto_libras.pdf`, as 2 páginas de `C:/Users/migue/Downloads/explicacao_tecnica_projeto_libras.pdf` e as 24 páginas de `C:/Users/migue/Downloads/RelatórioLibrasIA-LuizaLima-JuliaGabriela-MariaEduardaNunes-LaraEwellen.pdf`, incluindo os gráficos e a matriz de confusão dos anexos. As duas imagens enviadas pelo usuário resumem cinco propostas desses documentos; não foram tratadas como instruções para alterar a aplicação.

Os três PDFs descrevem o **mesmo protótipo**, portanto não são três resultados independentes. O projeto usa MediaPipe, 288 valores por quadro (33 pontos de pose, 21 por mão e dez faciais), janelas de 20 quadros, Conv1D + duas BiLSTMs + atenção, 13 classes incluindo `Desconhecido`, filtros de aceitação, formação de frase e síntese de voz. O relatório declara 66,47% de acurácia e 69,19% de acurácia balanceada em 498 **janelas** de teste (relatório, pp. 9–15). Os anexos (pp. 22–24) mostram treino quase perfeito, validação aproximadamente 89% no fim e confusões importantes para `Oi`; a matriz é coerente com o suporte de 498 janelas da tabela.

**Não comparar diretamente 66,47% com os 56/85 do Sinaliza.** São 13 contra 25 classes, 20 contra 48 quadros, janelas contra vídeos, dados/pessoas distintos e protocolos de split não equivalentes. O relatório lista cerca de 320 vídeos de origem e 498 janelas de teste, mas não explica se todas as janelas do mesmo vídeo ou do mesmo articulador permaneceram no mesmo split. Isso **não prova vazamento**; deixa a independência do teste sem verificação. A diferença entre treino/validação nos gráficos e o teste também pede cautela antes de atribuir o resultado à arquitetura.

## O que pode melhorar o Sinaliza

| Ideia dos PDFs | Situação atual no Sinaliza | Decisão e experimento necessário |
| --- | --- | --- |
| Validação prévia dos vídeos (pp. 1, 9) | `scripts/validate_landmarks.py` verifica `.npz`; `scripts/audit_mixed25_quality.py` audita 670 amostras, cobertura da mão, nitidez/escuro em quadros amostrados e produziu fila de 168 vídeos. | **Prioridade alta:** adicionar ou consolidar um preflight dos MP4 antes de reextrair: existência, decodificação de quadros, duração, resolução, classe, pessoa/sessão e contagem por classe. Revisar os 168 alertas por vídeo inteiro, sem excluir validação/teste após ver os resultados. |
| Rejeição e estabilidade (pp. 3, 11–12, 15–16) | Há EMA, cooldown de 1 s, limiar global 0,30 e bloqueio por rastreamento insuficiente. Não há margens e thresholds por classe calibrados, voto temporal formal ou classe negativa ativa. O filtro de movimento foi revertido por pedido do usuário; o piloto de negativos das bordas rejeitou indevidamente 5/85 sinais no teste exploratório. | **Prioridade alta, em etapas:** coletar negativos reais revisados (repouso, conversa, gesto incompleto e fora do vocabulário) de pessoas/sessões separadas; medir falsos positivos por minuto, sinais corretos recusados, F1 macro e atraso. Testar margens/votação por janela apenas depois de calibrar os escores da fusão; o escore top 5 atual não é uma probabilidade de rejeição calibrada. Não ativar filtros rígidos por cópia do outro projeto. |
| Conv1D + BiLSTM + atenção (pp. 2–3, 10–11) | O código já possui BiLSTM com atenção (`ml/models/bilstm.py`) e TCN com convoluções 1D (`ml/models/tcn.py`), mas a dupla ativa de 25 classes é Transformer. | **Prioridade média, como ablação:** implementar apenas a combinação Conv1D→BiLSTM→atenção e treiná-la no **mesmo** mapa de 25 classes, split congelado, 48 quadros, 697 features e normalização/augmentation equivalentes. Comparar acurácia, F1 macro por pessoa/classe, parâmetros e latência; não usar o teste já consultado para escolher hiperparâmetros. A arquitetura dos PDFs não garante ganho aqui. |
| Formação de frases e RAG (pp. 4, 7, 13, 17) | O frontend já acumula até 20 sinais numa lista separada por vírgulas, sem reconstruir gramática; não há endpoints de desfazer/limpar frase nem RAG. | **Prioridade média para a experiência:** primeiro permitir confirmar, corrigir, desfazer e limpar sinais. Só então testar um formador determinístico de frases curtas e editáveis. O PDF cita `rag_libras.py`, mas não fornece código, base de recuperação, método ou avaliação; não há evidência suficiente para adotar RAG ou apresentar uma frase gerada como tradução fiel de Libras. |
| Fala da frase completa (pp. 4, 8, 13) | `TranslationDisplay.tsx` já usa `speechSynthesis` em `pt-BR` para a palavra isolada. | **Prioridade baixa e implementação simples:** adicionar botão para falar a sequência confirmada ou frase sugerida, com controles de parar/repetir e verificação de suporte do navegador. Evitar leitura automática de palpites instáveis. |

O relatório do outro projeto também descreve feedback humano (pp. 4, 13). Isso é aplicável: salvar vídeo/landmarks, rótulo corrigido, pessoa/sessão, consentimento e versão do modelo para uma fila de revisão. O clique do usuário não deve retreinar o modelo nem alterar rótulos de teste automaticamente.

## Ordem sugerida e critérios de aceitação

1. **Qualidade dos dados:** concluir revisão humana dos vídeos sinalizados e preflight dos MP4; publicar contagem de vídeos válidos por classe/pessoa/split antes de qualquer novo treino.
2. **Rejeição e estabilidade:** gravar sessões negativas e positivas independentes, congelar splits por pessoa/sessão e definir metas mensuráveis para falsos positivos por minuto, recusas indevidas e latência entre gesto e resposta. Calibrar apenas com treino/validação novos; preservar um teste ainda não consultado.
3. **UX de frase e voz:** oferecer edição explícita antes de formar/falar texto. Testar com usuários para confirmar que a frase não sugere significado ausente nos sinais classificados.
4. **Arquitetura comparativa:** rodar Conv1D + BiLSTM + atenção somente se a qualidade e o protocolo estiverem estáveis. Registrar logs, seed, split, hashes, checkpoints e tempo por quadro. Promover um novo modelo apenas após superar a fusão atual sem degradar classes essenciais ou latência.

## Conclusão

O maior ganho provável não vem de trocar imediatamente Transformer por TensorFlow/Keras. As ideias mais úteis são uma validação bruta de vídeos mais cedo, negativos reais para rejeição e uma interface que permita corrigir sinais antes de formar ou falar frases. Conv1D + BiLSTM + atenção é um experimento legítimo, mas sua vantagem no Sinaliza permanece hipótese. Nenhum checkpoint, limiar ou fluxo ativo foi alterado nesta revisão.
