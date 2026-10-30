# Plano de qualidade, rejeição e fusão — vocabulário de 25 classes

## Escopo congelado antes da implementação

Vocabulário: as 15 classes de `configs/minds_libras_15_sem_cinco.json` e as dez de `configs/novas_10_palavras.json`. Base: `data/landmarks/mixed_25/` e divisão já registrada em `data/processed/mixed_25/` (500 treino, 85 validação, 85 teste). Nenhuma classe positiva externa será acrescentada. As pessoas de teste dessa base já foram avaliadas em experimentos anteriores: toda medição nesses vídeos continua **exploratória**.

## Etapas

1. **Auditoria dos dados.** Medir por vídeo a mão ativa estimada a partir do movimento de punho/cotovelo relativo ao ombro, a cobertura dos 21 pontos dessa mão durante o intervalo ativo, a continuidade da geometria, frames válidos, borrão e corte junto às bordas. Salvar CSV e JSON com limiares explícitos e uma fila de revisão. Uma falha automática é indício para inspeção; só uma revisão humana pode marcar uma gravação para correção ou exclusão. Preservar todos os vídeos e `.npz` originais.
2. **Exemplos de rejeição.** Como não existem gravações de repouso, conversa ou gestos incompletos, criar o formato de coleta e uma ferramenta que valide rótulo, duração, origem, pessoa, sessão e split antes de gerar landmarks. A classe `__nenhum_sinal__` será treinada somente após arquivos reais serem adicionados e revisados. O backend poderá expor um estado de rejeição conservador e explícito, mas seus limiares não serão descritos como calibrados sem exemplos negativos.
3. **Calibração e fusão.** Usar exclusivamente os 85 vídeos de validação para ajustar temperaturas dos dois Transformers de 25 classes e comparar regras de fusão, inclusive `full` isolado. Fixar checkpoints por SHA-256, ordem, top K, peso, temperaturas e critério de decisão em um manifesto versionável. Medir acurácia, F1 macro, erro de calibração e latência de inferência isolada e dupla no hardware local. O teste já consultado será relatado somente como análise secundária, sem reotimizar decisões.
4. **Backend.** Carregar os dois checkpoints compatíveis de 25 classes, usar o mesmo pré-processamento de treino para cada modalidade, responder com palavra, estado e tempos. A fusão será selecionável por configuração e só carregará quando os hashes e o mapa de classes forem compatíveis com o manifesto congelado. O modo de 8 classes atual continuará disponível como configuração.
5. **Augmentation.** Registrar uma configuração experimental com perturbações de escala, rotação e ruído mais fortes, mantendo máscaras e sem aplicar transformações em validação/teste. Treinar uma ablação apenas depois de revisar os alertas de qualidade; comparar no mesmo split e sem alterar o protocolo de teste.

## Critérios e limites

- Nenhum vídeo rotulado como sinal será convertido automaticamente em negativo.
- Nenhum dado com alerta automático será apagado ou corrigido sem revisão visual.
- Não chamar limiar de confiança ou temperatura de calibrado para rejeição sem exemplos negativos independentes.
- Seleção para implantação: uma fusão só supera `full` isolado se aumentar o F1 macro de validação em pelo menos 0,02, não reduzir os acertos nas dez classes novas e não reduzir os acertos gerais. Em empate ou ganho menor, usar `full` isolado para evitar a segunda inferência. A regra é definida antes do ajuste abaixo.
- A implementação deve gerar artefatos suficientes para reprodução: hashes, comandos, contagens, versões de configuração, CSV por vídeo e medições de latência.

## Diário de execução

1. Auditoria de landmarks executada nos 670 vídeos; 168 entraram na fila de revisão. Um piloto visual de dez vídeos confirmou a decodificação e não encontrou borrão pelo limiar automático. A revisão humana e as correções seguem pendentes; nenhuma amostra foi modificada.
2. Pipeline de negativos e treinamento de 26 classes preparado. O usuário confirmou que ainda não tem vídeos negativos; a ferramenta rejeita um formulário vazio. Não existe checkpoint de rejeição.
3. Calibração em validação e regra de fusão de 25 classes congeladas em `configs/fusion25_frozen.json`, com hashes de pesos/dados/splits. A regra escolhida foi pose/face top 5, reordenamento com peso 0,22 pose/face. No teste antigo, previamente consultado, 56/85 acertos exploratórios; sem ajuste posterior.
4. Backend opcional integrado por `SINALIZA_FUSION_MANIFEST_PATH`. Testes de paridade, hash, sessão e inicialização HTTP passaram. Medição em CPU: 6,29 ms p50 na dupla inferência de landmarks e 110,47 ms p50 por quadro JPEG + MediaPipe + inferência num vídeo gravado. O mesmo benchmark de landmarks em CUDA mediu 7,97 ms p50.
5. Ablação de augmentation mais forte registrada em configuração, ainda não treinada por depender da revisão dos 168 alertas. Relatório, comandos, limites e próximos passos em `docs/RELATORIO_QUALIDADE_FUSAO_25_CLASSES_2026-10-27.md`.
