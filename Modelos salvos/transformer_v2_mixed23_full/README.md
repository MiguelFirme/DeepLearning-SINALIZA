# Transformer v2 de 23 classes — entrada completa

Modelo padrão ativado em 13/11/2026. Arquivo de pesos: `best.pt`.

Treino: `transformer_v2_mixed23_seed42`, 440/75/75 vídeos por pessoa em
treino/validação/teste; 13 classes MINDS e dez do grupo. Seleção por F1 macro da
validação, época 69; early stopping na época 94.
Validação: 80.0%, F1 macro 0.8176.
O conjunto de teste não foi avaliado nesta ativação.

Pipeline v2: correção de aspecto, pose superior sem visibilidade, mãos globais e
locais, distâncias e velocidade por segundo; 725 features, 15 FPS, janela 3,2 s.
Transformer com dimensão 128, quatro cabeças, três camadas e dropout 0,15.
AdamW 1e-3/L2 0,01, warmup 5, cosine, sampler balanceado e augmentation no treino.
Não há classe `_nenhum_sinal` treinada neste checkpoint.

O checkpoint inclui mapa de classes e configuração de pré-processamento. Não
precisa dos dados brutos para inferência. `config/`, `data_split/` e `records/`
preservam o protocolo, configurações e logs. `checksums.sha256` verifica os arquivos.
Leia `docs/RESULTADO_TREINO_V2_MIXED23_2026-11-12.md` e
`docs/ATIVACAO_TRANSFORMER_V2_PADRAO_2026-11-13.md` para limites e execução.
