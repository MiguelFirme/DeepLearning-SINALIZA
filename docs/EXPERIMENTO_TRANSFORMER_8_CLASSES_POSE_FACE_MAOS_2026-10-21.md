# Experimento — Transformer de 8 classes com pose, face e mãos

## Pergunta

Adicionar os landmarks das mãos melhora o Transformer de oito classes treinado com pose e face no MINDS-Libras?

## Plano anterior ao treino

1. Usar os mesmos 320 arquivos `.npz`, manifesto e split por sinalizador do modelo `pose_face` arquivado: 240 vídeos de treino (Sinalizador01, 02, 05, 06, 08, 10), 40 de validação (Sinalizador11) e 40 de teste (Sinalizador12).
2. Usar `configs/transformer_minds_full.yaml`, já existente. Ele herda toda a configuração do Transformer `pose_face` e altera somente `data.feature_mode` para `full`, preservando arquitetura, augmentation, loss, otimizador, seed, 48 frames e critério de parada. Conferir programaticamente essa diferença antes do treino.
3. Treinar em experimento novo, sem sobrescrever checkpoints existentes. Escolher `best.pt` somente pela menor `val_loss`; testar uma vez o checkpoint escolhido. O sinalizador de teste já foi consultado em experimentos anteriores, portanto seu resultado será exploratório.
4. Comparar métricas e previsões por classe com o checkpoint `pose_face` no mesmo split. Se o modelo completo não melhorar na validação, mantê-lo separado do backend.
5. Salvar checkpoint, cadeia completa de configuração, splits, histórico, métricas, previsões, logs e README em `Modelos salvos/transformer_minds_8_classes_full/`. Conferir carregamento estrito e hashes.

## Risco conhecido antes do treino

A auditoria em `docs/AUDITORIA_MAOS_MINDS_8_CLASSES.md` mediu ao menos uma mão em 48,77% dos quadros de treino e 99,75% dos de validação. Essa mudança de distribuição pode prejudicar o modelo completo, apesar do ganho potencial de configuração e movimento dos dedos. As porcentagens incluem repouso; não são medidas da mão ativa. O ramo `hands_only` anterior atingiu 12,5% na validação. A hipótese de benefício deve ser testada; não será presumida.

## Execução e resultados — 21/10/2026

- Configurações efetivas verificadas: o único valor diferente entre controle e experimento é `data.feature_mode` (`pose_face` → `full`). Split idêntico de 240/40/40 vídeos, oito classes; treino em CUDA.
- Treino `experiments/transformer_minds_school8_full_seed42/`: Transformer de 121.018 parâmetros, parada na época 49, melhor checkpoint na época 34 por `val_loss=0,9659`. Na última época, `train_top1=98,75%`, `val_top1=50%`; o checkpoint escolhido tinha `val_top1=57,5%`.
- Validação do checkpoint: **23/40 (57,5%), F1 macro 0,495**. Controle `pose_face`: **40/40 (100%)**. Erros completos: `Aluno` 5/5 → `Medo`, `Ruim` 5/5 → `Conhecer`, `Vontade` 5/5 → `Conhecer`; `Medo` 2/5 → `Vontade`.
- Teste exploratório: **35/40 (87,5%), F1 macro 0,856**. Controle `pose_face`: **37/40 (92,5%)**. O teste não foi usado para selecionar checkpoint nem para retreinar.
- O arquivo arquivado em `Modelos salvos/transformer_minds_8_classes_full/` contém checkpoint, configuração, splits, logs, histórico, métricas, previsões e instruções de reprodução.
- Verificação final: o SHA-256 do checkpoint arquivado coincide com o original (`2541ef019002efddb5db186bb694dfa522bab0c236a0199b2bfed63c084c3313`); o backend o carrega com pesos estritos e `feature_mode=full`. Uma amostra real confirmou que a entrada difere de `pose_face` nos pontos das mãos. Os 23 checksums do pacote passaram; 11 testes de modalidades, normalização e backend passaram.

## Interpretação e decisão

A inclusão direta dos pontos das mãos prejudicou este experimento controlado, sobretudo no sinalizador de validação. O padrão é compatível com a diferença de detecção das mãos entre sinalizadores, mas também pode envolver diferenças de gravação e execução dos sinais. A causa exata não foi isolada. O desempenho de treino alto com validação baixa indica generalização insuficiente. **Manter o modelo `pose_face` no backend; preservar o modelo completo como resultado experimental.** Um próximo estudo deve inspecionar a qualidade da mão ativa nos quadros do gesto e avaliar estratégias de fusão somente com validação por pessoas novas.
