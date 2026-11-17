# Transformer v2 ampliado — 41 classes

Modelo padrão ativado em 17/11/2026. Pesos: `best.pt`.
Experimento: `transformer_v2_grupo_expandido_20261117_seed42`. Melhor época: 80, por F1 macro
da validação; treino encerrado na época 105.

Dados: 13 classes MINDS e 28 do grupo, 715 vídeos, splits por pessoa 529/93/93.
Validação: 75.27% (70/93), F1 macro 0.7512.
Teste não avaliado nesta ativação. A validação não é acurácia de webcam.

Entrada completa: pose superior, face e mãos; 725 features, 15 FPS, janela 3,2 s.
Transformer 128, quatro cabeças, três camadas, dropout 0,15. AdamW, augmentation
somente no treino, sampler balanceado, early stopping por F1 macro. A configuração
e o mapa completo de classes estão no checkpoint e arquivados neste diretório.
Este checkpoint não contém classe `_nenhum_sinal` treinada.

`config/`, `data_split/`, `records/` e `checksums.sha256` guardam a reprodução.
Não é preciso distribuir vídeos/landmarks para inferência. Consulte
`docs/ATIVACAO_TRANSFORMER_V2_41_CLASSES_2026-11-17.md`.
