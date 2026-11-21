# Transformer v2 de 39 classes

Retreino de 20/11/2026 sem Cinco, Vacina e Amarelo. Cinco já estava ausente.
Experimento: `transformer_v2_grupo39_20261120_seed42`. Peso treinado: `best.pt`.
635 vídeos, separados por pessoa em 469 treino, 83 validação e 83 teste.
Mesmo Transformer full de 725 features: pose, 40 pontos de face e mãos;
48 amostras a 15 FPS, janela de 3,2 s. d_model 128, 4 cabeças, 3 camadas,
dropout 0,15. AdamW, LR 0,001, weight_decay 0,01, cosine/warmup 5 épocas,
label smoothing 0,1, sampler balanceado e augmentation somente no treino.

Seleção por F1 macro de validação (paciência 25). Melhor época:
57; épocas executadas: 82.
Validação: top1 62.650602%, F1 macro 0.673681.
Teste reservado e não avaliado. Isso não é uma métrica de webcam.
Não há classe negativa `_nenhum_sinal` treinada.

Configurações, índices, mapa de classes, logs e métricas acompanham o modelo.
`checksums.sha256` permite conferir a integridade dos arquivos.
Reprodução e ativação: `docs/RETREINO_GRUPO39_2026-11-20.md` no repositório.
