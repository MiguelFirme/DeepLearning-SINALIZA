# Conv1D + BiLSTM + atenção: full

Treino do zero com 25 classes (15 sinais MINDS-Libras e dez gravados pelo grupo), 500/85/85 vídeos nos mesmos splits da dupla Transformer, seed 42 e 48 quadros. Entrada de 697 features derivadas dos landmarks MediaPipe; modo `full` com 40 pontos faciais. Projeção 64 → Conv1D 64 canais/kernel 3 → duas BiLSTMs de 32 unidades por direção → atenção temporal → classificador.

Validação: 49/85, F1 macro 0.462. Teste exploratório já consultado anteriormente: 63/85, F1 macro 0.649. O checkpoint foi escolhido exclusivamente pela menor `val_loss`; o teste não orientou o treino.

Arquivos: `best.pt`, `checksums.sha256`, `config/`, `data_split/` e `records/`. Vídeos e `.npz` não são duplicados. Veja `docs/RELATORIO_CONV_BILSTM_ATENCAO_MIXED25_2026-11-04.md` para a comparação completa.
