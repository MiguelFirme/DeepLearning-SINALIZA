# Conv1D + Transformer + atenção temporal: full

Treino do zero nas mesmas 25 classes e splits 500/85/85 dos modelos anteriores, seed 42 e 48 quadros. Entrada de 697 features derivadas de landmarks com 40 pontos faciais. Modo `full`. Projeção 64 → Conv1D 64/kernel 3 → duas camadas Transformer (quatro cabeças, feedforward 128) → atenção temporal → classificador. Total de 136,907 parâmetros.

Validação: 37/85, F1 macro 0.280. Teste exploratório anteriormente consultado: 46/85, F1 macro 0.513. O melhor checkpoint foi escolhido somente por `val_loss`; não houve escolha pelo teste.

Arquivos: `best.pt`, `checksums.sha256`, `config/`, `data_split/`, `records/`. Os vídeos e `.npz` não são duplicados. Veja `docs/RELATORIO_CONV_TRANSFORMER_ATENCAO_MIXED25_2026-11-04.md`.
