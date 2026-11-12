# Resultado do primeiro treino v2 mixed23

Verificação em 12/11/2026, a partir do `run_record.json`, histórico e `val_metrics.json` locais. Execução: `transformer_v2_mixed23_seed42`. Status registrado: **completed**. Não foi iniciado outro treino nem consultado o split de teste nesta análise.

## Execução e artefatos

- 23 classes, 590 vídeos: 440 treino, 75 validação e 75 teste.
- Pipeline v2 com 725 features, 15 FPS e janela de 3,2 s; Transformer de entrada completa (pose, face e mãos).
- Melhor checkpoint por F1 macro da validação: **época 69**.
- Early stopping: época **94**, após 25 épocas sem melhora no monitor.
- Duração registrada: **367,43 s**, aproximadamente 6 min 7 s.
- Checkpoint: `experiments/transformer_v2_mixed23_seed42/best.pt`.
- Histórico e métricas: no mesmo diretório; log e registro em `logs/experiments/transformer_v2_mixed23_seed42/`.
- Nenhum arquivo teve fallback de FPS ou dimensões nos três splits, segundo `metadata_issues`.

## Validação do melhor checkpoint

| Medida | Resultado |
| --- | ---: |
| Top 1 | **60/75 — 80,0%** |
| Top 3 | 67/75 — 89,3% |
| Top 5 | 71/75 — 94,7% |
| F1 macro | **0,8176** |
| MINDS | 50/65 — 76,9% |
| Palavras do grupo | 10/10 — 100% |

Erros completos da matriz de validação:

- Aproveitar → Comer: 4 vídeos (Aproveitar acertou 1/5).
- Filho → Aluno: 4 vídeos; Filho → Vacina: 1 vídeo (Filho acertou 0/5).
- Medo → Vacina: 5 vídeos (Medo acertou 0/5).
- Vontade → Amarelo: 1 vídeo (Vontade acertou 4/5).

## Interpretação e próximos passos

As dez palavras próprias acertaram todos os exemplos de validação, mas há **somente um vídeo por classe** e uma pessoa dessa origem no split; isso não comprova generalização para usuários novos. A validação MINDS usa cinco exemplos por classe e outra pessoa separada do treino.

Na época selecionada, `train_top1` foi 98,64% com amostragem balanceada/augmentation, enquanto `val_top1` foi 80%. Após essa época o treino chegou a 99,32% na última época e a validação caiu para 60%; o early stopping preservou o checkpoint melhor. A diferença entre treino/validação permanece e não deve ser interpretada como problema totalmente resolvido.

Os 75 vídeos de teste **não foram avaliados**, conforme `--skip-test`; não há `test_metrics.json`. Também faltam simulação de FPS com o checkpoint real e avaliação da webcam. Não se deve comparar diretamente este resultado de validação em 23 classes com o teste exploratório antigo de 25 classes.

O treino não adicionou exemplos `_nenhum_sinal`; portanto, ainda não é um modelo de rejeição treinada. O padrão da aplicação continua v1 até ativação explícita da v2. Para testar o checkpoint ao vivo:

```bash
SINALIZA_FUSION_MANIFEST_PATH='' SINALIZA_MODEL_PATH=experiments/transformer_v2_mixed23_seed42/best.pt ./.venv/Scripts/python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

O [guia de treino mixed23](TREINO_V2_MIXED23_GRUPO_2026-11-12.md) registra o protocolo e os comandos de reprodução; [PIPELINE_V2.md](PIPELINE_V2.md) descreve a simulação de FPS, adaptando o caminho do checkpoint para este experimento.
