# Teste da fusão híbrida: face16 + modelo completo de 40 pontos

Data: 04/11/2026. [Plano e protocolo fixado](PLANO_FUSAO_HIBRIDA_FACE16_FULL40_2026-11-04.md).

## Configuração

Foram usados os checkpoints já treinados `transformer_mixed25_pose_face16` (pose e 16 pontos faciais, sem mãos) e `transformer_mixed25_full` (pose, 40 pontos faciais e ambas as mãos). O vocabulário tem as mesmas 25 classes. Os vídeos e rótulos foram conferidos um a um: 85 de validação e 85 de teste. Nenhum checkpoint foi retreinado.

A regra herdada da fusão original propõe as cinco classes mais prováveis do primeiro modelo e as reordena por `0,22 × pose_face16 + 0,78 × full40`. As temperaturas originais (`1,028734578` e `0,792446596`) foram mantidas sem ajuste. O CSV de `pose_face16` já contém probabilidades calibradas; o do `full40` foi calibrado uma vez neste script.

## Resultados

| Dupla | Validação: acertos / 85 | Validação: F1 macro | Teste exploratório: acertos / 85 | Teste: F1 macro |
| --- | ---: | ---: | ---: | ---: |
| pose_face40 + full40, original | **50 (58,8%)** | **0,452** | **56 (65,9%)** | **0,480** |
| pose_face16 + full16 | 45 (52,9%) | 0,382 | 42 (49,4%) | 0,359 |
| **pose_face16 + full40, híbrida** | **46 (54,1%)** | **0,374** | **49 (57,6%)** | **0,430** |

Em relação à dupla original, a híbrida não corrigiu nenhum vídeo: perdeu quatro acertos na validação (`Cansado −1`, `Medo −2`, `Vacina −1`) e sete no teste (`Barulho −4`, `Filho −1`, `Medo −1`, `Vacina −1`). Nas dez classes novas, a híbrida acertou 3/10 na validação e 4/10 no teste, contra 4/10 e 4/10 da original. Em comparação com a dupla 16/16, a híbrida ganhou um acerto na validação e sete no teste, mas seu F1 macro na validação foi menor.

**Conclusão:** com a regra e calibração já congeladas, a combinação híbrida não supera a dupla original de 40 pontos. O modelo original permanece a melhor opção entre as três fusões avaliadas. Esse resultado não demonstra que 16 pontos sejam sempre piores: o modelo `pose_face16` isolado teve melhora no teste anterior, mas sua lista top 5 e a reordenação com `full40` causaram perdas nesta combinação.

O teste foi consultado em experimentos anteriores, então estes 85 vídeos não são confirmação independente. As temperaturas e o peso foram calibrados na dupla de 40 pontos; podem não ser os ideais para a híbrida. O resultado não mede generalização em novos sinalizadores nem acurácia de uso em câmera ao vivo.

## Artefatos e reprodução

- Script: `scripts/evaluate_hybrid_face16_full40.py`.
- Predições por vídeo: `experiments/fusion_mixed25_hybrid/val_pose_face_predictions.csv` e `test_pose_face_predictions.csv`.
- Métricas por classe e comparação pareada: `reports/fusion_mixed25_hybrid/comparison.json`.
- Manifesto experimental com hashes: `configs/fusion25_hybrid_face16_full40.json`.
- Teste de equivalência offline/backend: `tests/test_fusion25_hybrid_runtime.py`.

No Git Bash, a partir da raiz do repositório:

```bash
./.venv/Scripts/python.exe -m scripts.evaluate_hybrid_face16_full40
./.venv/Scripts/python.exe -m pytest tests/test_fusion25_hybrid_runtime.py tests/test_fusion25_face16_runtime.py tests/test_fusion25_protocol.py -q
```

O manifesto pode ser selecionado para uma avaliação manual via `SINALIZA_FUSION_MANIFEST_PATH=configs/fusion25_hybrid_face16_full40.json`. O modo padrão do backend continua sendo o de oito classes. Os seis testes focados passaram; a inferência real da dupla híbrida reproduziu as probabilidades, as candidatas e o escore do CSV offline em um vídeo de validação.
