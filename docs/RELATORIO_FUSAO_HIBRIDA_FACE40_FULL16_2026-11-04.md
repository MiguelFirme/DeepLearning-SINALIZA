# Teste da fusão invertida: pose + face40, modelo completo face16

Data: 04/11/2026. [Plano registrado antes da avaliação](PLANO_FUSAO_HIBRIDA_FACE40_FULL16_2026-11-04.md).

## Configuração

Usamos os checkpoints existentes `transformer_mixed25_pose_face` (40 pontos faciais, sem mãos) e `transformer_mixed25_full_face16` (16 pontos faciais, com mãos). O primeiro propôs as cinco classes mais prováveis; o segundo participou da reordenação. Mantivemos o escore `0,22 × pose_face40 + 0,78 × full_face16` e as temperaturas originais de `1,028734578` e `0,792446596`. Não houve novo treino, extração de landmarks nem ajuste de calibração.

Os 85 caminhos e rótulos de cada split foram comparados aos da dupla original e da híbrida anterior. Os CSVs de 16 pontos já continham probabilidades calibradas; somente as probabilidades brutas de 40 pontos foram calibradas neste script.

## Resultado

| Dupla | Validação: acertos / 85 | Validação: F1 macro | Teste exploratório: acertos / 85 | Teste: F1 macro |
| --- | ---: | ---: | ---: | ---: |
| pose_face40 + full40, original | **50 (58,8%)** | **0,452** | **56 (65,9%)** | **0,480** |
| pose_face16 + full40, híbrida anterior | 46 (54,1%) | 0,374 | 49 (57,6%) | 0,430 |
| **pose_face40 + full16, híbrida invertida** | **49 (57,6%)** | **0,401** | **48 (56,5%)** | **0,422** |

A híbrida invertida ficou a um acerto da original na validação, mas oito acertos abaixo no teste exploratório. Em comparação à original, corrigiu cinco vídeos e errou seis que antes estavam corretos na validação; no teste, corrigiu cinco e perdeu 13. Nas dez classes novas, teve 3/10 na validação e 5/10 no teste, contra 4/10 em ambos os splits da original.

Na validação, os maiores deslocamentos por classe em relação à original foram `Barulho −4` e `Vontade +5`. No teste exploratório, `Aluno −5`, `Conhecer −4`, `Vacina +3` e `Barulho −2`. Cada classe antiga possui apenas cinco vídeos por split, portanto a diferença por classe é muito sensível a poucos vídeos. O [JSON de comparação](../reports/fusion_mixed25_hybrid_face40_full16/comparison.json) inclui todas as classes e a análise pareada.

**Conclusão:** essa inversão ficou próxima da original em acurácia de validação, porém o F1 macro foi menor e o teste exploratório também foi pior. Entre as duplas avaliadas com a regra fixa, a original de 40 + 40 pontos continua com o melhor desempenho global. Não foi feita troca do modelo padrão da aplicação.

O conjunto de teste já foi consultado anteriormente e não confirma generalização independente. Os pesos e temperaturas foram escolhidos para a dupla original, podendo ser subótimos para esta combinação. O experimento também não mede pessoas inéditas nem latência da câmera ao vivo.

## Arquivos e reprodução

- Avaliação: `scripts/evaluate_hybrid_face40_full16.py`.
- Predições por vídeo: `experiments/fusion_mixed25_hybrid_face40_full16/val_pose_face_predictions.csv` e `test_pose_face_predictions.csv`.
- Relatório de métricas: `reports/fusion_mixed25_hybrid_face40_full16/comparison.json`.
- Manifesto experimental com hashes: `configs/fusion25_hybrid_face40_full16.json`.
- Teste de equivalência offline/backend: `tests/test_fusion25_hybrid_face40_full16_runtime.py`.

No Git Bash do VS Code, a partir da raiz:

```bash
./.venv/Scripts/python.exe -m scripts.evaluate_hybrid_face40_full16
./.venv/Scripts/python.exe -m pytest tests/test_fusion25_hybrid_face40_full16_runtime.py tests/test_fusion25_hybrid_runtime.py tests/test_fusion25_face16_runtime.py tests/test_fusion25_protocol.py -q
```

O manifesto pode ser selecionado para avaliação manual por `SINALIZA_FUSION_MANIFEST_PATH=configs/fusion25_hybrid_face40_full16.json`. Os sete testes focados passaram; a inferência da dupla reproduziu as probabilidades, as cinco candidatas e o escore salvos no CSV de validação.
