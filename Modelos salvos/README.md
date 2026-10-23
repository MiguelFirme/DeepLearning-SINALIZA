# Modelos salvos do Sinaliza

Este diretório arquiva os checkpoints `best.pt` selecionados por **menor perda de validação**. Os experimentos originais em `experiments/` continuam intactos.

| Modelo | Classes | Validação | Teste | Documentação |
|---|---:|---:|---:|---|
| Transformer MINDS `pose_face` | 8 | 40/40 (100%) | 37/40 (92,5%) | [README do modelo de 8 classes](transformer_minds_8_classes_pose_face/README.md) |
| Transformer MINDS `hands_only` | 8 | 5/40 (12,5%) | 6/40 (15%), exploratório | [README do modelo de mãos](transformer_minds_8_classes_hands_only/README.md) |
| Transformer MINDS `full` | 8 | 23/40 (57,5%) | 35/40 (87,5%), exploratório | [README do modelo completo](transformer_minds_8_classes_full/README.md) |
| Transformer MINDS `pose_face` | 15 | 49/75 (65,3%) | 41/75 (54,7%), exploratório | [README do modelo de 15 classes](transformer_minds_15_classes_pose_face/README.md) |
| Transformer MINDS `full` | 15 | 45/75 (60%) | 34/75 (45,3%), exploratório | [README do modelo de 15 classes com mãos](transformer_minds_15_classes_full/README.md) |
| Transformer MINDS `pose_face` | 20 | 37/100 (37%) | Não executado | [README do modelo de 20 classes](transformer_minds_20_classes_pose_face/README.md) |
| Transformer novas palavras `pose_face` | 5 | 5/5 (100%) | 3/5 (60%), exploratório | [README das cinco palavras com pose e face](transformer_novas_5_pose_face/README.md) |
| Transformer novas palavras `full` | 5 | 5/5 (100%) | 2/5 (40%), exploratório | [README das cinco palavras com mãos](transformer_novas_5_full/README.md) |
| Transformer misto `pose_face` | 25 | 37/85 (43,5%) | 27/85 (31,8%), exploratório | [README do modelo misto de pose e face](transformer_mixed25_pose_face/README.md) |
| Transformer misto `full` | 25 | 46/85 (54,1%) | 48/85 (56,5%), exploratório | [README do modelo misto com mãos](transformer_mixed25_full/README.md) |
| Transformer novas palavras `full` | 10 | 6/10 (60%) | 6/10 (60%), exploratório | [README do modelo das dez novas](transformer_novas_10_full/README.md) |
| Transformer cinco restantes `pose_face` | 5 | 4/5 (80%) | 4/5 (80%), exploratório | [README das cinco restantes com pose e face](transformer_novas_5_restantes_pose_face/README.md) |
| Transformer cinco restantes `full` | 5 | 4/5 (80%) | 5/5 (100%), exploratório | [README das cinco restantes com mãos](transformer_novas_5_restantes_full/README.md) |

Cada subpasta guarda o checkpoint, o mapa de rótulos, o manifesto e os índices do split, os YAMLs de configuração, o registro da execução, o histórico e as métricas disponíveis. `checksums.sha256` registra a integridade dos arquivos arquivados.

**Dados necessários para reprodução:** os landmarks `.npz` permanecem em `data/landmarks/`; os MP4 originais do MINDS ficam em `data/raw/minds_libras/`, e os das cinco novas palavras em `data/raw/<classe>/`. Eles não são duplicados aqui. O `.gitignore` permite versionar especificamente estes `best.pt`, mas os arquivos ainda precisam ser adicionados e enviados ao repositório para existir fora deste computador. Os resultados são de sinais isolados, não de tradução de frases nem de usuários novos fora dos datasets.
