# Modelos salvos do Sinaliza

Este diretório arquiva os checkpoints `best.pt` selecionados por **menor perda de validação**. Os experimentos originais em `experiments/` continuam intactos.

| Modelo | Classes | Validação | Teste | Documentação |
|---|---:|---:|---:|---|
| Transformer MINDS `pose_face` | 8 | 40/40 (100%) | 37/40 (92,5%) | [README do modelo de 8 classes](transformer_minds_8_classes_pose_face/README.md) |
| Transformer MINDS `hands_only` | 8 | 5/40 (12,5%) | 6/40 (15%), exploratório | [README do modelo de mãos](transformer_minds_8_classes_hands_only/README.md) |
| Transformer MINDS `full` | 8 | 23/40 (57,5%) | 35/40 (87,5%), exploratório | [README do modelo completo](transformer_minds_8_classes_full/README.md) |
| Transformer MINDS `pose_face` | 15 | 49/75 (65,3%) | 41/75 (54,7%), exploratório | [README do modelo de 15 classes](transformer_minds_15_classes_pose_face/README.md) |
| Transformer MINDS `pose_face` | 20 | 37/100 (37%) | Não executado | [README do modelo de 20 classes](transformer_minds_20_classes_pose_face/README.md) |

Cada subpasta guarda o checkpoint, o mapa de rótulos, o manifesto e os índices do split, os YAMLs de configuração, o registro da execução, o histórico e as métricas disponíveis. `checksums.sha256` registra a integridade dos arquivos arquivados.

**Dados necessários para reprodução:** os landmarks `.npz` permanecem em `data/landmarks/`; os MP4 originais permanecem em `data/raw/minds_libras/`. Eles não são duplicados aqui. O `.gitignore` permite versionar especificamente estes `best.pt`, mas os arquivos ainda precisam ser adicionados e enviados ao repositório para existir fora deste computador. Os resultados são de sinais isolados do MINDS-Libras, não de tradução de frases nem de usuários novos fora do dataset.
