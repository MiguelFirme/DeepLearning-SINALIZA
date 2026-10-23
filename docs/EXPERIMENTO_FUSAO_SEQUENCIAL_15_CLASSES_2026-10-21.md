# Experimento: fusão sequencial dos dois Transformers de 15 classes

## Plano antes da execução

1. Usar os checkpoints já arquivados `pose_face` e `full`, sem retreiná-los. Confirmar que os mapas de classes, manifesto e splits arquivados são idênticos e que as predições dos dois modelos se referem ao mesmo vídeo e rótulo. Usar os `.npz` originais e o pré-processamento de cada checkpoint, sem augmentation.
2. Medir os dois modelos individualmente nos 75 vídeos de validação (Sinalizador11). Na fusão direta, o modelo `pose_face` escolhe as candidatas top 3 ou top 5. O modelo `full` avalia o mesmo vídeo e reordena apenas essas candidatas pelo escore fixo `0,65 × P(pose_face) + 0,35 × P(full)`. A escolha das ponderações antecede a avaliação e favorece o modelo que já foi melhor na validação anterior. A ordem só tem efeito porque a primeira etapa restringe as classes elegíveis.
3. Relatar acurácia top 1, F1 macro, cobertura da classe real no top K da primeira etapa e predições individuais. A comparação principal é top 5; top 3 é análise secundária. Definir que a abordagem "funcionou" se top 5 superar os 49/75 acertos do `pose_face` na validação, sem redução do F1 macro. Somente nesse caso executar a ordem inversa como segundo experimento. Congelar parâmetros antes de consultar o teste.
4. Avaliar no teste (Sinalizador12) a fusão direta com os parâmetros fixos e, se aplicável, a inversa. Salvar métricas, probabilidades por vídeo, hashes dos checkpoints e comandos de reprodução. O teste já foi consultado por experimentos anteriores: tratá-lo como evidência exploratória, sem escolher pesos, top K ou ordem por seu resultado.

## Execução e resultados

Executado em 21/10/2026 com os dois checkpoints arquivados, nos mesmos `.npz` de 48 quadros amostrados por vídeo. O avaliador verificou igualdade dos mapas de classes, splits e manifesto, ausência de interseção com os outros splits, modos de entrada (`pose_face` e `full`) e alinhamento das predições por caminho/rótulo. Ambos carregaram os pesos estritamente. Nenhum vídeo foi aumentado ou usado para atualizar parâmetros.

| Método | Validação: 75 vídeos | F1 macro val | Teste exploratório: 75 vídeos | F1 macro teste |
|---|---:|---:|---:|---:|
| `pose_face` sozinho | 49/75 (65,3%) | 0,581 | 41/75 (54,7%) | 0,480 |
| `full` sozinho | 45/75 (60,0%) | 0,550 | 34/75 (45,3%) | 0,358 |
| `pose_face` → `full`, top 3 | 50/75 (66,7%) | 0,643 | 44/75 (58,7%) | 0,506 |
| **`pose_face` → `full`, top 5** | **52/75 (69,3%)** | **0,660** | **44/75 (58,7%)** | **0,506** |
| `full` → `pose_face`, top 3 | 50/75 (66,7%) | 0,625 | 42/75 (56,0%) | 0,481 |
| `full` → `pose_face`, top 5 | 52/75 (69,3%) | 0,660 | 44/75 (58,7%) | 0,506 |

O critério do plano foi satisfeito: top 5 direto aumentou acurácia e F1 macro na validação, então a ordem foi invertida e avaliada. Com top 5, **as 75 previsões foram exatamente iguais nas duas ordens**, tanto na validação quanto no teste. A inversão não trouxe ganho neste experimento. Com top 3, a ordem mudou 4 previsões na validação e 2 no teste. A classe correta apareceu entre as candidatas do modelo inicial em 67/75 vídeos de validação e 66/75 do teste para `pose_face` top 5; essa cobertura é o limite máximo de acertos possível para essa regra de seleção.

Em relação ao `pose_face` sozinho, a fusão direta top 5 corrigiu 8 erros e introduziu 5 erros na validação; no teste, corrigiu 6 e introduziu 3. O ganho líquido foi de **3 vídeos em cada split**. O escore combinado é uma média de probabilidades softmax **não calibradas**; o valor numérico não deve ser interpretado como confiança garantida de acerto. Quando a classe correta fica fora do top K inicial, o segundo modelo não pode recuperá-la. A fusão exige duas inferências e pode aumentar a latência da aplicação em tempo real; isso não foi medido aqui.

Os 75 vídeos de cada split pertencem a um único sinalizador (11 na validação, 12 no teste), com apenas 5 vídeos por classe. O teste do Sinalizador12 já foi usado em experimentos anteriores, inclusive na comparação entre esses modelos; **os números de teste são exploratórios e não constituem uma confirmação independente**. Não houve teste com câmera ao vivo, pessoas externas ao MINDS-Libras, frases ou classe "sem sinal". Por isso, o checkpoint ativo de 8 classes e o backend não foram alterados. O resultado justifica um piloto adicional com novos participantes antes de integrar esta fusão ao produto.

## Artefatos

- Código: `scripts/evaluate_two_stage_fusion.py`; testes: `tests/test_two_stage_fusion.py` (7 passaram).
- Saídas: `experiments/fusion_15_classes/{val,test}_{pose_face,full}_metrics.json` e arquivos correspondentes `_predictions.csv`. Os JSON contêm matriz de confusão, acurácia por classe, cobertura top K e SHA-256 dos checkpoints. Os CSV contêm caminho, classe real, top K candidato, probabilidades completas dos dois modelos, previsão final e acerto para cada vídeo.
- Checkpoint `pose_face`: SHA-256 `3a3e1c2be5885cfcc30028ade9cf81f496492506b77bca92a0b55127408b5799`. Checkpoint `full`: SHA-256 `7877a600af9ee6c00b18558d88416adf1b905091ed6e0e346ee4d30e757dccd6`.

## Reproduzir no Git Bash do VS Code

Execute na raiz do repositório, com Python 3.11 e as dependências de `requirements.txt` instaladas no `.venv`, os checkpoints arquivados e `data/landmarks/minds_libras_all20/` disponíveis:

```bash
COMMON=(
  --pose-checkpoint "Modelos salvos/transformer_minds_15_classes_pose_face/best.pt"
  --full-checkpoint "Modelos salvos/transformer_minds_15_classes_full/best.pt"
  --pose-split-dir "Modelos salvos/transformer_minds_15_classes_pose_face/data_split"
  --full-split-dir "Modelos salvos/transformer_minds_15_classes_full/data_split"
  --data-dir data/landmarks/minds_libras_all20
  --top-k 3 5 --pose-weight 0.65
  --output-dir experiments/fusion_15_classes
)
./.venv/Scripts/python.exe scripts/evaluate_two_stage_fusion.py "${COMMON[@]}" --split val --first pose_face
./.venv/Scripts/python.exe scripts/evaluate_two_stage_fusion.py "${COMMON[@]}" --split val --first full
./.venv/Scripts/python.exe scripts/evaluate_two_stage_fusion.py "${COMMON[@]}" --split test --first pose_face
./.venv/Scripts/python.exe scripts/evaluate_two_stage_fusion.py "${COMMON[@]}" --split test --first full
./.venv/Scripts/python.exe -m pytest tests/test_two_stage_fusion.py -q
```

Neste ambiente de execução, o `.venv` existente apontava para uma instalação de Python 3.11 removida. Para concluir a auditoria, foi usado um **Python 3.11.9 portátil oficial** em uma pasta temporária, com os pacotes já instalados no `.venv`; nem o checkpoint nem os dados foram modificados. Para reproduzir com os comandos acima em outra máquina, recrie o `.venv` caso o executável também esteja quebrado.
