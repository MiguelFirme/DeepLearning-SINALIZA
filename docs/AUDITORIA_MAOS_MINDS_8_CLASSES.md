# Auditoria de detecção das mãos — MINDS-Libras, 8 classes

## Plano

1. Ler as máscaras `mask.npy` dos 320 arquivos `.npz` listados no manifesto do checkpoint de 8 classes, preservando a separação por sinalizador.
2. Contar por vídeo os quadros com mão esquerda, direita, ambas, apenas uma e nenhuma detectada. Validar formato, contagem de arquivos e cobertura dos splits.
3. Agregar por classe, sinalizador e split; salvar dados por vídeo e resumos em CSV/JSON para análise e reprodução.
4. Interpretar os resultados para decidir como avaliar um futuro modelo de mãos e o peso de fusão. A máscara indica entrega de landmarks pelo MediaPipe, não prova que a mão estava fisicamente visível nem que os pontos estavam corretos.

## Procedimento

O script `scripts/audit_hand_coverage.ps1` lê somente o ZIP interno dos `.npz` e a matriz booleana `mask.npy`. Não modifica os arquivos de entrada e não executa o MediaPipe novamente. Usa o manifesto e os índices de split arquivados com o modelo de 8 classes. A análise visual dos quadros com detecção ruim é uma etapa posterior, porque os `.npz` não guardam imagens.

## Resultados

Executado em 18/10/2026 sobre os 320 `.npz` do manifesto arquivado com o Transformer de oito classes. Todos os arquivos existem, os splits cobrem as 320 entradas sem sobreposição (240/40/40), e as contagens de esquerda/direita/ambas/nenhuma fecham em todos os 44.011 quadros. As porcentagens abaixo são **ponderadas por quadro**; a contagem de vídeos com pelo menos metade dos quadros com ambas as mãos é apresentada separadamente.

| Split | Vídeos | Quadros | Esquerda | Direita | Ambas | Nenhuma | Vídeos com ambas em ≥50% |
|---|---:|---:|---:|---:|---:|---:|---:|
| Treino | 240 | 31.591 | 12,81% | 45,89% | 9,93% | 51,23% | 8/240 |
| Validação | 40 | 5.916 | 99,49% | 95,69% | 95,44% | 0,25% | 40/40 |
| Teste | 40 | 6.504 | 88,58% | 82,41% | 74,42% | 3,43% | 35/40 |

No treino, **150/240 vídeos não têm nenhum quadro com a mão esquerda detectada** e **155/240 não têm nenhum quadro com ambas detectadas**. Todos os vídeos têm ao menos uma mão detectada em algum quadro. Os seis sinalizadores de treino apresentam 4,36% a 15,20% de quadros com ambas as mãos; o sinalizador de validação chega a 95,44% e o de teste a 74,42%.

### Cobertura por classe

| Classe | Treino: ≥1 mão | Treino: esquerda | Treino: direita | Treino: ambas | Validação: ≥1 mão | Teste: ≥1 mão |
|---|---:|---:|---:|---:|---:|---:|
| Acontecer | 42,76% | 36,19% | 35,18% | 28,62% | 97,80% | 94,58% |
| Aluno | 52,17% | 7,21% | 51,31% | 6,35% | 100,00% | 100,00% |
| Banheiro | 55,10% | 51,08% | 45,64% | 41,62% | 100,00% | 93,93% |
| Barulho | 50,60% | 1,43% | 49,75% | 0,59% | 100,00% | 95,73% |
| Conhecer | 43,12% | 5,33% | 41,10% | 3,31% | 100,00% | 95,67% |
| Medo | **44,26%** | 0,32% | 43,94% | 0,00% | 100,00% | 97,23% |
| Ruim | 51,33% | 3,42% | 49,22% | 1,30% | 100,00% | 98,43% |
| Vontade | 48,86% | 1,20% | 48,11% | 0,46% | 100,00% | 96,38% |

## Interpretação e decisão para o próximo experimento

A articulação de `Medo` é predominantemente de **uma mão**, conforme observado pelo grupo. Portanto, os 0% de quadros com **ambas** as mãos detectadas no treino **não são, por si, uma falha para essa classe**. O dado relevante é a cobertura da mão ativa: no treino há detecção da direita em 43,94% dos quadros e de pelo menos uma mão em 44,26%; na validação, há pelo menos uma mão em 100%. Isso ainda não prova falha durante o gesto, pois o denominador inclui introdução e repouso.

A diferença entre sinalizadores domina a distribuição das máscaras. Um Transformer treinado apenas com esses pontos receberia predominantemente mão direita ou nenhuma mão nos quadros de treino, mas frequentemente duas mãos detectadas na validação, inclusive uma mão de repouso. A taxa de detecção também distingue classes no treino (por exemplo, `Banheiro` contra `Medo`), o que pode virar um atalho espúrio para o classificador. A fusão deve considerar **a mão necessária ao sinal no intervalo ativo**, e não exigir duas mãos para todas as palavras. **Ainda não há evidência de que um peso alto e fixo para mãos melhore a generalização.**

Esta contagem cobre o vídeo inteiro, inclusive preparação e repouso; não mede separadamente os quadros centrais do gesto. `mask=False` só indica que o MediaPipe não entregou pontos naquela parte, sem provar ausência física da mão. A auditoria também não mede a precisão geométrica de pontos quando `mask=True`. Antes de selecionar peso de fusão, a etapa seguinte deve inspecionar visualmente amostras de treino e validação e medir a cobertura da mão ativa no intervalo do gesto. Se a informação dessa mão estiver realmente ausente ou borrada nos vídeos de treino, novas gravações ou um conjunto mais consistente terão mais valor do que aumentar o peso desse ramo.

## Arquivos e reprodução

Os arquivos em `reports/hand_coverage_school8/` contêm `per_video.csv`, `by_class.csv`, `by_signer.csv`, `by_split.csv`, `by_split_class.csv`, `by_class_signer.csv` e `summary.json`. O JSON inclui SHA-256 do manifesto e dos splits usados. Por exemplo, `by_class.csv` mistura treino, validação e teste; para decidir sobre treino, usar `by_split_class.csv`.

No Git Bash do VS Code, a partir da raiz do projeto:

```bash
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/audit_hand_coverage.ps1
```

O comando regrava somente os relatórios, sem modificar `.npz`, checkpoints ou splits. Não requer Python nem MediaPipe.
