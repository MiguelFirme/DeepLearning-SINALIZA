# Plano de aumentação de landmarks — 2026-09-26

## Hipótese e referência

Na rodada de 398 classes com Articulador2 reservado, o Bi-LSTM atingiu
97,86% de top-1 no treino e 3,78% no teste. O principal problema observado é
a generalização entre pessoas. Aumentação moderada pode reduzir a dependência
das sequências exatas de treino, mas não substitui vídeos de mais pessoas.

## Desenho

1. Manter os splits, labels, arquitetura, épocas e semente da referência.
2. Aplicar mudanças aleatórias somente aos landmarks brutos dos articuladores
   de treino, antes de normalização e reamostragem para 48 quadros. Gerar nova
   vista a cada leitura, sem gravar arquivos sintéticos permanentes.
3. Começar com ruído pequeno, rotação discreta, recorte temporal curto e
   variação leve de velocidade. Manter máscaras de detecção sincronizadas.
4. Desativar espelhamento, ocultação artificial de landmarks, perda de quadros,
   jitter temporal independente por quadro, grandes rotações e cortes fortes.
   Podem mudar o sinal ou gerar trajetórias pouco plausíveis.
5. Fixar geradores aleatórios de treino para repetir a experiência.
6. Executar smoke test de carregamento/treino e então uma rodada completa.
   Comparar top-1, top-3, F1 macro e loss de treino com a referência.

## Interpretação

O Articulador2 já foi consultado para decidir este experimento, portanto seu
resultado passa a orientar desenvolvimento. Uma conclusão final exige vídeos
de pessoas novas, não usados na escolha das transformações. Melhora somente no
treino ou piora no teste indica que a aumentação não ajudou. Mesmo melhora no
teste não comprova desempenho de produção sem dados independentes adicionais.

## Implementação e verificação

- `configs/bilstm_augmented.yaml` herda `bilstm_focus.yaml`. Em 80% das
  leituras de treino, cada transformação pode ser sorteada separadamente:
  ruído gaussiano (30%, desvio 0,002), rotação (30%, até 3°), recorte temporal
  (20%, preservando ao menos 90% dos quadros) e velocidade (30%, 0,90–1,10×).
  As demais transformações estão desativadas.
- `scripts/train.py` semeia o gerador do `SequenceAugmentor` com a seed do
  experimento. O perfil usa `num_workers=0`; com essa configuração, a ordem
  de sorteio é reproduzível. Amostras de validação/teste não recebem aumento.
- Teste automatizado confirmou resultados idênticos com a mesma seed,
  preservação do array original e ausência de espelhamento/ocultação.
- Smoke test de uma época com 398 classes concluiu carregamento, treino,
  checkpoint e teste. Top-1 de 0,50% depois de uma época é somente verificação
  de funcionamento, não evidência de eficácia.
- O processo Python retornou `0xC0000409` ao encerrar após salvar os
  artefatos, como no treino sem aumentação. A causa nativa segue aberta.

Para a rodada comparativa no Git Bash do VS Code:

```bash
./.venv/Scripts/python.exe scripts/run_signer_cv.py \
  --config configs/bilstm_augmented.yaml --round 2 --num-workers 0
```

Isso criará `experiments/bilstm_augmented_signer_cv_allclasses_120epochs_seed42/`
e os splits correspondentes em `data/processed/signer_cv/`. O resultado da
referência está em
`experiments/bilstm_focus_signer_cv_allclasses_120epochs_seed42/articulador2/`.

## Resultado da rodada completa — Articulador2

O treino de 120 épocas foi concluído. SHA-256 de `splits.json` e
`label_map.json` confirma arquivos idênticos aos da referência.

| Métrica | Sem aumento | Com aumento |
|---|---:|---:|
| Top-1 treino, última época | 97,86% | 96,98% |
| Top-1 teste | 3,78% (15/397) | 2,52% (10/397) |
| Top-3 teste | 8,82% | 6,80% |
| Top-5 teste | 11,59% | 10,58% |
| F1 macro teste | 2,35% | 1,56% |
| Loss teste | 8,4739 | 8,4132 |

Nesta rodada, a receita moderada **não melhorou a identificação de sinais de
um articulador novo**: o top-1 caiu 1,26 ponto percentual (cinco acertos a
menos). A pequena queda da loss de teste não compensa a piora de top-1, top-3
e F1. O treino ainda chega perto de 97%, então permanece uma diferença grande
entre exemplos vistos e pessoa nova.

Os resultados de um único articulador e uma seed são exploratórios. Não há
evidência para adotar esta receita como padrão. Próximas ações com melhor
relação de custo e informação: auditoria de landmarks/normalização, matriz de
confusão e um baseline simples; depois, vídeos reais de mais pessoas e
repetições por classe. O processo Python repetiu o retorno `3221226505` após
salvar os artefatos, que foram preservados pelo executor.
