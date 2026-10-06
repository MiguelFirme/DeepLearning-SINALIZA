# Plano: regularização e avaliação sem vazamento

## Diagnóstico inicial

- O experimento escolar de 23 classes chegou a 100% no treino e 13,04% top-1 no articulador de teste; isso indica memorização e amostra insuficiente de pessoas independentes.
- O dataset já aplica transformações dentro de `__getitem__`, mas a configuração usada no diagnóstico as desliga. O `Trainer` já implementa AdamW, early stopping e ReduceLROnPlateau, mas a configuração de foco também os desliga.
- A divisão comum era estratificada por amostra. Variantes do mesmo vídeo precisam ser agrupadas antes da divisão; validação e teste não devem conter arquivos sintéticos.

## Implementação planejada

1. Permitir divisão por grupos de vídeo de origem, sessão ou articulador com `GroupShuffleSplit`; verificar explicitamente a disjunção dos grupos nos três conjuntos. Preservar o protocolo de holdout por articulador.
2. Impedir origem comum em splits distintos também no carregamento do treino; recusar amostras sintéticas em validação/teste.
3. Criar configuração de experimento com BiLSTM menor, LayerNorm antes das ativações, dropout 0,45, AdamW com `weight_decay=1e-4`, early stopping de 15 épocas por `val_loss` e scheduler plateau.
4. Ativar somente no treino augmentation dinâmica moderada de ruído, rotação e escala; usar vídeos originais para validação/teste.
5. Testar invariantes de split, exclusividade da augmentation, arquitetura e integração mínima de treinamento. Registrar comandos e limites dos resultados aqui.

## Critério de avaliação

O teste final deve usar um articulador independente e permanecer intocado até a seleção do checkpoint por validação. Uma única gravação real por classe no teste gera estimativa muito instável; melhoria de configuração não equivale a acurácia de produção.

## Implementado

- `ml/data/split.py`: divisão por grupos de vídeo, sessão ou articulador. Usa `GroupShuffleSplit` se `scikit-learn` estiver disponível; a `.venv` atual não o tem, então usa uma divisão local equivalente por grupos completos. Todo arquivo sintético herda o grupo do vídeo em `source_path`.
- `scripts/build_dataset.py`: padrão `--strategy group --group-key video`; sintéticos são removidos de validação e teste. O holdout por articulador continua disponível e permite treino/teste sem validação quando solicitado.
- `scripts/train.py`: confere manifesto, origem comum entre splits e ausência de sintéticos em validação/teste. Salva configuração, contagens, hashes e métricas em `logs/`.
- `ml/data/dataset.py` e `ml/training/trainer.py`: augmentation em `__getitem__` apenas no treino, com sequência aleatória distinta por época e worker. Validação/teste não recebem augmentor.
- `ml/models/bilstm.py`: `regularized: true` adiciona LayerNorm antes de GELU, dropout após projeção, saída recorrente e camada oculta do classificador. O formato antigo permanece selecionável para carregar checkpoints anteriores.
- `configs/bilstm_regularized_school.yaml`: BiLSTM de uma camada com 64 unidades por direção, projeção 64 e classificador 64, dropout 0,45, Adam com `weight_decay=1e-4` (L2), early stopping de 15 épocas por `val_loss`, `ReduceLROnPlateau` com paciência 5. Augmentation espacial moderada: ruído 0,002, escala 0,98–1,02 e rotação até 2°. O `Trainer` mantém AdamW como padrão para experimentos antigos e agora aceita `optimizer: adam`.

## Execução no Git Bash do VS Code

Na raiz do projeto:

```bash
./.venv/Scripts/python.exe scripts/build_dataset.py --landmarks data/landmarks --output data/processed/regularized_school_23 --strategy signer_holdout --essential-categories escola --train-signers Articulador1 --val-signer Articulador2 --test-signer Articulador3 --max-classes 23
./.venv/Scripts/python.exe scripts/train.py --config configs/bilstm_regularized_school.yaml --data-dir data/landmarks --processed-dir data/processed/regularized_school_23 --experiment bilstm_regularized_school_23_adam --num-workers 0
```

Os dois comandos já foram executados: 23 vídeos reais no treino (Articulador1), 23 na validação (Articulador2) e 23 no teste (Articulador3), sem sintéticos. O melhor checkpoint foi escolhido pela menor `val_loss`, e somente então o articulador de teste foi avaliado. Resultados ficam em `experiments/bilstm_regularized_school_23_adam/` e `logs/experiments/bilstm_regularized_school_23_adam/`.

Para reproduzir a verificação rápida sem acessar o teste:

```bash
./.venv/Scripts/python.exe -m pytest tests -q
./.venv/Scripts/python.exe scripts/train.py --config configs/bilstm_regularized_school.yaml --data-dir data/landmarks --processed-dir data/processed/regularized_school_23 --epochs 1 --experiment bilstm_regularized_school_smoke --skip-test --device cpu --num-workers 0
```

## Verificação realizada

- 88 testes passaram; há dois avisos de depreciação de FastAPI/Pydantic não ligados a este pipeline.
- O smoke test de uma época terminou em CPU, sem tocar o teste: 131.209 parâmetros, `train_loss=3,6790` e `val_loss=3,4433`. Essa rodada verifica funcionamento, não mede generalização.
- Treino completo com Adam na GPU: o early stopping interrompeu na época 43; melhor `val_loss=3,08857` na época 28. O teste obteve top-1 **1/23 = 4,35%**, top-3 **3/23 = 13,04%** e F1 macro **0,00543**. Uma rodada anterior com AdamW, preservada em `experiments/bilstm_regularized_school_23/`, também obteve 1/23. O resultado histórico de aproximadamente 14% top-1 usava mais arquivos sintéticos no treino e outra seleção de modelo; a comparação não isola o efeito de cada mudança. Articulador3 foi consultado em experimentos anteriores, portanto já não é um teste completamente virgem para decisões futuras.
- A regularização diminuiu a memorização, mas também prejudicou o aprendizado neste conjunto minúsculo. **Não houve melhora de generalização**. Não ajustar novamente hiperparâmetros com base neste teste: isso transformaria o articulador reservado em validação. Para um avanço confiável, coletar múltiplos vídeos reais por classe de mais pessoas e sessões, reservar pessoas inéditas para um novo teste e selecionar configurações apenas em validação.
