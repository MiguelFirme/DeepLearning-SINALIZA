# Plano e resultado: sensibilidade à detecção de mãos no MINDS-Libras

## Pergunta

O experimento de cinco classes acertou 3/25 vídeos da pessoa reservada para teste. Nos seis sinalizadores de treino, a mão esquerda foi detectada em média em 6,7% a 23,5% dos frames; na validação, em 100%, e no teste, em 88,6%. Queremos verificar se o modelo está usando esse padrão de ausência como atalho.

## Decisão

Não interpolar nem copiar coordenadas de uma mão ausente de outro vídeo: isso criaria gestos sem confirmação visual. Primeiro, fazer uma ablação reproduzível: zerar as duas mãos e suas máscaras **em todos os splits**, preservando pose e face. Esta ablação responde se o restante dos landmarks consegue classificar melhor uma pessoa não vista. O baseline com mãos permanece preservado.

## Execução planejada

1. Adicionar `data.feature_mode: pose_face` ao `LibrasDataset`, aplicando a máscara antes de augmentation e normalização; manter `full` como padrão. Registrar o modo no checkpoint e na avaliação.
2. Treinar com exatamente os mesmos 150 vídeos de treino, 25 de validação, rótulos e arquitetura do baseline. Usar `--skip-test` para comparar somente `val_loss`, `val_top1` e F1.
3. Se houver evidência de melhora na validação, investigar detecção de mãos e testar outras pessoas em um protocolo novo. Não selecionar um modelo com base no Sinalizador12 já consultado.

## Dados de referência

- Baseline com mãos: melhor época 1, `val_loss=1,8523`, `val_top1=20%`, F1 macro da validação `0,0833`; teste já consultado: 12% top-1.
- Referência aleatória uniforme para cinco classes equilibradas: 20% top-1 e loss de aproximadamente `ln(5)=1,609`.

## Resultado

Em 2026-10-05, a implementação adicionou `feature_mode=pose_face` ao dataset. As coordenadas das duas mãos e suas máscaras são zeradas **antes** de qualquer augmentation ou normalização; velocidades e distâncias dependentes delas também ficam zeradas. O modo `full` permanece padrão. O modo efetivo é salvo em `run_record.json` e no checkpoint; `scripts/evaluate.py` usa o modo do checkpoint. Os arquivos `.npz` originais não são alterados.

Com os mesmos 150 vídeos e a mesma arquitetura/configuração do baseline, o treino `bilstm_minds_school5_pose_face_seed42` usou `--skip-test`. O early stopping ocorreu na época 30, e a menor `val_loss=1,0430` foi na época 15. Nesse checkpoint, `val_top1=11/25=44%` e F1 macro `0,3500`, contra 20% e `0,0833` no baseline. O pico histórico de top-1 chegou a 52% na época 17, mas **não foi usado para escolher o checkpoint**: o critério permanece a menor `val_loss`.

A melhora é real na pessoa de validação, porém parcial: no checkpoint escolhido, 19/25 previsões são `Conhecer`, 5 são `Banheiro` e 1 é `Aluno`. Acertos por classe: `Aluno` 1/5, `Banheiro` 5/5, `Conhecer` 5/5, `Medo` 0/5, `Vontade` 0/5. Portanto, retirar as mãos reduz o desajuste de distribuição, mas ainda não resolve o reconhecimento das cinco classes. Posteriormente, a pedido do usuário, esse checkpoint foi avaliado no Sinalizador12: 15/25 (60%) e F1 macro 0,478; `Medo` e `Vontade` permaneceram em 0/5. Ver comparação exploratória com o Transformer em `docs/PLANO_TRANSFORMER_1D_MINDS.md`.

Próxima investigação recomendada: melhorar a detecção de mãos nos vídeos de treino com uma extração visual consistente para todas as pessoas, auditar as máscaras e comparar somente em validação. Até isso acontecer, o modelo `pose_face` é um diagnóstico, não um modelo pronto para uso. Não usar o resultado já conhecido do Sinalizador12 como critério de seleção.

### Auditoria adicional da proposta de misturar vídeos completos e incompletos

Nos 150 vídeos de treino, a proporção de frames com **ambas** as mãos detectadas varia fortemente por classe:

| Classe | Média de frames com ambas as mãos | Vídeos com ao menos 50% dos frames completos |
|---|---:|---:|
| Aluno | 6,3% | 0/30 |
| Banheiro | 41,1% | 8/30 |
| Conhecer | 3,2% | 0/30 |
| Medo | 0,0% | 0/30 |
| Vontade | 0,4% | 0/30 |

Portanto, aumentar a frequência dos vídeos com ambas as mãos detectadas favoreceria quase só `Banheiro` e confundiria qualidade de detecção com rótulo. `mask=False` significa que o MediaPipe Holistic não entregou landmarks para aquela mão naquele frame; não prova que a mão estava ausente do vídeo. O extrator usa `model_complexity=2`, `min_detection_confidence=0.5` e `min_tracking_confidence=0.5` (`ml/features/landmarks.py`). Quadros previamente inspecionados de `Banheiro` mostram mãos parcialmente sobrepostas e diferenças de roupa/contraste entre pessoas, mas não isolam a causa. Uma tentativa de extrair mais quadros diagnósticos em 2026-10-05 foi impedida por falha de capacidade na revisão automática de aprovação da ferramenta; a auditoria quantitativa acima foi concluída com os landmarks já disponíveis.

Comando de reprodução no Git Bash:

```bash
./.venv/Scripts/python.exe scripts/train.py --config configs/bilstm_minds_pose_face.yaml --data-dir data/landmarks/minds_libras_school5 --processed-dir data/processed/minds_libras_school5 --experiment bilstm_minds_school5_pose_face_seed42 --num-workers 0 --skip-test
```

## Próximo experimento: detector de mãos separado

### Plano

1. Manter intactos os 200 arquivos `.npz` e o split por sinalizador.
2. Comparar, nos **mesmos frames** de três sinais gravados pelos sinalizadores 01 (treino) e 11 (validação), a quantidade de mãos produzida pelo Holistic original com `MediaPipe Hands` em `static_image_mode=True`, `max_num_hands=2` e `min_detection_confidence=0.3`.
3. Inspecionar imagens com os novos pontos sobrepostos ao vídeo, especialmente nas classes `Medo` e `Vontade`, antes de aceitar qualquer ganho quantitativo. Verificar mão errada, pontos em roupa/corpo e trocas de lado. As coordenadas dentro da imagem não bastam para validar um gesto.
4. Só se os pontos forem plausíveis, considerar um extrator alternativo para **todos** os vídeos e comparar os modelos na pessoa de validação. Não usar o sinalizador 12 para escolher configuração.

O `min_detection_confidence` do **Holistic** se refere à detecção da pessoa, e `min_tracking_confidence` ao rastreamento da pose; diminuí-los não é um ajuste específico do detector de mãos. O `Hands` separado oferece limiar próprio de detecção de mãos. A implementação atual usa o Holistic com `model_complexity=2` (máxima complexidade de pose disponível nessa API).

Script de auditoria: `scripts/audit_hand_detector.py`. Saída JSON inclui contagem de frames com pelo menos uma mão, duas mãos, ganho/perda versus Holistic e índices dos frames. Imagens de prévia são geradas apenas em frames com ganho, até quatro por vídeo. Comando para Git Bash:

```bash
./.venv/Scripts/python.exe scripts/audit_hand_detector.py \
  --videos \
  data/raw/minds_libras/08BanheiroSinalizador01-1.mp4 \
  data/raw/minds_libras/08BanheiroSinalizador11-1.mp4 \
  data/raw/minds_libras/16MedoSinalizador01-1.mp4 \
  data/raw/minds_libras/16MedoSinalizador11-1.mp4 \
  data/raw/minds_libras/20VontadeSinalizador01-1.mp4 \
  data/raw/minds_libras/20VontadeSinalizador11-1.mp4 \
  --landmarks-dir data/landmarks/minds_libras_school5 \
  --output logs/minds_hand_detector_audit.json \
  --preview-dir logs/minds_hand_detector_previews \
  --samples-per-video 40
```

### Estado em 2026-10-05

O script foi adicionado; uma tentativa de execução pelo agente foi impedida por falha de capacidade na revisão automática. Depois, o usuário executou o comando no Git Bash e forneceu o relatório em `logs/minds_hand_detector_audit.json`. Nenhum `.npz` ou checkpoint foi alterado.

### Resultado da auditoria executada pelo usuário

Foram comparados 40 frames amostrados em cada um de seis vídeos, totalizando 240 frames de três classes e dois sinalizadores:

| Detector | Pelo menos uma mão | Ambas as mãos |
|---|---:|---:|
| Holistic em cache | 182/240 | 137/240 |
| Hands separado, confiança 0,3, modo imagem | 179/240 | 125/240 |

O `Hands` encontrou **mais** mãos que o Holistic em apenas 2 frames e **menos** em 17. Em `Medo/Sinalizador01`, ambos detectaram zero frames com duas mãos; em `Vontade/Sinalizador01`, idem. A lacuna mais importante do treino permanece. Houve 10 frames com ao menos um ponto do `Hands` fora dos limites da imagem. As mensagens sobre XNNPACK e feedback tensors no terminal não interromperam a execução.

As duas prévias de ganho foram inspecionadas. Em `Medo/Sinalizador01`, frame 64, o `Hands` marca uma mão baixa e borrada durante o movimento; o ganho é plausível, mas a geometria da mão não pode ser confirmada com segurança nesse frame. Em `Vontade/Sinalizador11`, frame 82, as duas mãos são visíveis, porém uma está baixa e em repouso; esse ganho de contagem não demonstra mais informação sobre o sinal. As imagens ficam em `logs/minds_hand_detector_previews/`.

**Decisão:** não substituir os landmarks atuais pelo `Hands` com essa configuração e não reextrair os 200 vídeos com base neste piloto. O teste usa apenas seis vídeos e modo estático, enquanto o Holistic original processou vídeos sequencialmente; portanto a comparação serve para descartar essa configuração como solução imediata, não para declarar um detector universalmente superior. Próximo diagnóstico: inspecionar vídeos de treino de `Medo` e `Vontade` nos frames em que o gesto ocorre e testar recortes de alta resolução guiados pela pose ou um detector de mãos atual com rastreamento, sempre com comparação visual e validação por sinalizador. Evitar preencher coordenadas ausentes artificialmente.

## Plano do piloto de recortes guiados pela pose

1. Usar os seis vídeos e os mesmos 40 índices por vídeo do piloto anterior. Ler pulsos e cotovelos dos landmarks de pose já extraídos, respeitando a visibilidade dos pontos.
2. Para cada lado, recortar uma região quadrada em resolução original em torno do pulso e da direção do antebraço. Executar `MediaPipe Hands` em cada recorte, associando no máximo uma mão a cada pulso. Evitar duplicar a mesma mão em dois recortes.
3. Comparar com o Holistic e com o detector em frame completo, mas **não** interpretar apenas a quantidade de mãos como qualidade. Salvar imagens dos ganhos e perdas, com recortes e pontos sobrepostos, para inspeção visual. Registrar frames em que o pulso está invisível ou fora do quadro.
4. Critério de avanço: ganhos plausíveis e recorrentes nos vídeos de treino de `Medo` e `Vontade`, sem perdas importantes nos vídeos de validação. Só depois implementar uma reextração completa, mantendo split por sinalizador e testando acurácia na validação. Não consultar o sinalizador 12 para escolher o método.

Este piloto não modifica `.npz`, splits ou checkpoints.

### Execução e resultado do piloto de recortes (2026-10-05)

Implementação: `scripts/audit_hand_crops.py`. Relatório: `logs/minds_hand_crop_audit.json`. Imagens separadas por `gain/`, `loss/` e `miss/` em `logs/minds_hand_crop_previews/`. Executado nos mesmos seis vídeos e 240 frames do piloto anterior com `Hands` em modo estático e confiança 0,3.

| Método | Pelo menos uma mão | Ambas as mãos | Frames com ganho/perda vs. Holistic |
|---|---:|---:|---:|
| Holistic original | 182/240 | 137/240 | referência |
| Hands no frame inteiro | 179/240 | 125/240 | 2 ganhos, 17 perdas |
| Hands em recortes guiados pela pose | 177/240 | 117/240 | 2 ganhos, 27 perdas |

O pulso ou cotovelo não passou no limiar de visibilidade em **164 de 480 regiões possíveis**, todas nos três vídeos do sinalizador 01: 40 em `Banheiro`, 63 em `Medo` e 61 em `Vontade`. Nos três vídeos do sinalizador 11 não houve região rejeitada por esse motivo. Em `Medo/01` e `Vontade/01`, o recorte ainda encontrou **zero** frames com duas mãos, assim como os outros dois métodos.

Inspeção de exemplos: o único ganho de `Medo/01` foi o mesmo frame 64 já detectado pelo `Hands` no quadro completo, com mão baixa e borrada; o recorte não recuperou uma mão nova. Um frame inicial sem detecção mostra as mãos fora da imagem, e um frame de perda mostra a mão em movimento borrada diante da roupa escura, com pose dos braços abaixo do limiar. Isso aponta para limitação visual e da pose nesses vídeos, mas não prova uma causa única. O ganho de `Vontade/11` também ocorreu no mesmo frame 82 do teste em quadro completo.

**Decisão:** não substituir o extrator nem reextrair o dataset com esses recortes. O método depende justamente dos pulsos que falham nos vídeos de treino. Próximo passo com maior potencial: auditar intervalos ativos de cada sinal, proporção de mãos realmente visíveis e qualidade de gravação por sinalizador; se a informação estiver ausente ou borrada, priorizar novas gravações com enquadramento, iluminação e velocidade adequados. O modelo `pose_face` permanece diagnóstico exploratório; o teste foi consultado posteriormente na comparação solicitada pelo usuário, registrada em `docs/PLANO_TRANSFORMER_1D_MINDS.md`.

Comando de reprodução no Git Bash:

```bash
./.venv/Scripts/python.exe scripts/audit_hand_crops.py \
  --reference-report logs/minds_hand_detector_audit.json \
  --video-dir data/raw/minds_libras \
  --landmarks-dir data/landmarks/minds_libras_school5 \
  --output logs/minds_hand_crop_audit.json \
  --preview-dir logs/minds_hand_crop_previews
```
