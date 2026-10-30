# Piloto: usar começo e fim dos vídeos como “nenhum sinal”

Data: 27/10/2026. Este trabalho responde à hipótese registrada antes da execução em [PLANO_NEGATIVOS_DAS_BORDAS_25_CLASSES_2026-10-27.md](PLANO_NEGATIVOS_DAS_BORDAS_25_CLASSES_2026-10-27.md). O vocabulário positivo permanece nas 15 classes selecionadas mais dez gravadas pelo grupo.

## Seleção dos candidatos

Foram examinados os landmarks dos 670 vídeos. A duração mediana foi de 132 quadros. Com 48 quadros nas bordas e filtros estritos de movimento, só apareceram 16 candidatos de treino e nenhum de validação. Por isso, o piloto usou trechos de **32 quadros** no primeiro ou último terço do vídeo, com pose presente em pelo menos 90% dos quadros, movimento médio e percentil 90 dos punhos abaixo de 20% do miolo, e pico principal de movimento fora do trecho. Os parâmetros foram fixados antes de consultar os candidatos do split de teste.

O [CSV de candidatos](../reports/idle_edges25/candidates.csv) tem 100 trechos de treino e 11 de validação. **Todos os 11 trechos de validação são do MINDS-Libras, Sinalizador11.** No total, 107/111 trechos são do MINDS-Libras e quatro das gravações novas. As [folhas de contato](../reports/idle_edges25/contact_sheets) mostram quatro quadros de cada trecho. A inspeção pela IA encontrou braços abaixados nessas amostras; não houve revisão humana dos 32 quadros inteiros. Os rótulos são **fracos/provisórios**.

Os trechos receberam o mesmo split e identificador de pessoa do vídeo original. O script `scripts/build_idle_edge_pilot.py` gerou 111 arquivos `.npz` derivados, sem alterar os originais. O manifesto experimental tem 600 amostras no treino (500 sinais + 100 repousos), 96 na validação (85 + 11) e preserva 85 positivos no teste, sem negativos de teste durante o treino.

## Modelo treinado

O Transformer `full` de 26 classes foi treinado do zero com `configs/transformer_mixed25_edge32_pilot.yaml`, seed 42, mesma estrutura de pré-processamento, loss CrossEntropy e pesos `sqrt_inverse`. Os segmentos de 32 quadros foram remostrados para 48 pelo pipeline existente. O treino parou na época 35, escolheu a época 23 pela menor perda de validação (1,7154) e levou cerca de 129 s em CUDA. Checkpoint, configuração resolvida, split e histórico estão em [Modelos salvos/transformer_mixed25_edge32_rejection_pilot](../Modelos%20salvos/transformer_mixed25_edge32_rejection_pilot/README.md). O log completo está em `logs/experiments/transformer_mixed25_edge32_rejection_pilot_seed42/train.log`.

| Validação | Resultado |
|---|---:|
| `full` anterior, sinais corretos | 46/85 |
| Piloto de 26 classes sozinho, sinais corretos | 29/85 |
| Piloto sozinho, repousos rejeitados | 11/11 |
| Piloto sozinho, sinais rejeitados por engano | 1/85 |
| Fusão congelada de 25 classes, sinais corretos | 50/85 |
| Fusão de 25 classes + piloto apenas como bloqueio, sinais corretos | 50/85 |

O piloto é ruim como tradutor isolado: perdeu 17 acertos positivos em relação ao `full` anterior. Como bloqueio, a regra fixa é “rejeitar se a saída mais provável do piloto for `__nenhum_sinal__`; caso contrário, manter a tradução da fusão de 25 classes”. Nessa validação pequena, rejeitou os 11 trechos de repouso e um sinal que a fusão já errava. Esses resultados estão em `reports/idle_edges25/pilot_evaluation/validation_metrics.json` e no CSV por vídeo ao lado.

## Teste exploratório depois de congelar a regra

O teste positivo original já havia sido consultado antes no projeto. Depois de congelar o filtro de mineração e a regra de bloqueio, foram encontrados 21 trechos de 32 quadros e seis janelas originais de 48 quadros no split de teste, todos do **mesmo Sinalizador12 do MINDS-Libras**. As seis prévias de 48 quadros mostram postura de repouso; resultados e arquivos por vídeo estão em `reports/idle_edges25/pilot_evaluation/` e `pilot_evaluation_48/`.

| Teste exploratório | Resultado |
|---|---:|
| Fusão de 25 classes sem bloqueio | 56/85 sinais corretos |
| Fusão + bloqueio do piloto | 55/85 sinais corretos |
| Sinais rejeitados indevidamente | 5/85 |
| Repousos de 32 quadros rejeitados | 20/21 |
| Repousos originais de 48 quadros rejeitados | 6/6 |

O bloqueio parece reconhecer repouso neste conjunto, inclusive nas seis janelas de 48 quadros. **Ainda não demonstra rejeição confiável na webcam.** Os negativos vêm das bordas dos mesmos vídeos rotulados como sinais, com cenários, iluminação e pessoas conhecidos. Os 32 quadros remostrados para 48 podem introduzir um padrão artificial que o modelo aprendeu. A classe negativa não inclui conversa, movimento casual, gestos incompletos nem sinais fora do vocabulário. Também há cinco rejeições indevidas no teste positivo. O bloqueio não foi ativado no backend.

## Reproduzir (Git Bash, raiz do projeto)

```bash
./.venv/Scripts/python.exe scripts/mine_idle_edges.py --length 32 --threshold 0.2 --preview-count 111
./.venv/Scripts/python.exe scripts/build_idle_edge_pilot.py --allow-weak-labels
./.venv/Scripts/python.exe scripts/train.py --config configs/transformer_mixed25_edge32_pilot.yaml --data-dir data/landmarks/mixed25_edge32_pilot --processed-dir data/processed/mixed25_edge32_pilot --experiment transformer_mixed25_edge32_rejection_pilot_seed42_repro --num-workers 0 --skip-test
./.venv/Scripts/python.exe scripts/evaluate_idle_edge_pilot.py --pilot experiments/transformer_mixed25_edge32_rejection_pilot_seed42_repro/best.pt
./.venv/Scripts/python.exe scripts/mine_idle_edges.py --splits test --length 32 --threshold 0.2 --preview-count 0 --output reports/idle_edges25/test_candidates_frozen.csv
./.venv/Scripts/python.exe scripts/evaluate_idle_edge_test.py --checkpoint experiments/transformer_mixed25_edge32_rejection_pilot_seed42_repro/best.pt
```

É preciso um `.venv` funcional e os vídeos/landmarks locais. Neste ambiente, `.venv/Scripts/python.exe` aponta para um Python removido; os comandos foram executados com um Python 3.11 portátil ligado às dependências instaladas.

## Decisão

**É tecnicamente possível treinar com começo/fim dos vídeos, e o piloto aprendeu a rejeitar esses repousos específicos.** Mantemos o checkpoint apenas para pesquisa. Antes de ativar a rejeição na aplicação: revisar clipes completos com o grupo, gravar negativos independentes de várias pessoas e cenários, incluir conversa e gestos fora do vocabulário, testar em janelas reais de 48 quadros e fixar um limite aceitável de rejeições indevidas em sinais. Depois, repetir a avaliação sem consultar o teste para ajustar a regra.
