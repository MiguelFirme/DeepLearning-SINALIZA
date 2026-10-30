# Relatório de qualidade, fusão e rejeição — 25 classes

Data: 27/10/2026. Este relatório registra o que foi executado depois do [plano](PLANO_QUALIDADE_REJEICAO_FUSAO_25_CLASSES_2026-10-27.md).

## Base e protocolo

O escopo é exatamente `configs/minds_libras_15_sem_cinco.json` mais `configs/novas_10_palavras.json`, sem sinais externos. São 670 vídeos, com 500/85/85 em treino/validação/teste. `configs/fusion25_frozen.json` fixa hashes SHA-256 dos dois checkpoints, label map, manifesto de dados, splits e predições de validação; também fixa temperaturas, ordem, top 5, peso e regra de escolha. O backend recusa divergência de pesos/vocabulário e verifica os dados de treinamento quando estiverem presentes. Um clone pode executar a inferência sem os vídeos e splits.

Os 85 vídeos de teste já haviam sido consultados anteriormente neste projeto. Os números abaixo são **exploratórios**, não uma estimativa independente de generalização. Não houve avaliação com participantes inéditos.

## Auditoria de qualidade

`scripts/audit_mixed25_quality.py` examinou os 670 arquivos de landmarks e produziu `reports/quality_mixed25/quality_all.csv`, `review_queue.csv` e `summary.json`. O detector estima a mão ativa a partir do movimento do punho relativo ao ombro, procura uma janela contígua de movimento, mede cobertura, continuidade, borda, pose, face e possível corte temporal. Esses alertas não são diagnósticos definitivos de sinal ruim.

| Medida | Resultado |
|---|---:|
| Vídeos com pelo menos um alerta | 168/670 |
| Cobertura da mão ativa abaixo de 50% | 124 |
| Mão ativa fora do enquadramento | 50 |
| Mão próxima da borda | 34 |
| Possível corte temporal | 7 |
| Alertas treino / validação / teste | 132 / 7 / 29 |

Mediana de cobertura da mão ativa: treino 0,719; validação 0,918; teste 0,860. A diferença entre pessoas/splits merece inspeção. A estimativa de mão ativa pode falhar em sinais de movimento pequeno, então não excluir automaticamente esses vídeos. Um piloto visual decodificou dois quadros de dez vídeos de treino mais suspeitos; todos foram legíveis, com escores de nitidez acima do limiar conservador de 20. O piloto não substitui a revisão de todo o gesto. A varredura visual em lote pode ser executada com `--video-samples 5`, mas é lenta neste computador. Nenhum vídeo/landmark original foi alterado ou excluído.

Para corrigir: revisar `review_queue.csv` olhando o vídeo inteiro e registrar `review_decision`, `reviewer` e `review_reason`. `keep` preserva; `exclude_train` permite criar um split derivado com `scripts/apply_quality_review.py`; `rerecord` e `reextract` exigem novos arquivos antes do treinamento. O script impede exclusão pós-hoc de validação/teste. Isso preserva a comparabilidade do protocolo. Um vídeo cortado, borrado ou ocluído requer nova gravação ou exclusão de treino revisada; interpolar pontos não recupera dedos invisíveis.

## Fusão congelada

Temperatura pose/face: 1,0287; temperatura completo: 0,7924. Primeiro modelo: pose/face; suas cinco candidatas são reordenadas pela soma ponderada das probabilidades dos dois modelos, com peso 0,22 para pose/face e 0,78 para completo. A regra foi escolhida na validação conforme o critério escrito no plano antes da calibração. A temperatura e o peso foram ajustados nos mesmos 85 vídeos usados na comparação; há risco de otimismo. Os escores resultantes da restrição top 5 não são probabilidades de rejeição calibradas.

| Conjunto | Modelo completo | Fusão congelada |
|---|---:|---:|
| Validação, acertos | 46/85 (54,1%) | 50/85 (58,8%) |
| Validação, F1 macro | 0,422 | 0,452 |
| Teste antigo, acertos | 48/85 (56,5%) | 56/85 (65,9%) |
| Teste antigo, F1 macro | 0,438 | 0,480 |
| Acertos nas dez classes novas, validação / teste | 4/10 / 4/10 | 4/10 / 4/10 |

Resultados detalhados: `configs/fusion25_frozen.json` e `reports/fusion25_frozen_test_exploratory.json`. A fusão melhora esse teste antigo, mas ainda não foi demonstrada em pessoas inéditas. Na aplicação, EMA e cooldown podem mudar a sequência de respostas em relação à avaliação offline, que classificou cada vídeo isoladamente.

## Latência medida

Em CPU, 30 repetições sobre uma janela de landmarks de validação: modelo completo 3,18 ms p50, fusão 6,29 ms p50 e 7,84 ms p95, incluindo pré-processamento e inferência. Na mesma janela e quantidade de repetições, uma RTX 4050 Laptop GPU obteve 3,98 ms p50 para o modelo completo e 7,97 ms p50 para a fusão. Neste caso pequeno, a transferência/sincronização torna a GPU mais lenta. `reports/fusion25_latency_cpu.json` e `reports/fusion25_latency_cuda.json` registram cada etapa. Num vídeo gravado de 60 quadros, convertido em JPEG 640×480 e processado pelo fluxo do backend em CPU, o tempo por quadro foi 110,47 ms p50 e 117,67 ms p95; a inferência dupla teve 7,58 ms p50 em duas chamadas e retornou `Banheiro` duas vezes. `reports/fusion25_web_pipeline_cpu.json` contém o registro. Esse ensaio não inclui navegador nem rede e não mede latência de ponta a ponta para uma pessoa ao vivo. O custo dominante foi a extração/processamento de quadros, não a segunda rede. O fluxo completo ainda não foi repetido com o modelo na GPU.

## Rejeição e augmentation

Ainda não há negativos reais revisados. `configs/negative_examples_template.csv` define o formulário de coleta: vídeo MP4, pessoa `origem:id`, sessão, subtipo (`repouso`, `conversa`, `gesto_incompleto`, `movimento_fora_vocabulario`), split e `reviewed=yes`. `scripts/prepare_rejection_dataset.py` exige vídeos e landmarks de pelo menos 24 quadros, grupos isolados e no mínimo 20/5/5 negativos de treino/validação/teste; cria a classe 26 `__nenhum_sinal__` sem alterar as 25 positivas. Esses mínimos só verificam que o pipeline está montado; uma avaliação confiável de falsos positivos exigirá mais vídeos, cenários e pessoas. Uma execução com o formulário vazio falhou antes de escrever arquivos, como esperado. **Nenhum modelo de rejeição foi treinado e o backend ainda não pode distinguir de forma confiável repouso de sinal fora do vocabulário.** O limiar atual de 0,30 é apenas um filtro heurístico.

`configs/transformer_mixed25_augmented_full.yaml` registra a ablação: ruído 0,004, escala 0,95–1,05 e rotação até 5°, amostrados dinamicamente apenas no treino. Validação/teste continuam sem augmentation. A ablação não foi treinada: primeiro é preciso revisar os 168 alertas e decidir as correções. O checkpoint congelado não foi substituído.

## Como reproduzir (Git Bash, raiz do projeto)

```bash
./.venv/Scripts/python.exe scripts/audit_mixed25_quality.py --video-samples 0
./.venv/Scripts/python.exe scripts/calibrate_fusion25.py
./.venv/Scripts/python.exe scripts/evaluate_frozen_fusion25.py
./.venv/Scripts/python.exe scripts/benchmark_fusion25.py --repetitions 20
./.venv/Scripts/python.exe -m pytest tests/test_fusion25_protocol.py tests/test_quality_pipeline25.py tests/test_web_video.py tests/test_augmentation.py -q
```

Para testar a fusão na aplicação, na raiz do projeto:

```bash
export SINALIZA_FUSION_MANIFEST_PATH=configs/fusion25_frozen.json
./.venv/Scripts/python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Em outro terminal, `cd frontend && npm run dev`. `GET /api/health` deve informar `num_classes: 25`. Sem a variável, o backend usa o modelo original de 8 classes. Os caminhos congelados são relativos à raiz do repositório.

Os comandos presumem um `.venv` funcional com as dependências do projeto. Neste ambiente de execução, o executável de `.venv/Scripts/` apontava para uma instalação Python 3.11 removida; as verificações acima foram feitas com um Python 3.11 portátil ligado aos pacotes existentes. Se o terminal do VS Code mostrar erro semelhante, recrie o ambiente virtual antes de executar os comandos.

Quando houver negativos, gravar e revisar os MP4 em `data/raw/negatives/`, preencher `data/negative_examples_reviewed.csv` a partir do template e executar:

```bash
./.venv/Scripts/python.exe scripts/extract_landmarks.py --input data/raw/negatives --output data/landmarks/negatives --workers 1
./.venv/Scripts/python.exe scripts/prepare_rejection_dataset.py
./.venv/Scripts/python.exe scripts/train.py --config configs/transformer_mixed25_rejection_full.yaml --data-dir data/landmarks/mixed25_rejection --processed-dir data/processed/mixed25_rejection --experiment transformer_mixed25_rejection_seed42 --num-workers 0 --skip-test
```

Após o treino, avaliar falsos positivos em negativos de pessoas inéditas e definir a integração de um checkpoint de 26 classes. Não usar o teste antigo para ajustar limiares. Para a ablação de augmentation, após a revisão dos dados, manter o mesmo split e usar `configs/transformer_mixed25_augmented_full.yaml` com um nome de experimento novo e `--skip-test`.

## Verificações

Foram executados os testes de protocolo/fusão, qualidade, integração web e augmentation: 12 passaram. Outros 12 testes de web, augmentation e fusão anterior também passaram. Uma chamada de inicialização da API confirmou `model_loaded=true` e `num_classes=25` com a variável de fusão. A paridade entre predições offline e backend foi verificada por teste numérico. Permanecem sem teste: câmera ao vivo, negativos, pessoas inéditas, fluxo completo com o modelo em CUDA e a ablação treinada.
