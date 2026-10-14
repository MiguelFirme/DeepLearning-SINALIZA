# Rejeição de cenas paradas na aplicação web

**Estado em 07/10/2026: experimento revertido a pedido do usuário.** O backend voltou a enviar janelas com pose/face detectada ao Transformer mesmo quando os braços estão parados. A confiança mínima voltou a **0,30**. Este documento preserva o plano e as medições históricas, não descreve um filtro ativo.

## Problema observado

Em 07/10/2026, o usuário relatou que a aplicação atribui palavras mesmo quando ele fica parado diante da câmera. O Transformer foi treinado apenas com oito classes positivas, sem a classe "nenhum sinal". A função softmax sempre distribui probabilidade entre as oito classes; por isso confiança alta não comprova que houve um sinal.

## Plano

1. Medir uma característica simples de atividade dos braços em sequências de landmarks dos vídeos MINDS-Libras, mantendo os splits já fixados.
2. Criar um filtro anterior ao Transformer: se a janela não tiver movimento suficiente ou detecção confiável, responder "nenhum sinal" e zerar o estado de suavização. Não modificar os pesos arquivados.
3. Fazer o frontend mostrar um estado de espera, sem lista de palavras, quando o backend rejeitar a janela.
4. Testar com quadros estáticos, um vídeo real de sinal e o build do frontend. Registrar o limiar e as suas limitações.

## Resultados

### Medição

Métrica estudada: amplitude robusta (percentis 10 e 90) da posição dos cotovelos e pulsos relativa ao ombro do mesmo lado, dividida pela largura dos ombros. Os landmarks vêm da pose de 25 pontos do extrator do projeto. O limiar experimental foi **0,35** e a confiança mínima experimental foi **0,50**; ambos foram revertidos. A menor confiança dos 40 vídeos de validação foi 0,583. Confiança softmax não é probabilidade calibrada de que houve um sinal.

- Treino MINDS-Libras, 240 clipes: amplitude mínima **1,536**, mediana **2,468**.
- Validação MINDS-Libras, 40 clipes: mínima **2,239**, mediana **2,641**.
- Primeiro quadro de dois vídeos repetido 48 vezes no MediaPipe: **0,017** e **0,033**.
- Portanto, 0,35 separou as duas cenas repetidas de todos os clipes positivos de treino e validação examinados. Isso não equivale a medir a especificidade em pessoas paradas diante de uma webcam real.

Reproduzir as medições no Git Bash:

```bash
./.venv/Scripts/python.exe scripts/audit_motion_gate.py --split train
./.venv/Scripts/python.exe scripts/audit_motion_gate.py --split val
./.venv/Scripts/python.exe scripts/audit_motion_gate.py --still-video data/raw/minds_libras/08BanheiroSinalizador11-1.mp4
```

### Comportamento durante o experimento, agora revertido

Durante a experiência, o backend media a amplitude dos braços antes do Transformer e devolvia `status: idle` em janelas abaixo de 0,35. O frontend mostrava “Nenhum sinal detectado”. Esse caminho foi removido porque, no teste do usuário com a webcam, restringiu demais as palavras reconhecidas. Permanece a verificação anterior de detecção de pose/face (`poor_tracking`). O limiar de confiança ativo é novamente 0,30 (`SINALIZA_CONFIDENCE_THRESHOLD`).

### Verificação

- Na versão experimental, 48 JPEGs estáticos produziam `idle`; esse resultado não é mais esperado.
- A versão experimental passou em oito testes direcionados e no build do frontend.
- `08BanheiroSinalizador11-1.mp4`, recodificado como quadros JPEG, manteve a predição **Banheiro** com confiança aproximada de **79,2%** durante o experimento. Isso não substituiu o teste com a webcam do usuário.

### Limites e próximo dado necessário

O Transformer não recebeu exemplos de "nenhum sinal" nem de gestos fora das oito classes. Sem o filtro de movimento, cenas paradas ou gestos aleatórios podem produzir palavras. Para uma solução futura, gravar exemplos negativos reais com a mesma webcam (pessoa parada, falando, acenando, arrumando óculos, movimentos incompletos) e avaliar taxa de falsos positivos por minuto antes de treinar uma classe de rejeição ou calibrar uma regra nova.
