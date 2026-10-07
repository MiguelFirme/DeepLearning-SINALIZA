# Rejeição de cenas paradas na aplicação web

## Problema observado

Em 07/10/2026, o usuário relatou que a aplicação atribui palavras mesmo quando ele fica parado diante da câmera. O Transformer foi treinado apenas com oito classes positivas, sem a classe "nenhum sinal". A função softmax sempre distribui probabilidade entre as oito classes; por isso confiança alta não comprova que houve um sinal.

## Plano

1. Medir uma característica simples de atividade dos braços em sequências de landmarks dos vídeos MINDS-Libras, mantendo os splits já fixados.
2. Criar um filtro anterior ao Transformer: se a janela não tiver movimento suficiente ou detecção confiável, responder "nenhum sinal" e zerar o estado de suavização. Não modificar os pesos arquivados.
3. Fazer o frontend mostrar um estado de espera, sem lista de palavras, quando o backend rejeitar a janela.
4. Testar com quadros estáticos, um vídeo real de sinal e o build do frontend. Registrar o limiar e as suas limitações.

## Resultados

### Medição

Métrica: amplitude robusta (percentis 10 e 90) da posição dos cotovelos e pulsos relativa ao ombro do mesmo lado, dividida pela largura dos ombros. Os landmarks vêm da pose de 25 pontos do extrator do projeto. A métrica vale zero quando a pose válida é insuficiente. O limiar inicial é **0,35** (`SINALIZA_MIN_ARM_MOTION`). A confiança mínima de exibição passou de 0,30 para **0,50** (`SINALIZA_CONFIDENCE_THRESHOLD`); a menor confiança dos 40 vídeos de validação foi 0,583. Confiança softmax não é probabilidade calibrada de que houve um sinal.

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

### Comportamento implementado

Antes de chamar o Transformer, o backend mede a presença de pose/face e a amplitude de braços dos 48 quadros. Se faltar detecção, devolve `status: poor_tracking`; se a amplitude for menor que 0,35, devolve `status: idle`. Ambos têm `sign: null` e `top_k: []`, e zeram a suavização anterior. O frontend apresenta uma mensagem de espera no lugar de palavras. Janelas com atividade seguem para o modelo. Se a confiança do modelo não atingir o limiar atual, a resposta é `uncertain` e a lista de candidatos não aparece.

### Verificação

- Teste WebSocket com 48 JPEGs estáticos: resposta `idle`, sem palavra ou candidatos. Depois de `reset` e 48 quadros com pulso móvel, a janela segue para o Transformer.
- Oito testes direcionados passaram e `npm run build` concluiu.
- `08BanheiroSinalizador11-1.mp4`, recodificado como quadros JPEG: o filtro preservou a predição **Banheiro** com confiança aproximada de **79,2%**.

### Limites e próximo dado necessário

Uma pessoa mexendo braços sem sinalizar ainda pode provocar uma classificação; o Transformer não recebeu exemplos de "nenhum sinal" nem de gestos fora das oito classes. Para melhorar a rejeição, gravar exemplos negativos reais com a mesma webcam (pessoa parada, falando, acenando, arrumando óculos, movimentos incompletos) e avaliar taxa de falsos positivos por minuto antes de calibrar outro limiar ou treinar uma classe de rejeição. O score atual pode precisar de ajuste conforme câmera, luz, enquadramento e velocidade dos sinais.
