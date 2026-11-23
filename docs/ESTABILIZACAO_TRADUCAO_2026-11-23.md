# Estabilização das palavras na interface

## Plano aprovado

Aplicar somente na apresentação do frontend: confirmar a mesma palavra em duas
previsões consecutivas válidas; exigir pelo menos 1.500 ms entre palavras novas;
não inserir novamente a última palavra aceita. Manter a última tradução na tela
enquanto se confirma a próxima, preservando os avisos atuais de rastreamento.
Reiniciar o filtro ao iniciar ou perder a conexão. Não reduzir o FPS, atrasar
overlays nem modificar o modelo/backend. Testar sequências de previsões e build.

## Resultado

Implementado em `frontend/src/utils/predictionStabilizer.ts`, aplicado às mensagens
de predição no hook WebSocket. `prediction` passa a conter somente a última
palavra aceita; `latestPrediction` mantém o estado atual para avisos da interface.
A frase e o painel de última tradução usam o resultado estável. Confiança e
candidatos são os da aceitação; a medição de processamento continua atualizando.

Regras:

- A primeira palavra também precisa de duas previsões válidas consecutivas.
- A partir da primeira aceitação, palavras diferentes só entram após 1.500 ms.
  O tempo é medido por `performance.now()`, sem depender do relógio do sistema.
- Uma palavra diferente reinicia a contagem. Incerteza, ausência de sinal e
  rastreamento insuficiente também interrompem uma confirmação pendente.
- Uma resposta de cooldown não é classificação; ela é ignorada na contagem.
- A mesma palavra aceita não é republicada, mesmo após 1,5 s. Pode voltar depois
  de outra palavra aceita. O filtro reinicia ao receber `ready`, perder conexão
  ou parar a câmera; o novo histórico de câmera mantém o comportamento existente.
- Não há fila de palpites nem publicação por timer. Depois do intervalo, só uma
  previsão recém-recebida e confirmada pode ser aceita, evitando palavras antigas.
- Enquanto se espera, a última palavra e a frase continuam visíveis. O estado
  pode indicar Confirmando sinal, incerteza ou ajuste do enquadramento.

O intervalo é **mínimo**, não um prazo de resposta exato: a chegada das previsões,
o cooldown já existente no backend e a necessidade de concordância também contam.
Não é garantia de maior acurácia do modelo; é um filtro de apresentação.

## Ajustar os valores

Sem nenhuma variável adicional, a aplicação já usa 2 confirmações e 1.500 ms.
Para alterar, configure no build do frontend (Vercel ou `.env.local` local):

```text
VITE_TRANSLATION_CONFIRMATIONS=2
VITE_TRANSLATION_MIN_INTERVAL_MS=1500
```

Confirmações exigem inteiro positivo; intervalo exige número finito não negativo.
Valores inválidos voltam aos padrões. Não coloque essas variáveis no backend.
Na Vercel, faça novo build/redeploy para incorporar as alterações. Não precisa
retreinar nem modificar configurações do Render para esta funcionalidade.

## Testes

```bash
cd frontend
npm run test:stability
npm run test:urls
VITE_API_URL=https://deeplearning-sinaliza.onrender.com/api npm run build
```

Resultado: 8 testes de estabilização e 9 testes de URLs passaram; build
TypeScript/Vite de produção passou. Casos: alternância, limite exato de 1.500 ms,
repetições, reentrada após outra palavra, estados sem sinal, troca de candidato
durante a espera, cooldown, reset e configuração personalizada.
Não foi realizado teste humano no navegador nesta alteração.

## Roteiro prático

1. Inicie a câmera e repita um sinal conhecido: a primeira previsão isolada não
   deve aparecer como tradução; aguarde a confirmação.
2. Mantenha o mesmo sinal: deve ficar uma palavra, sem repetições na frase.
3. Faça outro sinal: a palavra anterior deve permanecer até a nova confirmação,
   respeitando pelo menos 1,5 s desde a aceitação anterior.
4. Saia do enquadramento: o aviso de rastreamento deve atualizar, preservando a
   última tradução. Ao voltar, uma nova palavra deve exigir nova confirmação.
5. Pare e reinicie a câmera: candidatos da sessão anterior devem ser descartados.

## Arquivos

Criados: este documento, `frontend/src/utils/predictionStabilizer.ts` e
`frontend/tests/predictionStabilizer.test.mjs`.
Alterados: `frontend/src/hooks/useWebSocket.ts`, `frontend/src/pages/HomePage.tsx`,
`frontend/src/components/TranslationDisplay.tsx`, `frontend/package.json`,
`frontend/.env.example` e `README.md`.
