# Roteiro de teste prático da interface — 25 classes

Data: 04/11/2026; atualizado em 11/11/2026 para a fusão ConvBiLSTM ativa. Protocolo para executar diante da câmera. **Ainda não há resultados preenchidos neste documento.**

## Plano

1. Confirmar que a API carregou a dupla ConvBiLSTM ativa de 25 classes e que a câmera envia quadros com pontos rastreados.
2. Medir separadamente reconhecimento de sinais conhecidos, falsos palpites sem sinal, rastreamento e tempo de resposta.
3. Repetir tentativas em ordem alternada, anotar o que apareceu na tela e calcular taxas por pessoa e por classe.
4. Usar os achados para escolher quais erros investigar nos vídeos/landmarks, sem alterar os modelos durante esta rodada.

## Preparação — cerca de 5 minutos

Na raiz do repositório, abra dois terminais **Git Bash** do VS Code:

```bash
# Terminal 1: backend
./.venv/Scripts/python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

```bash
# Terminal 2: frontend
cd frontend
npm run dev
```

Abra `http://127.0.0.1:8000/api/health`: espere `model_loaded: true` e `num_classes: 25`. Em `http://127.0.0.1:8000/api/model/info`, espere `model_type: conv_bilstm_attention_fusion`. Abra `http://localhost:5173`, clique em **Iniciar Câmera**, permita o acesso e confirme **Modelo pronto** e **Pontos detectados**. Anote `FPS câmera` e `FPS análise`. Os dois podem ser diferentes. Se houver mais de uma câmera, escolha a mesma para toda a rodada.

Faça o teste com a pessoa inteira da cintura para cima, rosto, ombros, cotovelos, punhos e mãos visíveis, sem cortar as mãos no movimento. Use luz frontal estável e fundo sem outras pessoas. Não copie uma pose estática: execute o sinal de Libras que vocês conhecem ou que aparece no vídeo de referência. Registre pessoa, câmera, resolução se conhecida, iluminação, distância aproximada e data/hora.

**Atenção ao tempo:** a primeira classificação precisa de **48 quadros analisados pelo backend**, e novas classificações vêm a cada 12 quadros. O número de FPS da câmera não determina sozinho esse tempo. A tela mostra `Ida e volta/quadro`, `Backend/quadro` e `Modelos`: são tempos por quadro ou inferência, **não** o atraso completo entre começar um sinal e receber a palavra.

## Rodada 1 — funcionamento básico

1. Com a câmera desligada, confira **Câmera desligada**. Ligue-a e espere **Modelo pronto**.
2. Mova braços e mãos devagar. Confira se os pontos do rosto, tronco, braços e mãos acompanham a pessoa; anote partes que desaparecem ou atrasam. As mãos aparecem no overlay e são usadas pelo modelo `full` da dupla ativa.
3. Saia do enquadramento por tempo suficiente para renovar a janela. O estado esperado é **Rastreamento insuficiente**. Volte e confira se os pontos reaparecem.
4. Faça um sinal conhecido. Registre palavra exibida, porcentagem de `Confiança`, posição da palavra correta em **Top Candidatos**, e o intervalo entre iniciar o gesto e a **primeira resposta correta** usando cronômetro externo ou gravação de tela. A lista de candidatos só aparece quando a interface aceita uma palavra.
5. Clique **Parar**. Confira que os pontos desaparecem; inicie de novo para a rodada seguinte. Isso limpa o histórico da sessão e exige novamente 48 quadros analisados.

Se falhar antes de qualquer palavra, registre exatamente qual estado apareceu: **Conectando**, **Rastreamento insuficiente**, **Capturando sinais**, **Analisando sinal** ou **Erro na tradução**. A palavra correta não deve ser inferida a partir da barra de confiança sozinha.

## Rodada 2 — reconhecimento de sinais conhecidos

Comece com estas **oito classes existentes**: `Banheiro`, `Água`, `Ajudar`, `Aluno`, `Diretor`, `Comer`, `Beber`, `Vontade`. Elas cobrem necessidades, ambiente escolar e gestos possivelmente parecidos. Se o grupo não souber executar uma delas com segurança, use o vídeo de referência da classe ou marque **sem referência confiável**; não conte essa tentativa como erro do modelo.

Para cada classe, faça **três tentativas** (24 no total), alternando a ordem das classes em cada rodada. Em cada tentativa:

1. Clique **Parar** e **Iniciar Câmera** para limpar a janela anterior. Aguarde **Modelo pronto** e os pontos aparecerem. Mantenha 1 segundo de repouso.
2. Execute o sinal naturalmente, sem exagerar ou acelerar, por três repetições. Continue por até **12 segundos**, pois a janela precisa acumular quadros analisados. Não mude de sinal durante a tentativa.
3. Anote a **primeira palavra aceita**, a posição da palavra esperada no top 5 (ou `fora do top 5`), a maior confiança mostrada, o tempo até a primeira resposta correta e os dois FPS. Se nenhuma palavra sair, registre `sem resposta`.
4. Se os pontos da mão ativa sumirem durante o gesto, marque `rastreamento da mão falhou`. Esse caso deve ser separado de um erro do classificador.

Use a tabela abaixo como modelo; duplique as linhas para as 24 tentativas. `Top 1 correto` significa que a **palavra aceita pela interface** foi a esperada; `top 5` é apenas diagnóstico.

| Pessoa | Classe esperada | Tentativa | Primeira palavra aceita | Top 1 correto? | Posição no top 5 | Confiança exibida | Tempo até acerto | FPS câmera / análise | Pontos da mão ativa | Observações |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| P1 | Banheiro | 1 |  |  |  |  |  |  |  |  |
| P1 | Banheiro | 2 |  |  |  |  |  |  |  |  |
| P1 | Banheiro | 3 |  |  |  |  |  |  |  |  |

Se houver tempo, repita com as **25 classes**, três tentativas por classe, seguindo o mesmo procedimento. O vocabulário ativo é: Acontecer, Ajudar, Aluno, Amarelo, Amigo, Aprender, Aproveitar, Banco, Banheiro, Barulho, Beber, Brincar, Cansado, Certo, Cinco, Comer, Conhecer, Diretor, Espelho, Filho, Medo, Ruim, Vacina, Vontade e Água. Não teste palavras fora dessa lista como se fossem acertos esperados.

## Rodada 3 — palpites indevidos e robustez

Faça **três repetições de 15 segundos** de cada situação abaixo. Anote número de palavras aceitas, quais foram e a confiança. Estas situações são **diagnósticas**: o modelo não foi treinado com uma classe revisada de “nenhum sinal”, então uma palavra aqui revela uma limitação real, não um erro de execução do teste.

| Situação | O que fazer | O que observar |
| --- | --- | --- |
| Repouso | Ficar visível e parado, mãos em repouso | Palavras aceitas por minuto; repetição da mesma palavra |
| Conversa comum | Falar naturalmente sem sinalizar | Palpites e movimento dos pontos |
| Movimento sem sinal | Acenar ou ajustar os óculos/cabelo | Confusão com classe conhecida |
| Sinal incompleto | Começar um sinal conhecido e parar antes de concluí-lo | Aceitação precoce e palavra atribuída |
| Fora do vocabulário | Fazer um sinal correto que não esteja nas 25 classes | Palavra conhecida atribuída indevidamente |
| Sem pessoa | Sair completamente do enquadramento | `Rastreamento insuficiente` após renovar a janela |

Depois, repita `Banheiro` e `Água` em duas condições, **três vezes por condição**: (a) mais perto/mais longe sem cortar braços ou mãos; (b) iluminação frontal melhor/pior, sem apagar totalmente a luz. Altere uma condição por vez. Anote também quando o overlay perde mão, rosto ou pose.

## Rodada 4 — pessoas e sessões novas

Peça a **pelo menos duas pessoas que não participaram das gravações de treino** para fazer a rodada das oito classes, três tentativas cada, sem mostrar as previsões anteriores durante a execução. Se possível, repita outro dia. Registre resultados por pessoa; não misture tudo em uma única porcentagem. Evite ajustar gestos com base nas respostas durante a rodada, pois isso muda o teste.

## Fechamento e interpretação

Calcule para cada classe e pessoa:

- **Acerto top 1:** tentativas em que a primeira palavra aceita foi a esperada ÷ tentativas válidas.
- **Cobertura top 5:** tentativas com a palavra esperada entre os candidatos ÷ tentativas válidas. Uma palavra só é visível no top 5 quando houve resposta aceita.
- **Sem resposta:** tentativas válidas sem palavra aceita em 12 segundos ÷ tentativas válidas.
- **Falso palpite:** total de palavras aceitas nas situações sem sinal ÷ minutos observados. Registre também quais palavras surgiram.
- **Tempo até acerto:** mediana dos tempos medidos desde o início do gesto; informe quantas tentativas nunca acertaram dentro de 12 segundos.
- **Falha de rastreamento:** tentativas em que a mão ativa, rosto ou tronco sumiram do overlay durante parte relevante do gesto.

Não trate a porcentagem exibida como probabilidade calibrada de estar certo: o escore da fusão top 5 não foi calibrado para uso ao vivo. Os resultados offline de 85 vídeos não são meta automática para a webcam. Erros concentrados em uma pessoa sugerem diferença de sinalização, enquadramento ou domínio; erros concentrados em uma classe sugerem olhar os vídeos, landmarks e confusões dessa classe. Se `FPS análise` estiver baixo, a janela de 48 quadros demora mais a formar; isso pode explicar atraso sem ser lentidão dos modelos.

Ao terminar, guarde uma cópia preenchida deste roteiro em `reports/teste_pratico_interface/` com data e identificador da pessoa, sem publicar vídeos pessoais sem autorização. Inclua uma captura da tela e, se possível, uma gravação que mostre o gesto e a resposta. Compare somente rodadas com as mesmas condições e o mesmo manifesto de modelo (`configs/fusion25_conv_bilstm_default.json`).
