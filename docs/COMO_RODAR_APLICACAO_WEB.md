# Como rodar a aplicação web (Git Bash no VS Code)

## Plano e escopo desta verificação

1. Conferir os pontos de entrada, dependências e portas do frontend e backend.
2. Descrever uma inicialização local reproduzível, em dois terminais.
3. Registrar o estado da integração com o modelo, sem confundir interface aberta com tradução funcional.

Verificado no código em 07/10/2026. A integração com o modelo foi testada também com o Python 3.11 da `.venv` e um vídeo gravado; o acesso à webcam deve ser conferido no navegador do usuário.

## Pré-requisitos

- Python 3.11 e Node.js com npm.
- A `.venv` precisa apontar para uma instalação de Python que ainda exista. Confira com `./.venv/Scripts/python.exe --version`.
- Execute os comandos na raiz `DeepLearning-SINALIZA`, salvo quando houver `cd frontend`.

Se a `.venv` estiver quebrada, crie uma nova com uma instalação de Python 3.11 disponível. No Git Bash:

```bash
py -3.11 -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Se `py -3.11` não encontrar o interpretador, instale Python 3.11 ou use o caminho do `python.exe` instalado para criar a `.venv`. Não há necessidade de baixar nem treinar dataset para apenas abrir a interface e a API.

## Terminal 1: backend

No Git Bash, rode este comando **na raiz do projeto**. Se você estiver em
`DeepLearning-SINALIZA/backend`, volte primeiro com `cd ..`; a `.venv` e o
arquivo `requirements.txt` ficam na raiz.

```bash
cd ..
./.venv/Scripts/python.exe --version
./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

O `cd ..` acima só é necessário se o terminal estiver dentro de `backend`.
Se `uvicorn` ainda não estiver instalado, o comando de instalação resolve o
erro `No module named uvicorn`. Use Python 3.11 para esta `.venv`; confirme a
versão antes de instalar as dependências.

```bash
./.venv/Scripts/python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Abra `http://127.0.0.1:8000/api/health` e `http://127.0.0.1:8000/docs`. A resposta de health contém `model_loaded`. Se for `false`, a API está de pé, mas a predição não está disponível.

## Terminal 2: frontend

```bash
cd frontend
npm install
npm run dev
```

Abra `http://localhost:5173`. Permita o acesso à câmera quando solicitado. `npm install` só precisa ser repetido quando as dependências mudarem. No PowerShell, se a política de execução bloquear `npm.ps1`, use `npm.cmd install` e `npm.cmd run dev`; os comandos acima são para Git Bash.

Para parar, pressione `Ctrl+C` em cada terminal.

## Estado atual da tradução ao vivo

- O backend agora carrega automaticamente `Modelos salvos/transformer_minds_8_classes_pose_face/best.pt`; não é preciso copiar o checkpoint. Confira `model_loaded: true` e `num_classes: 8` em `/api/health`.
- O frontend envia JPEGs da câmera a `/api/ws/predict`; o MediaPipe e o pré-processamento do treinamento rodam no backend. Aguarde 48 quadros processados para a primeira classificação.
- A integração e o teste com vídeo gravado estão documentados em [PLANO_INTEGRACAO_WEB_8_CLASSES.md](PLANO_INTEGRACAO_WEB_8_CLASSES.md). A avaliação com a webcam do usuário ainda precisa ser feita no navegador.
- Cenas paradas agora recebem `Nenhum sinal detectado`. A medição do filtro, os testes e as limitações estão em [PLANO_REJEICAO_SEM_SINAL.md](PLANO_REJEICAO_SEM_SINAL.md).
- Docker Compose foi ajustado para incluir o checkpoint e encaminhar o WebSocket, mas não foi executado neste ambiente.
