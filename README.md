# SINALIZA

> Reconhecimento de sinais da Língua Brasileira de Sinais (Libras) com visão computacional e aprendizado profundo.

## Sobre o projeto

O **SINALIZA** tem como objetivo desenvolver um sistema capaz de reconhecer sinais em vídeos a partir dos movimentos das mãos, do corpo e das expressões faciais. A primeira etapa do projeto é o reconhecimento de sinais isolados; a tradução de sequências e frases é um objetivo posterior.

O pipeline extrai pontos de referência (*landmarks*) de cada quadro e analisa sua evolução temporal para identificar o sinal realizado.

## Estratégia de dados

O projeto utiliza dois datasets com papéis diferentes:

- **V-LIBRASIL — principal:** vídeos de palavras e expressões isoladas, usados no treinamento do modelo temporal;
- **Alfabeto em Libras — auxiliar:** imagens estáticas das configurações das mãos, mantidas em um pipeline separado e complementar.

Os dois conjuntos não são misturados diretamente porque possuem modalidades e objetivos diferentes. As fontes avaliadas estão em [Referencias/Datasets.txt](Referencias/Datasets.txt).

O conteúdo baixado deve ser preservado como foi fornecido. Os adaptadores do projeto identificam vídeos, imagens, rótulos e articuladores e geram manifestos padronizados.

## Pipeline principal

```text
Vídeo do V-LIBRASIL
        ↓
Indexação do rótulo e do articulador
        ↓
MediaPipe: mãos, pose e face
        ↓
Sequência temporal de landmarks
        ↓
BiLSTM (padrão) ou GRU
        ↓
Sinal reconhecido
```

O BiLSTM trazido do experimento em Colab é o modelo padrão. O GRU anterior continua disponível para comparação e, depois, ambos poderão ser comparados com o Transformer 1D proposto.

## Tecnologias

- **Python:** processamento, treinamento e inferência;
- **MediaPipe:** detecção dos landmarks das mãos, da pose e da face;
- **OpenCV:** leitura dos vídeos;
- **NumPy:** armazenamento das sequências processadas;
- **TensorFlow:** treinamento e execução dos modelos;
- **React:** interface planejada para captura de vídeo e exibição dos resultados.

## Estrutura do projeto

```text
DeepLearning-SINALIZA/
├── datasets/
│   ├── brutos/
│   │   ├── v_librasil/                 # dataset principal sem alterações
│   │   └── alfabeto_libras/            # dataset auxiliar sem alterações
│   ├── normalizados/                    # manifestos padronizados
│   ├── processados/
│   │   ├── v_librasil/
│   │   │   └── landmarks/              # sequências temporais NumPy
│   │   └── alfabeto_libras/
│   │       └── landmarks_maos/         # reservado ao pipeline auxiliar
│   ├── divisoes/
│   │   ├── v_librasil/                 # treino, validação e teste
│   │   └── alfabeto_libras/
│   └── README.md
├── models/                              # modelos e rótulos gerados
├── Referencias/
│   └── Datasets.txt
├── scripts/
│   ├── indexar_alfabeto.py             # cataloga o dataset auxiliar
│   ├── baixar_vlibrasil.py              # baixa o dataset pela API do Kaggle
│   ├── executar_pipeline.py             # download, processamento e treino
│   ├── processar_vlibrasil.py          # cataloga e processa o principal
│   ├── prever_video.py                 # reconhece um sinal em vídeo
│   └── treinar.py                       # treina e avalia BiLSTM ou GRU
├── src/
│   └── sinaliza/
│       ├── datasets/
│       │   ├── alfabeto_libras.py
│       │   ├── amostras.py
│       │   └── v_librasil.py
│       ├── config.py
│       ├── landmarks.py
│       ├── modelo.py
│       ├── processamento.py
│       └── sequencias.py
├── tests/
│   ├── test_adaptadores.py
│   └── test_sequencias.py
├── .env.example
├── .gitignore
├── pytest.ini
├── requirements-dev.txt
└── requirements.txt
```

Mais detalhes sobre os diretórios de dados estão em [datasets/README.md](datasets/README.md).

## Preparação do ambiente

É recomendado utilizar Python 3.11 em um ambiente virtual.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

Os caminhos padrão já apontam para as pastas em `datasets/brutos`. O arquivo de configuração é necessário apenas se os dados estiverem em outro local:

```powershell
Copy-Item .env.example .env
```

Nesse caso, descomente e informe os caminhos no arquivo `.env`:

```env
VLIBRASIL_DIR=C:/caminho/para/v-librasil
ALFABETO_LIBRAS_DIR=C:/caminho/para/alfabeto-libras
```

O caminho do alfabeto é necessário somente quando o pipeline auxiliar for utilizado. O arquivo `.env` não é versionado.

## Colocando os datasets no projeto

### Download direto do Kaggle

É possível baixar o V-LIBRASIL sem abrir o site nem usar o Colab:

```powershell
python scripts/baixar_vlibrasil.py
```

O comando usa o `kagglehub` e grava os arquivos em `datasets/brutos/v_librasil`. O dataset continua ocupando espaço no computador: fora do ambiente Kaggle, a biblioteca precisa baixar os vídeos localmente. Execuções posteriores reutilizam a cópia já baixada; use `--forcar` somente para baixar novamente.

Datasets públicos normalmente não exigem login. Se o Kaggle solicitar autenticação ou aceite de termos, gere um token nas configurações da conta Kaggle e siga a autenticação indicada pela mensagem da biblioteca.

O download do V-LIBRASIL pode ser descompactado diretamente em:

```text
datasets/brutos/v_librasil/
```

Não é necessário reorganizar os vídeos em pastas por sinal. O adaptador procura um `annotations.csv`; quando ele não existe, interpreta nomes como:

```text
Abacaxi_Articulador1.mp4
Abacaxi_Articulador2.mp4
Abacaxi_Articulador3.mp4
```

O dataset auxiliar pode ser descompactado em:

```text
datasets/brutos/alfabeto_libras/
```

Nesse caso, a pasta imediatamente acima de cada imagem será considerada seu rótulo.

## Executando o pipeline principal

Para executar todo o fluxo com o BiLSTM do notebook:

```powershell
python scripts/executar_pipeline.py --epocas 30
```

Se os vídeos já estiverem na pasta bruta, acrescente `--sem-download`.

### 1. Processar o V-LIBRASIL

Usando a pasta padrão do projeto (ou o caminho definido no `.env`):

```powershell
python scripts/processar_vlibrasil.py
```

Ou informando outro caminho:

```powershell
python scripts/processar_vlibrasil.py --entrada "C:/caminho/para/v-librasil"
```

Por padrão, a extração usa até quatro processos em paralelo e reaproveita arquivos
de landmarks válidos de execuções anteriores. Para controlar o paralelismo:

```powershell
python scripts/processar_vlibrasil.py --processos 4
```

Esse comando:

1. cataloga os vídeos em `datasets/normalizados/v_librasil.csv`;
2. extrai os landmarks de cada quadro;
3. redimensiona cada sequência para 60 quadros;
4. grava os arquivos em `datasets/processados/v_librasil/landmarks`;
5. cria `datasets/processados/v_librasil/manifesto.csv` para o treinamento.

O comprimento pode ser alterado com `--comprimento-sequencia`.

### 2. Treinar e avaliar

```powershell
python scripts/treinar.py --epocas 30
```

O BiLSTM é a arquitetura padrão. Para comparar com o baseline anterior, use `--arquitetura gru`. O treinamento utiliza somente o manifesto processado do V-LIBRASIL e salva:

```text
models/modelo_bilstm.keras
models/rotulos.json
models/modelo_atual.json
models/avaliacao/metricas.json
models/avaliacao/historico.json
models/avaliacao/curvas_aprendizado.png
models/avaliacao/matriz_confusao.npy
models/avaliacao/matriz_confusao.png
```

### 3. Reconhecer um vídeo

```powershell
python scripts/prever_video.py "C:/caminho/para/video.mp4"
```

### 4. Executar os testes

```powershell
pytest
```

## Pipeline auxiliar do alfabeto

Por enquanto, o adaptador auxiliar apenas cataloga as imagens:

```powershell
python scripts/indexar_alfabeto.py
```

O resultado é salvo em `datasets/normalizados/alfabeto_libras.csv`. A extração de landmarks das mãos e o classificador estático serão implementados separadamente e não alteram o treinamento principal.

## Observações sobre avaliação

O V-LIBRASIL possui poucas execuções de cada sinal. O baseline atual usa uma divisão aleatória estratificada apenas para validar o funcionamento técnico do pipeline. Antes de comparar modelos, as partições deverão considerar o identificador do articulador e ser gravadas em `datasets/divisoes/v_librasil`.

## Como contribuir

Contribuições são bem-vindas. Ao enviar uma alteração:

1. mantenha o V-LIBRASIL como fluxo principal;
2. não versione datasets, landmarks ou modelos gerados;
3. documente novos comandos e decisões;
4. inclua testes quando aplicável.

## Licença

Este repositório ainda não possui uma licença definida. Até que uma licença seja adicionada, os direitos sobre o código e os demais materiais permanecem reservados aos autores.
