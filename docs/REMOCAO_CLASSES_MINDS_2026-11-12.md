# Remoção de sete classes do dataset local

Data: 12/11/2026. Pedido: apagar vídeos e landmarks de Cinco, Espelho, América, Esquina, Sapo, Bala e Maca.

## Plano e execução

Foram selecionadas as classes exatas do manifesto MINDS (`America` e `Maca` são os nomes presentes nos arquivos). Antes da remoção, foram validados os caminhos absolutos, extensões, existência e tamanho de todos os alvos dentro de `data/raw/minds_libras` e `data/landmarks/minds_libras_all20`. A remoção usou caminhos literais, sem exclusão recursiva de diretórios. Checkpoints, relatórios e experimentos históricos foram preservados.

O plano com cada caminho e tamanho foi salvo em `reports/remocao_classes_minds_2026-11-12/remocao.json`; o status final é `concluido_verificado`. O manifesto e o CSV de anotações anteriores foram guardados nessa pasta como registro, sem duplicar os vídeos ou landmarks removidos.

## Resultado verificado

- Removidos: **280 vídeos MP4 e 280 landmarks NPZ**, 40 de cada tipo por classe.
- Espaço dos arquivos removidos: aproximadamente **16,19 GiB**.
- Restantes: **520 vídeos e 520 landmarks**, de **13 classes**, com 40 exemplos por classe.
- Manifesto e `annotations.csv` foram regenerados pelo construtor MINDS existente; ambos contêm 520 amostras. Todos os 560 alvos foram conferidos como ausentes.

Classes restantes: Acontecer, Aluno, Amarelo, Aproveitar, Banco, Banheiro, Barulho, Conhecer, Filho, Medo, Ruim, Vacina e Vontade.

Não havia splits locais em `data/processed` para reajustar nesta execução. Os splits arquivados antigos continuam históricos: não devem ser reutilizados como se este dataset ainda tivesse 20 ou 25 classes. Com as dez palavras próprias recuperadas, o novo vocabulário misto terá **23 classes**, exigindo novos manifestos, splits e treinamento. Excluir os dados não altera as palavras dos checkpoints já treinados, que ainda podem prever Cinco e Espelho.

## Downloads futuros

O índice remoto Kaggle foi preservado como registro completo da fonte. Repetir o download sem filtro baixaria novamente as classes removidas. Para manter somente as 13 classes:

```bash
./.venv/Scripts/python.exe scripts/download_minds_libras.py --output data/raw/minds_libras --jobs 4 --labels Acontecer Aluno Amarelo Aproveitar Banco Banheiro Barulho Conhecer Filho Medo Ruim Vacina Vontade
```

O diretório de landmarks mantém o nome histórico `minds_libras_all20`, mas seu manifesto atual tem 13 classes. O nome da pasta não define o vocabulário; o conteúdo do manifesto define.
