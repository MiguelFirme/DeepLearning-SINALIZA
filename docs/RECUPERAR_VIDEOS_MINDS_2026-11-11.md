# Recuperar vídeos e landmarks MINDS-Libras

Data: 11/11/2026. Fonte solicitada: https://www.kaggle.com/datasets/j0aopsantos/minds-libras

Plano: usar o downloader já existente para salvar os vídeos dentro de `data`, verificar tamanho/retomada e depois extrair os landmarks no formato bruto compatível com o protocolo arquivado. Nenhum download nem extração foi executado na preparação destas instruções. A página/API do Kaggle não pôde ser verificada pelo navegador de pesquisa nesta sessão; o `--dry-run` abaixo consulta o índice atual e mostra a quantidade e o espaço necessário antes do download.

No Git Bash do VS Code, a partir da raiz `DeepLearning-SINALIZA`:

```bash
mkdir -p data/raw/minds_libras data/landmarks/minds_libras_all20

# Consulta o índice e calcula o tamanho; não baixa os vídeos.
./.venv/Scripts/python.exe scripts/download_minds_libras.py --output data/raw/minds_libras --dry-run

# Baixa todas as classes disponíveis, com dois downloads simultâneos.
./.venv/Scripts/python.exe scripts/download_minds_libras.py --output data/raw/minds_libras --jobs 2

# Execute somente após concluir o download.
./.venv/Scripts/python.exe scripts/extract_landmarks.py --input data/raw/minds_libras --output data/landmarks/minds_libras_all20 --dataset-format minds_libras --workers 2

# Diagnostica o que ainda falta para o experimento misto de 25 classes.
./.venv/Scripts/python.exe scripts/restore_mixed25_dataset.py --check-only
```

Os vídeos ficam em `data/raw/minds_libras/`; `.npz`, `manifest.json` e `annotations.csv` extraídos ficam em `data/landmarks/minds_libras_all20/`. O downloader usa `curl`, retoma `.part` interrompidos e só converte para `.mp4` após verificar tamanho. Pode repetir o mesmo comando para continuar. A extração pula `.npz` existentes; confira o resumo final e eventuais erros antes de seguir.

Não use `--include-classes` para os MP4s planos do MINDS: essa opção filtra pastas de classe. `--dataset-format minds_libras` é obrigatório para obter as pessoas/classes dos nomes dos vídeos corretamente.

O MINDS recupera as classes públicas originais. As dez palavras da equipe **não vêm desse Kaggle**: são necessários os vídeos/landmarks de `novas_5_palavras` e `novas_5_restantes`, ou novas gravações com novos manifestos e splits. O recuperador não deve conseguir restaurar o mixed25 completo enquanto essas partes estiverem ausentes. Se forem novas gravações em vez dos arquivos originais, construa splits novos por pessoa; não atribua automaticamente os índices antigos a novos vídeos.

Depois de recuperar **todos os arquivos originais** esperados, execute `scripts/restore_mixed25_dataset.py` sem `--check-only` para recuperar os índices arquivados. Consulte [PIPELINE_V2.md](PIPELINE_V2.md) antes de treinar.
