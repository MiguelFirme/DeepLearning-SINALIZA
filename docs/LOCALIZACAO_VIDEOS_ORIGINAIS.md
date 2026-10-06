# Localização dos vídeos originais

Verificação em 2026-10-05, durante o download das 20 classes do MINDS-Libras. Nenhum arquivo foi apagado ou movido.

## MINDS-Libras (Kaggle)

- Pasta dos MP4: `C:\Users\migue\DeepLearning-SINALIZA\data\raw\minds_libras\`.
- Na inspeção: 342 MP4 completos, aproximadamente 18,98 GiB. O número e tamanho podem aumentar enquanto o download continua.
- A pasta também contém `.part` de downloads em andamento, `kaggle_file_index.json` e `download_state.json`.
- Landmarks derivados estão em `data/landmarks/minds_libras_school5/` e `data/landmarks/minds_libras_school8/`; são arquivos `.npz`, não vídeos originais.

## V-Librasil

- O script antigo `scripts/download_dataset.py` tem saída padrão `data/raw/`, mas a pasta `data/raw/` contém apenas `minds_libras/`.
- A busca de extensões de vídeo em todo `data/` encontrou somente os MP4 de `data/raw/minds_libras/`; portanto os vídeos originais do V-Librasil **não estão atualmente dentro deste projeto**.
- Localização encontrada depois pelo usuário: `C:\Users\migue\.cache\kagglehub\datasets\davimedio01\v-librasil\versions\1\videos UFPE (V-LIBRASIL)\data\`.
- Inspeção: **4.086 MP4**, aproximadamente **10,11 GiB**. Exemplo: `Abacaxi_Articulador1.mp4`. A pasta do dataset também contém `annotations.csv` e `error.csv` fora da subpasta `data`.
- O mesmo cache `kagglehub` contém outra entrada (`ziya07/multi-sensor-cnc-tool-wear-dataset`); não confundir a pasta inteira do cache com o V-Librasil.
- As buscas em `C:\Users\migue\Downloads`, `Documents` e `Videos` não tinham encontrado esses vídeos porque o cache está em `.cache\kagglehub`.

Para liberar espaço, primeiro concluir ou interromper o download em andamento. Apagar MP4 do MINDS-Libras durante o download pode fazer o script baixá-los novamente; a reextração de landmarks e os testes I3D/RGB também dependem dos vídeos originais. Os treinos `pose_face` já extraídos usam os `.npz`.
