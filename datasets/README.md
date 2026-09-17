# Organização dos datasets

O **V-LIBRASIL é o dataset principal** do projeto. O dataset de alfabeto é
complementar e possui um pipeline independente, pois contém imagens estáticas.

```text
datasets/
├── brutos/                      # downloads originais, sem reorganização manual
│   ├── v_librasil/
│   └── alfabeto_libras/
├── normalizados/                # manifestos padronizados pelos adaptadores
├── processados/
│   ├── v_librasil/
│   │   └── landmarks/           # sequências temporais
│   └── alfabeto_libras/
│       └── landmarks_maos/      # características estáticas (pipeline futuro)
└── divisoes/                    # partições reproduzíveis de treino/validação/teste
    ├── v_librasil/
    └── alfabeto_libras/
```

Os downloads podem ser colocados diretamente nas pastas `brutos`, mantendo a
estrutura fornecida pelo Kaggle. Os adaptadores são responsáveis por localizar
os arquivos e criar manifestos com nomes de campos consistentes.

O V-LIBRASIL também pode ser baixado diretamente para a pasta correta com
`python scripts/baixar_vlibrasil.py`. O KaggleHub reutiliza o download local nas
execuções seguintes.

## V-LIBRASIL

O adaptador aceita preferencialmente um `annotations.csv`. Se o CSV não estiver
presente, tenta interpretar nomes no formato `<sinal>_Articulador<número>.mp4`.
O manifesto normalizado contém `rotulo`, `articulador_id` e `origem`.

## Alfabeto auxiliar

O adaptador cataloga imagens a partir das pastas de classes. Esse material não é
misturado ao treinamento temporal do V-LIBRASIL.
