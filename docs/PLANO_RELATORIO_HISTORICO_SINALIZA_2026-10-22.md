# Plano do relatório histórico do Sinaliza

## Objetivo

Produzir um documento Word que permita à equipe e ao professor entender o que foi implementado, testado e medido até 22/10/2026, com resultados comparáveis e limitações claras.

## Fontes e critérios

1. Conferir README atual, documentação dos experimentos, métricas JSON/CSV e arquivos dos modelos arquivados. Usar resultados do checkpoint selecionado, não picos isolados de época; distinguir treino, validação e teste.
2. Organizar a evolução por fases: V-Librasil, preparação de dados e regularização, MINDS-Libras, ablação de mãos, transferência I3D, Transformers de 5/8/20/15 classes, fusão de 15 classes e integração web.
3. Explicitar diferenças de vocabulário, quantidade de pessoas e reutilização do sinalizador de teste. Registrar hipóteses e resultados negativos sem apresentá-los como falhas conclusivas de arquitetura.
4. Gerar um DOCX em `reports/`, com resumo executivo, cronologia, tabelas de comparação, estado atual, limitações, próximos passos e referências aos caminhos locais usados. Renderizar todas as páginas e revisar visualmente antes da entrega.

## Estado

Concluído em 22/10/2026. O relatório foi gerado em `reports/Relatorio_historico_Sinaliza_2026-10-22.docx` pelo script reproduzível `scripts/generate_project_report.py`. Foram conferidos os registros de métricas dos modelos de 8, 20 e 15 classes e da fusão, além dos documentos de V-Librasil, mãos, I3D, regularização e aplicação web. O arquivo final tem 63 parágrafos e cinco tabelas; a estrutura DOCX foi reaberta com sucesso, o estilo de título ficou sem borda e a seção web tem quebra de página explícita para o Word.

A rotina `render_docx.py` foi tentada, mas este Windows não dispõe de LibreOffice e a instância COM do Word não iniciou na sessão de automação. Como verificação visual alternativa, `docx-preview` foi executado localmente em Edge invisível, sem envio do arquivo a serviços externos; a prévia em PDF teve quatro páginas e todas foram inspecionadas. O navegador não respeita todas as regras de paginação do Word, por isso também foi verificado no XML que a seção web tem `page_break_before=True` e `keep_with_next=True`. As prévias e bibliotecas temporárias foram removidas após a revisão; apenas o DOCX final foi mantido em `reports/`.
