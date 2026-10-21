# Conferencia dos sete arquivos do frontend com o RAR recebido

## Plano

1. Localizar os sete caminhos indicados na imagem dentro de `C:\Users\migue\Downloads\DeepLearning-SINALIZA-main.rar`.
2. Extrair somente esses arquivos para uma pasta temporaria e comparar hashes SHA-256 com as versoes em `frontend/src/`.
3. Se houver diferencas, registrar quais arquivos diferem e copiar somente os sete arquivos solicitados do RAR para o projeto. Manter todos os demais arquivos intactos.
4. Recalcular os hashes e executar o build do frontend. Registrar o resultado aqui para que a verificacao seja repetivel.

Arquivos: `pages/HomePage.tsx`, `components/TranslationDisplay.tsx`, `components/TranslationDisplay.module.css`, `components/WebcamView.tsx`, `components/WebcamView.module.css`, `hooks/useCameraFrames.ts` e `services/api.ts`.

## Resultado em 14/10/2026

Fonte: `C:\Users\migue\Downloads\DeepLearning-SINALIZA-main.rar`, SHA-256 `4D66CB48C4E9AA1B4FEA1372130CC11171DF31EEFF41F02E91BAE1E13C89EB5E`. Todos os sete arquivos estavam no subdiretorio `DeepLearning-SINALIZA-main/frontend/src/` do RAR.

Na primeira comparacao, seis arquivos ja eram identicos. Apenas `hooks/useCameraFrames.ts` diferia: o RAR incluia o rastreamento visual local MediaPipe para desenhar os pontos de maos, pose e face sobre o video. Somente esse arquivo foi substituido pela versao do RAR. A copia do RAR faz rastreamento no navegador para visualizacao, alem do processamento ja realizado no backend; a comparacao de hashes nao mede o impacto disso no FPS em uso real.

Hash SHA-256 final, identico no projeto e no RAR:

| Arquivo em `frontend/src/` | SHA-256 |
|---|---|
| `pages/HomePage.tsx` | `A446574A316AAE9FC966BE9868A46D016B15E95BDDAD450FF1FDC6C0CAD9B800` |
| `components/TranslationDisplay.tsx` | `B83F0E1E1A29C6E2AA1BD9E68FD1A9BC4CE629B857056F9D23AE8D8CA12163F7` |
| `components/TranslationDisplay.module.css` | `C673ED46EC0983482F6394E06774E8855248C2129FFF73A463760B5781587F9C` |
| `components/WebcamView.tsx` | `0BA1BB98D63962B4FB0AD5BC479C3A4FFA6956998ED5B56C5AAB297419384080` |
| `components/WebcamView.module.css` | `E2BC118EFABB90DB635CE72F12E4E515E7F2EEE53DEAA84E94BE8848029BEDAE` |
| `hooks/useCameraFrames.ts` | `ED219BC89A63D6A2FE78F1711851F096FFCD40EF22D7ADF55CEBB73827BD8579` |
| `services/api.ts` | `08D2494395CD3EBA57AF7EA3734AE7EF52BBB6D00038EBB280AFAA8122EB77A4` |

A verificacao final comparou novamente os sete pares por `Get-FileHash -Algorithm SHA256`: **7 de 7 iguais**. `npm run build` passou, incluindo TypeScript e Vite (1588 modulos). A camera real nao foi testada nesta verificacao.

## Estado posterior

A igualdade de hashes acima foi verdadeira no momento da conferencia. Apos o relato de que a traducao e o overlay nao funcionavam, `useCameraFrames.ts`, `WebcamView.tsx`, `HomePage.tsx` e `api.ts` foram alterados para corrigir o fluxo. O estado atual desses arquivos nao e mais identico ao RAR. Causa, plano e verificacoes em [PLANO_CORRECAO_TRADUCAO_OVERLAY.md](PLANO_CORRECAO_TRADUCAO_OVERLAY.md).
