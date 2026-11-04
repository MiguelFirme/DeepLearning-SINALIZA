# Transformer mixed25 pose_face16

Treinado do zero com 25 classes: 15 do MINDS-Libras e dez gravadas pelo grupo. Mesmo split (500 treino, 85 validação, 85 teste), seed 42, 48 quadros, arquitetura de 122.123 parâmetros e parâmetros de treinamento do modelo de 40 pontos. Modo `pose_face16`: somente 16 pontos faciais em sobrancelhas, olhos e boca; os demais 24 pontos ficam zerados antes e depois da normalização. O modelo `pose_face16` também zera os pontos detalhados das mãos.

Validação: 41/85 (48.2%), F1 macro 0.289. Teste exploratório já consultado anteriormente: 37/85 (43.5%), F1 macro 0.321.

Conteúdo: `best.pt`; `config/` com toda a cadeia YAML; `data_split/` com manifesto, splits, mapa de classes e metadados; `records/` com histórico, registro efetivo, log e comparação. Os vídeos e `.npz` originais não são duplicados. O checkpoint foi escolhido pela menor perda de validação; o teste não orientou o treino. Veja `docs/PLANO_ABLACAO_FACE16_MIXED25_2026-11-04.md` para comandos e limites.
