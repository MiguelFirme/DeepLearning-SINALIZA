# Plano: Transformer v2 de 41 classes como padrão

Data: 17/11/2026. Pedido: usar o checkpoint ampliado recém-treinado como modelo principal.

1. Conferir o registro concluído de `transformer_v2_grupo_expandido_20261117_seed42`, o mapa de 41 classes, a época selecionada (80), os hashes dos dados de treino e os metadados do pipeline v2.
2. Arquivar o melhor checkpoint, configurações, splits, logs e métricas em `Modelos salvos/transformer_v2_grupo41_full`, preservando os pesos de 23 classes e os experimentos antigos.
3. Trocar `backend/config.py` para carregar esse arquivo sem seleção manual, mantendo `fusion_manifest_path=None`. Ajustar a imagem e o healthcheck Docker.
4. Testar inicialização FastAPI com os pesos reais, anúncio das 41 classes e inferência WebSocket temporal. Manter a compatibilidade explícita da fusão histórica.
5. Atualizar README e guias com o novo padrão, o comando de reinício e os limites. Nenhum novo treinamento nem avaliação dos 93 vídeos de teste será executado na ativação.
