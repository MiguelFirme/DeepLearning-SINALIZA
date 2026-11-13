# Plano: Transformer v2 de 23 classes como padrão

Data: 13/11/2026. Pedido: usar o novo `transformer_v2` automaticamente ao iniciar o backend.

1. Conferir o checkpoint `transformer_v2_mixed23_seed42/best.pt`, seus metadados v2, mapa de 23 classes e registro de treino concluído.
2. Arquivar pesos, configurações, logs, métricas de validação e splits em `Modelos salvos/transformer_v2_mixed23_full`, com hash do checkpoint e sem consultar os 75 vídeos de teste.
3. Definir o arquivo arquivado como `model_path` padrão e desabilitar o manifesto de fusão por padrão. Preservar a seleção dos modelos antigos por variável de ambiente.
4. Atualizar imagem Docker, healthcheck e documentação de execução. Verificar o carregamento real na inicialização FastAPI e o WebSocket v2 com relógio controlado e pesos reais. Preservar um teste explícito da fusão v1.
5. Registrar os resultados e limites, inclusive webcam humana ainda não medida e diferenças históricas já observadas após reextração dos landmarks.

O treino não será repetido. A troca usa o checkpoint selecionado na época 69 por F1 macro da validação, sem otimizar pelo conjunto de teste.
