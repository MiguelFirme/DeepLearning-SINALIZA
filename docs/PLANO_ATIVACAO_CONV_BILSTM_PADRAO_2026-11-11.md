# Plano: ativar a fusão Conv1D + BiLSTM + atenção como padrão

Data: 11/11/2026. Pedido do usuário: usar a nova dupla ConvBiLSTM de 25 classes como modelo principal da aplicação.

## Estado verificado antes da alteração

- O backend escolhe `configs/fusion25_frozen.json` por padrão, com dois Transformers antigos. `PredictionService` já carrega qualquer dupla compatível pelo `FusionBundle`, mas `/api/model/info` anuncia um tipo fixo `transformer_fusion`.
- Os checkpoints `conv_bilstm_attention_mixed25_pose_face` e `conv_bilstm_attention_mixed25_full`, o mapa de 25 classes e o manifesto experimental com hashes estão arquivados. A fusão preserva top 5 de pose/face e pesos 0,22/0,78.
- No estudo offline anterior, a nova fusão acertou 47/85 na validação contra 50/85 da original. No teste já consultado, acertou 65/85 contra 56/85. Este teste é exploratório e não substitui avaliação ao vivo com pessoas novas. A mudança de padrão decorre do pedido do usuário.

## Ações planejadas

1. Criar manifesto de implantação separado, com os mesmos hashes, temperaturas e regra da dupla ConvBiLSTM e mapa de classes do próprio arquivo de modelos. Preservar o manifesto experimental como registro.
2. Definir esse manifesto como padrão no backend e derivar `model_type` dos checkpoints carregados, mantendo o contrato da API.
3. Ajustar a imagem Docker para incluir os dois checkpoints novos e o mapa de classes.
4. Atualizar testes de configuração, inicialização e WebSocket; verificar predição com os pesos arquivados e arquivos de manifesto intactos.
5. Atualizar README, guia de execução e roteiro de teste prático. Registrar resultados e limites em relatório de ativação.

## Critérios de conclusão

- Sem variável de ambiente, `/api/health` responde `model_loaded: true`, `num_classes: 25` e `/api/model/info` identifica a fusão ConvBiLSTM.
- A inferência WebSocket recebe 48 quadros e continua devolvendo status, palavra ou incerteza, confiança, candidatos e overlay.
- O manifesto antigo continua disponível para reversão por `SINALIZA_FUSION_MANIFEST_PATH`.
- Testes relevantes e suíte completa passam; qualquer limite de Docker ou webcam real fica registrado.
