# Plano do pipeline v2

Pedido recebido em 11/11/2026. Preservar extração bruta (346 valores e máscara por parte), pesos e experimentos históricos. A v2 terá uma implementação compartilhada entre dataset e backend, escolhida por `pipeline_version: 2` no checkpoint.

## Fases e commits

1. Construtor espacial: aspecto, 17 pontos de pose sem visibilidade, lacunas curtas, ombros, forma local das mãos, cinco distâncias e velocidades por segundo. Dimensão: 297 coordenadas globais + 126 locais de mão + 5 distâncias + 297 velocidades = **725**.
2. Reamostragem temporal compartilhada, trecho ativo testável, janelas de treino/avaliação e sessão de inferência por tempo.
3. Augmentation bruta com máscara/timestamps sincronizados e espelhamento explícito de índices.
4. Configuração e treinamento v2, sampler balanceado, metadados completos, carregamento v1/v2, negativo opcional e validação por pessoa com métricas por origem.
5. Coleta na interface e backend, consentimento, gravação bruta e criação de manifestos/splits por pessoa.
6. Simulador de FPS pela sessão real, testes de invariâncias e documentação final com comandos.

Ao final de cada fase: executar `pytest tests -q`, registrar resultado e criar um commit. Dados ausentes serão cobertos por testes sintéticos; nenhuma métrica de reconhecimento será inventada. O pedido atual determina a fusão Transformer `configs/fusion25_frozen.json` como padrão v1, substituindo a escolha padrão da tarefa anterior; ConvBiLSTM continua selecionável pelo manifesto já arquivado.
