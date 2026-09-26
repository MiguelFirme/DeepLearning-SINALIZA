# Relatório de Treinamento — Sinaliza

## 1. Informações do Modelo
- **type**: BiLSTMModel
- **input_dim**: 697
- **num_classes**: 1364
- **total**: 4649670
- **trainable**: 4649670

## 2. Treinamento
- **checkpoint**: experiments\bilstm_art1_val2_test3\best.pt
- **best_epoch**: 22

## 3. Métricas de Teste

| Métrica | Valor |
|---------|-------|
| Top-1 Accuracy | 0.0007 |
| Top-3 Accuracy | 0.0022 |
| Top-5 Accuracy | 0.0037 |
| Precision (macro) | 0.0000 |
| Recall (macro) | 0.0007 |
| F1 Score (macro) | 0.0000 |
| F1 Score (weighted) | 0.0000 |
| Total de amostras | 1362 |

## 4. Pares Mais Confundidos

| Real | Predito | Ocorrências |
|------|---------|-------------|
| Abacaxi | Careca | 1 |
| Abanar | Careca | 1 |
| Abandonar | Careca | 1 |
| Abelha | Careca | 1 |
| Abençoar | Careca | 1 |
| Aborto | Careca | 1 |
| Abraço | Careca | 1 |
| Abrir a janela | Careca | 1 |
| Abrir a porta | Careca | 1 |
| Abóbora | Careca | 1 |

## 5. Classes com Menor Acurácia

| Classe | Acurácia |
|--------|----------|
| Abacaxi | 0.0000 |
| Abanar | 0.0000 |
| Abandonar | 0.0000 |
| Abelha | 0.0000 |
| Abençoar | 0.0000 |
| Aborto | 0.0000 |
| Abraço | 0.0000 |
| Abrir a janela | 0.0000 |
| Abrir a porta | 0.0000 |
| Abóbora | 0.0000 |
