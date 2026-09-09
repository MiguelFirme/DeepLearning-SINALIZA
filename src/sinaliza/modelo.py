"""Modelo baseline para classificação de sinais isolados."""

from __future__ import annotations

from tensorflow import keras


def construir_modelo_gru(sequence_length: int, feature_dim: int, class_count: int) -> keras.Model:
    """Cria e compila uma rede GRU pequena para o primeiro experimento."""
    inputs = keras.Input(shape=(sequence_length, feature_dim), name="landmarks")
    x = keras.layers.Masking(mask_value=0.0)(inputs)
    x = keras.layers.GRU(128, return_sequences=True)(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.GRU(64)(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(class_count, activation="softmax", name="class")(x)

    model = keras.Model(inputs, outputs, name="sinaliza_gru")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
