import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

<<<<<<< HEAD
#  1. Carregar e pré-processar o MNIST 
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
 
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0
 
# Adicionar dimensão de canal: (N, 28, 28) → (N, 28, 28, 1)
x_train = x_train[..., tf.newaxis]
x_test  = x_test[..., tf.newaxis]
 
#  2. Construir a CNN 
# Arquitetura enxuta para Edge AI:
#   - 3 blocos Conv2D com poucos filtros → modelo leve (~93 KB em .tflite)
#   - BatchNormalization: estabiliza ativações, acelera convergência nas épocas limitadas
#   - Dropout(0.3): reduz overfitting sem custo computacional adicional
#   - Sem camadas densas profundas: mantém o número de parâmetros baixo

model = keras.Sequential([
    layers.Conv2D(16, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax"),
])
 
model.summary()
 
#  3. Compilar e treinar 
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    verbose=1,
)
 
#  4. Métricas no conjunto de teste 
# Accuracy: fração de dígitos classificados corretamente (meta: >98%)
# Loss: valor da cross-entropy - quanto menor, mais confiante e correto o modelo
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
val_acc_final = history.history["val_accuracy"][-1]
gap = abs(accuracy - val_acc_final)
 
print(f"\n--- Resultados ---")
print(f"Acurácia (teste)    : {accuracy * 100:.2f}%")
print(f"Loss (teste)        : {loss:.4f}")
print(f"Acurácia (val)      : {val_acc_final * 100:.2f}%")
print(f"Gap val/teste       : {gap * 100:.2f}%  ", end="")
if gap < 0.02:
    print("(baixo overfitting)")
else:
    print("(possível overfitting)")

#  5. Salvar o modelo
model.save("model.h5")     
print("\nModelo salvo em model.h5")

import os

size_h5 = os.path.getsize("model.h5") / 1024
print(f"Tamanho do modelo (.h5): {size_h5:.2f} KB")
=======
#insira seu código aqui
>>>>>>> 899bbaa9c313b8c23de5f26b7e09e78f0ad4ebe5
