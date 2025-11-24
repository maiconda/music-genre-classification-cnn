import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# --- CONFIGURAÇÕES ---
# Onde estão as imagens geradas
DATASET_PATH = './spectrograms_img'

# Tamanho padrão para redimensionar as imagens (a rede precisa de entrada fixa)
IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 32 # Quantas imagens a rede vê por vez antes de atualizar os pesos

print("Carregando dataset...")

# 1. CARREGAMENTO DOS DADOS (PIPELINE)
# Keras tem uma função mágica que lê as pastas e já entende que o nome da pasta é a classe (label)
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2, # 20% para validação (prova), 80% para treino (estudo)
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print(f"Gêneros encontrados: {class_names}")

# Otimização de performance para carregar dados na memória cache
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 2. CONSTRUÇÃO DA REDE NEURAL (CNN)
num_classes = len(class_names)

model = models.Sequential([
    # Camada de normalização: converte pixels de 0-255 para 0-1 (ajuda na matemática)
    layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    
    # Bloco 1 de Convolução
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    # Bloco 2 de Convolução
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    # Bloco 3 de Convolução
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    # Dropout para evitar Overfitting (a rede "decora" em vez de aprender)
    layers.Dropout(0.2),
    
    # Achatamento (transforma matriz 2D em vetor 1D)
    layers.Flatten(),
    
    # Camadas Densas (Classificação final)
    layers.Dense(256, activation='relu'),
    layers.Dense(num_classes, activation='softmax') # Softmax dá a % de probabilidade
])

# 3. COMPILAÇÃO DO MODELO
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

model.summary()

# 4. TREINAMENTO
print("\nIniciando treinamento... (isso pode demorar)")
epochs = 15 # Quantas vezes a rede vai estudar todo o dataset
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# 5. VISUALIZAÇÃO DOS RESULTADOS
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(12, 6))

# Gráfico de Acurácia
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Acurácia de Treino')
plt.plot(epochs_range, val_acc, label='Acurácia de Validação')
plt.legend(loc='lower right')
plt.title('Performance de Treino vs Validação')

# Gráfico de Perda (Erro)
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Erro de Treino')
plt.plot(epochs_range, val_loss, label='Erro de Validação')
plt.legend(loc='upper right')
plt.title('Erro (Loss)')

plt.show()

# Salvar o modelo treinado para usar depois
model.save('modelo_musical_treinado.keras')
print("Modelo salvo como 'modelo_musical_treinado.keras'")