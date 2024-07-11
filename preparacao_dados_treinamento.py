import os
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


data_dir = 'hand_movement_data'
X = []
y = []

# Carrega os dados salvos
for file_name in os.listdir(data_dir):
    with open(os.path.join(data_dir, file_name), 'r') as f:
        data = json.load(f)
        X.append(data['landmarks'])
        y.append(data['label'])

# Converte para arrays numpy
X = np.array(X)
y = np.array(y)

# Codificação das etiquetas (label encoding)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Certifique-se de que X tenha o formato (n_amostras, n_features)
n_landmarks = 21  # MediaPipe usa 21 pontos de referência
n_features = n_landmarks * 2  # Cada ponto tem coordenadas x e y
X = X.reshape((X.shape[0], n_features))

print(X.shape)  # Deve ser (n_amostras, n_features)
print(y.shape)  # Deve ser (n_amostras,)

# Criação do modelo (para dados sem dimensão temporal)
model = Sequential([
    Dense(64, activation='relu', input_shape=(n_features,)),
    Dense(64, activation='relu'),
    Dense(len(np.unique(y)), activation='softmax')  # Número de classes
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Treinamento do modelo
model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2)

model.save('hand_movement_model.keras')
