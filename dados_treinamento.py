movements = ['oi', 'meu', 'nome', 'hugo']
import cv2
import mediapipe as mp
import numpy as np
import os
import time
import json

# Inicializa os módulos MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Diretório para salvar os dados
data_dir = 'hand_movement_data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)


# Função para salvar os dados
def save_data(landmarks, label):
    data = {'landmarks': landmarks.tolist(), 'label': label}
    timestamp = int(time.time() * 1000)
    with open(os.path.join(data_dir, f'data_{timestamp}.json'), 'w') as f:
        json.dump(data, f)


# Captura de vídeo
cap = cv2.VideoCapture(1)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Converte a imagem para RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Processa a imagem e detecta as mãos
    results = hands.process(image)

    # Desenha as landmarks das mãos
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Coleta as coordenadas das landmarks
            landmarks = [[landmark.x, landmark.y] for landmark in hand_landmarks.landmark]
            landmarks = np.array(landmarks).flatten()

            # Substitua 'up_and_down' pela etiqueta correta ao coletar dados
            save_data(landmarks, 'hugo')

    # Mostra a imagem com as landmarks
    cv2.imshow('Hand Tracking', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
