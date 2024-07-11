import cv2
import mediapipe as mp
import numpy as np
from keras.src.saving.saving_api import load_model

# Carregar o modelo salvo
model = load_model('hand_movement_model.keras')

# Inicializa os módulos MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Função para processar a imagem e extrair landmarks
def extract_landmarks(image):
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = [[landmark.x, landmark.y] for landmark in hand_landmarks.landmark]
            return np.array(landmarks).flatten()
    return None

# Captura de vídeo
cap = cv2.VideoCapture(1)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Extrai as landmarks da mão
    landmarks = extract_landmarks(frame)

    if landmarks is not None:
        # Prepara os dados de entrada para o modelo
        X_input = np.expand_dims(landmarks, axis=0)  # Adiciona uma dimensão extra para representar o batch

        # Faz a previsão
        prediction = model.predict(X_input)
        predicted_class = np.argmax(prediction)

        # Mapeia a classe prevista para o nome do movimento
        movements = ['hugo', 'meu', 'nome', 'oi']
        predicted_movement = movements[predicted_class]

        # Mostra a classe prevista na imagem
        cv2.putText(frame, f'Movement: {predicted_movement}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Mostra a imagem com a previsão
    cv2.imshow('Hand Movement Recognition', frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
