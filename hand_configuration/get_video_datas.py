import cv2
import os
import mediapipe as mp
import numpy as np

# Inicialização do Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Inicialização da captura de vídeo
cap = cv2.VideoCapture(0)

# Nome da pasta e contagem de imagens
folder_name = ""
image_count = 0

# Função para salvar a imagem
def save_image(image, folder_name):
    global image_count
    folder_path = os.path.join('hand_images', folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    image_path = os.path.join(folder_path, f"{image_count}.jpg")
    cv2.imwrite(image_path, image)
    image_count += 1

# Janela principal
cv2.namedWindow('Hand Tracking')
# cv2.setMouseCallback('Hand Tracking', on_mouse_click)

# Calcular a distância em pixels para 3 centímetros (ajuste conforme necessário)
pixel_per_cm = 30  # Aproximadamente 30 pixels por cm (dependendo da resolução da câmera e do monitor)
margin_cm = 1
margin_pixels = int(margin_cm * pixel_per_cm)

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

    hand_detected = False
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extrai a região da mão
            x_coords = [int(landmark.x * frame.shape[1]) for landmark in hand_landmarks.landmark]
            y_coords = [int(landmark.y * frame.shape[0]) for landmark in hand_landmarks.landmark]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            # Adiciona a margem de 3 centímetros
            x_min = max(0, x_min - margin_pixels)
            x_max = min(frame.shape[1], x_max + margin_pixels)
            y_min = max(0, y_min - margin_pixels)
            y_max = min(frame.shape[0], y_max + margin_pixels)

            # Verifica se a mão está completamente visível
            if x_min > 0 and y_min > 0 and x_max < frame.shape[1] and y_max < frame.shape[0]:
                hand_image = frame[y_min:y_max, x_min:x_max]
                hand_detected = True

                # Mostra a imagem da mão em uma nova janela
                cv2.imshow('Hand Only', hand_image)

    # Mostra a imagem com as landmarks
    if folder_name:
        cv2.putText(image, f"Folder: {folder_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Hand Tracking', image)

    # Leitura de texto em tempo de execução
    key = cv2.waitKey(1) & 0xFF
    if key == ord('t'):
        folder_name = input("Enter folder name: ")

    # Captura de foto
    if key == ord('c'):
        if hand_detected and folder_name:
            save_image(hand_image, folder_name)

    # Saída do loop ao pressionar 'q'
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
