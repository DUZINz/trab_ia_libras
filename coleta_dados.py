import cv2
import mediapipe as mp
import csv
import os

# Inicialização do MediaPipe e configurações
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

csv_file = 'alfabeto_libras.csv'
samples_per_letter = 200

header = ['letra']
for i in range(21):
    header += [f'x{i}', f'y{i}', f'z{i}']

if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
    print(f"Arquivo {csv_file} criado.")
else:
    print(f"Arquivo {csv_file} já existe. Novos dados serão adicionados.")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Não foi possível acessar a câmera.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=4),
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
            )

    cv2.imshow('Coleta de Dados - Libras', frame)
    key = cv2.waitKey(10) & 0xFF

    if key == ord('0'):
        break

    if ord('a') <= key <= ord('z'):
        letra = chr(key)
        print(f"Coletando dados para '{letra.upper()}'")
        for i in range(samples_per_letter):
            ret_sample, frame_sample = cap.read()
            if not ret_sample:
                print("Falha ao capturar imagem.")
                continue

            frame_sample = cv2.flip(frame_sample, 1)
            rgb_sample = cv2.cvtColor(frame_sample, cv2.COLOR_BGR2RGB)
            result_sample = hands.process(rgb_sample)

            if result_sample.multi_hand_landmarks:
                for hand_landmarks in result_sample.multi_hand_landmarks:
                    row = [letra]
                    for lm in hand_landmarks.landmark:
                        row.extend([lm.x, lm.y, lm.z])
                    with open(csv_file, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(row)
                    print(f"Amostra {i+1}/{samples_per_letter} salva.")
            else:
                print("Nenhuma mão detectada.")

            status = f"Coletando... {i + 1}/{samples_per_letter}"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('Coleta de Dados - Libras', frame)
            cv2.waitKey(1)

        print(f"Coleta para '{letra.upper()}' finalizada.")