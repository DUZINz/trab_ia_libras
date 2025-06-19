import cv2
import mediapipe as mp
import pickle
import numpy as np

# 1. Carregar o modelo treinado
with open('reconhecedor_libras.pkl', 'rb') as f:
    model = pickle.load(f)

# 2. Configurações do MediaPipe e OpenCV
mp_maos = mp.solutions.hands
mp_desenho = mp.solutions.drawing_utils
maos = mp_maos.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    sucesso, imagem = cap.read()
    if not sucesso:
        print("Falha na câmera.")
        break

    imagem = cv2.flip(imagem, 1)
    imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
    resultados = maos.process(imagem_rgb)

    if resultados.multi_hand_landmarks:
        for pontos_mao in resultados.multi_hand_landmarks:
            # Desenha os pontos na mão
            mp_desenho.draw_landmarks(
                imagem, pontos_mao, mp_maos.HAND_CONNECTIONS,
                mp_desenho.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=4),
                mp_desenho.DrawingSpec(color=(0, 255, 0), thickness=2)
            )

            # 3. Extrair coordenadas e preparar para o modelo
            pontos = []
            for marco in pontos_mao.landmark:
                pontos.extend([marco.x, marco.y, marco.z])
            
            # Converte para o formato que o modelo espera (1 amostra, 63 features)
            dados_para_previsao = np.array(pontos).reshape(1, -1)

            # 4. Fazer a Previsão
            previsao = model.predict(dados_para_previsao)
            letra_prevista = previsao[0].upper() # Pega a primeira (e única) previsão

            # 5. Mostrar o resultado na tela
            # Pega as coordenadas do pulso para posicionar o texto
            x_pulso = int(pontos_mao.landmark[0].x * imagem.shape[1])
            y_pulso = int(pontos_mao.landmark[0].y * imagem.shape[0])
            
            cv2.putText(imagem, letra_prevista, (x_pulso, y_pulso - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow('Reconhecedor de Libras - EDUARO', imagem)

    # Pressione 'q' para sair
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()