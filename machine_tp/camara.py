import cv2
import mediapipe as mp
import numpy as np
import math

# --- 1. IMPORTAMOS Y ENTRENAMOS EL MODELO (SOLO UNA VEZ) ---
# Importamos la función para entrenar desde tu propio script
from utiles.entrenando_tp import train_model

print("Entrenando el modelo, por favor espera...")
# Entrenamos el modelo al iniciar el programa
model = train_model()
print("¡Modelo entrenado y listo!")

# Creamos un diccionario para saber qué significa cada predicción del modelo
# IMPORTANTE: Asegúrate de que el orden sea el mismo que usaste para entrenar
# (por ejemplo, si tus datos de entrenamiento están como estrella, rectangulo, triangulo)
labels = {0: 'Estrella', 1: 'Rectangulo', 2: 'Triangulo'}


# --- 2. CONFIGURACIÓN DE MEDIAPIPE PARA DETECCIÓN DE MANOS ---
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils


# --- 3. INICIAMOS LA CAPTURA DE VIDEO ---
capture = cv2.VideoCapture(0)

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break

    # Guardamos las dimensiones del fotograma
    height, width, _ = frame.shape

    # Convertimos la imagen a RGB para MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Hacemos la detección
    image_rgb.flags.writeable = False
    results = holistic_model.process(image_rgb)
    image_rgb.flags.writeable = True

    # Convertimos de vuelta a BGR para mostrar con OpenCV
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Verificamos si se detectó una mano (usaremos solo la derecha por simplicidad)
    if results.right_hand_landmarks:
        hand_landmarks = results.right_hand_landmarks

        # --- 4. CREAR UNA IMAGEN DE LA FORMA DE LA MANO ---
        # Creamos una imagen negra (una máscara) del mismo tamaño que el video
        mask = np.zeros((height, width), dtype=np.uint8)

        # Obtenemos las coordenadas de los puntos de la mano y las escalamos al tamaño de la imagen
        points = np.array(
            [[int(landmark.x * width), int(landmark.y * height)] for landmark in hand_landmarks.landmark],
            dtype=np.int32
        )
        
        # Dibujamos un polígono relleno con los puntos de la mano sobre la máscara negra
        cv2.fillConvexPoly(mask, points, 255)

        # --- 5. CALCULAR MOMENTOS DE HU Y PREDECIR ---
        # Buscamos el contorno de la forma que acabamos de dibujar
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            cnt = contours[0]
            # Calculamos los momentos de Hu del contorno
            moments = cv2.moments(cnt)
            hu_moments = cv2.HuMoments(moments)

            # Aplicamos la transformación logarítmica (igual que en el entrenamiento)
            for i in range(0, 7):
                if hu_moments[i] != 0:
                    hu_moments[i] = -1 * math.copysign(1.0, hu_moments[i]) * math.log10(abs(hu_moments[i]))

            # Preparamos los momentos para el modelo y hacemos la predicción
            hu_moments_reshaped = hu_moments.reshape(1, -1)
            prediction = model.predict(hu_moments_reshaped)
            
            # Obtenemos el nombre de la forma predicha
            label_text = labels.get(prediction[0], "Desconocido")

            # --- 6. MOSTRAR EL RESULTADO EN PANTALLA ---
            # Escribimos la predicción en la esquina superior izquierda del video
            cv2.putText(image_bgr, label_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
        
        # Dibujamos los puntos de la mano en la imagen original
        mp_drawing.draw_landmarks(
            image_bgr,
            hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )

    # Mostramos la imagen final
    cv2.imshow("Reconocimiento de Formas con la Mano", image_bgr)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()