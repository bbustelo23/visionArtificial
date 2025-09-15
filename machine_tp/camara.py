import cv2
import numpy as np
import math

# Importamos la función para entrenar desde tu propio script
from utiles.entrenando_tp import train_model

# --- 1. ENTRENAR EL MODELO UNA SOLA VEZ ---
print("Entrenando el modelo, por favor espera...")
model = train_model()
print("¡Modelo entrenado y listo!")

# Diccionario para traducir la predicción del modelo a texto
labels = {0: 'Estrella', 1: 'Rectangulo', 2: 'Triangulo'}

# --- 2. INICIAR LA CÁMARA ---
capture = cv2.VideoCapture(0)
if not capture.isOpened():
    print("Error: No se pudo abrir la cámara.")
    print("Asegúrate de que no esté siendo usada por otro programa.")
    exit()
else:
    print("Cámara iniciada. Presiona 'q' para salir.")

while True:
    # Capturamos un fotograma (frame) de la cámara
    ret, frame = capture.read()

    # --- MODIFICACIÓN CLAVE ---
    # Si 'ret' es False, no se pudo capturar el fotograma.
    # Usamos 'continue' para saltar al siguiente intento en lugar de cerrar.
    if not ret:
        print("Advertencia: No se pudo capturar el fotograma. Reintentando...")
        continue # <-- ESTE ES EL CAMBIO IMPORTANTE

    # --- 3. PROCESAR LA IMAGEN PARA ENCONTRAR FORMAS ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- 4. ANALIZAR LA FORMA MÁS GRANDE ENCONTRADA ---
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) > 500:
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 3)
            moments = cv2.moments(cnt)
            hu_moments = cv2.HuMoments(moments)
            for i in range(0, 7):
                if hu_moments[i] != 0:
                    hu_moments[i] = -1 * math.copysign(1.0, hu_moments[i]) * math.log10(abs(hu_moments[i]))
            hu_moments_reshaped = hu_moments.reshape(1, -1)
            prediction = model.predict(hu_moments_reshaped)
            label_text = labels.get(prediction[0], "Desconocido")
            cv2.putText(frame, label_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)

    # --- 6. MOSTRAR EL VIDEO ---
    cv2.imshow('Reconocimiento de Figuras', frame)

    # Salimos del bucle SOLAMENTE si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberamos la cámara y cerramos las ventanas
capture.release()
cv2.destroyAllWindows()