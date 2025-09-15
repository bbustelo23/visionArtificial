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
# IMPORTANTE: Asegúrate de que el orden sea el mismo que usaste para entrenar
labels = {0: 'estrella_tp', 1: 'rectangulo_tp', 2: 'triangulo_tp'}

# --- 2. INICIAR LA CÁMARA ---
capture = cv2.VideoCapture(0)
if not capture.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

while True:
    # Capturamos un fotograma (frame) de la cámara
    ret, frame = capture.read()
    if not ret:
        print("No se pudo recibir el fotograma. Saliendo...")
        break

    # --- 3. PROCESAR LA IMAGEN PARA ENCONTRAR FORMAS ---
    
    # Convertimos la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Aplicamos un desenfoque para reducir el ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Creamos una imagen binaria (blanco y negro). 
    # THRESH_BINARY_INV convierte los objetos oscuros (como un dibujo en marcador) a blanco.
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

    # Buscamos los contornos de todas las formas blancas en la imagen
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- 4. ANALIZAR LA FORMA MÁS GRANDE ENCONTRADA ---
    if contours:
        # Encontramos el contorno con el área más grande
        cnt = max(contours, key=cv2.contourArea)

        # Solo procesamos contornos que sean suficientemente grandes para evitar ruido
        if cv2.contourArea(cnt) > 500:
            # Dibujamos el contorno en la imagen original para ver qué está detectando
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 3)

            # --- 5. CALCULAR HU MOMENTS Y PREDECIR ---
            moments = cv2.moments(cnt)
            hu_moments = cv2.HuMoments(moments)
            
            # Aplicamos la transformación logarítmica (igual que en el entrenamiento)
            for i in range(0, 7):
                if hu_moments[i] != 0:
                    hu_moments[i] = -1 * math.copysign(1.0, hu_moments[i]) * math.log10(abs(hu_moments[i]))

            # Preparamos los datos para el modelo y hacemos la predicción
            hu_moments_reshaped = hu_moments.reshape(1, -1)
            prediction = model.predict(hu_moments_reshaped)
            
            # Obtenemos el nombre de la forma
            label_text = labels.get(prediction[0], "Desconocido")
            
            # Mostramos el resultado en pantalla
            cv2.putText(frame, label_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)

    # --- 6. MOSTRAR EL VIDEO ---
    cv2.imshow('Reconocimiento de Figuras', frame)

    # Salimos del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberamos la cámara y cerramos las ventanas
capture.release()
cv2.destroyAllWindows()