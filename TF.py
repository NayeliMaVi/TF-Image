import cv2
from ultralytics import YOLO
import math as m
import mediapipe as mp

# Función para calcular el ángulo entre dos puntos
def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt((x2 - x1)**2 + (y2 - y1)**2) * y1))
    degree = int(180 / m.pi) * theta
    return degree

#cambiar la ruta xd
# Ruta al modelo YOLOv8
model_path = "D:\\VS\\TF-PI\\runs\\detect\\train\\weights\\best.pt"

# Cargar el modelo YOLOv8
model = YOLO(model_path)

# Inicializar MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se puede abrir la cámara.")
    exit()


with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("No se pueden leer los frames de la cámara.")
            break

        image_pose = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pose = cv2.flip(image_pose, 1)
        image_pose.flags.writeable = False
        results_pose = pose.process(image_pose)
        image_pose.flags.writeable = True
        image_pose = cv2.cvtColor(image_pose, cv2.COLOR_RGB2BGR)

        # Detección de postura con MediaPipe
        if results_pose.pose_landmarks:
            landmarks = results_pose.pose_landmarks.landmark

            # Puntos de interés para la postura
            earL = 8  # Oído izquierdo seria 7 pero mi cam reocnoce 8 
            shouldL = 12  # Hombro izquierdo seria 11 pero micam reconoce 12
            hipsL = 24  # Cadera izquierda seria 23 pero mi cam reconoce 24

            if landmarks[earL] and landmarks[shouldL] and landmarks[hipsL]:
                h, w, c = image_pose.shape
                earX = int(landmarks[earL].x * w)
                earY = int(landmarks[earL].y * h)
                SLpX = int(landmarks[shouldL].x * w)
                SLpY = int(landmarks[shouldL].y * h)
                hLX = int(landmarks[hipsL].x * w)
                hLY = int(landmarks[hipsL].y * h)

                # Dibujar puntos y líneas para la postura
                cv2.circle(image_pose, (SLpX, earY), 10, (255, 255, 0), -1)
                cv2.circle(image_pose, (earX, earY), 10, (255, 255, 0), -1)
                cv2.circle(image_pose, (SLpX, SLpY), 10, (255, 255, 0), -1)
                cv2.circle(image_pose, (hLX, hLY), 10, (255, 255, 0), -1)

                cv2.line(image_pose, (earX, earY), (SLpX, SLpY), (0, 0, 255), 9)
                cv2.line(image_pose, (SLpX, SLpY), (hLX, hLY), (0, 0, 255), 9)
                cv2.line(image_pose, (SLpX, SLpY), (SLpX, earY), (0, 0, 255), 9)

                
                angulo = findAngle(SLpX, SLpY, earX, earY)
                cv2.putText(image_pose, f"Inclinacion del cuello: {int(angulo)} grados", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                # Detectar mala postura 
                if angulo > 30:
                    cv2.putText(image_pose, "MALA POSTURA", (20, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Detección de objetos con YOLOv8
        results_yolo = model(frame)[0]

        for result in results_yolo.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > 0.25 and score < 0.30:  # Umbral de confianza para resaltar cajas en rojo
                cv2.rectangle(image_pose, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
                cv2.putText(image_pose, f'{model.names[int(class_id)].upper()}: {score:.2f}', (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        
        cv2.imshow('Detección de objeto y postura', image_pose)

        # Detener la ejecución si se presiona 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()