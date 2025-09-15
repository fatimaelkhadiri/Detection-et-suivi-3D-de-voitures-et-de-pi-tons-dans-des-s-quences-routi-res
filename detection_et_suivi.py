import cv2
import numpy as np
import torch
import os

# Charger YOLO (YOLOv5)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# Configuration complète des paramètres de YOLOv5
model.conf = 0.5  # Confiance minimale
model.iou = 0.45  # Seuil d'IoU
model.agnostic = False  # Suppression agnostique de classe
model.multi_label = False  # Détection multi-label désactivée
model.max_det = 1000  # Nombre maximum de détections
model.amp = True  # Précision mixte pour accélération
model.half = False  # Désactivation du mode 16 bits
model.imgsz = 640  # Taille d'entrée de l'image
model.classes = [0, 1, 2, 7]  # Personne, Cycliste, Voiture, Camion
model.device = 'cpu'  # Utilisation du CPU uniquement
model.dnn = False  # Désactivation de l'inférence avec OpenCV DNN
model.verbose = True  # Affichage des logs détaillés

# Paramètres de calibration avancés issus du fichier calib_cam_to_cam.txt
K_left = np.array([[9.842439e+02, 0.0, 6.900000e+02],
                   [0.0, 9.808141e+02, 2.331966e+02],
                   [0.0, 0.0, 1.0]])
D_left = np.array([-3.728755e-01, 2.037299e-01, 2.219027e-03, 1.383707e-03, -7.233722e-02])
R_left = np.array([[1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0],
                   [0.0, 0.0, 1.0]])
T_left = np.array([2.573699e-16, -1.059758e-16, 1.614870e-16])
baseline = 0.537  # Basé sur la composante X de T_01
focal_length = K_left[0, 0]  

# Chemins vers les dossiers des images stéréo
left_images_path = r'C:\Users\Fatima\Downloads\2011_09_26_drive_0005_sync\2011_09_26\2011_09_26_drive_0005_sync\image_00\data'
right_images_path = r'C:\Users\Fatima\Downloads\2011_09_26_drive_0005_sync\2011_09_26\2011_09_26_drive_0005_sync\image_01\data'

# Charger toutes les images des deux dossiers
left_images = sorted(os.listdir(left_images_path))
right_images = sorted(os.listdir(right_images_path))

# Vérifier que le nombre d'images est le même
assert len(left_images) == len(right_images), "Le nombre d'images doit être identique dans les deux dossiers."

# Créer un objet pour calculer la disparité
stereo = cv2.StereoSGBM_create(numDisparities=64, blockSize=15)

# Définir les couleurs pour chaque classe
colors = {
    'person': (0, 0, 255),  # Rouge
    'bicycle': (0, 140, 255),  # Orange
    'car': (255, 0, 0),  # Bleu
    'cyclist': (128, 0, 128)  # Violet foncé
}

def get_color(label):
    return colors.get(label, (0, 255, 0))  # Vert par défaut

# Traiter chaque paire d'images
for left_img_name, right_img_name in zip(left_images, right_images):
    # Charger les images
    img_left = cv2.imread(os.path.join(left_images_path, left_img_name))
    img_right = cv2.imread(os.path.join(right_images_path, right_img_name))

    # Conversion en niveaux de gris
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # Calcul de la disparité
    disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0

    # Détection avec YOLO sur l'image gauche
    results = model(img_left)
    detections = results.pandas().xyxy[0]  # Récupérer les détections sous forme de dataframe

    # Créer une image pour la vue de dessus
    bird_eye_view = np.zeros((500, 500, 3), dtype=np.uint8)

    # Ajouter l'icône représentant la position de la caméra
    cv2.circle(bird_eye_view, (250, 490), 10, (255, 255, 255), -1)
    cv2.putText(bird_eye_view, 'Camera', (260, 490), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

    # Stocker les coordonnées 3D des objets détectés
    detected_objects = []

    # Dessiner les objets détectés sur l'image et dans la vue de dessus
    for index, detection in detections.iterrows():
        x_min, y_min, x_max, y_max, confidence, class_name = (
            int(detection['xmin']), int(detection['ymin']),
            int(detection['xmax']), int(detection['ymax']),
            detection['confidence'], detection['name']
        )
        if class_name in ['umbrella', 'traffic light']:
            continue

        color = get_color(class_name)
        cv2.rectangle(img_left, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(img_left, f"{class_name} {confidence:.2f}", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Calculer la position centrale de l'objet
        center_x = int((x_min + x_max) / 2)
        center_y = int((y_min + y_max) / 2)

        # Obtenir la disparité au point central
        disparity_value = disparity[center_y, center_x]
        if disparity_value > 0:
            # Calculer la profondeur
            Z = (focal_length * baseline) / disparity_value
            X = (center_x - K_left[0, 2]) * Z / focal_length
            Y = (center_y - K_left[1, 2]) * Z / focal_length
            detected_objects.append((X, Y, Z, class_name))

# Dessiner tous les objets détectés sur le bird eye avec tolérance de distance
    unique_points = {}
    tolerance = 5.2  # Tolérance de distance pour regrouper les points proches

    for obj in detected_objects:
        added = False
        for key in unique_points:
            if np.linalg.norm(np.array(key) - np.array(obj[:3])) < tolerance:
                if set([unique_points[key], obj[3]]) == {'person', 'bicycle'}:
                    unique_points[key] = 'cyclist'
                added = True
                break
        if not added:
            unique_points[(obj[0], obj[1], obj[2])] = obj[3]

    for point, label in unique_points.items():
        bird_x = int(250 + point[0] * 10)
        bird_y = int(500 - point[2] * 10)
        color = get_color(label)
        cv2.circle(bird_eye_view, (bird_x, bird_y), 5, color, -1)
        cv2.putText(bird_eye_view, label, (bird_x + 10, bird_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))


    cv2.imshow("YOLO Detection", img_left)
    cv2.imshow("Bird Eye View", bird_eye_view)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
