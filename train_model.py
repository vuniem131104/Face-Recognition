import os
import numpy as np
import cv2
from PIL import Image
from sklearn.metrics import accuracy_score

def get_image_data(path):
    paths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    ids = []
    for image_path in paths:
        img = Image.open(image_path).convert('L')
        img_np = np.array(img, 'uint8')
        faces.append(img_np)
        ids.append(int(os.path.split(image_path)[1].split('.')[1]))
    return np.array(ids), faces

ids_train, faces_train = get_image_data('people/train/')
ids_test, faces_test = get_image_data('people/test/')
lbph_classifier = cv2.face.LBPHFaceRecognizer.create(grid_x=30, grid_y=30, radius=2)
lbph_classifier.train(faces_train, ids_train)
lbph_classifier.write('lbph_classifier.yml')
lbph_trained_classifier = cv2.face.LBPHFaceRecognizer.create()
lbph_trained_classifier.read('lbph_classifier.yml')
predicts = [lbph_trained_classifier.predict(image)[0] for image in faces_test]
print("LBPH: {:.2f}%".format(accuracy_score(list(ids_test), predicts)*100))
