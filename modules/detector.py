import cv2
import numpy as np

class FaceDetector:
    def __init__(self, prototxt, model, confidence=0.5):
        self.net = cv2.dnn.readNetFromCaffe(prototxt, model)
        self.conf = confidence

    def extract_faces(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0,
                                     (300,300), (104.0,177.0,123.0))
        self.net.setInput(blob)
        dets = self.net.forward()
        faces = []
        for i in range(dets.shape[2]):
            if dets[0,0,i,2] > self.conf:
                box = dets[0,0,i,3:7] * np.array([w,h,w,h])
                (sX,sY,eX,eY) = box.astype("int")
                sX,sY = max(0,sX), max(0,sY)
                eX,eY = min(w-1,eX), min(h-1,eY)
                faces.append((sX,sY,eX,eY, dets[0,0,i,2]))
        return faces