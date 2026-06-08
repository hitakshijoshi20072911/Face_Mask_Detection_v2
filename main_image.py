import cv2, argparse
from modules.detector import FaceDetector
from modules.classifier import MaskClassifier
from modules.logger import ComplianceLogger

parser = argparse.ArgumentParser()
parser.add_argument("--image", nargs='+', required=True)
parser.add_argument("--model", choices=["cnn","mbv2","cnn_tflite","mbv2_tflite"], default="cnn")
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

image_path = " ".join(args.image)

# Model selection (same as above)
if args.model == "cnn":
    MODEL_FILE = "models/custom_cnn/mask_detector_cnn.keras"
    LABEL_FILE = "models/custom_cnn/class_labels.json"
    IS_MOBILENET = False; USE_TFLITE = False
elif args.model == "mbv2":
    MODEL_FILE = "models/mobilenetv2/mask_detector_mbv2.keras"
    LABEL_FILE = "models/mobilenetv2/class_labels.json"
    IS_MOBILENET = True; USE_TFLITE = False
elif args.model == "cnn_tflite":
    MODEL_FILE = "models/custom_cnn/mask_detector_cnn.tflite"
    LABEL_FILE = "models/custom_cnn/class_labels.json"
    IS_MOBILENET = False; USE_TFLITE = True
elif args.model == "mbv2_tflite":
    MODEL_FILE = "models/mobilenetv2/mask_detector_mbv2.tflite"
    LABEL_FILE = "models/mobilenetv2/class_labels.json"
    IS_MOBILENET = True; USE_TFLITE = True

detector = FaceDetector(
    "models/face_detector/deploy.prototxt",
    "models/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
)
classifier = MaskClassifier(MODEL_FILE, LABEL_FILE,
                            is_mobilenet=IS_MOBILENET, use_tflite=USE_TFLITE)
logger = ComplianceLogger()

frame = cv2.imread(image_path)
if frame is None:
    print(f"Error: Could not read image '{image_path}'")
    exit()

frame = cv2.resize(frame, (800, 600))

faces = detector.extract_faces(frame)
for (x1,y1,x2,y2, conf) in faces:
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        continue
    label, score, full_preds = classifier.evaluate_face_raw(roi)

    if args.debug:
        print(f"[DEBUG] Raw predictions: {full_preds} -> {label} ({score*100:.1f}%)")

    if score < 0.4:
        continue

    color = (0,255,0) if label == "with_mask" else \
            (0,255,255) if label == "incorrect_mask" else (0,0,255)
    cv2.putText(frame, f"{label}: {score*100:.1f}%", (x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
    if label != "with_mask":
        logger.log(label, score, frame.copy())

cv2.imshow("Image Inference", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()