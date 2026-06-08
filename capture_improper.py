import cv2
import os

os.makedirs("dataset/train/incorrect_mask", exist_ok=True)
cap = cv2.VideoCapture(0)
count = 0

print("Press SPACE to capture, 'q' to quit.")
while True:
    ret, frame = cap.read()
    cv2.imshow("Capture", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        fname = f"dataset/train/incorrect_mask/my_improper_{count:04d}.jpg"
        cv2.imwrite(fname, frame)
        print(f"Saved {fname}")
        count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()