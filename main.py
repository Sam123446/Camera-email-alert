import cv2
import os
import numpy as np
import time
import threading
from email_alert import send_email  # your existing email function

# ================== PATHS ==================
KNOWN_FACES_DIR = "known_faces"
CAPTURES_DIR = "captures"
os.makedirs(CAPTURES_DIR, exist_ok=True)

# ================== LOAD FACE DETECTOR ==================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ================== TRAINING DATA ==================
faces = []
labels = []
label_map = {}
current_label = 0

print("🔍 Loading known faces...")

for file in os.listdir(KNOWN_FACES_DIR):
    if not file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(KNOWN_FACES_DIR, file)
    image = cv2.imread(image_path)

    if image is None:
        print(f"❌ Could not read {file}")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detected = face_cascade.detectMultiScale(gray, 1.2, 5)
    print(f"{file} → faces detected: {len(detected)}")

    for (x, y, w, h) in detected:
        faces.append(gray[y:y+h, x:x+w])
        labels.append(current_label)

    if len(detected) > 0:
        label_map[current_label] = os.path.splitext(file)[0]
        current_label += 1

if len(faces) == 0:
    raise Exception("❌ No faces detected. Use a clear front-facing image.")

print("✅ Training data ready")
print("Faces:", len(faces))
print("Labels:", label_map)

# ================== TRAIN LBPH MODEL ==================
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))
print("✅ LBPH training completed")

# ================== CAMERA ==================
cap = cv2.VideoCapture(0)

EMAIL_DELAY = 30
last_email_time = 0
email_lock = threading.Lock()
enroll_mode = False

# ================== EMAIL THREAD ==================
def email_worker(frame, path):
    with email_lock:
        cv2.imwrite(path, frame)
        send_email(path)

# ================== MAIN LOOP ==================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in detected_faces:
        face_roi = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(face_roi)

        if confidence < 70:
            name = label_map[label]
            color = (0, 255, 0)
        else:
            name = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{name} ({int(confidence)})",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    color, 2)

        # 🔔 Email alert for unknown face
        if name == "Unknown" and time.time() - last_email_time > EMAIL_DELAY:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            image_path = f"{CAPTURES_DIR}/unknown_{timestamp}.jpg"
            frame_copy = frame.copy()
            threading.Thread(
                target=email_worker,
                args=(frame_copy, image_path),
                daemon=True
            ).start()
            last_email_time = time.time()

        # 🟢 Face Enrollment
        if enroll_mode:
            face_img = frame[y:y+h, x:x+w]
            enroll_name = input("Enter name for this face: ").strip()
            if enroll_name:
                save_path = os.path.join(KNOWN_FACES_DIR, f"{enroll_name}.jpeg")
                cv2.imwrite(save_path, face_img)
                print(f"✅ Face saved as {save_path}")
            else:
                print("❌ Name not provided")
            enroll_mode = False

    cv2.imshow("Face Recognition Alert System", frame)

    # ================== KEY HANDLING ==================
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('e'):
        enroll_mode = True
        print("🟢 Enrollment mode ON - Look at camera")
        
cap.release()
cv2.destroyAllWindows()
