import cv2
import os
import time

NAMA_USER = 'unknown'  
TIPE_DATA = 'train'  
JUMLAH_MAKSIMAL = 20 

target_dir = f'dataset/{TIPE_DATA}/{NAMA_USER}'

if not os.path.exists(target_dir):
    print(f"[ERROR] Folder '{target_dir}' tidak ditemukan!")
    exit()

cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print("[ERROR] File detektor wajah Haar Cascade tidak dapat dimuat!")
    exit()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
time.sleep(2) 

count = 0

print(f"\n[INFO] Bersiaplah! Kamera akan mengambil {JUMLAH_MAKSIMAL} sampel wajah untuk '{NAMA_USER}'...")
print("[INFO] Pastikan wajah menghadap ke kamera dengan pencahayaan yang cukup.")
print("[INFO] Tekan tombol 'q' pada jendela kamera untuk keluar.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("[WARNING] Gagal menangkap gambar dari webcam, mencoba ulang...")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.05,  
        minNeighbors=3,    
        minSize=(40, 40)   
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face_crop = frame[y:y+h, x:x+w]
        face_resize = cv2.resize(face_crop, (150, 150))
        
        count += 1
        file_name = f"{target_dir}/{NAMA_USER}_{count}.jpg"
        cv2.imwrite(file_name, face_resize)
        
        cv2.putText(frame, f"Captured: {count}/{JUMLAH_MAKSIMAL}", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        time.sleep(0.3)

    cv2.imshow('Proses Pengumpulan Dataset Wajah', frame)

    if count >= JUMLAH_MAKSIMAL:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Pengambilan data dibatalkan oleh pengguna.")
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n[SUKSES] Berhasil menyimpan {count} foto wajah bersih di: {target_dir}")
