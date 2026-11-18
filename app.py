import os
import cv2
import torch
import numpy as np
# Memotong jalur impor global untuk menghindari konflik protobuf/tensorflow
import mediapipe.python.solutions.face_mesh as mp_mesh
from facenet_pytorch import InceptionResnetV1

# 1. Inisialisasi Face Mesh Secara Langsung
face_mesh = mp_mesh.FaceMesh(
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Indeks Titik Wajah (Landmarks) untuk Skripsi
# Rahang (Jawline): Titik kontur bawah wajah
JAWLINE_IDX = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 
    400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 
    54, 103, 67, 109
]

# Hidung (Nose Outline)
NOSE_IDX = [168, 6, 197, 195, 5, 4, 1, 240, 97, 2, 326, 327, 278, 360]

# Pola Tengkorak / Dahi Atas (Skull/Forehead Line)
SKULL_FOREHEAD_IDX = [54, 68, 69, 108, 151, 337, 299, 298, 284]



# 2. Inisialisasi Deep Learning FaceNet (Ekstraksi 512 Vektor Wajah)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# 3. Load Dataset Wajah (Hadi, dll.)
DATASET_PATH = "dataset/train"
known_embeddings = []
known_names = []

def load_dataset():
    print("Mengindeks dataset wajah... Mohon tunggu.")
    if not os.path.exists(DATASET_PATH):
        print(f"Folder {DATASET_PATH} tidak ditemukan!")
        return

    for name in os.listdir(DATASET_PATH):
        person_dir = os.path.join(DATASET_PATH, name)
        if not os.path.isdir(person_dir):
            continue
        
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Ekstraksi wajah untuk dijadikan database internal
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_img)
            
            if results.multi_face_landmarks:
                # Ambil koordinat bounding box terkecil
                h, w, _ = img.shape
                landmarks = results.multi_face_landmarks[0].landmark
                coords = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
                x_min = max(0, min([c[0] for c in coords]))
                y_min = max(0, min([c[1] for c in coords]))
                x_max = min(w, max([c[0] for c in coords]))
                y_max = min(h, max([c[1] for c in coords]))
                
                face_crop = rgb_img[y_min:y_max, x_min:x_max]
                if face_crop.size == 0:
                    continue
                
                # Preprocessing untuk FaceNet
                face_resized = cv2.resize(face_crop, (160, 160))
                face_tensor = torch.tensor(face_resized).permute(2, 0, 1).float().to(device)
                face_tensor = (face_tensor - 127.5) / 128.0 # Normalisasi
                face_tensor = face_tensor.unsqueeze(0)
                
                with torch.no_grad():
                    embedding = resnet(face_tensor).cpu().numpy().flatten()
                
                known_embeddings.append(embedding)
                known_names.append(name)
                
    print(f"Selesai! Berhasil memuat {len(known_embeddings)} sampel wajah.")

load_dataset()

# 4. Loop Utama Deteksi Kamera Real-time
cap = cv2.VideoCapture(0)

# Optimasi Performa Video Frame
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    # Flip frame biar seperti cermin
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Deteksi Wajah menggunakan Face Mesh
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            coords = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]
            
            # --- FITUR 1 & 2: Gambar Detail Struktur (Rahang & Hidung) ---
            # Gambar Garis Rahang (Jawline)
            for i in range(len(JAWLINE_IDX) - 1):
                cv2.line(frame, coords[JAWLINE_IDX[i]], coords[JAWLINE_IDX[i+1]], (0, 255, 255), 2)
            # Gambar Pola Hidung
            for i in range(len(NOSE_IDX) - 1):
                cv2.line(frame, coords[NOSE_IDX[i]], coords[NOSE_IDX[i+1]], (255, 0, 0), 1)
            # Gambar Pola Tengkorak / Kontur Luar Wajah
            for i in range(len(SKULL_FOREHEAD_IDX) - 1):
                cv2.line(frame, coords[SKULL_FOREHEAD_IDX[i]], coords[SKULL_FOREHEAD_IDX[i+1]], (255, 0, 255), 1)
                
            # Deteksi Topi secara logika sederhana (Mengecek dahi terhalang/tidak)
            # Jika dahi (titik atas) terputus atau tidak ideal, tandai kemungkinan menggunakan aksesoris kepala
                        # --- FITUR DETEKSI TOPI & ATURAN VERIFIKASI KEAMANAN ---
            # Mengambil koordinat dahi atas (titik 10) dan tengah dahi (titik 9)
            top_forehead = face_landmarks.landmark[10].y
            mid_forehead = face_landmarks.landmark[9].y

            pakai_topi = False
            # Jika dahi tertutup topi, jarak antara dahi atas dan tengah akan mengecil secara drastis
            if (mid_forehead - top_forehead) < 0.025 or top_forehead < 0.05:
                pakai_topi = True

            # Koordinat bounding box wajah
            x_min = max(0, min([c[0] for c in coords]))
            y_min = max(0, min([c[1] for c in coords]))
            x_max = min(w, max([c[0] for c in coords]))
            y_max = min(h, max([c[1] for c in coords]))

            name_result = "Tidak Dikenal"
            warna_kotak = (0, 0, 255) # Default merah (Akses Ditolak)

            if pakai_topi:
                # JIKA PAKAI TOPI: LANGSUNG BLOKIR AKSES SECARA SOP KEAMANAN
                name_result = "AKSES DITOLAK: Lepas Topi Anda!"
                warna_kotak = (0, 0, 255) # Merah
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), warna_kotak, 2)
                cv2.putText(frame, name_result, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, warna_kotak, 2)
                continue # Langsung skip proses CNN di bawahnya demi efisiensi performa
            
                        # --- PROSES UNTUK KONDISI NORMAL (TIDAK PAKAI TOPI) ---
            face_crop = rgb_frame[y_min:y_max, x_min:x_max]
            if face_crop.size == 0:
                continue
                
            face_resized = cv2.resize(face_crop, (160, 160))
            face_tensor = torch.tensor(face_resized).permute(2, 0, 1).float().to(device)
            face_tensor = (face_tensor - 127.5) / 128.0
            face_tensor = face_tensor.unsqueeze(0)
            
            with torch.no_grad():
                current_embedding = resnet(face_tensor).cpu().numpy().flatten()
            
            # Proses Pencocokan Berbasis Jarak Euclidean
            name_result = "Tidak Dikenal"
            warna_kotak = (0, 0, 255) # Merah untuk tidak dikenal
            
            if len(known_embeddings) > 0:
                distances = [np.linalg.norm(current_embedding - ke) for ke in known_embeddings]
                min_distance_idx = np.argmin(distances)
                
                # Threshold kecocokan CNN
                if distances[min_distance_idx] < 0.75:
                    name_result = f"Akses Diterima: {known_names[min_distance_idx]}"
                    warna_kotak = (0, 255, 0) # Hijau jika berhasil verifikasi
                else:
                    name_result = "Akses Ditolak: Salah Orang"
                    warna_kotak = (0, 0, 255) # Merah jika salah orang
            
            # GAMBAR KOTAK UNTUK KONDISI NORMAL
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), warna_kotak, 2)
            cv2.putText(frame, name_result, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, warna_kotak, 2)

            
    cv2.imshow('Projek Deteksi Wajah Premium', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
