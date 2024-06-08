import cv2

# Membuat objek CascadeClassifier untuk deteksi wajah
face_ref = cv2.CascadeClassifier("face_ref.xml")
# Mengaktifkan kamera webcam
camera = cv2.VideoCapture(0)


def face_detection(frame):
    # Konversi frame ke skala grayscale untuk deteksi wajah
    optimize_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Mendeteksi wajah di dalam frame
    faces = face_ref.detectMultiScale(optimize_frame, scaleFactor=1.1, minNeighbors=5)
    return faces


def drawer_box(frame, faces, name):
    for (x, y, w, h) in faces:
        # Membuat sebuah kotak di sekitar wajah
        cv2.rectangle(frame, (x, y), (x + w, y + h), (128, 0, 128), 4)
        # Menampilkan nama yang ada di atas wajah
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


def close_window():
    # Menghentikan penggunaan kamera
    camera.release()
    # Menutup semua jendela OpenCV
    cv2.destroyAllWindows()
    # Keluar dari program
    exit()

def main():
    name = "Rizka Magvira"
    while True:
        # Membaca frame dari kamera
        _, frame = camera.read()
        # Mendeteksi wajah di dalam frame
        faces = face_detection(frame)
        # Menggambar sebuah kotak sekitar wajah dan menulis nama
        drawer_box(frame, faces, name)
        # Menampilkan sebuah frame dengan label "Deteksi Wajah"
        cv2.imshow("Mendeteksi Wajah", frame)

        # Jika tombol 'q' ditekan, maka program sedang berjalan akan berhenti
        if cv2.waitKey(1) & 0xFF == ord('q'):
            close_window()


if __name__ == '__main__':
    main()