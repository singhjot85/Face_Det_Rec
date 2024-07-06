import os
import cv2
import face_recognition
import pickle

face_database = "Face Dataset"
cache_file = "face_encodings_cache.pkl"

def image_loader(directory_path):
    image_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    return image_files

def find_face_encodings(image):
    face_enc = face_recognition.face_encodings(image)  # Get face encodings from the image
    if face_enc:
        return face_enc[0]  # Return face encodings
    else:
        return None

def cache_encodings():
    encodings = {}
    paths = image_loader(face_database)
    for path in paths:
        image = cv2.imread(path)
        enc = find_face_encodings(image)
        if enc is not None:
            name= os.path.basename(path).split('.',1)[0]
            encodings[name] = enc
    with open(cache_file, 'wb') as f:
        pickle.dump(encodings, f)

def load_encodings():
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    else:
        return {}

def recognize_face(frame, cached_encodings):
    frame_encoding = find_face_encodings(frame)
    if frame_encoding is None:
        return "No face detected"
    
    for name, enc in cached_encodings.items():
        is_same = face_recognition.compare_faces([enc], frame_encoding)[0] 
        if is_same:
            return name
    return "Face detected, user not found"

def detect_bounding_box(vid, cached_encodings):
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
        custom_text = recognize_face(vid, cached_encodings)
        cv2.putText(vid, custom_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return vid

def process_video(video_path, cached_encodings, skip_frames=2):
    video_capture = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        result, video_frame = video_capture.read()
        if not result:
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        if frame_count % skip_frames == 0:
            frame = detect_bounding_box(video_frame, cached_encodings)
            if frame is not None:
                cv2.imshow("Face Recognition", frame)
        frame_count += 1
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    video_capture.release()
    cv2.destroyAllWindows()

def launch_camera(n,cached_encodings):
    video_capture = cv2.VideoCapture(n)
    while True:
        result, video_frame = video_capture.read()
        if not result:
            break
        
        frame = detect_bounding_box(video_frame, cached_encodings)
        if frame is not None:
            cv2.imshow("Face Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == 32: 
            cv2.imwrite("captured_image.jpg", video_frame)
            break
    video_capture.release()
    cv2.destroyAllWindows()

def main():
    if not os.path.exists(cache_file):
        cache_encodings()


    cached_encodings = load_encodings()

    n= 0 #Camera to be used: 0->webcam, 1->other connected device

    choice = input("Enter 'c' to use the camera or 'u' to upload an image/video path: ").lower()
    if choice == 'c':
        launch_camera(n,cached_encodings)
    elif choice == 'u':
        file_path = input("Enter the file path of the image/video: ")
        if os.path.isfile(file_path):
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image = cv2.imread(file_path)
                frame = detect_bounding_box(image, cached_encodings)
                cv2.imshow("Face Recognition", frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            elif file_path.lower().endswith(('.mp4', '.avi', '.mov')):
                process_video(file_path, cached_encodings)
            else:
                print("Unsupported file format.")
        else:
            print("File not found.")
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
