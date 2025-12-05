import cv2
import mediapipe as mp

# Loading images
normal = cv2.imread("hamster_normal.jpg")
peace = cv2.imread("hamster_peace_sign.jpg")
scared = cv2.imread("hamster_scared.jpg")

if normal is None or peace is None or scared is None:
    print("Error: Could not load the images")
    exit()

current_img = normal.copy()

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_faces=1
)

hands = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=2
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame,1)
    height, width, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_results = face_mesh.process(rgb_frame)
    mouth_open = False

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            upper_lip = face_landmarks.landmark[13]
            lower_lip = face_landmarks.landmark[14]

            mouth_distance = abs(lower_lip.y - upper_lip.y) * height
            if mouth_distance > 15:
                mouth_open = True

            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

    hand_results = hands.process(rgb_frame)
    finger_count = 0

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            lm = hand_landmarks.landmark

            tips = [8, 12, 16, 20]
            pips = [6, 10, 14, 18]

            thumb_tip = lm[4]
            thumb_ip = lm[3]

            if thumb_tip.x < thumb_ip.x:
                finger_count += 1

            for tip_idx, pip_idx in zip(tips, pips):
                if lm[tip_idx].y < lm[pip_idx].y:
                    finger_count += 1

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

    if finger_count == 2 and not mouth_open:
        current_img = peace.copy()
    elif mouth_open:
        current_img = scared.copy()
    else:
        current_img = normal.copy()
    
    h_img, w_img, _ = current_img.shape
    scale_factor = min(200 / w_img, 200 / h_img)
    new_w = int(w_img * scale_factor)
    new_h = int(h_img * scale_factor)
    resized_img = cv2.resize(current_img, (new_w, new_h))

    frame[10:10+new_h, width-new_w-10:width-10] = resized_img

    cv2.putText(frame, f'Fingers: {finger_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Mouth: {"Open" if mouth_open else "Closed"}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Hand and Mouth Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()