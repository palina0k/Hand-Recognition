import cv2
import mediapipe as mp
import math

# Loading images
normal = cv2.imread("hamster_normal.jpg")
peace = cv2.imread("hamster_peace_sign.jpg")
scared = cv2.imread("hamster_scared.jpg")
thinking = cv2.imread("thinking.jpg")

if normal is None or peace is None or scared is None or thinking is None:
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
    max_num_faces=1,
    refine_landmarks=True
)

hands = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=2
)

cap = cv2.VideoCapture(0)

def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def get_mouth_center(face_landmarks, width, height):
    upper_lip_points = [13, 14, 80, 81]
    lower_lip_points = [17, 84, 314, 178]
    
    mouth_points = []
    for idx in upper_lip_points + lower_lip_points:
        if idx < len(face_landmarks.landmark):
            landmark = face_landmarks.landmark[idx]
            mouth_points.append((int(landmark.x * width), int(landmark.y * height)))
    
    if mouth_points:
        center_x = sum([p[0] for p in mouth_points]) // len(mouth_points)
        center_y = sum([p[1] for p in mouth_points]) // len(mouth_points)
        return (center_x, center_y)
    
    return (int(face_landmarks.landmark[13].x * width), 
            int(face_landmarks.landmark[13].y * height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame,1)
    height, width, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_results = face_mesh.process(rgb_frame)
    mouth_open = False
    mouth_center = None
    finger_touching_mouth = False

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mouth_center = get_mouth_center(face_landmarks, width, height)
            
            upper_lip = face_landmarks.landmark[13]
            lower_lip = face_landmarks.landmark[14]

            mouth_distance = abs(lower_lip.y - upper_lip.y) * height
            if mouth_distance > 15:
                mouth_open = True

            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

    hand_results = hands.process(rgb_frame)
    finger_count = 0

    if hand_results.multi_hand_landmarks:

        for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            lm = hand_landmarks.landmark

            handedness = "right"
            if hand_results.multi_handedness:
                handedness = hand_results.multi_handedness[hand_idx].classification[0].label.lower()
            
            tips = [8, 12, 16, 20]
            pips = [6, 10, 14, 18]

            thumb_tip = lm[4]
            wrist = lm[0]
            index_mcp = lm[5]

            if handedness == "right":
                if thumb_tip.x < index_mcp.x:
                    finger_count += 1
            else:
                if thumb_tip.x > index_mcp.x:
                    finger_count += 1

            for tip_idx, pip_idx in zip(tips, pips):
                if lm[tip_idx].y < lm[pip_idx].y:
                    finger_count += 1

            finger_tips = [
                (int(lm[8].x * width), int(lm[8].y * height)),   # Index
                (int(lm[12].x * width), int(lm[12].y * height)),  # Middle
                (int(lm[16].x * width), int(lm[16].y * height)),  # Ring
                (int(lm[20].x * width), int(lm[20].y * height)),  # Pinky
                (int(lm[4].x * width), int(lm[4].y * height))     # Thumb
            ]
            
            touch_threshold = 50
            for finger_tip in finger_tips:
                distance = calculate_distance(finger_tip, mouth_center)
                if distance < touch_threshold:
                    finger_touching_mouth = True
                    break 
            

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

    if finger_touching_mouth:
        current_img = thinking.copy()
    elif finger_count == 2 and not mouth_open:
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