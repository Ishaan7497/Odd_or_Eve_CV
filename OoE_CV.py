import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

mp_draw = mp.solutions.drawing_utils


tip_ids = [4, 8, 12, 16, 20]

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

   
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    results = hands.process(rgb)

    total_fingers = 0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            landmarks = hand_landmarks.landmark
            fingers = []

            if landmarks[4].x > landmarks[3].x:
                fingers.append(1)
            else:
                fingers.append(0)

            for i in range(1,5):
                if landmarks[tip_ids[i]].y < landmarks[tip_ids[i]-2].y:
                    fingers.append(1)
                else:
                    fingers.append(0)

            total_fingers = fingers.count(1)

    cv2.putText(
        frame,
        f"Fingers: {total_fingers}",
        (50,100),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0,255,0),
        3
    )

    cv2.imshow("Hand Cricket Input", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()