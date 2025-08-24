import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL

# Ses Seviyesi Ayarları
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
min_volume, max_volume = volume.GetVolumeRange()[:2]

# Mediapipe ve El Takibi Modülleri
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.7,
)

# Kamera Ayarları
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

screen_width, screen_height = pyautogui.size()
previous_position = None

mode = ""


# Mesafe Hesaplama Fonksiyonu
def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


# Parmak Durumu Tespiti Fonksiyonu
def capture_fingers(hand_landmarks):
    fingers = []
    for tip_index in [mp_hands.HandLandmark.INDEX_FINGER_TIP,
                      mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                      mp_hands.HandLandmark.RING_FINGER_TIP,
                      mp_hands.HandLandmark.PINKY_TIP]:
        if hand_landmarks.landmark[tip_index].y < hand_landmarks.landmark[tip_index - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers


# Mouse Cursor Kontrolü
def control_cursor(index_finger_tip):
    global previous_position
    if previous_position is None or (
            abs(index_finger_tip[0] - previous_position[0]) > 5 or abs(index_finger_tip[1] - previous_position[1]) > 5):
        pyautogui.moveTo(index_finger_tip[0], index_finger_tip[1])
        previous_position = index_finger_tip


# Ses Seviyesi Kontrolü
def control_volume(thumb_tip, index_finger_tip, frame):
    distance = calculate_distance(thumb_tip, index_finger_tip)
    volume_level = np.interp(distance, [30, 450], [min_volume, max_volume])
    volume.SetMasterVolumeLevel(volume_level, None)

    vol_thumb_tip = (
        int(landmarks[mp_hands.HandLandmark.THUMB_TIP].x * frame.shape[1]),
        int(landmarks[mp_hands.HandLandmark.THUMB_TIP].y * frame.shape[0]),
    )
    vol_index_finger_tip = (
        int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1]),
        int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0]),
    )

    # Çizgiyi ve daireyi çizin
    cv2.line(frame, vol_thumb_tip, vol_index_finger_tip, (0, 255, 0), 3)

    center_point = (
        int((vol_thumb_tip[0] + vol_index_finger_tip[0]) / 2),
        int((vol_thumb_tip[1] + vol_index_finger_tip[1]) / 2),
    )

    # Daireyi çiz
    cv2.circle(frame, center_point, 10, (0, 0, 255), -1)

    # Ses Seviyesini Görselleştir
    cv2.putText(frame, f"Volume: {int(np.interp(volume_level, [min_volume, max_volume], [0, 100]))}%",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


# Scroll Kontrolü
def control_scroll(mode):
    if mode == "scrollUp":
        pyautogui.scroll(110)
    elif mode == "scrollDown":
        pyautogui.scroll(-110)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            fingers = capture_fingers(hand_landmarks)
            landmarks = hand_landmarks.landmark
            print(fingers)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Parmak İpuçları ve Koordinatlar
            index_finger_tip = (
                landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * screen_width,
                landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * screen_height,
            )
            thumb_tip = (
                landmarks[mp_hands.HandLandmark.THUMB_TIP].x * screen_width,
                landmarks[mp_hands.HandLandmark.THUMB_TIP].y * screen_height,
            )
            pinky_mcp = (landmarks[mp_hands.HandLandmark.PINKY_MCP].x * screen_width,
                         landmarks[mp_hands.HandLandmark.PINKY_MCP].y * screen_height)

            # El Modları
            if fingers == [1, 1, 1, 1]:
                mode = "cursor"
            elif fingers == [1, 0, 0, 1]:
                mode = "volume"
            elif fingers == [1, 0, 0, 0]:
                mode = "scrollUp"
            elif fingers == [1, 1, 0, 0]:
                mode = "scrollDown"
            elif fingers == [0, 0, 0, 0]:
                mode = "free"

            # Mod Kontrolü
            if mode == "cursor":
                control_cursor(index_finger_tip)
            elif mode == "volume":
                control_volume(thumb_tip, index_finger_tip, frame)
            elif mode in ["scrollUp", "scrollDown"]:
                control_scroll(mode)

            # Başparmak ve Pembe Parmağın Mesafesine Göre Tıklama
            if calculate_distance(pinky_mcp, thumb_tip) < 350 and mode == "cursor":
                pyautogui.click()

    # Görüntüyü Göster
    cv2.imshow("El Takibi", frame)

    # FPS ve Bekleme Süresi
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.de
