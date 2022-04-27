import cv2
import time
import numpy as np
from HandTrackingModule import HandDetector, Fingers
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

########################
wCam, hCam = 640, 480
########################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0

detector = HandDetector()

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# volume.GetMute()
# volume.GetMasterVolumeLevel()
minVol, maxVol, _ = volume.GetVolumeRange()
vol = 400
volBar = 400
volPer = 0
colorVolume = (255, 0, 0)
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands = detector.find_hands(img)

    # right_hand = detector.get_hand(hands, HandType.Right)

    for hand in hands:
        if hand.type == "Right":
            landmarks = hand.landmarks
            bounding_box = hand.bounding_box

            if landmarks:

                # Filter based on size
                area = (bounding_box[2] - bounding_box[0]) * (bounding_box[3] - bounding_box[1]) // 100

                if 250 < area < 1000:

                    # Find Distance between index and thumb
                    length, lineInfo = detector.find_distance(img, hand.get_finger(Fingers.Index),
                                                              hand.get_finger(Fingers.Thumb), True)

                    # Convert Volume
                    # Hand range 50 - 300
                    volBar = np.interp(length, [50, 200], [400, 150])
                    volPer = np.interp(length, [50, 200], [0, 100])

                    # Reduce Resolution to make it smoother
                    smoothness = 10
                    volPer = smoothness * round(volPer / smoothness)

                    # Check fingers up
                    fingers = hand.fingers_up()

                    # If pinky is down set volume
                    if not fingers[4]:
                        # volume.SetMasterVolumeLevelScale(volPer / 100, None)
                        cv2.circle(img, (lineInfo[4], lineInfo[5]), 7, (0, 255, 0), cv2.FILLED)
                        colorVolume = (0, 255, 0)
                    else:
                        colorVolume = (255, 0, 0)

    # Drawing
    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)}%', (40, 450), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    currentVolume = int(volume.GetMasterVolumeLevelScalar() * 100)
    cv2.putText(img, f'Vol Set: {int(volPer)}', (400, 50), cv2.FONT_HERSHEY_PLAIN, 2, colorVolume, 2)

    # Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

    cv2.imshow("Img", img)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
