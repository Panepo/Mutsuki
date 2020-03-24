import cv2
import numpy as np

cap = cv2.VideoCapture(0)
tracker = cv2.TrackerKCF_create()
traceStart = 0

while(True):
    (grabbed, img) = cap.read()
    if not grabbed:
        break

    if traceStart:
        ok, bbox = tracker.update(img)

        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(img, p1, p2, (0,255,0), 3)

    cv2.imshow("Tracking", img)

    getKey = cv2.waitKey(20) & 0xFF
    if getKey == ord('q'):
        break
    elif getKey == ord('c'):
        bbox = cv2.selectROI(img, False)
        ok = tracker.init(img, bbox)
        traceStart = 1
        cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()
