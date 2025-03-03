import cv2

############### Tracker Types #####################
# Uncomment the tracker you want to use
#tracker = cv2.TrackerBoosting_create()
tracker = cv2.TrackerMIL_create()
tracker = cv2.TrackerKCF_create()
#tracker = cv2.TrackerTLD_create()
# tracker = cv2.TrackerMedianFlow_create()
#tracker = cv2.legacy.TrackerCSRT_create()  # Updated for compatibility with modern OpenCV versions

########################################################

cap = cv2.VideoCapture(0)  # Default webcam

# TRACKER INITIALIZATION
success, frame = cap.read()
if not success:
    print("Failed to read from webcam. Exiting...")
    cap.release()
    cv2.destroyAllWindows()
    exit()

bbox = cv2.selectROI("Tracking", frame, False)
tracker.init(frame, bbox)

def drawBox(img, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 3)
    cv2.putText(img, "Tracking", (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

while True:
    timer = cv2.getTickCount()
    success, img = cap.read()
    if not success:
        print("Failed to capture frame. Exiting...")
        break

    success, bbox = tracker.update(img)

    if success:
        drawBox(img, bbox)
    else:
        cv2.putText(img, "Lost", (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    # Draw FPS and status box
    cv2.rectangle(img, (15, 15), (200, 90), (255, 0, 255), 2)
    cv2.putText(img, "Fps:", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    cv2.putText(img, "Status:", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    # Color based on FPS
    if fps > 60:
        myColor = (20, 230, 20)
    elif fps > 20:
        myColor = (230, 20, 20)
    else:
        myColor = (20, 20, 230)

    cv2.putText(img, str(int(fps)), (75, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, myColor, 2)

    cv2.imshow("Tracking", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Fixed condition
        break

cap.release()
cv2.destroyAllWindows()
