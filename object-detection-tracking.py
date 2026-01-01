import cv2
import numpy as np

def detect_stop_sign(frame):
    #Detect a red octagonal stop sign and return bounding box (x, y, w, h)

    # Step 1: Grayscale conversion
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Step 2: Gaussian Blur

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 3: Thresholding (Simple Binary)
    
    _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)


    cv2.imshow("Original",frame)
    cv2.imshow("Grayscale", gray)
    cv2.imshow("Gaussian Blur", blur)
    cv2.imshow("Thresholding", thresh)

    

    # CANNY EDGE DETECTION
    
    edges = cv2.Canny(blur, 50, 150)
    cv2.imshow("canny_edge", edges)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 120, 70]) #np.array([H_min, S_min, V_min])
    upper_red1 = np.array([10, 255, 255]) #np.array function takes a list as argument and converts it into array
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Creating red masks
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    red_only = cv2.bitwise_and(frame, frame, mask=red_mask)

    cv2.imshow("Original", frame)
    cv2.imshow("Red Mask", red_mask)
    cv2.imshow("Red Detection", red_only)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        if len(approx) == 8 and cv2.contourArea(cnt) > 500:
            cv2.drawContours(frame, [cnt], 0, (0, 255, 0), 3)
            
            
            x, y = approx.ravel()[0], approx.ravel()[1]


            cv2.putText(frame, "STOP Sign", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
            # Get bounding box coordinates of the detected octagon
            x, y, w, h = cv2.boundingRect(approx)

            cv2.imshow("Red Octagon Detection", frame)
            return (x, y, w, h)
            
        
            
    return None
    
# tracking 
# Start video capture 
cap = cv2.VideoCapture(0)  # Use 0 or path to video

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 240)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

tracker = None 
initBB = None
frame_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    display = frame.copy()
    
    if tracker is None or frame_counter % 60 == 0:  # Redetect every 60 frames
        bbox = detect_stop_sign(frame)
        if bbox: #if stop sign is detected
            initBB = bbox
            tracker = cv2.legacy.TrackerMOSSE_create()
            tracker.init(frame, initBB)
            print(f"Tracking initialized at {initBB}")
        else:
            cv2.putText(display, "STOP Sign Not Detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        
        success, box = tracker.update(frame)
        if success: 
            
            (x, y, w, h) = [int(v) for v in box] 
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display, "Tracking STOP Sign", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display, "Tracking Lost", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("MOSSE Tracker", display)
    if cv2.waitKey(1) & 0xFF == ord('m'):   
        break

cap.release()
cv2.destroyAllWindows()