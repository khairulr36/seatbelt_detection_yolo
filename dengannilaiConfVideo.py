import cv2
import torch
import numpy as np

path='besttt.pt'
path_video = 'tes1.mp4'
model = torch.hub.load('ultralytics/yolov5', 'custom', path, force_reload=True)
#cap=cv2.VideoCapture('http://192.168.1.2:4747/video')
cap=cv2.VideoCapture(path_video)

# Set threshold values
threshold = 0.0

# Loop through the video frames
while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the frame
    frame=cv2.resize(frame,(1020,500))
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect objects in the frame
    seatbelt_detected = False
    no_seatbelt_detected = False
    for pred in model(img).pred:
        # Extract objects with confidence score above threshold
        pred = [p for p in pred if p[4] >= threshold]
        if len(pred) > 0:
            for object in pred:
                conf, label = object[4], model.names[int(object[5])]
                if label == 'seatbelt' and conf >= threshold:
                    seatbelt_detected = True
                elif label != 'seatbelt' and conf >= threshold:
                    no_seatbelt_detected = True
                # Draw bounding box for object
                box = object[:4].detach().cpu().numpy().astype(int)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                # Display confidence score for object
                cv2.putText(frame, f"{label} {conf:.2f}", (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display seatbelt status text
    if seatbelt_detected:
        seatbelt_status = "Memakai sabuk pengaman"
    elif no_seatbelt_detected:
        seatbelt_status = "Tidak memakai sabuk pengaman"
    else:
        seatbelt_status = "Objek tidak terdeteksi"
    cv2.putText(frame, seatbelt_status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the image
    cv2.imshow("VIDEO", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()