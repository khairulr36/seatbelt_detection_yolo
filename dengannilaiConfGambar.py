import cv2
import torch
import numpy as np

path = 'besttt.pt'
path_image = 'sabuk.jpg'
model = torch.hub.load('ultralytics/yolov5', 'custom', path, force_reload=True)

# Load the image and preprocess it
img = cv2.imread(path_image)
img = cv2.resize(img, (1020, 500))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Set threshold values
threshold = 0.0

# Detect objects in the image
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
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            # Display confidence score for object
            cv2.putText(img, f"{label} {conf:.2f}", (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Display seatbelt status text
if seatbelt_detected:
    seatbelt_status = "Memakai sabuk pengaman"
elif no_seatbelt_detected:
    seatbelt_status = "Tidak memakai sabuk pengaman"
else:
    seatbelt_status = "Objek tidak terdeteksi"
cv2.putText(img, seatbelt_status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Display the image
cv2.imshow("IMAGE", img)
cv2.waitKey(0)
cv2.destroyAllWindows()