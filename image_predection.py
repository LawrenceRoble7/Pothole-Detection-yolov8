from ultralytics import YOLO
import cv2
import pandas as pd

img_path = input("Enter the image path: ")
image = cv2.imread(img_path)
if image is None:
    print("No image found, please try again!")

# Resize the image
width, height = [1280 // 2, 720 // 2]
image = cv2.resize(image, (width, height))

# Load the YOLOv8 model with the specified weights
yolov8_weights = "weights/yolov8n_pothole_weights.pt"
model = YOLO(yolov8_weights, "v8")

# Predict the bounding boxes on the provided image
pred = model.predict(source=[image], save=False)
PX_ = pd.DataFrame(pred[0].boxes.data).astype("float")

sumOfCls = []
for index_, row in PX_.iterrows():
    x1 = int(row[0])
    y1 = int(row[1])
    x2 = int(row[2])
    y2 = int(row[3])

    confidence = round(row[4], 5)

    w, h = x2 - x1, y2 - y1
    rec_pos = (x1, y1, x2, y2)  # Corrected rec_pos
    text_pos = (x1, y1 - 10)

    font = cv2.FONT_HERSHEY_COMPLEX
    box_color = (255, 100, 100)

    cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)  # Corrected rectangle parameters
    sumOfCls.append(x1)
    # Uncomment if you want to display confidence
    # cv2.putText(image, f"{confidence}%", text_pos, font, fontScale=0.5, color=(255, 255, 255), thickness=2)

cv2.putText(image, f"Total Classes: {len(sumOfCls)}", (15, 15), font, fontScale=0.5, color=(255, 255, 255), thickness=2)
cv2.imshow("Detected Potholes", image)

cv2.waitKey(0)
cv2.destroyAllWindows()