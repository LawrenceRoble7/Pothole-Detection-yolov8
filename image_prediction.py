from ultralytics import YOLO
import cv2
import pandas as pd

frame_name = ""
yolov8_weights = ""

if input("Choose weights (n or l): ") == 'n':
    frame_name = "YOLOv8-nano"
    yolov8_weights = "weights/yolov8n_pothole_weights.pt"
else:
    frame_name = "YOLOv8-large"
    yolov8_weights = "weights/yolov8l_pothole_weights.pt"

img_path = [
    "inferences/images/img1.jpg",
    "inferences/images/img2.png",
    "inferences/images/img3.jpg",
    "inferences/images/img4.jpg",
]

index = input("Choose one image (1 to 4): ")

model = YOLO(yolov8_weights, "v8")
image = cv2.imread(img_path[int(index) - 1])

# Resize the image
width, height = [1280 // 2, 720 // 2]
image = cv2.resize(image, (width, height))

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
    start_point, end_point = [(x1, y1), (x2, y2)]
    # text_pos = (x1, y1 - 10)

    font = cv2.FONT_HERSHEY_COMPLEX
    box_color = (255, 100, 100)

    cv2.rectangle(image, start_point, end_point, box_color, 2)  # Corrected rectangle parameters
    sumOfCls.append(x1)
    # Uncomment if you want to display confidence
    # cv2.putText(image, f"{confidence}%", text_pos, font, fontScale=0.5, color=(255, 255, 255), thickness=2)

cv2.putText(image, f"Total Classes: {len(sumOfCls)}", (15, 15), font, fontScale=0.5, color=(255, 255, 255), thickness=2)
cv2.imshow(frame_name, image)

cv2.waitKey(0)
cv2.destroyAllWindows()