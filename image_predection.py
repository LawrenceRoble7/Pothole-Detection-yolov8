from ultralytics import YOLO
import cv2
import pandas as pd

path = [
    r'C:\Users\Lawrence\Desktop\Pothole-Detection-yolov8\inferences\images\img1.jpg',
    r'C:\Users\Lawrence\Desktop\Pothole-Detection-yolov8\inferences\images\img2.png',
    r'C:\Users\Lawrence\Desktop\Pothole-Detection-yolov8\inferences\images\img3.jpg',
    r'C:\Users\Lawrence\Desktop\Pothole-Detection-yolov8\inferences\images\img4.jpg'
]
image = cv2.imread(path[1])
width, height = [1280 // 2, 720  // 2]
image = cv2.resize(image, (width, height))

yolov8_weights = "weights/yolov8n_pothole_weights.pt"
model = YOLO(yolov8_weights, "v8")
pred = model.predict(source=[image], save=False)
PX_ = pd.DataFrame(pred[0].boxes.data).astype("float")
sumOfCls = []
for index_, row in PX_.iterrows():
    x1 = int(row[0])
    y1 = int(row[1])
    x2 = int(row[2])
    y2 = int(row[3])

    confidence = round(row[4], 5)
    # detected_class_index = int(row[5])
    # class_ID_name = class_list[detected_class_index]

    w, h = x2 - x1, y2 - y1
    rec_pos = (x1, y1, w, h)
    text_pos = (x1, y1-10)

    font = cv2.FONT_HERSHEY_COMPLEX
    box_color = (255, 100, 100)

    cv2.rectangle(image, rec_pos, box_color, 2)
    sumOfCls.append(x1)
    # cv2.putText(image, f"{confidence}%", text_pos, font, fontScale=0.5, color=(255, 255, 255), thickness=2)

cv2.putText(image, f"Total Class: {len(sumOfCls)}", (15, 15), font, fontScale=0.5, color=(255, 255, 255), thickness=2)
cv2.imshow("Frame2", image) 

cv2.waitKey(0) 
cv2.destroyAllWindows() 
