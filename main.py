from ultralytics import YOLO
import cv2
import pandas as pd

def Split_Class_List(file_path):
    myFile = open(file_path, "r")
    data = myFile.read()
    class_list = data.split("\n")
    myFile.close()
    
    return class_list

def main():
    VIDEO_SOURCE_PATH = "inferences/videos/sample_video1.mp4"
    yolov8_weights = "weights/yolov8n_pothole_weights.pt"
    COCO_FILE_PATH = "utils/coco.names"

    model = YOLO(yolov8_weights, "v8")
    cap = cv2.VideoCapture(VIDEO_SOURCE_PATH)
    class_list = Split_Class_List(COCO_FILE_PATH) 

    if not cap.isOpened():
        print("Cannot open camera")
        exit()


    while True:
        success, frame = cap.read()

        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, (0, 0), fx=0.68, fy=0.68)
        pred = model.predict(source=[frame], save=False)
        PX_ = pd.DataFrame(pred[0].boxes.data).astype("float")

        for index_, row in PX_.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])

            confidence = round(row[4], 5)
            detected_class_index = int(row[5])
            class_ID_name = class_list[detected_class_index]
            start_point, end_point = [(x1, y1), (x2, y2)]
            text_pos = (x1, y1 - 10)

            font = cv2.FONT_HERSHEY_COMPLEX
            clsID_and_Conf = f"{class_ID_name} {confidence}%"
            box_color = (255, 100, 100)

            cv2.rectangle(frame, start_point, end_point, box_color, 2)
            cv2.putText(frame, clsID_and_Conf, text_pos, font, fontScale=0.5, color=(255, 255, 255), thickness=2)
        
        cv2.imshow("SAMPLE DETECTION", frame)

        if cv2.waitKey(1) & 0xFF == 27: # ESC
            break


    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()