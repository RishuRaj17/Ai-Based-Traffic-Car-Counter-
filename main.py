from ultralytics import YOLO
import cv2
import math

cap =  cv2.VideoCapture("video/traffic2.mp4")
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1880)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,900)

counter_line_position = 200
tracker= []
counter= 0

while True:
    success, frame = cap.read()
    model = YOLO("/yolo-weights/yolov8n.pt")
    results = model(frame, stream= True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            conf = math.ceil(box.conf[0]*100)

            c = box.cls
            name = model.names[int(c)]
            if name == "car" or name == "truck" and conf>35:
               cv2.putText(frame,name, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
               cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
               center = x1+15 , y1+15
               cv2.circle(frame,center,1,(0,0,255),2,cv2.FILLED)
               tracker.append(center)

               cv2.line(frame,(2,counter_line_position),(650,counter_line_position),(0,255,255),2)

            for (x,y) in tracker:
                if y<(counter_line_position+7) and y>(counter_line_position):
                    counter+=1
                tracker.remove((x,y))
                print(counter)
                cv2.putText(frame,"Vechicle counter:"+str(counter),(160,20),cv2.FONT_HERSHEY_COMPLEX,0.5,(1,1,1),1)

    cv2.imshow("Video",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()