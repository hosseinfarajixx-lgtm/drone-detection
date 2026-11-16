import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture('drone.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame)
    
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            name = model.names[cls]
            if name in ['airplane', 'bird']:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f'Drone? {name}', (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
    
    cv2.imshow('Drone Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()