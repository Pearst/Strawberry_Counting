import cv2
import numpy as np
import time

# Load Yolo
net = cv2.dnn.readNet("./strawberry_yolotiny_3000.weights", "./strawberry_yolotiny.cfg")

classes = ["Strawberry"]

layer_names = net.getLayerNames()
outputlayers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Initialize object tracking
tracker = cv2.TrackerCSRT_create()
tracking_started = False
tracked_object = None

# Loading web cam
camera = cv2.VideoCapture(0)
#camera = cv2.VideoCapture("4.mp4")



detected_objects = []  # Store detected objects

font = cv2.FONT_HERSHEY_PLAIN
start_time = time.time()
frame_count = 0

while True:
    success, img = camera.read()
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(outputlayers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    if boxes:
        if not tracking_started:
            x, y, w, h = boxes[0]
            tracked_object = (x, y, x + w, y + h)
            tracking_started = tracker.init(img, tuple(tracked_object))
        else:
            success, tracked_object = tracker.update(img)
            if success:
                x, y, w, h = map(int, tracked_object)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                tracking_started = False

    count = len(boxes)  # Count of detected objects
    cv2.putText(img, f'Count: {count}', (10, 30), font, 2, (0, 255, 0), 2)

    # Calculate and display FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    cv2.putText(img, f'FPS: {fps:.2f}', (width - 160, 30), font, 2, (0, 255, 0), 2)

    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        if class_ids[i] < len(classes):
            label = str(classes[class_ids[i]])
        else:
            label = 'Unknown'  # Handle unknown or out-of-range classes
        color = colors[class_ids[i] % len(colors)]  # Use modulo to ensure a valid color
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

    cv2.imshow("Image", img)

    key = cv2.waitKey(1)
    if key == 27:
        break

camera.release()
cv2.destroyAllWindows()