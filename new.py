from roboflowoak import RoboflowOak
import cv2
import time
import numpy as np

def adjust_fly_path(predictions, frame, speed =20):
    # calculate the center of the frame
    center_x = frame.shape[1] // 2
    center_y = frame.shape[0] // 2


    # iterate over the detected objects
    for pred in predictions:
        x1 = pred.x
        y1 = pred.y
        x2 = pred.x + pred.width
        y2 = pred.y + pred.height

        # calculate the center of the object
        object_center_x = (x1 + x2) // 2
        object_center_y = (y1 + y2) // 2

        # calculate the distance between the center of the frame and the center of the object
        distance_x = center_x - object_center_x
        distance_y = center_y - object_center_y
        # calculate the angle between the obstacle and the center of the frame
        angle = np.arctan2(distance_y, distance_x)

        # adjust the fly path based on the angle
        if abs(angle) > np.pi / 4:
            if angle > 0:
                print("Adjusting fly path to the right")
                # adjust the fly path to the right

            else:
                print("Adjusting fly path to the left")
                # adjust the fly path to the left

        if abs(distance_y) > frame.shape[0] // 10:
            if distance_y < 0:
                print("Adjusting fly path downwards")
                # adjust the fly path downwards

            else:
                print("Adjusting fly path upwards")
                # adjust the fly path upwards

if __name__ == '__main__':
    # instantiate a video capture object
    cap = cv2.VideoCapture(0)

    # instantiating an object (rf) with the RoboflowOak module
    rf = RoboflowOak(model="object-detect-81cl8", confidence=0.05, overlap=0.5,
    version="1", api_key="e9LwkOXSTafyixWudFZh", rgb=True,
    depth=True, device="rgb", blocking=True)

    # Running our model and displaying the video output with detections
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()

        # The rf.detect() function runs the model inference
        result, frame, _, depth = rf.detect()

        predictions = result["predictions"]

        # printing out the predictions in a more readable format
        for pred in predictions:
            print("x: {}, y: {}, width: {}, height: {}, depth: {}, confidence: {}, class: {}".format(
                pred.x, pred.y, pred.width, pred.height,
                pred.depth, pred.confidence, pred.class_name
            ))

        # adjust the fly path based on the detected objects
        adjust_fly_path(predictions, frame)

        # displaying the video feed as successive frames
        cv2.imshow("Drone Camera", frame)

        # how to close the OAK inference window / stop inference: CTRL+q or CTRL+c
        if cv2.waitKey(1) == ord('q'):
            break

    # release the camera and close all windows
    cap.release()
    rf.close()
    cv2.destroyAllWindows()
