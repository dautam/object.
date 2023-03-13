from roboflowoak import RoboflowOak
import cv2
import time
import numpy as np
if __name__ == '__main__':
    # instantiating an object (rf) with the RoboflowOak module
    rf = RoboflowOak(model="staircase-r5ko1", confidence=0.05, overlap=0.5,
                     version="1", api_key="SMyDfda4TDc4FN43nxEm", rgb=True,
                     depth=True, device="rgb", blocking=True)

    # Running our model and displaying the video output with detections
    while True:
        t0 = time.time()

        # The rf.detect() function runs the model inference
        result, frame, raw_frame, depth = rf.detect()

        staircase_detections = [pred for pred in result["predictions"] if pred.class_name == "staircase"]

        # printing out the staircase detections in a more readable format
        for pred in staircase_detections:
            print("Staircase detected: x: {}, y: {}, width: {}, height: {}, depth: {}, confidence: {}".format(
                pred.x, pred.y, pred.width, pred.height, pred.depth, pred.confidence))
            # setting parameters for depth calculation
            max_depth = np.amax(depth)
            cv2.imshow("depth", depth / max_depth)
        # displaying the video feed as successive frames
        cv2.imshow("Staircase Detection", frame)

        # how to close the OAK inference window / stop inference: CTRL+q or CTRL+c
        if cv2.waitKey(1) == ord('q'):
            break

    # release the camera and close all windows
    cap.release()
    rf.close()
    cv2.destroyAllWindows()
