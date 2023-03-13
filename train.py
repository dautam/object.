import depthai
import cv2
import numpy as np

# initialize the OAK-D camera
device = depthai.Device()
pipeline = device.create_pipeline(config={
    "streams": ["metaout", "previewout"],
    "ai": {
        "blob_file": "/user/tamdau/Desktop/detect/best.blob",
        "blob_file_config": "/Users/tamdau/Desktop/detect/best.blob",
        "calc_dist_to_bb": True,
        "keep_aspect_ratio": True,
        "camera_input": "right"
    }
})

# set up the drone flight path
path = [(0, 0), (100, 0), (100, 100), (0, 100)]

while True:
    # get the next frame from the camera
    data_packets = pipeline.get_available_data_packets()
    if not data_packets:
        continue
    data_packet = data_packets[0]
    if data_packet.stream_name == 'previewout':
        frame = data_packet.get_preview_frame()

        # run object detection on the frame
        detections = pipeline.get_output_queue("object_detection").try_get()
        if detections is not None:
            for detection in detections.detections:
                label = detection.label
                x1, y1, x2, y2 = detection.xmin, detection.ymin, detection.xmax, detection.ymax
                distance_to_object = detection.spatialCoordinates.z
                # adjust the drone flight path to avoid the object
                if label == "person" and distance_to_object < 2.0:
                    path = [(0, 0), (100, 0), (100, 50), (150, 50), (150, 100), (0, 100)]
                elif label == "car" and distance_to_object < 5.0:
                    path = [(0, 0), (100, 0), (100, 100), (0, 100)]

        # draw the flight path on the frame
        for i in range(len(path) - 1):
            cv2.line(frame, path[i], path[i + 1], (0, 0, 255), 2)

        # display the frame
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == ord('q'):
            break

# clean up resources
pipeline.close()
device.close()
cv2.destroyAllWindows()
