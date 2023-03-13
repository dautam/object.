import cv2
import depthai as dai

# Set up the neural network
model_path = '/Users/tamdau/Desktop/model/best.blob'
model_name = 'best'

# Create the pipeline
pipeline = dai.Pipeline()

# Define the input and output nodes
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(640, 640)
cam_rgb.setInterleaved(False)
cam_rgb.setFps(30)

nn = pipeline.createYoloDetectionNetwork()
nn.setBlobPath(model_path)
nn.setNumClasses(1)
nn.setCoordinateSize(4)
nn.setAnchors(
    [
        12, 16, 19, 36, 40, 28,
        36, 75, 76, 55, 72, 146,
        142, 110, 192, 243, 459, 401
    ]
)
nn.setIouThreshold(0.5)
nn.setConfidenceThreshold(0.5)

# Define the anchor masks for each output layer
nn.setAnchorMasks({
    "side_0": [3, 4, 5],
    "side_1": [0, 1, 2],
    "side_2": [6, 7, 8]
})

xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName('rgb')

xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName('detections')

# Connect the nodes
cam_rgb.preview.link(nn.input)
nn.passthrough.link(xout_rgb.input)
nn.out.link(xout_nn.input)

# Start the pipeline
with dai.Device(pipeline) as device:
    # Get the output streams
    q_rgb = device.getOutputQueue('rgb', 1, False)
    q_nn = device.getOutputQueue('detections', 1, False)

    # Process the frames
    while True:
        # Get the RGB frame from the camera
        in_rgb = q_rgb.get()
        frame = in_rgb.getCvFrame()

        # Get the detections from the neural network
        detections = q_nn.get().detections

        # Draw the detections on the frame
        for detection in detections:
            x1 = int(detection.xmin * frame.shape[1])
            y1 = int(detection.ymin * frame.shape[0])
            x2 = int(detection.xmax * frame.shape[1])
            y2 = int(detection.ymax * frame.shape[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw the label
            label = f"{model_name}: {detection.label}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break

# Clean up
cv2.destroyAllWindows()
