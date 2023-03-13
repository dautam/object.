import cv2
import depthai as dai
import numpy as np

# Load the trained model checkpoint
model_path = 'home/tamdau/Desktop/training/train1'
model = cv2.dnn.readNet(model_path)

# Define the inference pipeline
pipeline = dai.Pipeline()
cam_rgb = pipeline.createColorCamera()
mono_left = pipeline.createMonoCamera()
stereo = pipeline.createStereoDepth()
nn = pipeline.createNeuralNetwork()

cam_rgb.setPreviewSize(300, 300)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
stereo.setOutputDepth(True)
stereo.setConfidenceThreshold(200)
nn.setBlobPath(model_path)
nn.setNumInferenceThreads(2)

cam_rgb.preview.link(nn.input)
mono_left.out.link(stereo.left)
stereo.depth.link(nn.inputDepth)

# Start the pipeline
with dai.Device(pipeline) as device:
    # Define the input and output queues
    in_q = device.getInputQueue('nn')
    out_q = device.getOutputQueue('nn', maxSize=1, blocking=True)

    # Define the ground truth annotations or initialize the human observer
    # ...

    # Capture test images or video frames from the OAK-D camera
    while True:
        # Get the RGB data
        in_rgb = out_q.get().getFirstLayerFp16()
        in_rgb = cv2.cvtColor(in_rgb, cv2.COLOR_BGR2RGB)
        in_rgb = cv2.resize(in_rgb, (300, 300))

        # Preprocess the input data to match the input format of the model checkpoint
        blob = cv2.dnn.blobFromImage(in_rgb, size=(300, 300), swapRB=True, crop=False)

        # Perform inference on the input data using the defined pipeline
        model.setInput(blob)
        detections = model.forward()

        # Process the output data to extract meaningful information, such as the presence and location of obstacles
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                class_id = int(detections[0, 0, i, 1])
                x_min = int(detections[0, 0, i, 3] * 300)
                y_min = int(detections[0, 0, i, 4] * 300)
                x_max = int(detections[0, 0, i, 5] * 300)
                y_max = int(detections[0, 0, i, 6] * 300)

                # Evaluate the model's performance metrics
                # ...

        # Display the results
        cv2.imshow('RGB', in_rgb)
        if cv2.waitKey(1) == ord('q'):
            break

