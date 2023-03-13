import cv2
import numpy as np
import tensorflow as tf
import depthai as dai

# Load the pre-trained staircase detection model
model = tf.keras.models.load_model('staircase_detection_model.h5')

# Define a function to preprocess the input image and prepare it for input into the model
def prepare_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (224, 224))
    image = np.array(image, dtype='float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Create a pipeline
pipeline = dai.Pipeline()

# Define a node to capture an image from the camera
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(544, 320)
cam_rgb.setInterleaved(False)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

# Define a node to convert the color image to grayscale
color_to_gray = pipeline.createColorCamera()
color_to_gray.setPreviewSize(544, 320)
color_to_gray.setInterleaved(False)
color_to_gray.setBoardSocket(dai.CameraBoardSocket.RGB)
color_to_gray.setColorOrder(dai.ColorCameraProperties.ColorOrder.GRAY)

# Define a node to run the neural network model
nn = pipeline.createNeuralNetwork()
nn.setBlobPath("/Users/tamdau/Desktop/stair)
cam_rgb.preview.link(nn.input)

# Define a node to display the output
xout = pipeline.createXLinkOut()
xout.setStreamName("nn")
nn.out.link(xout.input)

# Connect the camera nodes
color_to_gray.input.setBlocking(False)
color_to_gray.video.link(nn.input)
cam_rgb.video.link(color_to_gray.input)

# Start the pipeline
with dai.Device(pipeline) as device:
    # Define a queue to receive the neural network results
    q = device.getOutputQueue("nn", maxSize=1, blocking=False)

    # Define a window to display the output image
    cv2.namedWindow("Staircase Detection", cv2.WINDOW_NORMAL)

    while True:
        # Capture an image from the camera
        in_rgb = device.getOutputQueue("cam_rgb").get()
        frame = in_rgb.getCvFrame()

        # Process the image with the neural network model
        img = prepare_image(frame)
        nn_input = dai.NNData()
        nn_input.setLayer("input", img)
        q.send(nn_input)
        nn_output = q.tryGet()

        # Draw bounding boxes around the detected staircases
        if nn_output is not None:
            detections = nn_output.getFirstLayerFp16()
            for detection in detections:
                if detection[2] > 0.5:
                    x1, y1, x2, y2 = map(int, detection[3:7] * np.array(frame.shape[1::-1]))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Staircase Detection', frame)

        # Wait for the user to press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Preprocess the captured image
image_path = 'drone_image.jpg'
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2


# Preprocess the captured image
image_path = 'drone_image.jpg'
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
kernel = np.ones((5, 5), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

# Find contours in the image and classify them using the pre-trained CNN model
contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    roi = gray_image[y:y + h, x:x + w]
    roi = cv2.resize(roi, (224, 224))
    roi = np.array(roi, dtype='float32')
    roi /= 255
    roi = np.expand_dims(roi, axis=0)
    prediction = model.predict(roi)
    if prediction[0][0] > 0.5:
        # If the prediction is above a threshold, draw a bounding box around the detected staircase
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the image
        cv2.imshow('Drone Camera', image)

        # Wait for the user to press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cv2.destroyAllWindows()
