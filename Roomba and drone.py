import serial
import socket
import cv2
import numpy as np
import time
import tensorflow as tf
import depthai as dai
# Open a serial connection to the Roomba Create 2
ser = serial.Serial('/dev/ttyUSB0', baudrate=115200)

# Start the Roomba Create 2
ser.write(b'\x80')
time.sleep(0.1)
ser.write(b'\x83')

# Wait for the Roomba Create 2 to finish cleaning and turn it off
time.sleep(3600)
ser.write(b'\xAD')
time.sleep(0.1)
ser.write(b'\x80')

# Connect to the drone and send takeoff command
host = ''
port = 9000
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((host, port))

print("Send command")
sock.sendto(b'command', ('192.168.10.1', 8889))
time.sleep(5)  # Wait for the drone to respond
print("Send takeoff")
sock.sendto(b'takeoff', ('192.168.10.1', 8889))
time.sleep(5)  # Wait for the drone to take off

# Fly to the stair
# Start the video stream
cap = cv2.VideoCapture('udp://@0.0.0.0:11111') #need to modify the video stream URI ('udp://@0.0.0.0:11111')
# depending on the configuration of your drone

# Wait for the stream to start
time.sleep(2)

# Loop through the frames in the video stream
while True:
    # Capture a frame
    ret, frame = cap.read()

    # Check if the frame was successfully captured
    if not ret:
        break

    # Display the frame
    cv2.imshow('Drone Camera', frame)

    # Wait for the user to press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    else:
        break

# Save the captured image
cv2.imwrite('drone_image.jpg', frame)

# Release the camera
cap.release()

# Load the pre-trained staircase detection model
model = tf.keras.models.load_model('staircase_detection_model.h5')

# Define a function to preprocess the input image and prepare it for input into the model
def prepare_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (224, 224))
    image = np.array(image, dtype='float32')
    image /= 255
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
nn.setBlobPath("model.blob")
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
                x1 = int(detection[3] * frame.shape[1])
                y1 = int(detection[4] * frame.shape[0])
                x2 = int(detection[5] * frame.shape[1])
                y2 = int(detection[6] * frame.shape[0])
                if detection[2] > 0.5:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

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
    roi = gray_image[y:y+h, x:x+w]
    roi = cv2.resize(roi, (224, 224))
    roi = np.array(roi, dtype='float32')
    roi /= 255
    roi = np.expand_dims(roi, axis=0)
    prediction = model.predict(roi)
    if prediction[0][0] > 0.5:
        # If the prediction is above a threshold, draw a bounding box around the detected staircase
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the frame
        cv2.imshow('Drone Camera', frame)

        # Wait for the user to press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cv2.destroyAllWindows()

# Send landing command
print("landing")
sock.sendto(b'land', ('192.168.10.1', 8889))
time.sleep(5)  # Wait for the drone to start landing

# Send emergency stop command (if necessary)
sock.sendto(b'emergency', ('192.168.10.1', 8889))
time.sleep(5)  # Wait for the drone to stop

# Close the socket
sock.close()