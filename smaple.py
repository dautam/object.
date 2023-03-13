import cv2
import depthai
import numpy as np

# Load the staircase detection model
model_path = "/Users/tamdau/Desktop/staircase/stair.blob"
pipeline = depthai.Pipeline()
nn_node = pipeline.createNeuralNetwork()
nn_node.setBlobPath(model_path)
nn_node.setNumInferenceThreads(2)
nn_node.input.setBlocking(False)
nn_node.setJsonPacketPath(model_json)

# Start the pipeline
device = depthai.Device(pipeline)
q_nn = device.getOutputQueue(name="nn")

# Start the camera
cap = cv2.VideoCapture(0)
while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to the input size of the model
    input_size = (nn_node.getInputsSize()[0].dimX, nn_node.getInputsSize()[0].dimY)
    frame_resized = cv2.resize(frame, input_size)

    # Convert the frame to the format expected by the model
    frame_nn = np.ascontiguousarray(frame_resized.transpose(2, 0, 1))

    # Send the frame to the model for inference
    nn_node.send(frame_nn)
    nn_inference = q_nn.tryGet()

    # Get the results of the inference
    if nn_inference is not None:
        results = np.array(nn_inference.getFirstLayerFp16())
        # Process the results here
        # For example, you could draw bounding boxes around the detected staircases
        # and display the annotated frame on the screen
        # You could also perform additional post-processing on the results to filter out false positives
        # or compute the distance to the staircase using the depth information from the OAK-D camera

    # Display the annotated frame on the screen
    cv2.imshow("Staircase detection", frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
device.close()
