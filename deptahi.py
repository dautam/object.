import cv2
import depthai as dai

# Define the DepthAI pipeline
pipeline = dai.Pipeline()

# Initialize the camera

cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setIspScale(2,3) # 1080P -> 720P
stillEncoder = pipeline.create(dai.node.VideoEncoder)

# Configure the camera properties
cam_rgb.setFps(30)
cam_rgb.setInterleaved(False)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

# Define the Neural Network for depth detection
nn_path = 'depth.blob'
nn_shape = dai.Dimensions((1, 3, 300, 300))

model_nn = dai.EdgeDetectionNetwork()
model_nn.setBlobPath(nn_path)
model_nn.setConfidenceThreshold(0.5)
model_nn.setNumThreads(2)
model_nn.input.setBlocking(False)
model_nn.setNumPoolFrames(1)
model_nn.input.setQueueSize(1)
model_nn.input.setShape(nn_shape)

# Link the camera output to the Neural Network input
cam_rgb.preview.link(model_nn.input)
xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
model_nn.passthrough.link(xout_nn.input)

# Connect to the OAK-D device and start the pipeline
with dai.Device(pipeline) as device:

    # Output queues will be used to get the depth frames from the device
    q_depth = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    # Continuously get the depth frames from the queue and display them
    while True:
        nn_in = device.getInputQueue('nn_in')
        nn_data = dai.ImgFrame()
        nn_data.setTimestamp(dai.Timestamp(mono_time()))
        nn_data.setWidth(300)
        nn_data.setHeight(300)
        nn_data.setData(np.array([[255, 0, 0] * 300 * 300], dtype=np.uint8).reshape(300, 300, 3))
        nn_in.send(nn_data)
        in_nn = q_depth.tryGet()
        if in_nn is not None:
            depth_frame = in_nn.getDepthFrame()
            depth_data = depth_frame.getData()

            # Get the depth values for the pixels in the center of the image
            center_x = int(depth_frame.getWidth() / 2)
            center_y = int(depth_frame.getHeight() / 2)
            center_index = center_y * depth_frame.getWidth() + center_x
            center_depth = depth_data[center_index]

            # Calculate the distance to the ground and stair steps
            # Assume the distance to the ground is 0 and the height of a stair step is 0.2 meters
            ground_distance = center_depth
            stair_step_distance = ground_distance + 0.2

            # Display the depth values and distances
            cv2.imshow("depth", depth_data)
            print("Ground distance:", ground_distance, "meters")
            print("Stair step distance:", stair_step_distance, "meters")

        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
