import pyzed.sl as sl
from time import sleep
import os
from pprint import pprint
import cv2
import numpy as np

zed = sl.Camera()
init_params = sl.InitParameters()
init_params.sdk_verbose = True # Enable verbose logging
init_params.depth_mode = sl.DEPTH_MODE.ULTRA # Set the depth mode to performance (fastest)
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
init_params.coordinate_units = sl.UNIT.CENTIMETER  # Use millimeter units

# Open the camera
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    print("Error {}, exit program".format(err)) # Display the error
    exit()

# save intrinsics
lc = zed.get_camera_information().calibration_parameters.left_cam
intrinsics = np.array([[lc.fx, 0, lc.cx, 0], [0, lc.fy, lc.cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
np.save(os.path.join(os.getcwd(), f"intrinsics.npy"), intrinsics)

# start positional tracking
tracking_parameters = sl.PositionalTrackingParameters()
err = zed.enable_positional_tracking(tracking_parameters)
if err != sl.ERROR_CODE.SUCCESS:
    print("Error {}, exit program".format(err)) # Display the error
    exit()

i = 0
image = sl.Mat()
depth = sl.Mat()
zed_pose = sl.Pose()
runtime_parameters = sl.RuntimeParameters()
for i in range(100):
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
        state = zed.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD)
        cv2.imwrite(os.path.join(os.getcwd(), "rgb_images", f"{i}_rgb.png"), image.get_data())
        with open(os.path.join(os.getcwd(), "depth_maps", f"{i}_depth.npy"), "wb") as depth_file:
            np.save(depth_file, depth.get_data())
        with open(os.path.join(os.getcwd(), "poses", f"{i}_pose.npy"), "wb") as pose_file:
            np.save(pose_file, zed_pose.pose_data(sl.Transform()).m)
        i += 1
        
    sleep(1)
        