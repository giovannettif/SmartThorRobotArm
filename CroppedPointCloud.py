# Convert the cropped depth map of the object into a point cloud.
# The coordinates will still be from the camera reference in the entire scene.
# Extract these coordinates and send to Inverse Kinematics.

# TODO: Using the image segmentation mask, crop out the segmented area from the generated depth_heatmap.png and convert to a 3D Point Cloud for Grasp Detection

# Currently have not implemented this as I am first working on implementing RL Inverse Kinematics