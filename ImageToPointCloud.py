import numpy as np
from transformers import pipeline
from PIL import Image
import open3d as o3d
import matplotlib.pyplot as plt

input_image = "Bird-View.jpg"

def main():
    # Load the depth estimation model
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
    
    # Open input image
    image = Image.open(input_image).convert('RGB')
    

    depth = pipe(image)["depth"]
    depth_image = np.array(depth)
    
    f_x = f_y = 500 # Focal length
    c_x = depth_image.shape[1] // 2  # Image center X
    c_y = depth_image.shape[0] // 2  # Image center Y

    image_array = np.array(image)
    height, width = depth_image.shape
    points = []
    colors = []
    
    # For flipping the images
    for v in range(height):
        for u in range(width):
            Z = depth_image[v, u]
            
            if Z == 0:
                continue
            
            # Convert (u, v, Z) to 3D (X, Y, Z)
            X = (u - c_x) * Z / f_x
            Y = (v - c_y) * Z / f_y
            
            # Flip the Y and Z axis to match Open3D's coordinate system
            X = -X   # Flip X-axis (since in depth map, X increases to the right)
            Y = Y  # Flip Y-axis (since in depth map, Y increases downward)
            Z = -Z  # Flip Z-axis (depth map has positive Z going toward the camera)
            
            # Get the RGB value for this pixel
            R, G, B = image_array[v, u]
            
            # Append point and corresponding color
            points.append([X, Y, Z])
            colors.append([R / 255.0, G / 255.0, B / 255.0])  # Normalize to [0, 1] for Open3D
    
    # Convert lists to numpy arrays
    points = np.array(points)
    colors = np.array(colors)
    
    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Flip the X-axis of the point cloud (invert X)
    pcd.points = o3d.utility.Vector3dVector(-points)  # Flip X by negating the X values
    
    # Save
    o3d.io.write_point_cloud(f"{input_image}-.ply", pcd)
    
    # Visualize the point cloud and dpeth maps
    o3d.visualization.draw_geometries([pcd])
    plt.imshow(depth_image, cmap='plasma')
    plt.colorbar()
    plt.title("Depth Heatmap")
    plt.savefig(f"{input_image}-depth_heatmap.png")
    plt.show()

if __name__ == "__main__":
    main()
