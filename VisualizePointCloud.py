import open3d as o3d

def load_and_visualize_point_cloud(file_path):
    # Load the point cloud from the file
    pcd = o3d.io.read_point_cloud(file_path)
    
    # Check if the point cloud is loaded successfully
    if pcd.is_empty():
        print("Failed to load point cloud.")
        return
    
    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd], window_name="Point Cloud Viewer")

if __name__ == "__main__":
    point_cloud_file = "Bird-View.jpg-.ply"
    
    # Load and visualize the point cloud
    load_and_visualize_point_cloud(point_cloud_file)