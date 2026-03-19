import matplotlib.pylab as plt
import open3d as o3d
import numpy as np

# Read the .obj file
mesh = o3d.io.read_triangle_mesh("datasets/meet_room_z_up/meet_room.obj")

# Get the centroid of the mesh
mesh_center = np.asarray(mesh.get_center())

# Translate the mesh to the origin of the coordinate system
mesh.translate(-mesh_center)

# Get the axis-aligned bounding box of the mesh
bounding_box = mesh.get_axis_aligned_bounding_box()

# Scale the bounding box in the Z direction
scaling_factor = 0.8  # Adjust this factor to change the scaling
z_center = (bounding_box.max_bound[2] + bounding_box.min_bound[2]) / 2
z_len = bounding_box.max_bound[2] - bounding_box.min_bound[2]

scaled_max_bound = bounding_box.max_bound.copy()
print(scaled_max_bound)
# scaled_max_bound[2] = z_center + scaling_factor * z_len / 2
scaled_max_bound[2] = scaling_factor * bounding_box.max_bound[2]
print(scaled_max_bound)

scaling_factor = 0.88
scaled_min_bound = bounding_box.min_bound.copy()
print(scaled_min_bound)

# scaled_min_bound[2] = z_center - scaling_factor * z_len / 2
scaled_min_bound[2] = scaling_factor * bounding_box.min_bound[2]
print(scaled_min_bound)

# Create a new bounding box with the scaled values
scaled_bounding_box = o3d.geometry.AxisAlignedBoundingBox(scaled_min_bound, scaled_max_bound)

# Crop the mesh using the scaled bounding box
cropped_mesh = mesh.crop(scaled_bounding_box)

cropped_mesh.translate(mesh_center)

# Visualize the original mesh and the cropped mesh
# o3d.visualization.draw_geometries([mesh, cropped_mesh])

o3d.io.write_triangle_mesh("meet_room_clip.obj", cropped_mesh)