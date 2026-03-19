import matplotlib.pylab as plt
import open3d as o3d
import torch
import numpy as np
import os

def project_mesh_to_plane(mesh_path, plane_normal, plane_point, mesh_scale=None):
    """
    return:
        projected points: [n, 2, 2[x,y]]
    """
    # Read OBJ file
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    
    if mesh_scale is not None:
        v = np.asarray(mesh.vertices)
        v = v * mesh_scale

        mesh.vertices = o3d.utility.Vector3dVector(v)
    
    plane_normal = torch.tensor(plane_normal, dtype=torch.float32, device='cuda')
    plane_point = torch.tensor(plane_point, dtype=torch.float32, device='cuda')

    # Calculate D in the plane equation
    plane_D = -(plane_normal.dot(plane_point))

    vertices = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32, device="cuda")
    faces = torch.tensor(np.asarray(mesh.triangles), dtype=torch.int64, device="cuda")

    triangles = vertices[faces] # (n_triangles, 3[point], 3[xyz])
    print("triangle shape", triangles.shape)

    # Calculate intersection point between edge and plane
    def edge_plane_intersection(edges, plane_normal, plane_D):
        v1, v2 = edges[0], edges[1]
        edge_vector = v2 - v1
        
        dot_product = torch.einsum("ijk, k -> ij", edge_vector, plane_normal) # (n, 3)

        parallel_mask = torch.abs(dot_product) < 1e-9 # (n, 3)
        dot_product = torch.where(parallel_mask, torch.tensor(1.0).float().cuda(), dot_product)
        t = -(torch.einsum("ijk, k -> ij", v1, plane_normal) + plane_D) / dot_product
        intersection_point = v1 + t.unsqueeze(2) * edge_vector

        intersect_mask = (t >= 0) & (t <= 1) & ~parallel_mask # (n, 3)
        valid_mask = torch.any(intersect_mask, dim=-1)
        intersect_mask = intersect_mask[valid_mask]
        projected_point = intersection_point[valid_mask, :, :2]
        return projected_point, intersect_mask

    # Extract edges from vertices
    edges = torch.stack((triangles, torch.roll(triangles, -1, dims=1)), dim=0)

    # Calculate intersection points for all edges
    projected_points, intersect_mask = edge_plane_intersection(edges, plane_normal, plane_D)
    projected_points = projected_points.cpu().numpy()
    intersect_mask = intersect_mask.cpu().numpy()

    # Connect the points to form line segments
    final_projected_points = []
    for i in range(projected_points.shape[0]):
        points = projected_points[i]
        mask = intersect_mask[i]
        if mask.sum() == 2:
            points = points[mask]
        elif mask.sum() == 3:
            # compare [0,1], [1,2], [2,0], if pos i points is close, then choose other pos points
            # for example, if points[1] and points[2] is close or equal which means points[1]
            # is close or equal to some points, then we can just choose points[2] and points[3]
            is_close = np.isclose(points, np.roll(points, -1, axis=0), atol=1e-5).all(axis=-1)
            points = points[~is_close]
            if points.shape[0] == 3:
                points = points[:2]
        elif mask.sum() == 1:
            continue
        else:
            raise Exception(f"The number of intersection points [{mask.sum()}] is not handled!")
        final_projected_points.append(points)
        # plt.plot(points[:, 0], points[:, 1])
    
    final_projected_points = np.stack(final_projected_points, axis=0)
    
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('Projected 2D Shape')
    # plt.grid(True)

    # plt.savefig("3.png")
    
    return final_projected_points
    
def project_mesh_to_plane_np(mesh_path, plane_normal, plane_point):
    """
    return:
        projected points: [n, 2, 2[x,y]]
    """
    # Read OBJ file
    mesh = o3d.io.read_triangle_mesh(mesh_path)

    # Calculate D in the plane equation
    plane_D = -np.dot(plane_normal, plane_point)

    # Store projected points
    projected_points = []

    # Helper function to calculate intersection point between edge and plane
    def edge_plane_intersection(edge, plane_normal, plane_D):
        vertex1, vertex2 = edge
        edge_vector = vertex2 - vertex1
        dot_product = np.dot(plane_normal, edge_vector)

        if np.abs(dot_product) < 1e-9:
            return None  # Edge is parallel to the plane

        t = - (np.dot(plane_normal, vertex1) + plane_D) / dot_product
        intersection_point = vertex1 + t * edge_vector

        if 0 <= t <= 1:
            return intersection_point[:2]  # Take the first two coordinates, i.e., x and y
        else:
            return None  # Intersection point is not within the range of the edge

    # Iterate over each triangle
    for face in np.asarray(mesh.triangles):
        vertices = np.asarray(mesh.vertices)[face]

        intersection_points = []
        # Iterate over each edge of the triangle
        for j in range(3):
            edge = (vertices[j], vertices[(j + 1) % 3])
            
            # Calculate the intersection point between the edge and the plane
            intersection_point = edge_plane_intersection(edge, plane_normal, plane_D)

            if intersection_point is not None:
                intersection_points.append(intersection_point)

        if len(intersection_points) > 0:
            if len(intersection_points) != 2:
                print(len(intersection_points))
                raise Exception("The number of intersection points is not equal to 2")
            projected_points.append(np.stack(intersection_points, axis=0))
    # Connect the points to form line segments
    # for i in range(0, len(projected_points)):
    #     plt.plot(projected_points[i][:, 0], projected_points[i][:, 1])

    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('Connected Projected 2D Shape')
    # plt.grid(True)

    # plt.savefig("2.png")
    final_projected_points = np.stack(projected_points, axis=0)
    
    return final_projected_points

def scale_mesh(path, out_dir):

    # Load a mesh from a file
    mesh = o3d.io.read_triangle_mesh(path)

    # Get the center of the mesh
    center = mesh.get_center()

    # Scaling factor
    scale_factor = 2.0

    # Translate the object to the origin, scale, and then translate back to the original position
    mesh.translate(-center)
    mesh.scale(scale_factor, center=(0, 0, 0))
    mesh.translate(center)

    mesh_name = os.path.basename(path)
    name, ext = os.path.splitext(mesh_name)

    # Save the scaled mesh
    o3d.io.write_triangle_mesh(os.path.join(out_dir, f"{name}_scaled{ext}"), mesh)


if __name__ == "__main__":
    
    # Define plane normal and a point it passes through
    plane_normal = torch.tensor([0, 0, 1.0], device="cuda")  # [A, B, C] For example, [0, 0, 1]
    plane_point = torch.tensor([0, 0, 0.0], device="cuda")  # [x, y, z] For example, [0, 0, 0]

    project_mesh_to_plane("mesh_z_up_cropped.obj", plane_normal, plane_point)
    
    # Define plane normal and a point it passes through
    plane_normal = np.array([0, 0, 1])  # [A, B, C] For example, [0, 0, 1]
    plane_point = np.array([0, 0, 0])  # [x, y, z] For example, [0, 0, 0]

    project_mesh_to_plane_np("mesh_z_up_cropped.obj", plane_normal, plane_point)