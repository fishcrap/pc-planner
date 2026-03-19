import open3d as o3d

mesh = o3d.io.read_triangle_mesh("robots/cylinder/cylinder.obj")

mesh.textures = []
mesh.translate(-mesh.get_center())

o3d.io.write_triangle_mesh("robots/cylinder/cylinder_centered.obj", mesh)