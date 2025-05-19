import numpy as np

def estimate_volume(mesh):
    """
    Estimate the volume of a closed mesh using the discrete triangular volume calculation method based on triangular grids
    """
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    volume = 0.0
    for tri in triangles:
        p0 = vertices[tri[0]]
        p1 = vertices[tri[1]]
        p2 = vertices[tri[2]]
        volume += np.dot(p0, np.cross(p1, p2))
    return abs(volume) / 6.0