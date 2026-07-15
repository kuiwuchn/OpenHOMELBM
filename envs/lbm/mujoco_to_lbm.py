import numpy as np
import mujoco
from typing import Tuple, Optional, Dict


def extract_mesh_projection_2d(model: mujoco.MjModel, geom_id: int, 
                               n_samples: Optional[int] = None) -> np.ndarray:
    """
    Project a MuJoCo mesh geometry onto the xy plane.
    
    Args:
        model: MuJoCo model.
        geom_id: Geometry ID.
        n_samples: Optional number of resampled points.
    
    Returns:
        numpy array: Projected vertices shaped (N, 2).
    """
    # Get the mesh data ID.
    dataid = model.geom_dataid[geom_id]
    
    if dataid < 0:
        print(f"警告: 几何体 {geom_id} 没有有效的 mesh 数据")
        # Return a fallback circle.
        angles = np.linspace(0, 2*np.pi, n_samples or 20, endpoint=False)
        return np.column_stack([0.1 * np.cos(angles), 0.1 * np.sin(angles)])
    
    # Read vertices from model.mesh_vert.
    mesh_vertadr = model.mesh_vertadr[dataid]  # Start offset.
    mesh_vertnum = model.mesh_vertnum[dataid]  # Vertex count.
    
    # Extract 3D vertices.
    vertices_3d = model.mesh_vert[mesh_vertadr:mesh_vertadr + mesh_vertnum].reshape(-1, 3)
    
    # Project onto the xy plane.
    vertices_2d = vertices_3d[:, :2]  # Keep x and y.
    
    # Resample to a fixed count when requested.
    if n_samples is not None and len(vertices_2d) != n_samples:
        vertices_2d = resample_polygon(vertices_2d, n_samples)
    
    return vertices_2d


def resample_polygon(vertices: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Resample vertices uniformly along a polygon boundary.
    
    Args:
        vertices: Original vertices shaped (N, 2).
        n_samples: Target point count.
    
    Returns:
        numpy array: Resampled vertices shaped (n_samples, 2).
    """
    if len(vertices) < 2:
        return vertices
    
    # Use the convex hull as the outer boundary.
    try:
        from scipy.spatial import ConvexHull
        if len(vertices) >= 3:
            hull = ConvexHull(vertices)
            vertices = vertices[hull.vertices]
    except:
        pass
    
    # Compute cumulative arc length.
    vertices_closed = np.vstack([vertices, vertices[0]])  # Close the polygon.
    segments = np.diff(vertices_closed, axis=0)
    segment_lengths = np.linalg.norm(segments, axis=1)
    cumulative_length = np.concatenate([[0], np.cumsum(segment_lengths)])
    total_length = cumulative_length[-1]
    
    if total_length < 1e-10:
        # Return a circle when all points coincide.
        angles = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
        radius = 0.1
        return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    
    # Sample uniformly over total arc length.
    sample_positions = np.linspace(0, total_length, n_samples, endpoint=False)
    
    # Interpolate new vertices.
    new_vertices = []
    for pos in sample_positions:
        # Find the containing segment.
        segment_idx = np.searchsorted(cumulative_length[1:], pos)
        segment_idx = min(segment_idx, len(vertices) - 1)
        
        # Interpolate within the segment.
        t = (pos - cumulative_length[segment_idx]) / (segment_lengths[segment_idx] + 1e-10)
        t = np.clip(t, 0, 1)
        
        point = vertices[segment_idx] + t * segments[segment_idx]
        new_vertices.append(point)
    
    return np.array(new_vertices)


def get_mesh_info(model: mujoco.MjModel, mesh_name: str) -> Dict:
    """
    Return details for a named mesh.
    
    Args:
        model: MuJoCo model.
        mesh_name: Mesh name.
    
    Returns:
        dict: Mesh metadata and geometry arrays.
    """
    # Find the mesh ID.
    mesh_id = -1
    for i in range(model.nmesh):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, i)
        if name == mesh_name:
            mesh_id = i
            break
    
    if mesh_id < 0:
        raise ValueError(f"未找到 mesh: {mesh_name}")
    
    # Extract mesh metadata.
    mesh_vertadr = model.mesh_vertadr[mesh_id]
    mesh_vertnum = model.mesh_vertnum[mesh_id]
    mesh_faceadr = model.mesh_faceadr[mesh_id]
    mesh_facenum = model.mesh_facenum[mesh_id]
    
    # Extract vertices.
    vertices = model.mesh_vert[mesh_vertadr:mesh_vertadr + mesh_vertnum].reshape(-1, 3)
    
    # Extract triangular faces.
    faces = model.mesh_face[mesh_faceadr:mesh_faceadr + mesh_facenum].reshape(-1, 3)
    
    # Compute the bounding box.
    bbox_min = np.min(vertices, axis=0)
    bbox_max = np.max(vertices, axis=0)
    
    return {
        'mesh_id': mesh_id,
        'vertices': vertices,
        'faces': faces,
        'num_vertices': mesh_vertnum,
        'num_faces': mesh_facenum,
        'bbox_min': bbox_min,
        'bbox_max': bbox_max,
        'center': (bbox_min + bbox_max) / 2,
        'size': bbox_max - bbox_min
    }


def get_geom_vertices_2d(model: mujoco.MjModel, data: mujoco.MjData, 
                         geom_name: str, n_samples: int = 20) -> np.ndarray:
    """
    Project a MuJoCo geometry onto the xy plane.
    
    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        geom_name: Geometry name.
        n_samples: Sample count for curved geometry.
    
    Returns:
        numpy array: Projected vertices shaped (N, 2).
    """
    # Get the geometry ID.
    geom_id = model.geom(geom_name).id
    geom_type = model.geom_type[geom_id]
    geom_size = model.geom_size[geom_id]
    
    # Read the world position and orientation.
    geom_pos = data.geom_xpos[geom_id][:2]  # xy coordinates.
    geom_mat = data.geom_xmat[geom_id].reshape(3, 3)  # Rotation matrix.
    
    # Extract rotation around the z-axis.
    rotation_angle = np.arctan2(geom_mat[1, 0], geom_mat[0, 0])
    
    vertices = []
    
    # Generate vertices by geometry type.
    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        # Build four box corners from its size.
        half_x, half_y = geom_size[0], geom_size[1]
        local_vertices = np.array([
            [-half_x, -half_y],
            [half_x, -half_y],
            [half_x, half_y],
            [-half_x, half_y]
        ])
        
    elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE or geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        # Sample a circle for spheres and cylinders.
        radius = geom_size[0]
        angles = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
        local_vertices = np.column_stack([
            radius * np.cos(angles),
            radius * np.sin(angles)
        ])
        
    elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        # Approximate a capsule as a rounded rectangle.
        radius = geom_size[0]
        half_length = geom_size[1]
        
        # Recover fromto capsule length from geom_rbound.
        if half_length < 1e-6:
            # Capsule rbound equals half-length plus radius.
            rbound = model.geom_rbound[geom_id]
            half_length = rbound - radius
            if half_length < 0:
                half_length = 0.0
        
        # MuJoCo capsules extend along their local z-axis.
        capsule_axis_3d = geom_mat[:, 2]  # Capsule axis.
        capsule_axis_2d = capsule_axis_3d[:2]  # Project onto xy.
        axis_len = np.linalg.norm(capsule_axis_2d)
        
        if axis_len > 1e-6:
            # Normalize the in-plane capsule direction.
            capsule_dir = capsule_axis_2d / axis_len
            # Measure capsule rotation from +y.
            capsule_angle = np.arctan2(capsule_dir[0], capsule_dir[1])  # Relative to +y.
        else:
            # A perpendicular capsule projects to a circle.
            capsule_angle = 0.0
        
        # Build two semicircles around the local y-axis.
        n_half = n_samples // 2
        
        # Upper semicircle at +y.
        angles_top = np.linspace(0, np.pi, n_half, endpoint=False)
        top_vertices = np.column_stack([
            radius * np.cos(angles_top),
            half_length + radius * np.sin(angles_top)
        ])
        
        # Lower semicircle at -y.
        angles_bottom = np.linspace(np.pi, 2*np.pi, n_half, endpoint=False)
        bottom_vertices = np.column_stack([
            radius * np.cos(angles_bottom),
            -half_length + radius * np.sin(angles_bottom)
        ])
        
        local_vertices_capsule = np.vstack([top_vertices, bottom_vertices])
        
        # Rotate capsule-local points into geometry space.
        cos_a, sin_a = np.cos(capsule_angle), np.sin(capsule_angle)
        capsule_rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        local_vertices = local_vertices_capsule @ capsule_rot.T
        
    elif geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        # Sample the ellipsoid's xy ellipse.
        a, b = geom_size[0], geom_size[1]  # xy semi-axes.
        angles = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
        local_vertices = np.column_stack([
            a * np.cos(angles),
            b * np.sin(angles)
        ])
        
    elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
        # Extract and project MuJoCo mesh vertices.
        local_vertices = extract_mesh_projection_2d(model, geom_id, n_samples)
        
    else:
        # Use a small circle for unsupported types.
        print(f"警告: 几何体类型 {geom_type} 未完全支持，使用默认圆形")
        radius = 0.1
        angles = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
        local_vertices = np.column_stack([
            radius * np.cos(angles),
            radius * np.sin(angles)
        ])
    
    # Apply rotation and translation.
    rotation_matrix = np.array([
        [np.cos(rotation_angle), -np.sin(rotation_angle)],
        [np.sin(rotation_angle), np.cos(rotation_angle)]
    ])
    
    global_vertices = local_vertices @ rotation_matrix.T + geom_pos
    
    return global_vertices


def get_body_vertices_2d(model: mujoco.MjModel, data: mujoco.MjData, 
                         body_name: str, n_samples: int = 20) -> np.ndarray:
    """
    Combine all body geometries projected onto the xy plane.
    
    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        body_name: Body name.
        n_samples: Samples per geometry.
    
    Returns:
        numpy array: Combined vertices shaped (N, 2).
    """
    body_id = model.body(body_name).id
    
    # Find all geometries attached to the body.
    geom_vertices_list = []
    
    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] == body_id:
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            vertices = get_geom_vertices_2d(model, data, geom_name, n_samples)
            geom_vertices_list.append(vertices)
    
    if not geom_vertices_list:
        raise ValueError(f"刚体 '{body_name}' 没有关联的几何体")
    
    # Combine all geometry vertices.
    all_vertices = np.vstack(geom_vertices_list)
    
    return all_vertices


def compute_convex_hull_2d(vertices: np.ndarray) -> np.ndarray:
    """
    Compute the convex hull of 2D points.
    
    Args:
        vertices: Vertices shaped (N, 2).
    
    Returns:
        numpy array: Counterclockwise hull vertices.
    """
    from scipy.spatial import ConvexHull
    
    if len(vertices) < 3:
        return vertices
    
    try:
        hull = ConvexHull(vertices)
        hull_vertices = vertices[hull.vertices]
        return hull_vertices
    except:
        # Return the original vertices if hull creation fails.
        print("警告: 凸包计算失败，返回原始顶点")
        return vertices


def extract_lbm_polygon_from_mujoco(model: mujoco.MjModel, 
                                   data: mujoco.MjData,
                                   body_or_geom_name: str,
                                   n_samples: int = 20,
                                   use_convex_hull: bool = True,
                                   normalize: bool = True,
                                   is_body: bool = True) -> Dict:
    """
    Convert MuJoCo object data into LBM solver input.

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        body_or_geom_name: Body or geometry name.
        n_samples: Sample count.
        use_convex_hull: Whether to compute a convex hull.
        normalize: Whether to center the vertices.
        scale: Normalization scale.
        is_body: Whether the name identifies a body.

    Returns:
        dict: Vertices, position, angle, and scale factor.
    """
    # Extract projected vertices.
    if is_body:
        vertices = get_body_vertices_2d(model, data, body_or_geom_name, n_samples)
        body_id = model.body(body_or_geom_name).id
        position = data.body(body_id).xipos[:2]
        
        # Extract rotation around z.
        quat = data.body(body_id).xquat  # w, x, y, z
        angle = np.arctan2(2*(quat[0]*quat[3] + quat[1]*quat[2]), 
                          1 - 2*(quat[2]**2 + quat[3]**2))
    else:
        vertices = get_geom_vertices_2d(model, data, body_or_geom_name, n_samples)
        geom_id = model.geom(body_or_geom_name).id
        position = data.geom_xpos[geom_id][:2]
        
        # Extract the rotation angle.
        geom_mat = data.geom_xmat[geom_id].reshape(3, 3)
        angle = np.arctan2(geom_mat[1, 0], geom_mat[0, 0])
    
    # Compute the convex hull.
    if use_convex_hull and len(vertices) > 3:
        vertices = compute_convex_hull_2d(vertices)
    
    if normalize:
        centroid = np.mean(vertices, axis=0)
        vertices = vertices - centroid

    return {
        'vertices': vertices,
        'position': position,
        'angle': angle,
    }
