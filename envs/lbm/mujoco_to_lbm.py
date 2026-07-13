import numpy as np
import mujoco
from typing import Tuple, Optional, Dict


def extract_mesh_projection_2d(model: mujoco.MjModel, geom_id: int, 
                               n_samples: Optional[int] = None) -> np.ndarray:
    """
    从 MuJoCo mesh 几何体提取 xy 平面投影
    
    参数:
        model: MuJoCo 模型
        geom_id: 几何体 ID
        n_samples: 可选的重采样点数量，如果为 None 则使用所有投影顶点
    
    返回:
        numpy array: 投影顶点坐标 (N, 2)
    """
    # 获取 mesh 数据 ID
    dataid = model.geom_dataid[geom_id]
    
    if dataid < 0:
        print(f"警告: 几何体 {geom_id} 没有有效的 mesh 数据")
        # 返回默认圆形
        angles = np.linspace(0, 2*np.pi, n_samples or 20, endpoint=False)
        return np.column_stack([0.1 * np.cos(angles), 0.1 * np.sin(angles)])
    
    # 获取 mesh 的顶点数据
    # MuJoCo 的 mesh 数据存储在 model.mesh_vert 中
    mesh_vertadr = model.mesh_vertadr[dataid]  # 起始地址
    mesh_vertnum = model.mesh_vertnum[dataid]  # 顶点数量
    
    # 提取顶点（3D 坐标）
    vertices_3d = model.mesh_vert[mesh_vertadr:mesh_vertadr + mesh_vertnum].reshape(-1, 3)
    
    # 投影到 xy 平面
    vertices_2d = vertices_3d[:, :2]  # 只取 x, y 坐标
    
    # 如果需要重采样到固定数量的点
    if n_samples is not None and len(vertices_2d) != n_samples:
        vertices_2d = resample_polygon(vertices_2d, n_samples)
    
    return vertices_2d


def resample_polygon(vertices: np.ndarray, n_samples: int) -> np.ndarray:
    """
    重采样多边形顶点到指定数量
    沿着多边形边界均匀采样
    
    参数:
        vertices: 原始顶点 (N, 2)
        n_samples: 目标采样点数量
    
    返回:
        numpy array: 重采样后的顶点 (n_samples, 2)
    """
    if len(vertices) < 2:
        return vertices
    
    # 计算凸包以获得外轮廓
    try:
        from scipy.spatial import ConvexHull
        if len(vertices) >= 3:
            hull = ConvexHull(vertices)
            vertices = vertices[hull.vertices]
    except:
        pass
    
    # 计算累积弧长
    vertices_closed = np.vstack([vertices, vertices[0]])  # 闭合多边形
    segments = np.diff(vertices_closed, axis=0)
    segment_lengths = np.linalg.norm(segments, axis=1)
    cumulative_length = np.concatenate([[0], np.cumsum(segment_lengths)])
    total_length = cumulative_length[-1]
    
    if total_length < 1e-10:
        # 所有点重合，返回均匀分布的圆
        angles = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
        radius = 0.1
        return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    
    # 在总弧长上均匀采样
    sample_positions = np.linspace(0, total_length, n_samples, endpoint=False)
    
    # 插值得到新的顶点
    new_vertices = []
    for pos in sample_positions:
        # 找到对应的线段
        segment_idx = np.searchsorted(cumulative_length[1:], pos)
        segment_idx = min(segment_idx, len(vertices) - 1)
        
        # 在线段内插值
        t = (pos - cumulative_length[segment_idx]) / (segment_lengths[segment_idx] + 1e-10)
        t = np.clip(t, 0, 1)
        
        point = vertices[segment_idx] + t * segments[segment_idx]
        new_vertices.append(point)
    
    return np.array(new_vertices)


def get_mesh_info(model: mujoco.MjModel, mesh_name: str) -> Dict:
    """
    获取 mesh 的详细信息
    
    参数:
        model: MuJoCo 模型
        mesh_name: mesh 名称
    
    返回:
        字典包含 mesh 的详细信息
    """
    # 查找 mesh ID
    mesh_id = -1
    for i in range(model.nmesh):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, i)
        if name == mesh_name:
            mesh_id = i
            break
    
    if mesh_id < 0:
        raise ValueError(f"未找到 mesh: {mesh_name}")
    
    # 提取 mesh 信息
    mesh_vertadr = model.mesh_vertadr[mesh_id]
    mesh_vertnum = model.mesh_vertnum[mesh_id]
    mesh_faceadr = model.mesh_faceadr[mesh_id]
    mesh_facenum = model.mesh_facenum[mesh_id]
    
    # 提取顶点
    vertices = model.mesh_vert[mesh_vertadr:mesh_vertadr + mesh_vertnum].reshape(-1, 3)
    
    # 提取面（三角形）
    faces = model.mesh_face[mesh_faceadr:mesh_faceadr + mesh_facenum].reshape(-1, 3)
    
    # 计算边界框
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
    获取 MuJoCo 几何体在 xy 平面上的投影顶点
    
    参数:
        model: MuJoCo 模型
        data: MuJoCo 数据
        geom_name: 几何体名称
        n_samples: 采样点数量（对于圆形/圆柱等）
    
    返回:
        numpy array: 顶点坐标数组，形状为 (N, 2)，表示 xy 平面上的点
    """
    # 获取几何体 ID
    geom_id = model.geom(geom_name).id
    geom_type = model.geom_type[geom_id]
    geom_size = model.geom_size[geom_id]
    
    # 获取几何体的全局位置和方向
    geom_pos = data.geom_xpos[geom_id][:2]  # xy 坐标
    geom_mat = data.geom_xmat[geom_id].reshape(3, 3)  # 旋转矩阵
    
    # 提取 xy 平面上的旋转角度（绕 z 轴）
    rotation_angle = np.arctan2(geom_mat[1, 0], geom_mat[0, 0])
    
    vertices = []
    
    # 根据几何体类型生成顶点
    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        # 盒子：使用尺寸信息生成4个顶点
        half_x, half_y = geom_size[0], geom_size[1]
        local_vertices = np.array([
            [-half_x, -half_y],
            [half_x, -half_y],
            [half_x, half_y],
            [-half_x, half_y]
        ])
        
    elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE or geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        # 球体或圆柱：生成圆形采样点
        radius = geom_size[0]
        angles = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
        local_vertices = np.column_stack([
            radius * np.cos(angles),
            radius * np.sin(angles)
        ])
        
    elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        # 胶囊体：近似为圆角矩形
        radius = geom_size[0]
        half_length = geom_size[1]
        
        # 对于使用 fromto 定义的 capsule，geom_size[1] 可能是 0
        # 需要从 geom_rbound 计算实际半长度
        if half_length < 1e-6:
            # geom_rbound 是几何体的包围球半径
            # 对于 capsule: rbound = half_length + radius
            rbound = model.geom_rbound[geom_id]
            half_length = rbound - radius
            if half_length < 0:
                half_length = 0.0
        
        # 获取 capsule 的局部方向（从 geom_xmat 的 z 轴，即第三列）
        # MuJoCo capsule 默认沿局部 z 轴延伸
        # 但我们在 2D 中工作，需要从 fromto 推断方向
        # geom_pos 存储的是 capsule 中心，geom_xmat 的第三列是 capsule 的轴向
        capsule_axis_3d = geom_mat[:, 2]  # z 轴方向（capsule 延伸方向）
        capsule_axis_2d = capsule_axis_3d[:2]  # 投影到 xy 平面
        axis_len = np.linalg.norm(capsule_axis_2d)
        
        if axis_len > 1e-6:
            # capsule 在 xy 平面有方向分量
            capsule_dir = capsule_axis_2d / axis_len
            # 计算 capsule 相对于 y 轴的旋转角度
            capsule_angle = np.arctan2(capsule_dir[0], capsule_dir[1])  # 相对于 +y 轴
        else:
            # capsule 垂直于 xy 平面，退化为圆
            capsule_angle = 0.0
        
        # 生成两个半圆和两条直线（在 capsule 局部坐标系中，沿 y 轴）
        n_half = n_samples // 2
        
        # 上半圆（+y 方向端点）
        angles_top = np.linspace(0, np.pi, n_half, endpoint=False)
        top_vertices = np.column_stack([
            radius * np.cos(angles_top),
            half_length + radius * np.sin(angles_top)
        ])
        
        # 下半圆（-y 方向端点）
        angles_bottom = np.linspace(np.pi, 2*np.pi, n_half, endpoint=False)
        bottom_vertices = np.column_stack([
            radius * np.cos(angles_bottom),
            -half_length + radius * np.sin(angles_bottom)
        ])
        
        local_vertices_capsule = np.vstack([top_vertices, bottom_vertices])
        
        # 将 capsule 局部坐标旋转到几何体局部坐标
        # capsule_angle 是 capsule 轴相对于 y 轴的角度
        cos_a, sin_a = np.cos(capsule_angle), np.sin(capsule_angle)
        capsule_rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        local_vertices = local_vertices_capsule @ capsule_rot.T
        
    elif geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        # 椭球：使用三个尺寸参数生成椭圆采样点
        a, b = geom_size[0], geom_size[1]  # xy 半轴
        angles = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
        local_vertices = np.column_stack([
            a * np.cos(angles),
            b * np.sin(angles)
        ])
        
    elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
        # 网格：从 MuJoCo mesh 数据中提取并投影到 xy 平面
        local_vertices = extract_mesh_projection_2d(model, geom_id, n_samples)
        
    else:
        # 其他类型：默认生成小圆
        print(f"警告: 几何体类型 {geom_type} 未完全支持，使用默认圆形")
        radius = 0.1
        angles = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
        local_vertices = np.column_stack([
            radius * np.cos(angles),
            radius * np.sin(angles)
        ])
    
    # 应用旋转和平移变换
    rotation_matrix = np.array([
        [np.cos(rotation_angle), -np.sin(rotation_angle)],
        [np.sin(rotation_angle), np.cos(rotation_angle)]
    ])
    
    global_vertices = local_vertices @ rotation_matrix.T + geom_pos
    
    return global_vertices


def get_body_vertices_2d(model: mujoco.MjModel, data: mujoco.MjData, 
                         body_name: str, n_samples: int = 20) -> np.ndarray:
    """
    获取 MuJoCo 刚体（body）的所有几何体在 xy 平面上的组合投影
    
    参数:
        model: MuJoCo 模型
        data: MuJoCo 数据
        body_name: 刚体名称
        n_samples: 每个几何体的采样点数量
    
    返回:
        numpy array: 组合顶点坐标数组，形状为 (N, 2)
    """
    body_id = model.body(body_name).id
    
    # 获取该刚体的所有几何体
    geom_vertices_list = []
    
    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] == body_id:
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            vertices = get_geom_vertices_2d(model, data, geom_name, n_samples)
            geom_vertices_list.append(vertices)
    
    if not geom_vertices_list:
        raise ValueError(f"刚体 '{body_name}' 没有关联的几何体")
    
    # 合并所有几何体的顶点
    all_vertices = np.vstack(geom_vertices_list)
    
    return all_vertices


def compute_convex_hull_2d(vertices: np.ndarray) -> np.ndarray:
    """
    计算二维点集的凸包
    
    参数:
        vertices: 顶点数组，形状为 (N, 2)
    
    返回:
        numpy array: 凸包顶点，按逆时针顺序排列
    """
    from scipy.spatial import ConvexHull
    
    if len(vertices) < 3:
        return vertices
    
    try:
        hull = ConvexHull(vertices)
        hull_vertices = vertices[hull.vertices]
        return hull_vertices
    except:
        # 如果凸包计算失败，返回原始顶点
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
    从 MuJoCo 提取物体信息并转换为 LBM 求解器输入格式
    
    参数:
        model: MuJoCo 模型
        data: MuJoCo 数据
        body_or_geom_name: 刚体或几何体名称
        n_samples: 采样点数量
        use_convex_hull: 是否计算凸包
        normalize: 是否规范化到原点
        scale: 规范化缩放因子
        is_body: True 表示输入的是 body 名称，False 表示是 geom 名称
    
    返回:
        字典包含:
            - 'vertices': 顶点坐标 (N, 2)
            - 'position': 物体位置 (x, y)
            - 'angle': 物体角度（弧度）
            - 'scale_factor': 建议的 LBM 缩放因子
    """
    # 提取顶点
    if is_body:
        vertices = get_body_vertices_2d(model, data, body_or_geom_name, n_samples)
        body_id = model.body(body_or_geom_name).id
        position = data.body(body_id).xipos[:2]
        
        # 提取旋转角度（绕 z 轴）
        quat = data.body(body_id).xquat  # w, x, y, z
        angle = np.arctan2(2*(quat[0]*quat[3] + quat[1]*quat[2]), 
                          1 - 2*(quat[2]**2 + quat[3]**2))
    else:
        vertices = get_geom_vertices_2d(model, data, body_or_geom_name, n_samples)
        geom_id = model.geom(body_or_geom_name).id
        position = data.geom_xpos[geom_id][:2]
        
        # 提取旋转角度
        geom_mat = data.geom_xmat[geom_id].reshape(3, 3)
        angle = np.arctan2(geom_mat[1, 0], geom_mat[0, 0])
    
    # 计算凸包
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