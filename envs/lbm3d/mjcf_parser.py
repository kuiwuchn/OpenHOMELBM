"""
MJCF Parser for LBM 3D

解析 MuJoCo MJCF XML 文件，提取 body 几何体信息并生成 trimesh 对象。
支持的几何体类型：box, sphere, ellipsoid, cylinder, capsule, mesh
"""
import xml.etree.ElementTree as ET
import trimesh
import numpy as np
from typing import List, Dict, Optional, Any
from scipy.spatial.transform import Rotation as R
import os


def parse_mjcf(mjcf_path: str, mesh_search_paths: Optional[List[str]] = None) -> Dict:
    """
    解析 MJCF XML 文件，返回 bodies 和 joints 信息
    
    Args:
        mjcf_path: MJCF XML 文件路径
        mesh_search_paths: mesh 文件搜索路径列表
        
    Returns:
        {
            'model_name': str,
            'bodies': [
                {
                    'name': str,
                    'meshes': List[trimesh.Trimesh],  # 该 body 的所有几何体
                    'combined_mesh': trimesh.Trimesh,  # 合并后的几何体
                    'parent': str,  # 父 body 名称
                    'pos': np.array,  # 相对于父 body 的位置
                    'quat': np.array,  # 相对于父 body 的四元数 (w,x,y,z)
                },
                ...
            ],
            'joints': [
                {
                    'name': str,
                    'type': str,
                    'body': str,  # 所属 body
                    'axis': np.array,
                    'range': (float, float),
                },
                ...
            ],
        }
    """
    if not os.path.exists(mjcf_path):
        raise FileNotFoundError(f"MJCF file not found: {mjcf_path}")
    
    # 设置搜索路径
    search_paths = mesh_search_paths or []
    mjcf_dir = os.path.dirname(os.path.abspath(mjcf_path))
    if mjcf_dir not in search_paths:
        search_paths.insert(0, mjcf_dir)
    
    # 解析 XML
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    
    # 解析 compiler 元素中的 meshdir 属性
    compiler_elem = root.find('compiler')
    if compiler_elem is not None:
        meshdir = compiler_elem.get('meshdir')
        if meshdir is not None:
            meshdir_abs = os.path.normpath(os.path.join(mjcf_dir, meshdir))
            if meshdir_abs not in search_paths:
                search_paths.insert(0, meshdir_abs)
    model_name = root.get('model', 'unnamed_model')
    
    # 解析 asset 中的 mesh 定义
    mesh_assets = {}
    asset_elem = root.find('asset')
    if asset_elem is not None:
        for mesh_elem in asset_elem.findall('mesh'):
            mesh_name = mesh_elem.get('name')
            mesh_file = mesh_elem.get('file')
            mesh_scale = mesh_elem.get('scale', '1 1 1')
            scale = np.array([float(x) for x in mesh_scale.split()])
            mesh_assets[mesh_name] = {'file': mesh_file, 'scale': scale}
    
    # 递归解析 worldbody
    bodies = []
    joints = []
    
    worldbody = root.find('worldbody')
    if worldbody is not None:
        _parse_body_recursive(worldbody, None, bodies, joints, mesh_assets, search_paths)
    
    return {
        'model_name': model_name,
        'bodies': bodies,
        'joints': joints,
    }


def _parse_body_recursive(parent_elem, parent_name: Optional[str], 
                          bodies: List, joints: List,
                          mesh_assets: Dict, search_paths: List[str]):
    """递归解析 body 元素"""
    for body_elem in parent_elem.findall('body'):
        body_name = body_elem.get('name', f'body_{len(bodies)}')
        
        # 解析位置和姿态
        pos_str = body_elem.get('pos', '0 0 0')
        pos = np.array([float(x) for x in pos_str.split()])
        
        quat_str = body_elem.get('quat', '1 0 0 0')  # MuJoCo: (w, x, y, z)
        quat = np.array([float(x) for x in quat_str.split()])
        
        # 也支持 euler 角度
        euler_str = body_elem.get('euler')
        if euler_str is not None:
            euler = np.array([float(x) for x in euler_str.split()])
            # 转换为四元数
            r = R.from_euler('xyz', euler, degrees=True)
            quat_xyzw = r.as_quat()
            quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        
        # 解析该 body 的所有 geom
        meshes = []
        for geom_elem in body_elem.findall('geom'):
            mesh = _parse_geom(geom_elem, mesh_assets, search_paths)
            if mesh is not None:
                meshes.append(mesh)
        
        # 合并所有 mesh
        combined_mesh = None
        if len(meshes) == 1:
            combined_mesh = meshes[0]
        elif len(meshes) > 1:
            combined_mesh = trimesh.util.concatenate(meshes)
        
        if combined_mesh is not None:
            bodies.append({
                'name': body_name,
                'meshes': meshes,
                'combined_mesh': combined_mesh,
                'parent': parent_name,
                'pos': pos,
                'quat': quat,
            })
        
        # 解析 joints
        for joint_elem in body_elem.findall('joint'):
            joint_info = _parse_joint(joint_elem, body_name)
            if joint_info is not None:
                joints.append(joint_info)
        
        # 递归解析子 body
        _parse_body_recursive(body_elem, body_name, bodies, joints, mesh_assets, search_paths)


def _parse_geom(geom_elem, mesh_assets: Dict, search_paths: List[str]) -> Optional[trimesh.Trimesh]:
    """解析单个 geom 元素，返回 trimesh 对象"""
    geom_type = geom_elem.get('type', 'sphere')
    
    # 解析 geom 的局部位置和姿态
    pos_str = geom_elem.get('pos', '0 0 0')
    pos = np.array([float(x) for x in pos_str.split()])
    
    quat_str = geom_elem.get('quat', '1 0 0 0')
    quat = np.array([float(x) for x in quat_str.split()])
    
    euler_str = geom_elem.get('euler')
    if euler_str is not None:
        euler = np.array([float(x) for x in euler_str.split()])
        r = R.from_euler('xyz', euler, degrees=True)
        quat_xyzw = r.as_quat()
        quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    
    # 创建几何体
    mesh = None
    
    if geom_type == 'box':
        size_str = geom_elem.get('size', '0.5 0.5 0.5')
        half_size = np.array([float(x) for x in size_str.split()])
        mesh = trimesh.creation.box(extents=half_size * 2)
        
    elif geom_type == 'sphere':
        size_str = geom_elem.get('size', '0.5')
        radius = float(size_str.split()[0])
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=radius)
        
    elif geom_type == 'ellipsoid':
        size_str = geom_elem.get('size', '0.5 0.5 0.5')
        radii = np.array([float(x) for x in size_str.split()])
        # 创建单位球，然后缩放
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        mesh.vertices *= radii
        
    elif geom_type == 'cylinder':
        size_str = geom_elem.get('size', '0.5 0.5')
        sizes = [float(x) for x in size_str.split()]
        radius = sizes[0]
        
        # 检查 fromto 属性
        fromto_str = geom_elem.get('fromto')
        if fromto_str is not None:
            fromto = np.array([float(x) for x in fromto_str.split()])
            p1 = fromto[:3]
            p2 = fromto[3:]
            length = np.linalg.norm(p2 - p1)
            
            # 创建圆柱体
            mesh = trimesh.creation.cylinder(radius=radius, height=length, sections=32)
            
            # 计算旋转：从 Z 轴（trimesh默认）到 (p2-p1) 方向
            direction = (p2 - p1) / length if length > 1e-6 else np.array([0, 0, 1])
            z_axis = np.array([0, 0, 1])
            
            if np.allclose(direction, z_axis):
                rot_matrix = np.eye(3)
            elif np.allclose(direction, -z_axis):
                rot_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            else:
                axis = np.cross(z_axis, direction)
                axis = axis / np.linalg.norm(axis)
                angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))
                rot = R.from_rotvec(axis * angle)
                rot_matrix = rot.as_matrix()
            
            mesh.vertices = mesh.vertices @ rot_matrix.T
            
            # 移动到中点
            center = (p1 + p2) / 2
            mesh.vertices += center
            
            return mesh
        else:
            half_length = sizes[1] if len(sizes) > 1 else sizes[0]
            mesh = trimesh.creation.cylinder(radius=radius, height=half_length * 2, sections=32)
        
    elif geom_type == 'capsule':
        size_str = geom_elem.get('size', '0.5')
        sizes = [float(x) for x in size_str.split()]
        radius = sizes[0]
        
        # 检查 fromto 属性
        fromto_str = geom_elem.get('fromto')
        if fromto_str is not None:
            fromto = np.array([float(x) for x in fromto_str.split()])
            p1 = fromto[:3]
            p2 = fromto[3:]
            length = np.linalg.norm(p2 - p1)
            
            # 创建胶囊体
            mesh = trimesh.creation.capsule(height=length, radius=radius, count=[16, 8])
            
            # 计算旋转：从 Z 轴（trimesh默认）到 (p2-p1) 方向
            direction = (p2 - p1) / length if length > 1e-6 else np.array([0, 0, 1])
            z_axis = np.array([0, 0, 1])
            
            if np.allclose(direction, z_axis):
                rot_matrix = np.eye(3)
            elif np.allclose(direction, -z_axis):
                rot_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            else:
                axis = np.cross(z_axis, direction)
                axis = axis / np.linalg.norm(axis)
                angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))
                rot = R.from_rotvec(axis * angle)
                rot_matrix = rot.as_matrix()
            
            mesh.vertices = mesh.vertices @ rot_matrix.T
            
            # 移动到中点
            center = (p1 + p2) / 2
            mesh.vertices += center
            
            return mesh
        else:
            half_length = sizes[1] if len(sizes) > 1 else sizes[0]
            mesh = trimesh.creation.capsule(height=half_length * 2, radius=radius, count=[16, 8])
            
    elif geom_type == 'mesh':
        mesh_name = geom_elem.get('mesh')
        if mesh_name and mesh_name in mesh_assets:
            asset = mesh_assets[mesh_name]
            mesh_path = _resolve_mesh_path(asset['file'], search_paths)
            if mesh_path:
                mesh = trimesh.load(mesh_path)
                if not np.allclose(asset['scale'], 1.0):
                    mesh.vertices *= asset['scale']
    
    if mesh is None:
        return None
    
    # 应用局部变换
    # 旋转
    quat_scipy = [quat[1], quat[2], quat[3], quat[0]]  # scipy: (x, y, z, w)
    rotation = R.from_quat(quat_scipy)
    mesh.vertices = rotation.apply(mesh.vertices)
    
    # 平移
    mesh.vertices += pos
    
    return mesh


def _parse_joint(joint_elem, body_name: str) -> Optional[Dict]:
    """解析单个 joint 元素"""
    joint_name = joint_elem.get('name', f'joint_{body_name}')
    joint_type = joint_elem.get('type', 'hinge')
    
    axis_str = joint_elem.get('axis', '0 0 1')
    axis = np.array([float(x) for x in axis_str.split()])
    
    range_str = joint_elem.get('range', '-180 180')
    range_vals = [float(x) for x in range_str.split()]
    joint_range = (range_vals[0], range_vals[1]) if len(range_vals) >= 2 else (-180, 180)
    
    return {
        'name': joint_name,
        'type': joint_type,
        'body': body_name,
        'axis': axis,
        'range': joint_range,
    }


def _resolve_mesh_path(mesh_uri: str, search_paths: List[str]) -> Optional[str]:
    """解析 mesh 文件路径"""
    if mesh_uri is None:
        return None
        
    # 绝对路径
    if os.path.isabs(mesh_uri):
        return mesh_uri if os.path.exists(mesh_uri) else None
    
    # 相对路径
    for sp in search_paths:
        full_path = os.path.join(sp, mesh_uri)
        if os.path.exists(full_path):
            return full_path
    
    return None


def get_body_world_positions(mjcf_data: Dict) -> Dict[str, np.ndarray]:
    """
    计算每个 body 的世界坐标位置（相对于 worldbody）
    
    Returns:
        {body_name: world_position}
    """
    bodies = mjcf_data['bodies']
    
    # 建立 name -> body 映射
    body_map = {b['name']: b for b in bodies}
    
    # 计算世界位置
    world_positions = {}
    
    for body in bodies:
        pos = body['pos'].copy()
        parent_name = body['parent']
        
        # 向上遍历到 worldbody
        while parent_name is not None and parent_name in body_map:
            parent = body_map[parent_name]
            # 应用父 body 的旋转
            parent_quat = parent['quat']
            quat_scipy = [parent_quat[1], parent_quat[2], parent_quat[3], parent_quat[0]]
            rotation = R.from_quat(quat_scipy)
            pos = rotation.apply(pos) + parent['pos']
            parent_name = parent['parent']
        
        world_positions[body['name']] = pos
    
    return world_positions


def parse_mjcf_to_meshes(mjcf_path: str, mesh_search_paths: Optional[List[str]] = None) -> Dict[str, trimesh.Trimesh]:
    """
    解析 MJCF XML 文件，返回 body 名称到 mesh 的映射
    
    Args:
        mjcf_path: MJCF XML 文件路径
        mesh_search_paths: mesh 文件搜索路径列表
        
    Returns:
        {body_name: trimesh.Trimesh}
    """
    mjcf_data = parse_mjcf(mjcf_path, mesh_search_paths)
    
    meshes = {}
    for body in mjcf_data['bodies']:
        if body['combined_mesh'] is not None:
            meshes[body['name']] = body['combined_mesh']
    
    return meshes


def parse_mjcf_as_urdf_format(mjcf_path: str) -> Dict:
    """
    解析 MJCF 并返回与 parse_urdf 兼容的格式
    
    Returns:
        {
            'robot_name': str,
            'links': [
                {
                    'name': str,
                    'mesh': trimesh.Trimesh,
                    'origin_xyz': np.array,
                    'origin_rpy': np.array,
                    'mass': float,
                    'inertia': np.array,
                },
                ...
            ],
            'joints': [
                {
                    'name': str,
                    'type': str,
                    'parent': str,
                    'child': str,
                    'origin_xyz': np.array,
                    'origin_rpy': np.array,
                    'axis': np.array,
                },
                ...
            ],
        }
    """
    mjcf_data = parse_mjcf(mjcf_path)
    
    # 转换 bodies -> links
    links = []
    for body in mjcf_data['bodies']:
        links.append({
            'name': body['name'],
            'mesh': body['combined_mesh'],
            'origin_xyz': np.zeros(3),  # MJCF 中 geom 已经包含了偏移
            'origin_rpy': np.zeros(3),
            'mass': 1.0,  # 默认值
            'inertia': np.eye(3) * 0.01,
        })
    
    # 转换 joints：从 MJCF 的 body 层级关系构建
    joints = []
    for body in mjcf_data['bodies']:
        if body['parent'] is not None:
            # 将四元数转换为 RPY
            quat = body['quat']
            quat_scipy = [quat[1], quat[2], quat[3], quat[0]]
            rpy = R.from_quat(quat_scipy).as_euler('xyz')
            
            joints.append({
                'name': f"joint_{body['parent']}_{body['name']}",
                'type': 'revolute',
                'parent': body['parent'],
                'child': body['name'],
                'origin_xyz': body['pos'],
                'origin_rpy': rpy,
                'axis': np.array([0, 0, 1]),
            })
    
    return {
        'robot_name': mjcf_data['model_name'],
        'links': links,
        'joints': joints,
    }
