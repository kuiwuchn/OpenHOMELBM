"""
MJCF Parser for LBM 3D

Parse MuJoCo MJCF XML and build trimesh geometry.
Supported types: box, sphere, ellipsoid, cylinder, capsule, and mesh.
"""
import xml.etree.ElementTree as ET
import trimesh
import numpy as np
from typing import List, Dict, Optional, Any
from scipy.spatial.transform import Rotation as R
import os


def parse_mjcf(mjcf_path: str, mesh_search_paths: Optional[List[str]] = None) -> Dict:
    """
    Parse MJCF XML and return body and joint data.
    
    Args:
        mjcf_path: Path to the MJCF XML file.
        mesh_search_paths: Mesh search paths.
        
    Returns:
        {
            'model_name': str,
            'bodies': [
                {
                    'name': str,
                    'meshes': List[trimesh.Trimesh],  # Body geometries.
                    'combined_mesh': trimesh.Trimesh,  # Combined geometry.
                    'parent': str,  # Parent body name.
                    'pos': np.array,  # Parent-relative position.
                    'quat': np.array,  # Parent-relative quaternion.
                },
                ...
            ],
            'joints': [
                {
                    'name': str,
                    'type': str,
                    'body': str,  # Owning body.
                    'axis': np.array,
                    'range': (float, float),
                },
                ...
            ],
        }
    """
    if not os.path.exists(mjcf_path):
        raise FileNotFoundError(f"MJCF file not found: {mjcf_path}")
    
    # Configure mesh search paths.
    search_paths = mesh_search_paths or []
    mjcf_dir = os.path.dirname(os.path.abspath(mjcf_path))
    if mjcf_dir not in search_paths:
        search_paths.insert(0, mjcf_dir)
    
    # Parse the XML tree.
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    
    # Read meshdir from the compiler element.
    compiler_elem = root.find('compiler')
    if compiler_elem is not None:
        meshdir = compiler_elem.get('meshdir')
        if meshdir is not None:
            meshdir_abs = os.path.normpath(os.path.join(mjcf_dir, meshdir))
            if meshdir_abs not in search_paths:
                search_paths.insert(0, meshdir_abs)
    model_name = root.get('model', 'unnamed_model')
    
    # Parse mesh assets.
    mesh_assets = {}
    asset_elem = root.find('asset')
    if asset_elem is not None:
        for mesh_elem in asset_elem.findall('mesh'):
            mesh_name = mesh_elem.get('name')
            mesh_file = mesh_elem.get('file')
            mesh_scale = mesh_elem.get('scale', '1 1 1')
            scale = np.array([float(x) for x in mesh_scale.split()])
            mesh_assets[mesh_name] = {'file': mesh_file, 'scale': scale}
    
    # Parse worldbody recursively.
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
    """Parse a body element recursively."""
    for body_elem in parent_elem.findall('body'):
        body_name = body_elem.get('name', f'body_{len(bodies)}')
        
        # Parse position and orientation.
        pos_str = body_elem.get('pos', '0 0 0')
        pos = np.array([float(x) for x in pos_str.split()])
        
        quat_str = body_elem.get('quat', '1 0 0 0')  # MuJoCo: (w, x, y, z)
        quat = np.array([float(x) for x in quat_str.split()])
        
        # Support Euler angles.
        euler_str = body_elem.get('euler')
        if euler_str is not None:
            euler = np.array([float(x) for x in euler_str.split()])
            # Convert to a quaternion.
            r = R.from_euler('xyz', euler, degrees=True)
            quat_xyzw = r.as_quat()
            quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        
        # Parse all body geometries.
        meshes = []
        for geom_elem in body_elem.findall('geom'):
            mesh = _parse_geom(geom_elem, mesh_assets, search_paths)
            if mesh is not None:
                meshes.append(mesh)
        
        # Combine all meshes.
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
        
        # Parse joints.
        for joint_elem in body_elem.findall('joint'):
            joint_info = _parse_joint(joint_elem, body_name)
            if joint_info is not None:
                joints.append(joint_info)
        
        # Parse child bodies recursively.
        _parse_body_recursive(body_elem, body_name, bodies, joints, mesh_assets, search_paths)


def _parse_geom(geom_elem, mesh_assets: Dict, search_paths: List[str]) -> Optional[trimesh.Trimesh]:
    """Parse one geometry element into a trimesh object."""
    geom_type = geom_elem.get('type', 'sphere')
    
    # Parse the local position and orientation.
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
    
    # Create the geometry.
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
        # Create and scale a unit sphere.
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        mesh.vertices *= radii
        
    elif geom_type == 'cylinder':
        size_str = geom_elem.get('size', '0.5 0.5')
        sizes = [float(x) for x in size_str.split()]
        radius = sizes[0]
        
        # Check the fromto attribute.
        fromto_str = geom_elem.get('fromto')
        if fromto_str is not None:
            fromto = np.array([float(x) for x in fromto_str.split()])
            p1 = fromto[:3]
            p2 = fromto[3:]
            length = np.linalg.norm(p2 - p1)
            
            # Create the cylinder.
            mesh = trimesh.creation.cylinder(radius=radius, height=length, sections=32)
            
            # Rotate the default z-axis onto p2-p1.
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
            
            # Move to the midpoint.
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
        
        # Check the fromto attribute.
        fromto_str = geom_elem.get('fromto')
        if fromto_str is not None:
            fromto = np.array([float(x) for x in fromto_str.split()])
            p1 = fromto[:3]
            p2 = fromto[3:]
            length = np.linalg.norm(p2 - p1)
            
            # Create the capsule.
            mesh = trimesh.creation.capsule(height=length, radius=radius, count=[16, 8])
            
            # Rotate the default z-axis onto p2-p1.
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
            
            # Move to the midpoint.
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
    
    # Apply the local transform.
    # Rotate.
    quat_scipy = [quat[1], quat[2], quat[3], quat[0]]  # scipy: (x, y, z, w)
    rotation = R.from_quat(quat_scipy)
    mesh.vertices = rotation.apply(mesh.vertices)
    
    # Translate.
    mesh.vertices += pos
    
    return mesh


def _parse_joint(joint_elem, body_name: str) -> Optional[Dict]:
    """Parse one joint element."""
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
    """Resolve a mesh file path."""
    if mesh_uri is None:
        return None
        
    # Check an absolute path.
    if os.path.isabs(mesh_uri):
        return mesh_uri if os.path.exists(mesh_uri) else None
    
    # Check relative paths.
    for sp in search_paths:
        full_path = os.path.join(sp, mesh_uri)
        if os.path.exists(full_path):
            return full_path
    
    return None


def get_body_world_positions(mjcf_data: Dict) -> Dict[str, np.ndarray]:
    """
    Compute each body's world position.
    
    Returns:
        {body_name: world_position}
    """
    bodies = mjcf_data['bodies']
    
    # Map names to bodies.
    body_map = {b['name']: b for b in bodies}
    
    # Compute world positions.
    world_positions = {}
    
    for body in bodies:
        pos = body['pos'].copy()
        parent_name = body['parent']
        
        # Traverse upward to worldbody.
        while parent_name is not None and parent_name in body_map:
            parent = body_map[parent_name]
            # Apply the parent rotation.
            parent_quat = parent['quat']
            quat_scipy = [parent_quat[1], parent_quat[2], parent_quat[3], parent_quat[0]]
            rotation = R.from_quat(quat_scipy)
            pos = rotation.apply(pos) + parent['pos']
            parent_name = parent['parent']
        
        world_positions[body['name']] = pos
    
    return world_positions


def parse_mjcf_to_meshes(mjcf_path: str, mesh_search_paths: Optional[List[str]] = None) -> Dict[str, trimesh.Trimesh]:
    """
    Parse MJCF XML and map body names to meshes.
    
    Args:
        mjcf_path: Path to the MJCF XML file.
        mesh_search_paths: Mesh search paths.
        
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
    Parse MJCF into a parse_urdf-compatible structure.
    
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
    
    # Convert bodies to links.
    links = []
    for body in mjcf_data['bodies']:
        links.append({
            'name': body['name'],
            'mesh': body['combined_mesh'],
            'origin_xyz': np.zeros(3),  # MJCF geometry includes its offset.
            'origin_rpy': np.zeros(3),
            'mass': 1.0,  # Default value.
            'inertia': np.eye(3) * 0.01,
        })
    
    # Build joints from the MJCF body hierarchy.
    joints = []
    for body in mjcf_data['bodies']:
        if body['parent'] is not None:
            # Convert the quaternion to RPY.
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
