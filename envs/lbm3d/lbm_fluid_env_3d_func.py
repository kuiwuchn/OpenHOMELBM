"""
Warp kernels for 3D LBM Fluid Environment
Contains all GPU kernels used in lbm_fluid_env_3d.py
"""
import warp as wp
from .lbm_core_3d import HomeFlow3D


# ============== Body State Extraction Kernels ==============

@wp.kernel
def extract_body_states_3d(
    xipos_full: wp.array2d(dtype=wp.vec3),  # (nworld, nbody) MuJoCo body COM positions
    xquat_full: wp.array2d(dtype=wp.quat),  # (nworld, nbody) MuJoCo quaternions
    body_ids: wp.array(dtype=wp.int32),  # (n_solids,)
    positions_out: wp.array3d(dtype=wp.float32),  # (nworld, n_solids, 3)
    quaternions_out: wp.array3d(dtype=wp.float32),  # (nworld, n_solids, 4)
):
    """
    Extract states of specified bodies from the full xipos/xquat arrays for all worlds.
    2D launch: (nworld, n_solids)
    """
    world_idx, idx = wp.tid()
    body_id = body_ids[idx]
    
    # Extract position (body COM from specified world)
    pos = xipos_full[world_idx, body_id]
    positions_out[world_idx, idx, 0] = pos[0]
    positions_out[world_idx, idx, 1] = pos[1]
    positions_out[world_idx, idx, 2] = pos[2]
    
    # Extract quaternion (w, x, y, z format)
    quat = xquat_full[world_idx, body_id]
    quaternions_out[world_idx, idx, 0] = quat[0]  # w
    quaternions_out[world_idx, idx, 1] = quat[1]  # x
    quaternions_out[world_idx, idx, 2] = quat[2]  # y
    quaternions_out[world_idx, idx, 3] = quat[3]  # z


# ============== MuJoCo to LBM Coordinate Conversion ==============

@wp.kernel
def convert_and_update_solid_batch_3d(
    flows: wp.array(dtype=HomeFlow3D),  # (nworld,) array of flow objects
    solid_ids: wp.array(dtype=wp.int32),  # (n_solids,) solid IDs in LBM
    mujoco_positions: wp.array3d(dtype=wp.float32),  # (nworld, n_solids, 3)
    mujoco_quaternions: wp.array3d(dtype=wp.float32),  # (nworld, n_solids, 4)
    mujoco_origins: wp.array(dtype=wp.vec3),  # (n_solids,) - shared across all worlds
    lbm_origins: wp.array(dtype=wp.vec3),  # (n_solids,) - shared across all worlds
    scales: wp.array(dtype=wp.float32),  # (n_solids,) - shared across all worlds
):
    """
    Convert MuJoCo coordinates to LBM and update all worlds in parallel.
    Also updates mesh_transforms for ray casting.
    2D launch: (nworld, n_solids)
    """
    world_idx, body_idx = wp.tid()
    
    flow = flows[world_idx]
    solid_id = solid_ids[body_idx]
    scale = scales[body_idx]
    
    # Convert position: (mujoco_pos - mujoco_origin) * scale + lbm_origin
    mujoco_pos_x = mujoco_positions[world_idx, body_idx, 0]
    mujoco_pos_y = mujoco_positions[world_idx, body_idx, 1]
    mujoco_pos_z = mujoco_positions[world_idx, body_idx, 2]
    
    mujoco_origin = mujoco_origins[body_idx]
    lbm_origin = lbm_origins[body_idx]
    
    lbm_x = (mujoco_pos_x - mujoco_origin[0]) * scale + lbm_origin[0]
    lbm_y = (mujoco_pos_y - mujoco_origin[1]) * scale + lbm_origin[1]
    lbm_z = (mujoco_pos_z - mujoco_origin[2]) * scale + lbm_origin[2]
    
    lbm_pos = wp.vec3(lbm_x, lbm_y, lbm_z)
    flow.solid_position[solid_id] = lbm_pos
    
    # Copy quaternion directly (w, x, y, z)
    w = mujoco_quaternions[world_idx, body_idx, 0]
    x = mujoco_quaternions[world_idx, body_idx, 1]
    y = mujoco_quaternions[world_idx, body_idx, 2]
    z = mujoco_quaternions[world_idx, body_idx, 3]
    
    flow.solid_quaternion[solid_id] = wp.vec4(w, x, y, z)
    
    # Update mesh_transforms for ray casting
    # Save current transform to last (for velocity calculation)
    is_initialized = flow.mesh_transforms_initialized[solid_id]
    if is_initialized > 0:
        flow.mesh_transforms_last[solid_id] = flow.mesh_transforms[solid_id]
    
    # Create new transform: wp.quat uses (x, y, z, w) order internally
    new_transform = wp.transform(lbm_pos, wp.quat(x, y, z, w))
    flow.mesh_transforms[solid_id] = new_transform
    
    # If first time, also set last to current (zero velocity)
    if is_initialized == 0:
        flow.mesh_transforms_last[solid_id] = new_transform
        flow.mesh_transforms_initialized[solid_id] = 1


# ============== Force Extraction Kernels ==============

@wp.kernel
def extract_forces_torques_batch_3d(
    flows: wp.array(dtype=HomeFlow3D),  # (nworld,) array of flow objects
    solid_ids: wp.array(dtype=wp.int32),  # (n_solids,) solid IDs
    scales: wp.array(dtype=wp.float32),  # (n_solids,) scaling factors
    forces_out: wp.array3d(dtype=wp.float32),  # (nworld, n_solids, 3)
    torques_out: wp.array3d(dtype=wp.float32),  # (nworld, n_solids, 3)
):
    """
    Extract forces and torques for all worlds in parallel.
    DEPRECATED: Use extract_forces_torques_physical_3d for proper physical unit conversion.
    2D launch: (nworld, n_solids)
    """
    world_idx, body_idx = wp.tid()
    
    flow = flows[world_idx]
    solid_id = solid_ids[body_idx]
    scale = scales[body_idx]
    
    # Get force from flow
    force = flow.solid_force[solid_id]
    torque = flow.solid_torque[solid_id]
    
    # Convert forces and torques back to MuJoCo coordinate system (divide by scale)
    forces_out[world_idx, body_idx, 0] = force[0] / scale
    forces_out[world_idx, body_idx, 1] = force[1] / scale
    forces_out[world_idx, body_idx, 2] = force[2] / scale
    
    scale_sq = scale * scale
    torques_out[world_idx, body_idx, 0] = torque[0] / scale_sq
    torques_out[world_idx, body_idx, 1] = torque[1] / scale_sq
    torques_out[world_idx, body_idx, 2] = torque[2] / scale_sq


@wp.kernel
def extract_forces_torques_physical_3d(
    flows: wp.array(dtype=HomeFlow3D),  # (nworld,) array of flow objects
    solid_ids: wp.array(dtype=wp.int32),  # (n_solids,) solid IDs
    force_conversion: wp.float32,  # Physical force conversion factor: rho × dx^4 / dt^2
    torque_conversion: wp.float32,  # Physical torque conversion factor: rho × dx^5 / dt^2
    forces_out: wp.array3d(dtype=wp.float32),  # (nworld, n_solids, 3)
    torques_out: wp.array3d(dtype=wp.float32),  # (nworld, n_solids, 3)
):
    """
    Extract forces and torques with proper physical unit conversion.
    
    Physical conversion based on dimensional analysis:
    - LBM uses dimensionless units: rho_lbm=1, dx_lbm=1, dt_lbm=1
    - Force:  F_physical = F_lbm × rho_fluid × dx^4 / dt^2  [N]
    - Torque: τ_physical = τ_lbm × rho_fluid × dx^5 / dt^2  [N·m]
    
    Where:
    - rho_fluid: fluid density (e.g., 1000 kg/m³ for water)
    - dx: physical grid spacing = 1/coordinate_scale (m)
    - dt: MuJoCo timestep (s)
    
    2D launch: (nworld, n_solids)
    """
    world_idx, body_idx = wp.tid()
    
    flow = flows[world_idx]
    solid_id = solid_ids[body_idx]
    
    # Get raw LBM forces (dimensionless)
    force = flow.solid_force[solid_id]
    torque = flow.solid_torque[solid_id]
    
    # Apply physical unit conversion
    forces_out[world_idx, body_idx, 0] = force[0] * force_conversion
    forces_out[world_idx, body_idx, 1] = force[1] * force_conversion
    forces_out[world_idx, body_idx, 2] = force[2] * force_conversion
    
    torques_out[world_idx, body_idx, 0] = torque[0] * torque_conversion
    torques_out[world_idx, body_idx, 1] = torque[1] * torque_conversion
    torques_out[world_idx, body_idx, 2] = torque[2] * torque_conversion


@wp.kernel
def fill_xfrc_3d_kernel(
    xfrc_buffer: wp.array2d(dtype=wp.spatial_vectorf),  # (nworld, nbody)
    body_ids: wp.array(dtype=wp.int32),  # (n_solids,)
    forces: wp.array3d(dtype=wp.float32),  # (nworld, n_solids, 3)
    torques: wp.array3d(dtype=wp.float32),  # (nworld, n_solids, 3)
):
    """
    Fill xfrc_applied buffer for all worlds.
    2D launch: (nworld, n_solids)
    
    Forces and torques are already converted to physical units (N and N·m)
    by extract_forces_torques_physical_3d kernel.
    
    xfrc_applied = [force_x, force_y, force_z, torque_x, torque_y, torque_z]
    """
    world_idx, idx = wp.tid()
    body_id = body_ids[idx]
    
    fx = forces[world_idx, idx, 0]
    fy = forces[world_idx, idx, 1]
    fz = forces[world_idx, idx, 2]
    tx = torques[world_idx, idx, 0]
    ty = torques[world_idx, idx, 1]
    tz = torques[world_idx, idx, 2]
    
    xfrc_buffer[world_idx, body_id] = wp.spatial_vector(
        wp.vec3(fx, fy, fz),
        wp.vec3(tx, ty, tz),
    )


# ============== Solid Position Extraction ==============

@wp.kernel
def extract_solid_position_3d_kernel(
    flows: wp.array(dtype=HomeFlow3D),  # (nworld,)
    solid_id: int,
    solid_positions_dst: wp.array2d(dtype=wp.float32),  # (nworld, 3)
):
    """
    Extract a specific solid's position from all LBM solvers.
    1D launch: nworld
    """
    world_idx = wp.tid()
    flow = flows[world_idx]
    pos = flow.solid_position[solid_id]
    solid_positions_dst[world_idx, 0] = pos[0]
    solid_positions_dst[world_idx, 1] = pos[1]
    solid_positions_dst[world_idx, 2] = pos[2]


@wp.kernel
def extract_all_solid_positions_3d_kernel(
    flows: wp.array(dtype=HomeFlow3D),  # (nworld,)
    solid_ids: wp.array(dtype=wp.int32),  # (n_solids,)
    solid_positions_dst: wp.array3d(dtype=wp.float32),  # (nworld, n_solids, 3)
):
    """
    Extract all solids' positions from all LBM solvers.
    2D launch: (nworld, n_solids)
    """
    world_idx, idx = wp.tid()
    solid_id = solid_ids[idx]
    flow = flows[world_idx]
    pos = flow.solid_position[solid_id]
    solid_positions_dst[world_idx, idx, 0] = pos[0]
    solid_positions_dst[world_idx, idx, 1] = pos[1]
    solid_positions_dst[world_idx, idx, 2] = pos[2]
