"""
Warp kernels for LBM Fluid Environment
Contains all GPU kernels used in lbm_fluid_env.py
"""
import warp as wp
from .lbm_core import HomeFlow


@wp.kernel
def extract_body_states(
    xipos_full: wp.array2d(dtype=wp.vec3),  # (batch, nbody) MuJoCo body COM positions
    xquat_full: wp.array2d(dtype=wp.quat),  # (batch, nbody) MuJoCo quaternions
    body_ids: wp.array(dtype=wp.int32),  # (n,)
    positions_out: wp.array3d(dtype=wp.float32),  # (nworld, n, 3)
    quaternions_out: wp.array3d(dtype=wp.float32),  # (nworld, n, 4)
):
    """
    Extract states of specified bodies from the full xipos/xquat arrays for all worlds
    2D launch: (nworld, n_bodies)
    """
    world_idx, idx = wp.tid()
    body_id = body_ids[idx]
    
    # Extract position (body COM from specified world)
    pos = xipos_full[world_idx, body_id]
    positions_out[world_idx, idx, 0] = pos[0]
    positions_out[world_idx, idx, 1] = pos[1]
    positions_out[world_idx, idx, 2] = pos[2]
    
    # Extract quaternion (from specified world)
    # quatf format is (w, x, y, z)
    quat = xquat_full[world_idx, body_id]
    quaternions_out[world_idx, idx, 0] = quat[0]  # w
    quaternions_out[world_idx, idx, 1] = quat[1]  # x
    quaternions_out[world_idx, idx, 2] = quat[2]  # y
    quaternions_out[world_idx, idx, 3] = quat[3]  # z


@wp.kernel
def fill_xfrc_kernel(
    xfrc_buffer: wp.array2d(dtype=wp.spatial_vectorf),  # (nworld, nbody)
    body_ids: wp.array(dtype=wp.int32),
    forces: wp.array3d(dtype=wp.float32),  # (nworld, n, 3)
    torques: wp.array3d(dtype=wp.float32),  # (nworld, n, 3)
):
    """
    Fill xfrc_applied buffer for all worlds
    2D launch: (nworld, n_bodies)
    spatial_vector format: [torque_x, torque_y, torque_z, force_x, force_y, force_z]
    """
    world_idx, idx = wp.tid()
    body_id = body_ids[idx]
    
    # Create spatial_vector (force, torque)
    xfrc_buffer[world_idx, body_id] = wp.spatial_vector(
        wp.vec3(forces[world_idx, idx, 0], forces[world_idx, idx, 1], forces[world_idx, idx, 2]),      # force
        wp.vec3(torques[world_idx, idx, 0], torques[world_idx, idx, 1], torques[world_idx, idx, 2]),  # torque
    )


@wp.kernel
def extract_solid_position_kernel(
    flows: wp.array(dtype=HomeFlow),  # (nworld,) array of flow objects
    solid_id: int,  # Which solid to extract
    solid_positions_dst: wp.array2d(dtype=wp.float32),  # Destination: (nworld, 2)
):
    """
    Extract a specific solid's position from all LBM solvers to 2D buffer
    1D launch: nworld
    """
    world_idx = wp.tid()
    flow = flows[world_idx]
    pos = flow.solid_position[solid_id]
    solid_positions_dst[world_idx, 0] = pos[0]
    solid_positions_dst[world_idx, 1] = pos[1]


@wp.kernel
def extract_solid_position_2d_kernel(
    flows: wp.array(dtype=HomeFlow),  # (nworld,) array of flow objects
    solid_id: int,  # Which solid to extract
    solid_positions_dst: wp.array3d(dtype=wp.float32),  # Destination: (nworld, n_solids, 2)
):
    """
    Extract a specific solid's position from all LBM solvers to 3D buffer
    1D launch: nworld
    """
    world_idx = wp.tid()
    flow = flows[world_idx]
    pos = flow.solid_position[solid_id]
    solid_positions_dst[world_idx, solid_id, 0] = pos[0]
    solid_positions_dst[world_idx, solid_id, 1] = pos[1]


@wp.kernel
def compute_butterfly_obs(
    qfrc_applied: wp.array2d(dtype=wp.float32),  # (nworld, nv)
    qpos: wp.array2d(dtype=wp.float32),  # (nworld, nq)
    qvel: wp.array2d(dtype=wp.float32),  # (nworld, nv)
    solid_positions: wp.array2d(dtype=wp.float32),  # (nworld, 2) - center body position in LBM coords
    obs_out: wp.array2d(dtype=wp.float32),  # (nworld, 19)
    nx: int,
    ny: int,
    target_x: float,
    target_y: float,
):
    """
    Compute observation for butterfly environment
    1D launch: nworld
    """
    world_idx = wp.tid()
    
    # Extract generalized forces
    fx = qfrc_applied[world_idx, 0]
    fy = qfrc_applied[world_idx, 1]
    tau_z = qfrc_applied[world_idx, 5]
    tau_left = qfrc_applied[world_idx, 6]
    tau_right = qfrc_applied[world_idx, 7]
    
    # Extract position
    x = qpos[world_idx, 0]
    y = qpos[world_idx, 1]
    
    # Extract rotation angle from quaternion
    qw = qpos[world_idx, 3]
    qx = qpos[world_idx, 4]
    qy_q = qpos[world_idx, 5]
    qz = qpos[world_idx, 6]
    theta_z = wp.atan2(2.0 * (qw * qz + qx * qy_q), 1.0 - 2.0 * (qy_q * qy_q + qz * qz))
    
    # Extract velocities
    vx = qvel[world_idx, 0]
    vy = qvel[world_idx, 1]
    omega_z = qvel[world_idx, 5]
    
    # Extract joint states
    joint_left = qpos[world_idx, 7]
    joint_right = qpos[world_idx, 8]
    joint_left_vel = qvel[world_idx, 6]
    joint_right_vel = qvel[world_idx, 7]
    
    # Get normalized LBM position
    current_x = solid_positions[world_idx, 0] / float(nx)
    current_y = solid_positions[world_idx, 1] / float(ny)
    
    # Fill observation
    obs_out[world_idx, 0] = fx
    obs_out[world_idx, 1] = fy
    obs_out[world_idx, 2] = tau_z
    obs_out[world_idx, 3] = tau_left
    obs_out[world_idx, 4] = tau_right
    obs_out[world_idx, 5] = x
    obs_out[world_idx, 6] = y
    obs_out[world_idx, 7] = theta_z
    obs_out[world_idx, 8] = vx
    obs_out[world_idx, 9] = vy
    obs_out[world_idx, 10] = omega_z
    obs_out[world_idx, 11] = joint_left
    obs_out[world_idx, 12] = joint_right
    obs_out[world_idx, 13] = joint_left_vel
    obs_out[world_idx, 14] = joint_right_vel
    obs_out[world_idx, 15] = current_x
    obs_out[world_idx, 16] = current_y
    obs_out[world_idx, 17] = target_x
    obs_out[world_idx, 18] = target_y


@wp.kernel
def compute_butterfly_reward(
    solid_positions: wp.array2d(dtype=wp.float32),  # (nworld, 2) - center body position
    rewards_out: wp.array(dtype=wp.float32),  # (nworld,)
    nx: int,
    ny: int,
    target_x: float,
    target_y: float,
    reward_weight: float,
):
    """
    Compute reward for butterfly environment
    1D launch: nworld
    """
    world_idx = wp.tid()
    
    # Get current position and normalize
    current_x = solid_positions[world_idx, 0] / float(nx)
    current_y = solid_positions[world_idx, 1] / float(ny)
    
    # Calculate distance to target
    dx = current_x - target_x
    dy = current_y - target_y
    dist = wp.sqrt(dx * dx + dy * dy)
    
    # Exponential reward
    rewards_out[world_idx] = reward_weight * wp.exp(-dist)


@wp.kernel
def check_butterfly_termination(
    solid_positions: wp.array3d(dtype=wp.float32),  # (nworld, n_solids, 2) - all solid positions
    solid_max_radii: wp.array(dtype=wp.float32),  # (n_solids,)
    terminated_out: wp.array(dtype=wp.uint8),  # (nworld,) - use uint8 for bool
    nx: int,
    ny: int,
    target_x: float,
    target_y: float,
    boundary_margin: float,
    n_solids: int,
):
    """
    Check termination condition for butterfly environment
    1D launch: nworld
    """
    world_idx = wp.tid()
    
    # Check if reached target (center body = solid 0)
    center_x = solid_positions[world_idx, 0, 0] / float(nx)
    center_y = solid_positions[world_idx, 0, 1] / float(ny)
    dx = center_x - target_x
    dy = center_y - target_y
    
    if wp.sqrt(dx * dx + dy * dy) < 0.0001:
        terminated_out[world_idx] = wp.uint8(1)
        return
    
    # Check boundary collision for all solids
    x_min = 0.0
    x_max = float(nx)
    y_min = 0.0
    y_max = float(ny)
    
    for solid_id in range(n_solids):
        pos_x = solid_positions[world_idx, solid_id, 0]
        pos_y = solid_positions[world_idx, solid_id, 1]
        max_radius = solid_max_radii[solid_id]
        
        if (pos_x - max_radius < x_min + boundary_margin or
            pos_x + max_radius > x_max - boundary_margin or
            pos_y - max_radius < y_min + boundary_margin or
            pos_y + max_radius > y_max - boundary_margin):
            terminated_out[world_idx] = wp.uint8(1)
            return
    
    # Not terminated
    terminated_out[world_idx] = wp.uint8(0)


@wp.kernel
def extract_forces_torques_batch(
    flows: wp.array(dtype=HomeFlow),  # (nworld,) array of flow objects
    solid_ids: wp.array(dtype=wp.int32),  # (n_bodies,) solid IDs
    scales: wp.array(dtype=wp.float32),  # (n_bodies,) scaling factors - shared across all worlds
    forces_out: wp.array3d(dtype=wp.float32),  # (nworld, n_bodies, 3)
    torques_out: wp.array3d(dtype=wp.float32)  # (nworld, n_bodies, 3)
):
    """
    Extract forces and torques for all worlds in parallel
    2D launch: (nworld, n_bodies)
    """
    world_idx, body_idx = wp.tid()
    
    flow = flows[world_idx]
    solid_id = solid_ids[body_idx]
    scale = scales[body_idx]
    
    # Convert forces and torques back to MuJoCo coordinate system (divide by scale)
    forces_out[world_idx, body_idx, 0] = flow.solid_forcex[solid_id] / scale
    forces_out[world_idx, body_idx, 1] = flow.solid_forcey[solid_id] / scale
    forces_out[world_idx, body_idx, 2] = 0.0
    
    torques_out[world_idx, body_idx, 0] = 0.0
    torques_out[world_idx, body_idx, 1] = 0.0
    torques_out[world_idx, body_idx, 2] = flow.torque[solid_id] / scale


@wp.kernel
def convert_and_update_solid_batch_2d(
    flows: wp.array(dtype=HomeFlow),  # (nworld,) array of flow objects
    solid_ids: wp.array(dtype=wp.int32),  # (n_bodies,) solid IDs
    mujoco_positions: wp.array3d(dtype=wp.float32),  # (nworld, n_bodies, 3)
    mujoco_quaternions: wp.array3d(dtype=wp.float32),  # (nworld, n_bodies, 4)
    mujoco_origins: wp.array2d(dtype=wp.float32),  # (n_bodies, 2) - shared across all worlds
    lbm_origins: wp.array2d(dtype=wp.float32),  # (n_bodies, 2) - shared across all worlds
    scales: wp.array(dtype=wp.float32)  # (n_bodies,) - shared across all worlds
):
    """
    Convert MuJoCo coordinates to LBM and update all worlds in parallel
    2D launch: (nworld, n_bodies)
    """
    world_idx, body_idx = wp.tid()
    
    flow = flows[world_idx]
    solid_id = solid_ids[body_idx]
    
    # Convert position: (mujoco_pos - mujoco_origin) * scale + lbm_origin
    mujoco_pos_x = mujoco_positions[world_idx, body_idx, 0]
    mujoco_pos_y = mujoco_positions[world_idx, body_idx, 1]
    
    mujoco_origin_x = mujoco_origins[body_idx, 0]
    mujoco_origin_y = mujoco_origins[body_idx, 1]
    
    lbm_origin_x = lbm_origins[body_idx, 0]
    lbm_origin_y = lbm_origins[body_idx, 1]
    
    scale = scales[body_idx]
    
    lbm_x = (mujoco_pos_x - mujoco_origin_x) * scale + lbm_origin_x
    lbm_y = (mujoco_pos_y - mujoco_origin_y) * scale + lbm_origin_y
    
    flow.solid_position[solid_id] = wp.vec2(lbm_x, lbm_y)
    
    # Calculate angle from quaternion
    w = mujoco_quaternions[world_idx, body_idx, 0]
    x = mujoco_quaternions[world_idx, body_idx, 1]
    y = mujoco_quaternions[world_idx, body_idx, 2]
    z = mujoco_quaternions[world_idx, body_idx, 3]
    
    angle = wp.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    flow.solid_angle[solid_id] = angle
