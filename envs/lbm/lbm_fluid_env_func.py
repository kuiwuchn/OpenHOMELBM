"""Warp kernels shared by two-dimensional coupled environments."""

import warp as wp

from .lbm_core import HomeFlow


@wp.kernel
def extract_body_states(
    xipos_full: wp.array2d(dtype=wp.vec3),
    xquat_full: wp.array2d(dtype=wp.quat),
    body_ids: wp.array(dtype=wp.int32),
    positions_out: wp.array3d(dtype=wp.float32),
    quaternions_out: wp.array3d(dtype=wp.float32),
):
    """Extract selected MuJoCo body poses for every world."""
    world_idx, body_idx = wp.tid()
    body_id = body_ids[body_idx]
    position = xipos_full[world_idx, body_id]
    quaternion = xquat_full[world_idx, body_id]
    for axis in range(3):
        positions_out[world_idx, body_idx, axis] = position[axis]
    for axis in range(4):
        quaternions_out[world_idx, body_idx, axis] = quaternion[axis]


@wp.kernel
def fill_xfrc_kernel(
    xfrc_buffer: wp.array2d(dtype=wp.spatial_vectorf),
    body_ids: wp.array(dtype=wp.int32),
    forces: wp.array3d(dtype=wp.float32),
    torques: wp.array3d(dtype=wp.float32),
):
    """Write projected fluid forces into MuJoCo's spatial-force buffer."""
    world_idx, body_idx = wp.tid()
    body_id = body_ids[body_idx]
    xfrc_buffer[world_idx, body_id] = wp.spatial_vector(
        wp.vec3(
            forces[world_idx, body_idx, 0],
            forces[world_idx, body_idx, 1],
            forces[world_idx, body_idx, 2],
        ),
        wp.vec3(
            torques[world_idx, body_idx, 0],
            torques[world_idx, body_idx, 1],
            torques[world_idx, body_idx, 2],
        ),
    )


@wp.kernel
def extract_forces_torques_batch(
    flows: wp.array(dtype=HomeFlow),
    solid_ids: wp.array(dtype=wp.int32),
    scales: wp.array(dtype=wp.float32),
    forces_out: wp.array3d(dtype=wp.float32),
    torques_out: wp.array3d(dtype=wp.float32),
):
    """Convert planar LBM loads to three-dimensional MuJoCo loads."""
    world_idx, body_idx = wp.tid()
    flow = flows[world_idx]
    solid_id = solid_ids[body_idx]
    scale = scales[body_idx]
    forces_out[world_idx, body_idx, 0] = flow.solid_forcex[solid_id] / scale
    forces_out[world_idx, body_idx, 1] = flow.solid_forcey[solid_id] / scale
    forces_out[world_idx, body_idx, 2] = 0.0
    torques_out[world_idx, body_idx, 0] = 0.0
    torques_out[world_idx, body_idx, 1] = 0.0
    torques_out[world_idx, body_idx, 2] = flow.torque[solid_id] / scale


@wp.kernel
def convert_and_update_solid_batch_2d(
    flows: wp.array(dtype=HomeFlow),
    solid_ids: wp.array(dtype=wp.int32),
    mujoco_positions: wp.array3d(dtype=wp.float32),
    mujoco_quaternions: wp.array3d(dtype=wp.float32),
    mujoco_origins: wp.array2d(dtype=wp.float32),
    lbm_origins: wp.array2d(dtype=wp.float32),
    scales: wp.array(dtype=wp.float32),
):
    """Project MuJoCo poses into the two-dimensional LBM plane."""
    world_idx, body_idx = wp.tid()
    flow = flows[world_idx]
    solid_id = solid_ids[body_idx]
    scale = scales[body_idx]

    lbm_x = (
        mujoco_positions[world_idx, body_idx, 0]
        - mujoco_origins[body_idx, 0]
    ) * scale + lbm_origins[body_idx, 0]
    lbm_y = (
        mujoco_positions[world_idx, body_idx, 1]
        - mujoco_origins[body_idx, 1]
    ) * scale + lbm_origins[body_idx, 1]
    flow.solid_position[solid_id] = wp.vec2(lbm_x, lbm_y)

    w = mujoco_quaternions[world_idx, body_idx, 0]
    x = mujoco_quaternions[world_idx, body_idx, 1]
    y = mujoco_quaternions[world_idx, body_idx, 2]
    z = mujoco_quaternions[world_idx, body_idx, 3]
    flow.solid_angle[solid_id] = wp.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
