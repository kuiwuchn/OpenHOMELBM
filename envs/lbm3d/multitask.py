"""Shared task definitions and observation kernels for 3D coupled environments."""

import warp as wp

from .lbm_core_3d import HomeFlow3D


TASK_FORWARD = 0
TASK_TURN_LEFT = 1
TASK_TURN_RIGHT = 2
TASK_ASCEND = 3
TASK_DESCEND = 4

NUM_TASKS = 5
TASK_NAMES = ["forward", "turn_left", "turn_right", "ascend", "descend"]


@wp.func
def _cross_vec3(a: wp.vec3, b: wp.vec3) -> wp.vec3:
    return wp.vec3(
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


@wp.func
def quat_rotate_vec(
    qw: wp.float32,
    qx: wp.float32,
    qy: wp.float32,
    qz: wp.float32,
    vector: wp.vec3,
) -> wp.vec3:
    """Rotate a body-frame vector into the world frame."""
    qv = wp.vec3(qx, qy, qz)
    twice_cross = _cross_vec3(qv, vector)
    twice_cross = wp.vec3(
        2.0 * twice_cross[0],
        2.0 * twice_cross[1],
        2.0 * twice_cross[2],
    )
    correction = _cross_vec3(qv, twice_cross)
    return wp.vec3(
        vector[0] + qw * twice_cross[0] + correction[0],
        vector[1] + qw * twice_cross[1] + correction[1],
        vector[2] + qw * twice_cross[2] + correction[2],
    )


@wp.func
def dot_vec3(a: wp.vec3, b: wp.vec3) -> wp.float32:
    """Return the dot product of two three-dimensional vectors."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@wp.kernel
def compute_multitask_obs_kernel(
    qfrc_applied: wp.array2d(dtype=wp.float32),
    qpos: wp.array2d(dtype=wp.float32),
    qvel: wp.array2d(dtype=wp.float32),
    flows: wp.array(dtype=HomeFlow3D),
    task_ids: wp.array(dtype=wp.int32),
    time_val: wp.array(dtype=wp.float32),
    obs_out: wp.array2d(dtype=wp.float32),
    nx: float,
    ny: float,
    nz: float,
    n_joints: int,
    n_tasks: int,
):
    """Build the shared rigid-state, LBM-position, task, and phase observation."""
    world_idx = wp.tid()
    flow = flows[world_idx]
    index = 0

    for i in range(6):
        obs_out[world_idx, index] = qfrc_applied[world_idx, i]
        index += 1
    for i in range(n_joints):
        obs_out[world_idx, index] = qfrc_applied[world_idx, 6 + i]
        index += 1
    for i in range(7):
        obs_out[world_idx, index] = qpos[world_idx, i]
        index += 1
    for i in range(6):
        obs_out[world_idx, index] = qvel[world_idx, i]
        index += 1
    for i in range(n_joints):
        obs_out[world_idx, index] = qpos[world_idx, 7 + i]
        index += 1
    for i in range(n_joints):
        obs_out[world_idx, index] = qvel[world_idx, 6 + i]
        index += 1

    center = flow.solid_position[0]
    obs_out[world_idx, index] = center[0] / nx
    obs_out[world_idx, index + 1] = center[1] / ny
    obs_out[world_idx, index + 2] = center[2] / nz
    index += 3

    task = task_ids[world_idx]
    for i in range(n_tasks):
        obs_out[world_idx, index] = wp.where(i == task, 1.0, 0.0)
        index += 1

    time = time_val[world_idx]
    obs_out[world_idx, index] = wp.sin(time)
    obs_out[world_idx, index + 1] = wp.cos(time)
