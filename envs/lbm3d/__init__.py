"""
3D LBM Environment Package

This package provides 3D LBM (Lattice Boltzmann Method) environments
with multi-world support for parallel training.

Uses gym (not gymnasium) for compatibility with dreamer_vec_wrapper.
"""

from .lbm_core_3d import HomeFlow3D
from .lbm_func_3d import (
    InitBoundary3D,
    InitFlow3D,
    ResetSingleWorldFlow3D,
    ResetSingleWorldSolidTransform3D,
    ResetSingleWorldForces3D,
    stream_and_collide_3d,
    apply_bc_3d,
    Swap_Mom_3D,
    init_force_3d,
)
from .lbm_solver_3d import LBM_Solver3D
from .lbm_fluid_env_3d import LBMFluidEnv3D 
from .lbm_fluid_env_3d_func import (
    extract_body_states_3d,
    convert_and_update_solid_batch_3d,
    extract_forces_torques_batch_3d,
    fill_xfrc_3d_kernel,
    extract_all_solid_positions_3d_kernel,
)
from .fish import FishLBMEnv3D
from .fish_mg import FishLBMEnv3DMultigoal
from .starfish import Starfish3DLBMEnvMultigoal
from .clownfish import Clownfish3DLBMEnv
from .torus import Torus3DLBMEnv, TorusMultiTaskEnv
from .mjcf_parser import (
    parse_mjcf,
    parse_mjcf_to_meshes,
    parse_mjcf_as_urdf_format,
    get_body_world_positions,
)

__all__ = [
    # Core
    'HomeFlow3D',
    # Functions
    'InitBoundary3D',
    'InitFlow3D',
    'ResetSingleWorldFlow3D',
    'ResetSingleWorldSolidTransform3D',
    'ResetSingleWorldForces3D',
    'stream_and_collide_3d',
    'apply_bc_3d',
    'Swap_Mom_3D',
    'init_force_3d',
    # Solver
    'LBM_Solver3D',
    # Environments
    'LBMFluidEnv3D',
    'VecLBMFluidEnv3D',
    'FishLBMEnv3D',
    'FishLBMEnv3DMultigoal',
    'Starfish3DLBMEnvMultigoal',
    'Clownfish3DLBMEnv',
    'Torus3DLBMEnv',
    'TorusMultiTaskEnv',
    # Environment Kernels
    'extract_body_states_3d',
    'convert_and_update_solid_batch_3d',
    'extract_forces_torques_batch_3d',
    'fill_xfrc_3d_kernel',
    'extract_all_solid_positions_3d_kernel',
    # MJCF Parser
    'parse_mjcf',
    'parse_mjcf_to_meshes',
    'parse_mjcf_as_urdf_format',
    'get_body_world_positions',
]
