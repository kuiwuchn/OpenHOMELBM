"""
3D LBM Core Data Structure with Multi-World Support

This module defines the HomeFlow3D struct for 3D LBM simulation
with support for multiple parallel worlds (nworld).
"""
import warp as wp
import numpy as np


# D3Q27 lattice constants - use wp.types.vector instead of wp.constant(wp.array(...))
cx_d3q27 = wp.types.vector(length=27, dtype=wp.float32)(
    0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1
)

cy_d3q27 = wp.types.vector(length=27, dtype=wp.float32)(
    0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1
)

cz_d3q27 = wp.types.vector(length=27, dtype=wp.float32)(
    0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1
)

w_d3q27 = wp.types.vector(length=27, dtype=wp.float32)(
    8.0/27.0,
    2.0/27.0, 2.0/27.0, 2.0/27.0, 2.0/27.0, 2.0/27.0, 2.0/27.0,
    1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0,
    1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0,
    1.0/216.0, 1.0/216.0, 1.0/216.0, 1.0/216.0,
    1.0/216.0, 1.0/216.0, 1.0/216.0, 1.0/216.0
)

indexd3q27Inv = wp.types.vector(length=27, dtype=wp.int32)(
    0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24, 23, 26, 25
)

cs2 = wp.constant(wp.float32(1.0/3.0))

ML_FLUID = wp.constant(wp.int32(0))
ML_SOLID = wp.constant(wp.int32(1))
ML_WALL = wp.constant(wp.int32(2))


@wp.struct
class HomeFlow3D:
    """3D LBM flow field data structure for a single world."""
    nx: int
    ny: int
    nz: int
    
    time_step: float
    grid_length: float
    velocity_scale: float

    bc_type: wp.types.vector(length=6, dtype=wp.int32)
    bc_value: wp.array1d(dtype=wp.vec3)

    flag: wp.array(dtype=wp.int32, ndim=3)
    
    # Flow field variables
    rho: wp.array(dtype=wp.float32, ndim=3)
    u: wp.array(dtype=wp.vec3, ndim=3)
    Sxx: wp.array(dtype=wp.float32, ndim=3)
    Syy: wp.array(dtype=wp.float32, ndim=3)
    Szz: wp.array(dtype=wp.float32, ndim=3)
    Sxy: wp.array(dtype=wp.float32, ndim=3)
    Sxz: wp.array(dtype=wp.float32, ndim=3)
    Syz: wp.array(dtype=wp.float32, ndim=3)

    rho_post: wp.array(dtype=wp.float32, ndim=3)
    u_post: wp.array(dtype=wp.vec3, ndim=3)
    Sxx_post: wp.array(dtype=wp.float32, ndim=3)
    Syy_post: wp.array(dtype=wp.float32, ndim=3)
    Szz_post: wp.array(dtype=wp.float32, ndim=3)
    Sxy_post: wp.array(dtype=wp.float32, ndim=3)
    Sxz_post: wp.array(dtype=wp.float32, ndim=3)
    Syz_post: wp.array(dtype=wp.float32, ndim=3)

    # Per-cell force fields
    forcex: wp.array(dtype=wp.float32, ndim=3)
    forcey: wp.array(dtype=wp.float32, ndim=3)
    forcez: wp.array(dtype=wp.float32, ndim=3)

    vis_shear: wp.float32

    # Multi-object support
    n_objects: int
    
    # Per-object arrays
    mass: wp.array(dtype=wp.float32, ndim=1)
    MoI: wp.array(dtype=wp.float32, ndim=1)
    
    # Mesh state - per object
    mesh_ids: wp.array(dtype=wp.uint64, ndim=1)
    mesh_transforms: wp.array(dtype=wp.transform, ndim=1)
    mesh_transforms_last: wp.array(dtype=wp.transform, ndim=1)
    mesh_transforms_initialized: wp.array(dtype=wp.int32, ndim=1)
    mesh_scale_sizes: wp.array(dtype=wp.vec3, ndim=1)
    
    linear_v: wp.array(dtype=wp.vec3, ndim=1)
    angle_v: wp.array(dtype=wp.vec3, ndim=1)

    solid_force: wp.array(dtype=wp.vec3, ndim=1)
    solid_position: wp.array(dtype=wp.vec3, ndim=1)
    solid_torque: wp.array(dtype=wp.vec3, ndim=1)
    solid_quaternion: wp.array(dtype=wp.vec4, ndim=1)
    solid_inertia: wp.array(dtype=wp.mat33, ndim=1)
    solid_inertia_inv: wp.array(dtype=wp.mat33, ndim=1)

    # Render buffers
    u_img_xy: wp.array2d(dtype=wp.float32)  # XY plane (top-down view, z slice)
    u_img_xz: wp.array2d(dtype=wp.float32)  # YZ plane (side view from right, x slice)  
    u_img_xz_front: wp.array2d(dtype=wp.float32)  # XZ plane (front view, y slice)

    def Initialize(self, nx, ny, nz, n_objects=1):
        """
        Initialize the 3D LBM flow field.
        
        Args:
            nx, ny, nz: Grid dimensions
            n_objects: Number of solid objects
        """
        self.nx, self.ny, self.nz = nx, ny, nz
        self.n_objects = n_objects

        self.time_step = 1.0
        self.grid_length = 1.0 / nx
        self.velocity_scale = self.time_step / self.grid_length

        self.vis_shear = 0.1
        self.bc_type = wp.types.vector(length=6, dtype=wp.int32)(1, 1, 1, 1, 1, 1)
        self.bc_value = wp.array((
            wp.vec3(0., 0., 0.),  # left (x=0)
            wp.vec3(0., 0., 0.),  # right (x=nx-1)
            wp.vec3(0., 0., 0.),  # top (y=ny-1)
            wp.vec3(0., 0., 0.),  # bottom (y=0)
            wp.vec3(0., 0., 0.),  # front (z=nz-1)
            wp.vec3(0., 0., 0.),  # back (z=0)
        ), dtype=wp.vec3)


        self.flag = wp.zeros((nx, ny, nz), dtype=wp.int32)

        self.rho = wp.ones((nx, ny, nz), dtype=wp.float32)
        self.u = wp.zeros((nx, ny, nz), dtype=wp.vec3)
        self.Sxx = wp.zeros((nx, ny, nz), dtype=wp.float32)
        self.Syy = wp.zeros((nx, ny, nz), dtype=wp.float32)
        self.Szz = wp.zeros((nx, ny, nz), dtype=wp.float32)
        self.Sxy = wp.zeros((nx, ny, nz), dtype=wp.float32)
        self.Sxz = wp.zeros((nx, ny, nz), dtype=wp.float32)
        self.Syz = wp.zeros((nx, ny, nz), dtype=wp.float32)

        self.rho_post = wp.ones((nx, ny, nz), dtype=wp.float32)
        self.u_post = wp.zeros((nx, ny, nz), dtype=wp.vec3)
        self.Sxx_post = wp.zeros((nx, ny, nz), dtype=wp.float32)
        self.Syy_post = wp.zeros((nx, ny, nz), dtype=wp.float32)
        self.Szz_post = wp.zeros((nx, ny, nz), dtype=wp.float32)
        self.Sxy_post = wp.zeros((nx, ny, nz), dtype=wp.float32)
        self.Sxz_post = wp.zeros((nx, ny, nz), dtype=wp.float32)
        self.Syz_post = wp.zeros((nx, ny, nz), dtype=wp.float32)

        self.forcex = wp.zeros((nx, ny, nz), dtype=wp.float32)
        self.forcey = wp.zeros((nx, ny, nz), dtype=wp.float32)
        self.forcez = wp.zeros((nx, ny, nz), dtype=wp.float32)

        # Multi-object arrays
        self.mass = wp.zeros((n_objects,), dtype=wp.float32)
        self.MoI = wp.zeros((n_objects,), dtype=wp.float32)

        self.mesh_ids = wp.zeros((n_objects,), dtype=wp.uint64)
        self.mesh_transforms = wp.zeros((n_objects,), dtype=wp.transform)
        self.mesh_transforms_last = wp.zeros((n_objects,), dtype=wp.transform)
        self.mesh_transforms_initialized = wp.zeros((n_objects,), dtype=wp.int32)
        self.mesh_scale_sizes = wp.ones((n_objects,), dtype=wp.vec3)

        self.linear_v = wp.zeros((n_objects,), dtype=wp.vec3)
        self.angle_v = wp.zeros((n_objects,), dtype=wp.vec3)

        self.solid_force = wp.zeros((n_objects,), dtype=wp.vec3)
        self.solid_position = wp.zeros((n_objects,), dtype=wp.vec3)
        self.solid_torque = wp.zeros((n_objects,), dtype=wp.vec3)
        self.solid_quaternion = wp.zeros((n_objects,), dtype=wp.vec4)
        self.solid_inertia = wp.zeros((n_objects,), dtype=wp.mat33)
        self.solid_inertia_inv = wp.zeros((n_objects,), dtype=wp.mat33)

        # Render buffers
        self.u_img_xy = wp.zeros((nx, ny), dtype=wp.float32)  # XY plane slice (top-down view)
        self.u_img_xz = wp.zeros((ny, nz), dtype=wp.float32)  # YZ plane slice (side view from right)
        self.u_img_xz_front = wp.zeros((nx, nz), dtype=wp.float32)  # XZ plane slice (front view)
