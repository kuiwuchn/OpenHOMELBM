"""Core data structures for the two-dimensional D2Q9 LBM solver.

The :class:`HomeFlow` Warp struct owns the fluid fields and rigid-boundary
state for one simulation world. Grid arrays use the project-native
``(nx, ny)`` storage order; rendering helpers transpose them for display.
"""

from typing import Any, Optional

import warp as wp

from .solid_func import generate_unit_circle_points

__all__ = ["HomeFlow"]


@wp.struct
class HomeFlow:
    """Store the D2Q9 fluid and immersed-solid state for one 2D world.

    Allocate the arrays by calling :meth:`Initialize` before launching LBM
    kernels. Each :class:`~envs.lbm.lbm_solver.LBM_Solver` owns one instance
    per parallel world.
    """

    nx: int  # number of grid cells in x direction
    ny: int  # number of grid cells in y direction

    flag: wp.array2d(dtype=wp.int32)

    # 6 variants each node (1+2+3)
    rho: wp.array2d(dtype=wp.float32) # density（1）
    u: wp.array2d(dtype=wp.vec2) # flow.uocity（2）
    Sxx: wp.array2d(dtype=wp.float32) # Sxx, Eq.(29)
    Syy: wp.array2d(dtype=wp.float32) # Syy, Eq.(29)
    Sxy: wp.array2d(dtype=wp.float32) # Sxy, Eq.(29)

    rho_post: wp.array2d(dtype=wp.float32)
    u_post: wp.array2d(dtype=wp.vec2)
    Sxx_post: wp.array2d(dtype=wp.float32)
    Syy_post: wp.array2d(dtype=wp.float32)
    Sxy_post: wp.array2d(dtype=wp.float32)

    w_d2q9: wp.types.vector(length=9, dtype=wp.float32)
    cx_d2q9: wp.types.vector(length=9, dtype=wp.float32)
    cy_d2q9: wp.types.vector(length=9, dtype=wp.float32)
    indexd2q9Inv_gpu: wp.types.vector(length=9, dtype=wp.int32)

    forcex: wp.array2d(dtype=wp.float32) # force in x direction
    forcey: wp.array2d(dtype=wp.float32) # force in y direction
    vis_shear: wp.float32
    
    bc_type : wp.types.vector(length=4, dtype=wp.int32)
    bc_value : wp.array1d(dtype=wp.vec2)

    #------------solid info--------------#
    n_objects: int  # number of objects
    max_segments_per_object: int  # maximum number of line segments per object
    
    # Per-object arrays (size: n_objects)
    # linear_v: wp.array(dtype=wp.vec2)
    # angle_v: wp.array(dtype=wp.float32)
    solid_forcex: wp.array(dtype=wp.float32) # force in x direction
    solid_forcey: wp.array(dtype=wp.float32) # force in y direction
    torque: wp.array(dtype=wp.float32)
    mass: wp.array(dtype=wp.float32)  # Changed to array for per-object mass
    MoI: wp.array(dtype=wp.float32)  # Changed to array for per-object MoI
    
    solid_position: wp.array(dtype=wp.vec2)
    solid_angle: wp.array(dtype=wp.float32)
    solid_scale: wp.array(dtype=wp.float32)  # Changed to array for per-object scale
    solid_mass_center: wp.array(dtype=wp.vec2)  # Changed to array for per-object mass center
    solid_max_radius: wp.array(dtype=wp.float32)  # Changed to array for per-object max radius
    
    # Geometry arrays (2D: [n_objects, max_segments_per_object])
    solid_line: wp.array2d(dtype=wp.vec2)  # Original line segments for each object
    solid_line_transformed: wp.array2d(dtype=wp.vec2)  # Transformed line segments
    solid_line_transformed_last: wp.array2d(dtype=wp.vec2)  # Transformed line segments in last step
    solid_line_num: wp.array(dtype=wp.int32)  # Number of segments per object
    #-------------------------------------#
    time_step: float
    grid_length: float
    velocity_scale: float

    u_img: wp.array2d(dtype=wp.float32) # velocity magnitude or vorticity field
    u_img_rgb: wp.array2d(dtype=wp.vec3) # 3-channel velocity field (magnitude, ux, uy)
    small_size: float
    small_u_img: wp.array2d(dtype=wp.float32) # flow.uocity（2）
    
    def Initialize(self, nx: int, ny: int, n_objects: int = 1) -> None:
        """Allocate fluid and rigid-body buffers for one simulation world.

        Args:
            nx: Number of lattice cells along the x axis.
            ny: Number of lattice cells along the y axis.
            n_objects: Number of immersed rigid objects whose forces and
                transforms are tracked independently.

        Notes:
            Arrays are allocated on Warp's current device. The solver launches
            initialization kernels after constructing all worlds.
        """
        self.time_step = 1.0
        self.nx, self.ny = nx, ny
        self.grid_length = 1.0/nx
        self.velocity_scale = self.time_step / self.grid_length
        
        self.flag = wp.zeros((nx, ny), dtype=wp.int32)

        self.rho = wp.ones((nx, ny), dtype=wp.float32) 
        self.u = wp.zeros((nx, ny), dtype=wp.vec2)
        self.Sxx = wp.zeros((nx, ny), dtype=wp.float32)
        self.Syy = wp.zeros((nx, ny), dtype=wp.float32)
        self.Sxy = wp.zeros((nx, ny), dtype=wp.float32)
        
        self.rho_post = wp.ones((nx, ny), dtype=wp.float32)
        self.u_post = wp.zeros((nx, ny), dtype=wp.vec2)
        self.Sxx_post = wp.zeros((nx, ny), dtype=wp.float32)
        self.Syy_post = wp.zeros((nx, ny), dtype=wp.float32)
        self.Sxy_post = wp.zeros((nx, ny), dtype=wp.float32)

        self.u_img = wp.zeros((nx, ny), dtype=wp.float32)
        self.u_img_rgb = wp.zeros((nx, ny), dtype=wp.vec3)
        self.small_size = 5.0
        self.small_u_img = wp.zeros((int(nx/self.small_size), int(ny/self.small_size)), dtype=wp.float32)

        self.w_d2q9 = wp.types.vector(length=9, dtype=wp.float32)(4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36)
        self.cx_d2q9 = wp.types.vector(length=9, dtype=wp.float32)(0, 1, -1, 0, 0, 1, -1, -1, 1)
        self.cy_d2q9 = wp.types.vector(length=9, dtype=wp.float32)(0, 0, 0, 1, -1, 1, 1, -1, -1)
        self.indexd2q9Inv_gpu = wp.types.vector(length=9, dtype=wp.int32)(0, 2, 1, 4, 3, 7, 8, 5, 6)
        self.forcex = wp.zeros((nx, ny), dtype=wp.float32)
        self.forcey = wp.zeros((nx, ny), dtype=wp.float32)
        # self.vis_shear = 0.00255
        # self.vis_shear = 0.001        # self.vis_shear = 1.0
        self.vis_shear = 0.01
        self.bc_type = wp.types.vector(length=4, dtype=wp.int32)(1,1,1,1)
        self.bc_value = wp.array((wp.vec2(0., 0.),  # left
                                        wp.vec2(0., -0.0), # top
                                        wp.vec2(0., 0.),  # right
                                        wp.vec2(0., 0.), # bottom
                                        ), dtype=wp.vec2)

        
        # Initialize multi-object support
        self.n_objects = n_objects
        
        # Allocate per-object arrays
        # self.linear_v = wp.zeros(n_objects, dtype=wp.vec2)
        # self.angle_v = wp.zeros(n_objects, dtype=wp.float32)
        self.solid_forcex = wp.zeros(n_objects, dtype=wp.float32)
        self.solid_forcey = wp.zeros(n_objects, dtype=wp.float32)
        self.torque = wp.zeros(n_objects, dtype=wp.float32)
        self.mass = wp.zeros(n_objects, dtype=wp.float32)
        self.MoI = wp.zeros(n_objects, dtype=wp.float32)
        
        self.solid_position = wp.zeros(n_objects, dtype=wp.vec2)
        self.solid_angle = wp.zeros(n_objects, dtype=wp.float32)
        self.solid_scale = wp.zeros(n_objects, dtype=wp.float32)
        self.solid_mass_center = wp.zeros(n_objects, dtype=wp.vec2)
        self.solid_max_radius = wp.zeros(n_objects, dtype=wp.float32)
        self.solid_line_num = wp.zeros(n_objects, dtype=wp.int32)
        
        # Initialize with default circle geometry for all objects
        # This will be overridden by configure_solid method
        self.max_segments_per_object = 10  # Default, will be updated
        self.solid_line = wp.zeros((n_objects, self.max_segments_per_object), dtype=wp.vec2)
        self.solid_line_transformed = wp.zeros((n_objects, self.max_segments_per_object), dtype=wp.vec2)
        self.solid_line_transformed_last = wp.zeros((n_objects, self.max_segments_per_object), dtype=wp.vec2)
        
        # print(f"Initialized HomeFlow with {n_objects} objects")
    
    def configure_solid(
        self,
        solid_id: int,
        lines: Any,
        position: Any,
        angle: float = 0.0,
        scale: float = 0.3,
        mass: float = 100000.0,
        mass_center: Optional[Any] = None,
    ) -> None:
        """Configure one immersed solid from a closed 2D polygon.
        
        Args:
            solid_id: Object index in ``[0, n_objects)``.
            lines: Polygon vertices with shape ``(n_segments, 2)``. Accepts a
                Warp array, NumPy array, or array-like object.
            position: Initial center in LBM lattice coordinates as ``(x, y)``.
            angle: Initial counter-clockwise angle in radians.
            scale: Geometry scale, conventionally expressed relative to
                ``nx`` by the MuJoCo-to-LBM mapping.
            mass: Rigid-body mass used when deriving the moment of inertia.
            mass_center: Optional local center of mass. Defaults to ``(0, 0)``.

        Raises:
            ValueError: If ``solid_id`` is outside the allocated object range.

        Notes:
            This method may reallocate all polygon buffers when the new solid
            has more vertices than the current shared capacity. It synchronizes
            Warp before returning.
        """
        import numpy as np
        import math
        
        if solid_id < 0 or solid_id >= self.n_objects:
            raise ValueError(f"solid_id {solid_id} out of range [0, {self.n_objects})")
        
        # Convert lines to numpy array
        if isinstance(lines, wp.array):
            lines_np = lines.numpy()
        elif isinstance(lines, np.ndarray):
            lines_np = lines
        else:
            lines_np = np.array(lines, dtype=np.float32)
        
        num_segments = len(lines_np)
        
        # Update max_segments_per_object if needed
        if num_segments > self.max_segments_per_object:
            # print(f"Expanding max_segments_per_object from {self.max_segments_per_object} to {num_segments}")
            old_max = self.max_segments_per_object
            
            self.max_segments_per_object = num_segments
            new_solid_line = wp.zeros((self.n_objects, num_segments), dtype=wp.vec2)
            new_solid_line_transformed = wp.zeros((self.n_objects, num_segments), dtype=wp.vec2)
            new_solid_line_transformed_last = wp.zeros((self.n_objects, num_segments), dtype=wp.vec2)
            
            # Copy existing data using kernel
            wp.launch(copy_solid_lines_kernel, dim=(self.n_objects, old_max), 
                     inputs=[self.solid_line, new_solid_line, old_max])
            
            self.solid_line = new_solid_line
            self.solid_line_transformed = new_solid_line_transformed
            self.solid_line_transformed_last = new_solid_line_transformed_last
        
        # Create warp array from lines
        lines_array = wp.array(lines_np, dtype=wp.vec2)
        
        # Process mass center
        if mass_center is None:
            mc = wp.vec2(0.0, 0.0)
        elif isinstance(mass_center, wp.vec2):
            mc = mass_center
        else:
            mc = wp.vec2(float(mass_center[0]), float(mass_center[1]))
        
        # Use a conservative squared radius because cut-cell tests compare
        # squared distances in lattice coordinates.
        max_dist = 0.0
        for i in range(num_segments):
            p = lines_np[i]
            dist = ((p[0] - mc[0])**2 + (p[1] - mc[1])**2)**0.5
            if dist > max_dist:
                max_dist = dist
        max_radius = (max_dist + math.sqrt(2.0))**2
        
        # Process position
        if isinstance(position, tuple) or isinstance(position, list):
            pos = wp.vec2(float(position[0]), float(position[1]))
        elif isinstance(position, wp.vec2):
            pos = position
        else:
            pos = wp.vec2(float(position[0]), float(position[1]))
        
        moi = float(mass) * (float(scale) * float(self.nx))**2.0 / 8.0
        
        # Launch kernels to configure this solid
        # First, copy line segments
        wp.launch(copy_line_segments_kernel, dim=num_segments, inputs=[
            self, solid_id, lines_array
        ])
        
        # Then, set all other properties
        wp.launch(configure_solid_properties_kernel, dim=1, inputs=[
            self, solid_id, num_segments, pos, mc, max_radius,
            float(angle), float(scale), float(mass), moi
        ])
        
        wp.synchronize()


@wp.kernel
def copy_solid_lines_kernel(old_lines: wp.array2d(dtype=wp.vec2), 
                            new_lines: wp.array2d(dtype=wp.vec2),
                            old_max: int):
    i, j = wp.tid()
    if j < old_max:
        new_lines[i, j] = old_lines[i, j]

@wp.kernel
def copy_line_segments_kernel(flow: HomeFlow, 
                              solid_id: int,
                              lines: wp.array(dtype=wp.vec2)):
    idx = wp.tid()
    flow.solid_line[solid_id, idx] = lines[idx]

@wp.kernel
def configure_solid_properties_kernel(flow: HomeFlow, 
                                     solid_id: int,
                                     num_segments: int,
                                     position: wp.vec2,
                                     mass_center: wp.vec2,
                                     max_radius: float,
                                     angle: float,
                                     scale: float,
                                     mass: float,
                                     moi: float):
    tid = wp.tid()
    if tid == 0:
        # Set all the properties for this solid
        flow.solid_line_num[solid_id] = num_segments
        flow.solid_position[solid_id] = position
        flow.solid_mass_center[solid_id] = mass_center
        flow.solid_max_radius[solid_id] = max_radius
        flow.solid_angle[solid_id] = angle
        flow.solid_scale[solid_id] = scale
        flow.mass[solid_id] = mass
        flow.MoI[solid_id] = moi
