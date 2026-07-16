"""High-level two-dimensional LBM solver with parallel-world support."""

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import warp as wp

from .lbm_core import HomeFlow
from .lbm_func import (
    InitBoundary,
    InitFlow,
    Swap_Mom,
    apply_bc,
    init_force,
    precompute_transformed_segments,
    stream_and_collide,
)
from .mujoco_to_lbm import extract_lbm_polygon_from_mujoco


class LBM_Solver:
    """Advance one or more D2Q9 fluid worlds on a Warp device.

    Args:
        nx: Number of lattice cells along the x axis.
        ny: Number of lattice cells along the y axis.
        solid_num: Number of immersed solids per world.
        nworld: Number of worlds stepped together.
        device: Warp device name or device object. Defaults to Warp's preferred
            device.

    Attributes:
        flows: Per-world :class:`~envs.lbm.lbm_core.HomeFlow` objects.
        device: Warp device used for kernel launches.
        mujoco_mappings: Coordinate mappings created for immersed solids.

    Notes:
        The class name is retained for compatibility with existing configs.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        solid_num: int = 1,
        nworld: int = 1,
        device: Optional[Any] = None,
    ) -> None:
        """Initialize fluid fields, boundaries, and batch metadata.

        Args:
            nx: Number of lattice cells along the x axis.
            ny: Number of lattice cells along the y axis.
            solid_num: Number of immersed solids per world.
            nworld: Number of parallel simulation worlds.
            device: Warp device name or object. Uses the preferred device when
                omitted.
        """
        self.nx = nx
        self.ny = ny
        self.solid_num = solid_num
        self.nworld = nworld

        self.captured = False
        self.captured_graph = None
        
        # Initialize multiple flow objects (one per world)
        self.flows = [HomeFlow() for _ in range(nworld)]
        for flow in self.flows:
            flow.Initialize(nx, ny, n_objects=solid_num)
        
        self.total_steps = 500000
        self.per_frame_t = 500
        if device is None:
            device = wp.get_preferred_device()
        self.device = wp.get_device(device)
        
        # Create Warp array of flows for batch operations
        # (needs to be created before batch initialization kernels)
        self.flows_wp = wp.array(self.flows, dtype=HomeFlow, device=self.device)
        
        # Initialize boundaries and flows for all worlds using batch kernels
        wp.launch(InitBoundary, dim=(nworld, nx, ny), inputs=[self.flows_wp], device=self.device)
        wp.launch(InitFlow, dim=(nworld, nx, ny), inputs=[self.flows_wp], device=self.device)
        
        self.mujoco_mappings = {}  # {solid_id: {'mujoco_origin': array, 'lbm_origin': array, 'scale': float}}
        
        # Coordinate transformation parameters (shared across all worlds)
        # Will be initialized in finalize_mappings
        self.mujoco_origins_wp = None  # (n_bodies, 2)
        self.lbm_origins_wp = None     # (n_bodies, 2)
        self.scales_wp = None          # (n_bodies,)
        self.solid_ids_wp = None       # (n_bodies,)
        
        self.solid_id_to_index = {}  # {solid_id: index in arrays}

    def step(self) -> None:
        """Advance every world by one LBM time step.

        The first call captures the stream/collide and boundary-condition
        kernels into a Warp graph. Later calls replay that graph. Hydrodynamic
        forces are cleared before each step.
        """
        for flow in self.flows:
            init_force(flow)
        if not self.captured:
            with wp.ScopedCapture() as capture: 
                # Precompute transformed segments (batch kernel)
                wp.launch(
                    precompute_transformed_segments,
                    dim=(self.nworld, self.flows[0].n_objects, self.flows[0].max_segments_per_object),
                    inputs=[self.flows_wp],
                    device=self.device
                )
                
                # Stream and collide (batch kernel)
                wp.launch(
                    stream_and_collide,
                    dim=(self.nworld, self.nx, self.ny),
                    inputs=[self.flows_wp],
                    device=self.device
                )
                
                # Apply boundary conditions (batch kernel)
                wp.launch(
                    apply_bc,
                    dim=(self.nworld, self.nx, self.ny),
                    inputs=[self.flows_wp],
                    device=self.device
                )
                
                wp.launch(
                    Swap_Mom,
                    dim=(self.nworld,),
                    inputs=[self.flows_wp],
                    device=self.device
                )
            self.captured = True
            self.captured_graph = capture.graph
        else:
            wp.capture_launch(self.captured_graph)

    def create_solid_from_mujoco(self,
                                 solid_id: int,
                                 model: Any,
                                 data: Any,
                                 body_or_geom_name: str,
                                 lbm_position: Optional[Tuple[float, float]] = None,
                                 lbm_scale: Optional[float] = 0.3,
                                 n_samples: int = 20,
                                 is_body: bool = True,
                                 **kwargs: Any) -> Dict[str, Any]:
        """Project one 3D MuJoCo body or geometry into every 2D LBM world.

        Args:
            solid_id: Object index in ``[0, solid_num)``.
            model: MuJoCo or MuJoCo-Warp model containing the source body or
                geometry.
            data: State associated with ``model``.
            body_or_geom_name: MuJoCo body or geometry name to project.
            lbm_position: Initial ``(x, y)`` center in lattice coordinates.
                Defaults to the grid center.
            lbm_scale: MuJoCo-to-LBM scale relative to ``nx``.
            n_samples: Number of vertices in the resampled polygon.
            is_body: Interpret ``body_or_geom_name`` as a body when true and a
                geometry when false.
            **kwargs: Additional solid properties forwarded to
                :meth:`HomeFlow.configure_solid`.

        Returns:
            Planar projection metadata including normalized polygon vertices,
            source position, and orientation.

        Notes:
            Call :meth:`finalize_mappings` after all solids have been created.
        """
        # Extract polygon information (this function needs original MuJoCo objects)
        polygon_info = extract_lbm_polygon_from_mujoco(
            model, data, body_or_geom_name, 
            n_samples=n_samples,
            use_convex_hull=True,
            normalize=True,
            is_body=is_body
        )
        
        # If LBM position not specified, use grid center
        if lbm_position is None:
            lbm_position = (self.nx * 0.5, self.ny * 0.5)
        
        # Configure solid for all flows (all worlds have same solid configuration)
        for flow in self.flows:
            flow.configure_solid(
                solid_id=solid_id,
                lines=polygon_info['vertices'],
                position=lbm_position,
                angle=polygon_info['angle'],
                scale=lbm_scale,
                **kwargs
            )
        
        # Store MuJoCo mapping information
        mujoco_scale = lbm_scale * self.nx
        self.mujoco_mappings[solid_id] = {
            'mujoco_origin': polygon_info['position'].copy(),  # Initial position in MuJoCo
            'lbm_origin': np.array(lbm_position, dtype=np.float32),  # Initial position in LBM
            'scale': mujoco_scale
        }
        
        return polygon_info
    
    def finalize_mappings(self, solid_ids: Sequence[int]) -> None:
        """Finalize coordinate-mapping arrays for batch coupling.
        
        Args:
            solid_ids: Solid IDs in the same order used by the MuJoCo coupling
                buffers.

        Raises:
            ValueError: If a requested solid was not created through
                :meth:`create_solid_from_mujoco`.
        """
        self.n = len(solid_ids)
        
        mujoco_origins = []
        lbm_origins = []
        scales = []
        
        for idx, solid_id in enumerate(solid_ids):
            if solid_id not in self.mujoco_mappings:
                raise ValueError(f"Solid {solid_id} not initialized via create_solid_from_mujoco")
            
            mapping = self.mujoco_mappings[solid_id]
            mujoco_origins.append(mapping['mujoco_origin'][:2])  # (2,)
            lbm_origins.append(mapping['lbm_origin'][:2])  # (2,)
            scales.append(mapping['scale'])
            self.solid_id_to_index[solid_id] = idx
        
        # Create Warp arrays (shared across all worlds)
        self.mujoco_origins_wp = wp.array(np.array(mujoco_origins, dtype=np.float32), dtype=wp.float32, device=self.device)
        self.lbm_origins_wp = wp.array(np.array(lbm_origins, dtype=np.float32), dtype=wp.float32, device=self.device)
        self.scales_wp = wp.array(np.array(scales, dtype=np.float32), dtype=wp.float32, device=self.device)
        self.solid_ids_wp = wp.array(solid_ids, dtype=wp.int32, device=self.device)
        
        self.flows_wp = wp.array(self.flows, dtype=HomeFlow, device=self.device)
