from .lbm_func import *
from .lbm_core import HomeFlow
import numpy as np
import mujoco
import mujoco_warp as mjw
from typing import Optional, Tuple, Dict
from .mujoco_to_lbm import extract_lbm_polygon_from_mujoco

class LBM_Solver:
    def __init__(self, nx: int, ny: int, solid_num: int = 1, nworld: int = 1, device=None):
        '''Initialize the LBM solver for multiple worlds.
        
        Args:
            nx, ny: Grid dimensions
            solid_num: Number of solid objects
            nworld: Number of parallel worlds (default: 1)
            device: Warp device to use
        '''
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

    def step(self):
        """Perform a single time step for all worlds using batch kernels."""
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
                                 model,  # mjw.MjModel (Warp) or mujoco.MjModel
                                 data,  # mjw.MjData (Warp) or mujoco.MjData
                                 body_or_geom_name: str,
                                 lbm_position: Optional[Tuple[float, float]] = None,
                                 lbm_scale: Optional[float] = 0.3,
                                 n_samples: int = 20,
                                 is_body: bool = True,
                                 **kwargs) -> Dict:
        """
        Create LBM solid from MuJoCo model for all worlds
        
        Note: model and data can be mujoco_warp or original mujoco types
        Internal conversion will be performed as needed
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
    
    def finalize_mappings(self, solid_ids: list):
        """
        Complete all solid creation and create Warp arrays for batch operations
        
        Args:
            solid_ids: List of solid IDs to use (in order)
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
