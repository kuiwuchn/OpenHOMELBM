"""
3D LBM Solver with Multi-World Support

This module provides the LBM_Solver3D class for parallel 3D LBM simulations.
"""
import warp as wp
import numpy as np
import trimesh
from typing import Any, Dict, Optional, Sequence, Tuple

from .lbm_core_3d import HomeFlow3D
from .lbm_func_3d import (
    InitBoundary3D, InitFlow3D, ResetSingleWorldFlow3D,
    stream_and_collide_3d, apply_bc_3d, Swap_Mom_3D, init_force_3d,
    init_force_3d_batch
)


class LBM_Solver3D:
    """Advance one or more D3Q27 fluid worlds on a Warp device.

    Args:
        nx: Number of lattice cells along the x axis.
        ny: Number of lattice cells along the y axis.
        nz: Number of lattice cells along the z axis.
        solid_num: Number of immersed meshes per world.
        nworld: Number of worlds stepped together.
        device: Warp device name or device object. Defaults to Warp's preferred
            device.

    Notes:
        Mesh geometry and MuJoCo-to-LBM coordinate mappings are shared across
        worlds; positions, orientations, and hydrodynamic loads are per-world.
    """
    
    def __init__(self, nx: int, ny: int, nz: int,
                 solid_num: int = 1, nworld: int = 1,
                 device: Optional[Any] = None) -> None:
        """Initialize D3Q27 fields, boundaries, and batch metadata.
        
        Args:
            nx: Number of lattice cells along the x axis.
            ny: Number of lattice cells along the y axis.
            nz: Number of lattice cells along the z axis.
            solid_num: Number of immersed meshes per world.
            nworld: Number of parallel simulation worlds.
            device: Warp device name or object.
        """
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.solid_num = solid_num
        self.nworld = nworld
        
        self.captured = False
        self.captured_graph = None
        
        if device is None:
            device = wp.get_preferred_device()
        self.device = wp.get_device(device)
        
        # Initialize multiple flow objects (one per world)
        self.flows = [HomeFlow3D() for _ in range(nworld)]
        for flow in self.flows:
            flow.Initialize(nx, ny, nz, n_objects=solid_num)
        
        # Create Warp array of flows for batch operations
        self.flows_wp = wp.array(self.flows, dtype=HomeFlow3D, device=self.device)
        
        # Initialize boundaries and flows for all worlds
        wp.launch(InitBoundary3D, dim=(nworld, nx, ny, nz), inputs=[self.flows_wp], device=self.device)
        wp.launch(InitFlow3D, dim=(nworld, nx, ny, nz), inputs=[self.flows_wp], device=self.device)
        
        # Per-object mesh data (shared across all worlds)
        self.meshes = [None] * solid_num
        self.mesh_wps = [None] * solid_num
        
        # MuJoCo mappings
        self.mujoco_mappings = {}
        
        # Coordinate transformation parameters
        self.mujoco_origins_wp = None
        self.lbm_origins_wp = None
        self.scales_wp = None
        self.solid_ids_wp = None
        self.solid_id_to_index = {}
    
    def step(self) -> None:
        """Advance every world by one LBM time step.

        Clear per-solid forces, then capture or replay the Warp graph containing
        stream/collide, boundary-condition, and moment-swap kernels.
        """
        # Clear forces for all worlds in parallel
        wp.launch(
            init_force_3d_batch,
            dim=(self.nworld,),
            inputs=[self.flows_wp],
            device=self.device
        )
        
        if not self.captured:
            with wp.ScopedCapture() as capture:
                wp.launch(
                    stream_and_collide_3d,
                    dim=(self.nworld, self.nx, self.ny, self.nz),
                    inputs=[self.flows_wp],
                    device=self.device
                )
                
                wp.launch(
                    apply_bc_3d,
                    dim=(self.nworld, self.nx, self.ny, self.nz),
                    inputs=[self.flows_wp],
                    device=self.device
                )
                
                wp.launch(
                    Swap_Mom_3D,
                    dim=(self.nworld,),
                    inputs=[self.flows_wp],
                    device=self.device
                )
            self.captured = True
            self.captured_graph = capture.graph
        else:
            wp.capture_launch(self.captured_graph)
    
    def create_solid_from_mesh(
        self,
        solid_id: int,
        mesh: trimesh.Trimesh,
        lbm_position: Optional[Tuple[float, float, float]] = None,
        lbm_scale: float = 0.15,
        init_quaternion: Tuple[float, float, float, float] = (1, 0, 0, 0),
        mujoco_origin: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Create the same triangular mesh solid in every LBM world.
        
        Args:
            solid_id: Object index in ``[0, solid_num)``.
            mesh: Triangular mesh expressed in its body-local frame.
            lbm_position: Initial center ``(x, y, z)`` in lattice coordinates.
                Defaults to the domain center.
            lbm_scale: Mesh scale relative to ``nx``.
            init_quaternion: Initial ``(w, x, y, z)`` orientation.
            mujoco_origin: Optional MuJoCo body origin used by later coordinate
                conversion. Defaults to the zero vector.
            
        Returns:
            Mesh metadata containing the local mesh, scaled vertices, effective
            lattice scale, initial position, and quaternion.

        Notes:
            The mesh is deliberately not recentered: its body-local origin must
            remain aligned with the MuJoCo joint or body pivot.
        """
        # Keep mesh vertices in the original MJCF body-local frame.
        #
        # For articulated models, the body origin is the joint
        # anchor/pivot, while the mesh geometry is already authored relative to
        # that pivot in the MJCF local frame. Re-centering to center_mass breaks
        # the hinge anchor and makes child links appear to stretch / grow when
        # lbm_scale changes because link offsets and geometry origins no longer
        # match.
        mesh_local = mesh.copy()
        
        # Calculate actual scale
        lbm_scale_actual = lbm_scale * self.nx
        scaled_vertices = mesh_local.vertices * lbm_scale_actual
        
        # Default position is center of domain
        if lbm_position is None:
            lbm_position = (self.nx * 0.5, self.ny * 0.5, self.nz * 0.5)
        
        # Store mesh in body-local frame
        self.meshes[solid_id] = mesh_local
        
        # Create warp mesh
        vertices = np.array(scaled_vertices, dtype=np.float32)
        indices = np.array(mesh_local.faces, dtype=np.int32).flatten()

        # Bounding-sphere radius (in LBM cell units) for narrow-band culling.
        # Vertices are already scaled and expressed in the body-local frame whose
        # origin is the mesh transform's rotation center, so the max vertex norm
        # is a rotation-invariant conservative bound on the solid's extent.
        if vertices.shape[0] > 0:
            bound_radius = float(np.linalg.norm(vertices, axis=1).max())
        else:
            bound_radius = 0.0
        
        with wp.ScopedDevice(self.device):
            pos_wp = wp.array(vertices, requires_grad=True, dtype=wp.vec3)
            indices_wp = wp.array(indices, dtype=wp.int32)
            mesh_wp = wp.Mesh(points=pos_wp, indices=indices_wp)
            self.mesh_wps[solid_id] = mesh_wp
        
        # Update all flows with mesh info
        for flow in self.flows:
            mesh_ids_np = flow.mesh_ids.numpy()
            mesh_ids_np[solid_id] = mesh_wp.id
            flow.mesh_ids = wp.array(mesh_ids_np, dtype=wp.uint64)
            
            scale_sizes_np = flow.mesh_scale_sizes.numpy()
            scale_sizes_np[solid_id] = (1.0, 1.0, 1.0)
            flow.mesh_scale_sizes = wp.array(scale_sizes_np, dtype=wp.vec3)
            
            positions_np = flow.solid_position.numpy()
            positions_np[solid_id] = lbm_position
            flow.solid_position = wp.array(positions_np, dtype=wp.vec3)
            
            quaternions_np = flow.solid_quaternion.numpy()
            quaternions_np[solid_id] = init_quaternion
            flow.solid_quaternion = wp.array(quaternions_np, dtype=wp.vec4)

            bound_radius_np = flow.solid_bound_radius.numpy()
            bound_radius_np[solid_id] = bound_radius
            flow.solid_bound_radius = wp.array(bound_radius_np, dtype=wp.float32)

            # Update mesh transform
            self._update_mesh_transform(flow, solid_id)
        
        # Update flows_wp
        self.flows_wp = wp.array(self.flows, dtype=HomeFlow3D, device=self.device)
        
        # Store MuJoCo mapping
        if mujoco_origin is None:
            mujoco_origin = np.zeros(3, dtype=np.float32)
        
        self.mujoco_mappings[solid_id] = {
            'mujoco_origin': np.array(mujoco_origin, dtype=np.float32),
            'lbm_origin': np.array(lbm_position, dtype=np.float32),
            'scale': lbm_scale_actual,
        }
        
        return {
            'mesh': mesh_local,
            'vertices': scaled_vertices,
            'scale': lbm_scale_actual,
            'lbm_position': lbm_position,
            'quaternion': init_quaternion,
        }
    
    def _update_mesh_transform(self, flow: HomeFlow3D, solid_id: int):
        """Update mesh transform for a specific solid in a flow."""
        pos = flow.solid_position.numpy()[solid_id]
        quat = flow.solid_quaternion.numpy()[solid_id]  # w, x, y, z
        
        transforms_np = flow.mesh_transforms.numpy()
        transforms_last_np = flow.mesh_transforms_last.numpy()
        initialized_np = flow.mesh_transforms_initialized.numpy()
        
        new_transform = wp.transform(
            wp.vec3(pos[0], pos[1], pos[2]),
            wp.quat(quat[1], quat[2], quat[3], quat[0])
        )
        
        if initialized_np[solid_id] == 0:
            transforms_np[solid_id] = new_transform
            transforms_last_np[solid_id] = new_transform
            initialized_np[solid_id] = 1
        else:
            transforms_last_np[solid_id] = transforms_np[solid_id]
            transforms_np[solid_id] = new_transform
        
        flow.mesh_transforms = wp.array(transforms_np, dtype=wp.transform)
        flow.mesh_transforms_last = wp.array(transforms_last_np, dtype=wp.transform)
        flow.mesh_transforms_initialized = wp.array(initialized_np, dtype=wp.int32)
    
    def update_solids_batch(
        self,
        world_idx: int,
        solid_ids: Sequence[int],
        positions: np.ndarray,
        quaternions: np.ndarray
    ) -> None:
        """Update the poses of multiple solids in one world.
        
        Args:
            world_idx: Target world index.
            solid_ids: Solid IDs corresponding to the first array axis.
            positions: MuJoCo positions with shape ``(N, 3)``.
            quaternions: MuJoCo quaternions with shape ``(N, 4)`` in
                ``(w, x, y, z)`` order.
        """
        flow = self.flows[world_idx]
        
        lbm_positions_np = flow.solid_position.numpy()
        lbm_quats_np = flow.solid_quaternion.numpy()
        
        for i, solid_id in enumerate(solid_ids):
            if solid_id in self.mujoco_mappings:
                mapping = self.mujoco_mappings[solid_id]
                mujoco_pos = positions[i]
                lbm_pos = (mujoco_pos - mapping['mujoco_origin']) * mapping['scale'] + mapping['lbm_origin']
                lbm_positions_np[solid_id] = lbm_pos
            else:
                lbm_positions_np[solid_id] = positions[i]
            
            lbm_quats_np[solid_id] = quaternions[i]
        
        flow.solid_position = wp.array(lbm_positions_np, dtype=wp.vec3)
        flow.solid_quaternion = wp.array(lbm_quats_np, dtype=wp.vec4)
        
        for solid_id in solid_ids:
            self._update_mesh_transform(flow, solid_id)
    
    def get_forces_and_torques(
        self,
        world_idx: int,
        solid_ids: Optional[Sequence[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get forces and torques on solids for a specific world.
        
        Args:
            world_idx: Source world index.
            solid_ids: Solid IDs to read. Defaults to all configured solids.
            
        Returns:
            A ``(forces, torques)`` pair. Both arrays have shape ``(N, 3)`` and
            are converted from lattice coordinates when a MuJoCo mapping is
            available.
        """
        if solid_ids is None:
            solid_ids = list(range(self.solid_num))
        
        flow = self.flows[world_idx]
        n = len(solid_ids)
        forces = np.zeros((n, 3), dtype=np.float32)
        torques = np.zeros((n, 3), dtype=np.float32)
        
        solid_forces = flow.solid_force.numpy()
        solid_torques = flow.solid_torque.numpy()
        
        for i, solid_id in enumerate(solid_ids):
            if solid_id in self.mujoco_mappings:
                scale = self.mujoco_mappings[solid_id]['scale']
                forces[i] = solid_forces[solid_id] / scale
                # Torque needs to be divided by scale² (force × distance)
                torques[i] = solid_torques[solid_id] / (scale * scale)
            else:
                forces[i] = solid_forces[solid_id]
                torques[i] = solid_torques[solid_id]
        
        return forces, torques
    
    def reset_world(self, world_idx: int) -> None:
        """Reset one world's fluid field and mesh-transform history."""
        reset_mask = np.zeros(self.nworld, dtype=np.int32)
        reset_mask[world_idx] = 1
        reset_mask_wp = wp.array(reset_mask, dtype=wp.int32, device=self.device)
        
        wp.launch(
            ResetSingleWorldFlow3D,
            dim=(self.nworld, self.nx, self.ny, self.nz),
            inputs=[self.flows_wp, reset_mask_wp],
            device=self.device
        )
        
        # Reset mesh transform initialized flag
        flow = self.flows[world_idx]
        initialized_np = flow.mesh_transforms_initialized.numpy()
        initialized_np[:] = 0
        flow.mesh_transforms_initialized = wp.array(initialized_np, dtype=wp.int32)
    
    def reset_worlds(self, world_indices: Sequence[int]) -> None:
        """Reset selected worlds' fluid fields and transform histories."""
        reset_mask = np.zeros(self.nworld, dtype=np.int32)
        for idx in world_indices:
            reset_mask[idx] = 1
        reset_mask_wp = wp.array(reset_mask, dtype=wp.int32, device=self.device)
        
        wp.launch(
            ResetSingleWorldFlow3D,
            dim=(self.nworld, self.nx, self.ny, self.nz),
            inputs=[self.flows_wp, reset_mask_wp],
            device=self.device
        )
        
        for idx in world_indices:
            flow = self.flows[idx]
            initialized_np = flow.mesh_transforms_initialized.numpy()
            initialized_np[:] = 0
            flow.mesh_transforms_initialized = wp.array(initialized_np, dtype=wp.int32)
    
    def finalize_mappings(self, solid_ids: Sequence[int]) -> None:
        """Finalize coordinate-mapping arrays for batch coupling.
        
        Args:
            solid_ids: Solid IDs in the same order used by coupling buffers.

        Raises:
            ValueError: If a requested solid has no MuJoCo mapping.
        """
        self.n = len(solid_ids)
        
        mujoco_origins = []
        lbm_origins = []
        scales = []
        
        for idx, solid_id in enumerate(solid_ids):
            if solid_id not in self.mujoco_mappings:
                raise ValueError(f"Solid {solid_id} not initialized")
            
            mapping = self.mujoco_mappings[solid_id]
            mujoco_origins.append(mapping['mujoco_origin'])
            lbm_origins.append(mapping['lbm_origin'])
            scales.append(mapping['scale'])
            self.solid_id_to_index[solid_id] = idx
        
        self.mujoco_origins_wp = wp.array(np.array(mujoco_origins, dtype=np.float32), dtype=wp.float32, device=self.device)
        self.lbm_origins_wp = wp.array(np.array(lbm_origins, dtype=np.float32), dtype=wp.float32, device=self.device)
        self.scales_wp = wp.array(np.array(scales, dtype=np.float32), dtype=wp.float32, device=self.device)
        self.solid_ids_wp = wp.array(solid_ids, dtype=wp.int32, device=self.device)
        
        # Refresh flows_wp
        self.flows_wp = wp.array(self.flows, dtype=HomeFlow3D, device=self.device)
