"""
3D LBM Fluid Environment with Multi-World Support

Base environment class for 3D LBM simulations with MuJoCo integration.
Uses gym (not gymnasium) for compatibility with dreamer_vec_wrapper.
All data processing uses Warp kernels - numpy only at entry/exit points.

Supports MJCF format only (no URDF).

Physical Unit Conversion:
    LBM uses dimensionless units (rho=1, dx=1, dt=1).
    Physical conversion is based on dimensional analysis:
    
    Force:  F_physical = F_lbm × rho_fluid × dx_physical^4 / dt_physical^2
    Torque: τ_physical = τ_lbm × rho_fluid × dx_physical^5 / dt_physical^2
    
    Where:
    - rho_fluid: fluid density (default: 1000 kg/m³ for water)
    - dx_physical: physical grid spacing = 1 / coordinate_scale (m)
    - dt_physical: MuJoCo timestep (s)
"""
import gym
from gym import spaces
import numpy as np
import warp as wp
import mujoco
import mujoco_warp as mjw
import trimesh
from typing import Optional, Tuple, List, Dict, Any

from .lbm_solver_3d import LBM_Solver3D
from .lbm_func_3d import (
    ResetSingleWorldFlow3D,
    ResetSingleWorldSolidTransform3D,
    ResetSingleWorldForces3D,
)
from .lbm_fluid_env_3d_func import (
    extract_body_states_3d,
    convert_and_update_solid_batch_3d,
    extract_forces_torques_physical_3d,
    fill_xfrc_3d_kernel,
    extract_all_solid_positions_3d_kernel,
)
from .mjcf_parser import parse_mjcf


class LBMFluidEnv3D(gym.Env):
    """
    Base 3D LBM Fluid Environment with multi-world support.
    
    This environment couples MuJoCo rigid body dynamics with 3D LBM fluid simulation.
    Supports parallel simulation of multiple worlds for vectorized training.
    
    Uses link_config for simplified configuration - only requires link_name and optional lbm_position.
    Automatically extracts mesh from MJCF using mjcf_parser.
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(
        self,
        mjcf_path: str,
        link_config: Optional[List[Dict[str, Any]]] = None,
        root_link: Optional[str] = None,
        root_position: Optional[Tuple[float, float, float]] = None,
        nx: int = 64,
        ny: int = 64,
        nz: int = 64,
        lbm_scale: float = 0.1,
        nworld: int = 1,
        max_episode_steps: int = 1000,
        per_frame_steps: int = 10,
        fluid_density: float = 1000.0,
        device: Optional[str] = None,
    ):
        """
        Initialize the 3D LBM Fluid Environment.
        
        Args:
            mjcf_path: Path to MuJoCo MJCF XML file
            link_config: Link configuration list (optional, auto-generated if not provided)
                Each entry: {'link_name': str, 'lbm_position': tuple, 'is_static': bool}
                is_static=True means the body has no joints and won't receive fluid forces.
            root_link: Name of root link for auto-positioning (optional)
            root_position: LBM grid position of root link (optional, default: center)
            nx: Number of lattice cells along the x axis.
            ny: Number of lattice cells along the y axis.
            nz: Number of lattice cells along the z axis.
            lbm_scale: Scale factor for geometry in LBM grid
            nworld: Number of parallel worlds
            max_episode_steps: Maximum steps per episode
            per_frame_steps: LBM-MuJoCo coupling iterations per environment step
            fluid_density: Physical fluid density in kg/m³ (default: 1000.0 for water)
            device: Warp device (None for auto-detect)
            
        Example (auto-generate with root_link):
            env = LBMFluidEnv3D(
                mjcf_path='model.xml',
                root_link='head',
                root_position=(32, 45, 32),
            )
            
        Example (explicit link_config with static obstacle):
            link_config = [
                {'link_name': 'head', 'lbm_position': (32, 45, 32)},
                {'link_name': 'body'},  # lbm_position auto-calculated
                {'link_name': 'obstacle', 'lbm_position': (32, 80, 32), 'is_static': True},
            ]
            env = LBMFluidEnv3D(mjcf_path='model.xml', link_config=link_config)
        """
        super().__init__()
        
        self.mjcf_path = mjcf_path
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.lbm_scale = lbm_scale
        self.nworld = nworld
        self.max_episode_steps = max_episode_steps
        self.per_frame_steps = per_frame_steps
        self.fluid_density = fluid_density  # Physical fluid density (kg/m³)
        self.coordinate_scale = lbm_scale * nx  # For coordinate conversion
        
        # Get device
        if device is None:
            device = wp.get_preferred_device()
        self.device = wp.get_device(device)
        
        # Parse MJCF to get body meshes
        self.mjcf_data = parse_mjcf(mjcf_path)
        self.body_meshes = {b['name']: b['combined_mesh'] for b in self.mjcf_data['bodies'] if b['combined_mesh'] is not None}
        
        # Load MuJoCo model
        self.mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        
        # ============== Physical Unit Conversion ==============
        self.mujoco_timestep = self.mj_model.opt.timestep  # seconds
        self.dx_physical = 1.0 / self.coordinate_scale  # physical grid spacing (m)
        
        # Force conversion: rho × dx^4 / dt^2
        self.force_conversion = (
            self.fluid_density 
            * (self.dx_physical ** 4) 
            / (self.mujoco_timestep ** 2)
        )
        
        # Torque conversion: rho × dx^5 / dt^2
        self.torque_conversion = (
            self.fluid_density 
            * (self.dx_physical ** 5) 
            / (self.mujoco_timestep ** 2)
        )
        
        # Create MuJoCo Warp models for parallel simulation
        self.mjw_model = mjw.put_model(self.mj_model)
        self.mjw_data = mjw.make_data(self.mj_model, nworld=nworld)
        
        # Get body information
        self.body_names = [self.mj_model.body(i).name for i in range(self.mj_model.nbody)]
        self.n_bodies = self.mj_model.nbody
        
        # Build set of body names that have joints (dynamic bodies)
        self._bodies_with_joints = self._find_dynamic_bodies()
        
        # Auto-generate or use provided link_config
        if link_config is None:
            self.link_config = self._auto_generate_link_config(root_link, root_position)
        else:
            self.link_config = link_config
        
        self.solid_num = len(self.link_config)
        
        # Build name to body_id mapping
        self._link_name_to_body_id = {}
        for i, name in enumerate(self.body_names):
            self._link_name_to_body_id[name] = i
        
        # Validate link_config and add body_id
        for i, cfg in enumerate(self.link_config):
            link_name = cfg['link_name']
            if link_name not in self._link_name_to_body_id:
                raise ValueError(f"Link '{link_name}' not found in MuJoCo model")
            cfg['body_id'] = self._link_name_to_body_id[link_name]
            cfg['solid_id'] = i
            # Default is_static to False if not specified
            if 'is_static' not in cfg:
                cfg['is_static'] = False
        
        # ============== Separate dynamic vs static solids ==============
        self.dynamic_link_config = [c for c in self.link_config if not c['is_static']]
        self.static_link_config = [c for c in self.link_config if c['is_static']]
        self.n_dynamic = len(self.dynamic_link_config)
        self.n_static = len(self.static_link_config)
        
        # Forward kinematics to initialize state
        mjw.forward(self.mjw_model, self.mjw_data)
        
        # Record initial solid positions/angles for reset (LBM grid coords)
        initial_solid_positions = np.array(
            [cfg.get("lbm_position", (nx * 0.5, ny * 0.5, nz * 0.5)) for cfg in self.link_config], 
            dtype=np.float32
        )
        initial_solid_quaternions = np.zeros((self.solid_num, 4), dtype=np.float32)
        initial_solid_quaternions[:, 0] = 1.0  # w=1, x=y=z=0
        self.init_solid_positions_wp = wp.array(initial_solid_positions, dtype=wp.vec3, device=self.device)
        self.init_solid_quaternions_wp = wp.array(initial_solid_quaternions, dtype=wp.vec4, device=self.device)
        
        # Initialize LBM solver
        self.lbm_solver = LBM_Solver3D(
            nx=nx, ny=ny, nz=nz,
            solid_num=self.solid_num,
            nworld=nworld,
            device=device
        )
        
        # ---- ALL solids (for LBM mesh update: position → LBM) ----
        self.body_ids_list = [cfg["body_id"] for cfg in self.link_config]
        self.solid_ids_list = [cfg["solid_id"] for cfg in self.link_config]
        self.body_ids_wp = wp.array(self.body_ids_list, dtype=wp.int32, device=self.device)
        self.solid_ids_wp = wp.array(self.solid_ids_list, dtype=wp.int32, device=self.device)
        
        # ---- DYNAMIC solids only (for force feedback: LBM → MuJoCo) ----
        self.dynamic_body_ids_list = [cfg["body_id"] for cfg in self.dynamic_link_config]
        self.dynamic_solid_ids_list = [cfg["solid_id"] for cfg in self.dynamic_link_config]
        self.dynamic_body_ids_wp = wp.array(self.dynamic_body_ids_list, dtype=wp.int32, device=self.device)
        self.dynamic_solid_ids_wp = wp.array(self.dynamic_solid_ids_list, dtype=wp.int32, device=self.device)
        
        # Pre-allocate position and quaternion buffers for ALL solids
        n_all = len(self.link_config)
        self.positions_buffer = wp.zeros((nworld, n_all, 3), dtype=wp.float32, device=self.device)
        self.quaternions_buffer = wp.zeros((nworld, n_all, 4), dtype=wp.float32, device=self.device)
        
        # Pre-allocate force and torque buffers for DYNAMIC solids only
        self.forces_buffer = wp.zeros((nworld, self.n_dynamic, 3), dtype=wp.float32, device=self.device)
        self.torques_buffer = wp.zeros((nworld, self.n_dynamic, 3), dtype=wp.float32, device=self.device)
        
        # Solid positions buffer for observation/termination (all solids)
        self.solid_positions_buffer = wp.zeros((nworld, n_all, 3), dtype=wp.float32, device=self.device)
        
        # Use CUDA Graph to accelerate single MuJoCo step
        self.mujoco_single_step_graph = None
        self.graph_initialized = False
        
        # Episode tracking
        self.step_counts = np.zeros(nworld, dtype=np.int32)
        self.episode_rewards = np.zeros(nworld, dtype=np.float32)
        
        # Define action space: MuJoCo controller input
        n_ctrl = self.mjw_model.nu
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.nworld, n_ctrl),
            dtype=np.float32
        )
        self.action_scale = 1.0
        
        # Define observation space (to be overridden by subclasses)
        self.observation_space = self._create_observation_space()
        
        # Coordinate mapping arrays (will be initialized by _reset_lbm_solver)
        self._mujoco_origins_wp = None  # (n_solids, 3) - all solids
        self._lbm_origins_wp = None  # (n_solids, 3) - all solids
        self._scales_wp = None  # (n_solids,) - all solids
    
    def _find_dynamic_bodies(self) -> set:
        """
        Identify which MuJoCo bodies are dynamic (have joints or are ancestors/descendants 
        of jointed bodies).
        
        A body is considered dynamic if:
        1. It directly has a joint, OR
        2. It is an ancestor of a body that has a joint (e.g. root body with freejoint), OR
        3. It is a descendant of a body that has a joint (child in kinematic chain)
        
        Bodies that are static (no joint in their kinematic subtree) will be treated as 
        obstacles in the LBM simulation - they block fluid but don't receive forces back.
        
        Returns:
            Set of body names that are dynamic.
        """
        # Find bodies that directly have joints
        bodies_with_joints = set()
        for j in range(self.mj_model.njnt):
            body_id = self.mj_model.jnt_bodyid[j]
            body_name = self.mj_model.body(body_id).name
            bodies_with_joints.add(body_name)
        
        # Start with jointed bodies
        dynamic_bodies = set(bodies_with_joints)
        
        # Mark all ancestors of jointed bodies as dynamic
        for body_name in list(bodies_with_joints):
            for i in range(self.mj_model.nbody):
                if self.mj_model.body(i).name == body_name:
                    parent_id = self.mj_model.body_parentid[i]
                    while parent_id > 0:  # 0 is worldbody
                        dynamic_bodies.add(self.mj_model.body(parent_id).name)
                        parent_id = self.mj_model.body_parentid[parent_id]
                    break
        
        # Mark all descendants of dynamic bodies as dynamic
        changed = True
        while changed:
            changed = False
            for i in range(self.mj_model.nbody):
                name = self.mj_model.body(i).name
                parent_id = self.mj_model.body_parentid[i]
                if parent_id > 0:
                    parent_name = self.mj_model.body(parent_id).name
                    if parent_name in dynamic_bodies and name not in dynamic_bodies:
                        dynamic_bodies.add(name)
                        changed = True
        
        return dynamic_bodies
    
    def _auto_generate_link_config(
        self,
        root_link: Optional[str] = None,
        root_position: Optional[Tuple[float, float, float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Auto-generate link_config from MJCF data.
        
        Uses the kinematic chain defined by joints to compute relative positions
        of each link, then converts to LBM grid coordinates.
        
        Args:
            root_link: Name of root link (default: first link with geometry)
            root_position: LBM grid position of root link (default: center of domain)
        """
        link_config = []
        bodies = self.mjcf_data['bodies']
        joints = self.mjcf_data['joints']
        
        if len(bodies) == 0:
            return link_config
        
        # Get all link names with geometry
        link_names = [b['name'] for b in bodies]
        
        # Determine root link
        if root_link is None:
            root_link = link_names[0]
        elif root_link not in link_names:
            raise ValueError(f"Root link '{root_link}' not found. Available: {link_names}")
        
        # Default root position: center of domain (slightly forward in Y)
        if root_position is None:
            root_position = (self.nx / 2, self.ny * 0.6, self.nz / 2)
        
        # Build parent-child relationships from joints
        parent_to_children = {}
        child_to_parent = {}
        
        for joint in joints:
            parent = joint.get('parent') or joint.get('body')
            child = joint.get('child') or joint.get('body')
            
            # For MJCF, use body hierarchy from parsed data
            pass
        
        # For MJCF, use the parent field from body data
        for body in bodies:
            parent = body.get('parent')
            if parent is not None:
                offset = body.get('pos', np.zeros(3))
                if parent not in parent_to_children:
                    parent_to_children[parent] = []
                parent_to_children[parent].append((body['name'], offset))
                child_to_parent[body['name']] = (parent, offset)
        
        # Compute world positions for each link (in MJCF meters)
        link_world_positions = {}
        link_world_positions[root_link] = np.array([0.0, 0.0, 0.0])
        
        # BFS to compute positions
        visited = {root_link}
        queue = [root_link]
        
        while queue:
            current = queue.pop(0)
            current_pos = link_world_positions[current]
            
            # Check children (forward kinematics)
            if current in parent_to_children:
                for child, offset in parent_to_children[current]:
                    if child not in visited and child in link_names:
                        link_world_positions[child] = current_pos + np.array(offset)
                        visited.add(child)
                        queue.append(child)
            
            # Check parent (backward)
            if current in child_to_parent:
                parent, offset = child_to_parent[current]
                if parent not in visited and parent in link_names:
                    link_world_positions[parent] = current_pos - np.array(offset)
                    visited.add(parent)
                    queue.append(parent)
        
        # Convert to LBM coordinates
        # Scale factor: lbm_scale * nx
        scale = self.lbm_scale * self.nx
        root_pos_array = np.array(root_position)
        
        for link_name in link_names:
            if link_name in link_world_positions:
                # Calculate LBM position relative to root
                mujoco_offset = link_world_positions[link_name]
                lbm_pos = root_pos_array + mujoco_offset * scale
                
                # Auto-detect static bodies (no joints in their kinematic subtree)
                is_static = link_name not in self._bodies_with_joints
                
                link_config.append({
                    'link_name': link_name,
                    'lbm_position': tuple(lbm_pos.tolist()),
                    'is_static': is_static,
                })
            else:
                # Body not reachable via BFS (e.g. isolated static obstacle).
                # Compute its LBM position from its MuJoCo world-frame xpos
                # relative to the root body.
                if link_name in self.body_names:
                    body_id = self.body_names.index(link_name)
                    body_mj_pos = self.mj_data.xpos[body_id]
                    root_body_id = self.body_names.index(root_link) if root_link in self.body_names else 0
                    root_mj_pos = self.mj_data.xpos[root_body_id]
                    mujoco_offset = body_mj_pos - root_mj_pos
                    lbm_pos = root_pos_array + mujoco_offset * scale
                    
                    link_config.append({
                        'link_name': link_name,
                        'lbm_position': tuple(lbm_pos.tolist()),
                        'is_static': True,  # isolated body = static
                    })
        
        # Sort: dynamic bodies first, static bodies last.
        # This ensures solid_position[0] = first dynamic body (e.g. head),
        # which many kernels rely on.
        dynamic_cfgs = [c for c in link_config if not c['is_static']]
        static_cfgs = [c for c in link_config if c['is_static']]
        link_config = dynamic_cfgs + static_cfgs
        
        return link_config
    
    def _create_observation_space(self) -> spaces.Space:
        """Create observation space (default returns qpos and qvel)."""
        nq = self.mjw_model.nq
        nv = self.mjw_model.nv
        obs_dim = nq + nv
        
        return spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.nworld, obs_dim),
            dtype=np.float32
        )
    
    def _get_obs(self) -> np.ndarray:
        """Get current observation (default returns qpos and qvel)."""
        obs = np.zeros((self.nworld, self.mjw_model.nq + self.mjw_model.nv), dtype=np.float32)
        qpos_all = self.mjw_data.qpos.numpy()
        qvel_all = self.mjw_data.qvel.numpy()
        for world_idx in range(self.nworld):
            obs[world_idx] = np.concatenate([qpos_all[world_idx], qvel_all[world_idx]])
        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information."""
        return {}
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> np.ndarray:
        """Reset environment."""
        if seed is not None:
            np.random.seed(seed)
        
        # Reset MuJoCo state using mjw.make_data for proper state reset
        self.mjw_data = mjw.make_data(self.mj_model, nworld=self.nworld)
        mjw.forward(self.mjw_model, self.mjw_data)
        
        # Reset LBM solver
        self._reset_lbm_solver()
        
        # Reset step counters for all worlds
        self.step_counts = np.zeros(self.nworld, dtype=np.int32)
        self.episode_rewards = np.zeros(self.nworld, dtype=np.float32)
        
        # Reset CUDA Graph (need to recapture after state change)
        self.graph_initialized = False
        self.mujoco_single_step_graph = None
        
        observation = self._get_obs()
        
        return observation
    
    def _reset_lbm_solver(self):
        """Reset LBM solver and create rigid bodies from MJCF meshes."""
        # Create temporary MuJoCo data for geometry extraction
        temp_data = mujoco.MjData(self.mj_model)
        mujoco.mj_forward(self.mj_model, temp_data)
        
        # Create new solver managing all worlds
        self.lbm_solver = LBM_Solver3D(
            nx=self.nx, ny=self.ny, nz=self.nz,
            solid_num=self.solid_num,
            nworld=self.nworld,
            device=self.device
        )
        
        # Arrays for coordinate mappings
        n_solids = len(self.link_config)
        mujoco_origins = np.zeros((n_solids, 3), dtype=np.float32)
        lbm_origins = np.zeros((n_solids, 3), dtype=np.float32)
        scales = np.zeros(n_solids, dtype=np.float32)
        
        # Create rigid bodies according to configuration
        for config in self.link_config:
            solid_id = config["solid_id"]
            link_name = config["link_name"]
            body_id = config["body_id"]
            lbm_position = config.get("lbm_position", (self.nx * 0.5, self.ny * 0.5, self.nz * 0.5))
            
            # Get mesh from parsed MJCF
            if link_name not in self.body_meshes:
                raise ValueError(f"No mesh found for link '{link_name}' in MJCF")
            mesh = self.body_meshes[link_name]
            
            # Get MuJoCo body position
            mj_pos = temp_data.xpos[body_id].copy()
            
            # Store coordinate mappings
            mujoco_origins[solid_id] = mj_pos
            lbm_origins[solid_id] = np.array(lbm_position, dtype=np.float32)
            scales[solid_id] = self.lbm_scale * self.nx
            
            # Create solid from mesh
            self.lbm_solver.create_solid_from_mesh(
                solid_id=solid_id,
                mesh=mesh,
                lbm_position=lbm_position,
                lbm_scale=self.lbm_scale,
                mujoco_origin=mj_pos,
            )
        
        # Initialize coordinate mapping arrays (use vec3 for origins)
        self._mujoco_origins_wp = wp.array(mujoco_origins, dtype=wp.vec3, device=self.device)
        self._lbm_origins_wp = wp.array(lbm_origins, dtype=wp.vec3, device=self.device)
        self._scales_wp = wp.array(scales, dtype=wp.float32, device=self.device)
        
        # Finalize mappings and create Warp arrays
        self.lbm_solver.finalize_mappings(self.solid_ids_list)
    
    def partial_reset(self, reset_mask: np.ndarray) -> np.ndarray:
        """
        Reset only specific worlds indicated by reset_mask.
        
        Args:
            reset_mask: Boolean array of shape (nworld,) where True indicates world needs reset
            
        Returns:
            np.ndarray: New observations for all worlds
        """
        if not np.any(reset_mask):
            return self._get_obs()
        
        # Convert reset_mask to int32 for Warp kernel
        reset_mask_int = reset_mask.astype(np.int32)
        reset_mask_wp = wp.array(reset_mask_int, dtype=wp.int32, device=self.device)
        
        # Reset MuJoCo states for indicated worlds using mjw.reset_data
        mjw.reset_data(self.mjw_model, self.mjw_data, wp.array(reset_mask_int, dtype=wp.bool, device=self.device))
        
        # Forward kinematics for all worlds
        mjw.forward(self.mjw_model, self.mjw_data)
        
        # Reset LBM flow field for indicated worlds
        wp.launch(
            ResetSingleWorldFlow3D,
            dim=(self.nworld, self.nx, self.ny, self.nz),
            inputs=[self.lbm_solver.flows_wp, reset_mask_wp],
            device=self.device,
        )
        
        # Reset solid mesh transforms for indicated worlds
        # In 3D, we use mesh transforms (not line segments), so dim is (nworld, n_objects)
        wp.launch(
            ResetSingleWorldSolidTransform3D,
            dim=(self.nworld, self.solid_num, 1),  # Third dim is just for kernel compatibility
            inputs=[self.lbm_solver.flows_wp, reset_mask_wp],
            device=self.device,
        )
        
        # Reset solid pose (LBM space) for indicated worlds
        for world_idx in range(self.nworld):
            if reset_mask[world_idx]:
                flow = self.lbm_solver.flows[world_idx]
                wp.copy(flow.solid_position, self.init_solid_positions_wp)
                wp.copy(flow.solid_quaternion, self.init_solid_quaternions_wp)
        
        # Reset solid forces for indicated worlds
        wp.launch(
            ResetSingleWorldForces3D,
            dim=self.nworld,
            inputs=[self.lbm_solver.flows_wp, reset_mask_wp],
            device=self.device,
        )
        
        # Reset step counters for indicated worlds
        for world_idx in range(self.nworld):
            if reset_mask[world_idx]:
                self.step_counts[world_idx] = 0
                self.episode_rewards[world_idx] = 0
        
        return self._get_obs()
    
    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Execute one step of environment interaction.
        
        Args:
            action: Action vector (nworld, action_dim)
            
        Returns:
            observation: New observation (nworld, obs_dim)
            reward: Reward (nworld,)
            done: Whether done (nworld,)
            info: Additional information
        """
        # Handle action shape - expect (nworld, action_dim)
        action = np.clip(action, self.action_space.low, self.action_space.high) * self.action_scale
        wp.copy(self.mjw_data.ctrl, wp.array(action, dtype=wp.float32, device=self.device))
        
        # Execute physics simulation step (including LBM-MuJoCo coupling)
        self._simulation_step()
        
        # Update step counts
        self.step_counts += 1
        
        # Get new observation
        observation = self._get_obs()
        
        # Check numerical stability
        instability_mask = (
            self._check_numerical_stability()
            if hasattr(self, "enable_stability_check") and self.enable_stability_check
            else None
        )
        
        # Convert NaN/Inf for network input
        if np.any(np.isnan(observation)) or np.any(np.isinf(observation)):
            observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Check termination condition
        terminated = self._is_terminated(instability_mask)
        truncated = np.array(self.step_counts >= self.max_episode_steps)
        
        # Compute reward
        reward = self._compute_reward(instability_mask)
        reward[terminated] -= 1.0
        
        # Combine terminated and truncated into done (gym API)
        done = terminated | truncated
        
        # Get additional information
        info = self._get_info()
        info["terminated"] = terminated
        info["truncated"] = truncated
        
        return observation, reward, done, info
    
    def _simulation_step(self):
        """
        Execute one complete physics simulation step.
        Including bidirectional coupling between LBM fluid simulation and MuJoCo rigid body dynamics.
        
        Static solids participate in LBM (block fluid, generate forces on fluid) but do NOT
        receive force feedback from LBM to MuJoCo (they are fixed obstacles).
        """
        n_all = len(self.link_config)
        
        for _ in range(self.per_frame_steps):
            # 1. Extract rigid body states from MuJoCo for ALL solids (dynamic + static)
            xipos_full = self.mjw_data.xipos  # (nworld, nbody, 3) - body COM position
            xquat_full = self.mjw_data.xquat  # (nworld, nbody, 4)
            
            wp.launch(
                extract_body_states_3d,
                dim=(self.nworld, n_all),
                inputs=[
                    xipos_full,
                    xquat_full,
                    self.body_ids_wp,
                    self.positions_buffer,
                    self.quaternions_buffer,
                ],
                device=self.device,
            )
            
            # 2. Update ALL solid positions in LBM (both dynamic and static need mesh updates)
            wp.launch(
                convert_and_update_solid_batch_3d,
                dim=(self.nworld, n_all),
                inputs=[
                    self.lbm_solver.flows_wp,
                    self.solid_ids_wp,
                    self.positions_buffer,
                    self.quaternions_buffer,
                    self._mujoco_origins_wp,
                    self._lbm_origins_wp,
                    self._scales_wp,
                ],
                device=self.device,
            )
            
            # 3. LBM fluid solver step for all worlds
            self.lbm_solver.step()
            
            # 4. Extract forces/torques only for DYNAMIC solids (not static obstacles)
            if self.n_dynamic > 0:
                wp.launch(
                    extract_forces_torques_physical_3d,
                    dim=(self.nworld, self.n_dynamic),
                    inputs=[
                        self.lbm_solver.flows_wp,
                        self.dynamic_solid_ids_wp,
                        self.force_conversion,
                        self.torque_conversion,
                        self.forces_buffer,
                        self.torques_buffer,
                    ],
                    device=self.device,
                )
            
            # 5. Apply fluid forces to MuJoCo only for DYNAMIC bodies
            self.mjw_data.xfrc_applied.zero_()
            
            if self.n_dynamic > 0:
                wp.launch(
                    fill_xfrc_3d_kernel,
                    dim=(self.nworld, self.n_dynamic),
                    inputs=[
                        self.mjw_data.xfrc_applied,
                        self.dynamic_body_ids_wp,
                        self.forces_buffer,
                        self.torques_buffer,
                    ],
                    device=self.device,
                )
            
            self.mjw_data.qfrc_applied.zero_()
            # Use xfrc_accumulate to convert spatial forces to generalized forces
            mjw.xfrc_accumulate(self.mjw_model, self.mjw_data, self.mjw_data.qfrc_applied)
            
            # 6. MuJoCo step (all worlds in parallel)
            if not self.graph_initialized:
                with wp.ScopedCapture() as capture:
                    mjw.step(self.mjw_model, self.mjw_data)
                self.graph_initialized = True
                self.mujoco_single_step_graph = capture.graph
            else:
                wp.capture_launch(self.mujoco_single_step_graph)
            wp.synchronize()
    
    def _compute_reward(self, instability_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute reward function. Override in subclasses."""
        raise NotImplementedError("Please implement reward function in subclass")
    
    def _is_terminated(self, instability_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Check whether to terminate. Override in subclasses."""
        raise NotImplementedError("Please implement termination condition in subclass")
    
    def _check_numerical_stability(self) -> Optional[np.ndarray]:
        """Check numerical stability. Override in subclasses if needed."""
        return None
    
    def _extract_solid_positions(self):
        """Extract all solid positions using Warp kernel."""
        n_solids = len(self.link_config)
        wp.launch(
            extract_all_solid_positions_3d_kernel,
            dim=(self.nworld, n_solids),
            inputs=[
                self.lbm_solver.flows_wp,
                self.solid_ids_wp,
                self.solid_positions_buffer,
            ],
            device=self.device,
        )
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render the first world when a renderer is implemented.

        Args:
            mode: Gym render mode. The base class accepts ``'human'`` and
                ``'rgb_array'`` but leaves rendering to subclasses or the
                JSON-driven realtime tools.

        Returns:
            ``None`` in the base implementation.
        """
        if mode == 'human':
            pass
        elif mode == 'rgb_array':
            pass
        return None
    
    def close(self) -> None:
        """Release renderer resources in subclasses."""
        pass
    
    @property
    def unwrapped(self) -> "LBMFluidEnv3D":
        """Return the environment without wrapper layers."""
        return self
