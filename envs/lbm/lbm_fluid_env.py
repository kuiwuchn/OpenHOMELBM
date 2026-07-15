import gym
from gym import spaces
import numpy as np
import mujoco
import mujoco_warp as mjw
import warp as wp
from .lbm_solver import LBM_Solver
from .lbm_func import get_u_img, precompute_transformed_segments
from typing import Optional, Tuple, Dict, Any, List

from concurrent.futures import ThreadPoolExecutor, as_completed
from .lbm_fluid_env_func import *


class LBMFluidEnv(gym.Env):
    def __init__(
        self,
        xml_path: str,
        solid_config: List[Dict[str, Any]],
        nx: int = 400,
        ny: int = 600,
        lbm_scale: float = 0.2,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 1000,
        per_frame_steps: int = 500,
        nworld: int = 1,
    ):
        """
        Initialize LBM fluid simulation environment

        Args:
            xml_path: MuJoCo XML model path
            solid_config: Rigid body configuration list, each element is a dict
            nx, ny: LBM grid dimensions
            lbm_scale: MuJoCo to LBM scaling ratio
            render_mode: Rendering mode ('human', 'rgb_array', None)
            max_episode_steps: Maximum steps per episode
            per_frame_steps: Sub-simulation steps per environment step (LBM-MuJoCo coupling iterations)
            nworld: Number of parallel environments (default: 1)

        Example:
            solid_config = [
                {
                    'solid_id': 0,
                    'body_id': 1,
                    'body_or_geom_name': 'center',
                    'lbm_position': (200, 300),
                },
                # ... more rigid bodies
            ]
        """
        super().__init__()

        self.xml_path = xml_path
        self.solid_config = solid_config
        self.nx = nx
        self.ny = ny
        self.lbm_scale = lbm_scale
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.per_frame_steps = per_frame_steps
        self.nworld = nworld

        self.solid_num = len(solid_config)

        # Render options
        self.render_vorticity = False  # True: render vorticity, False: velocity field
        self.render_solid_boundary = False  # True: render solid bounds only

        # Record initial solid positions/angles for reset (LBM grid coords)
        initial_solid_positions = np.array(
            [cfg["lbm_position"] for cfg in solid_config], dtype=np.float32
        )
        initial_solid_angles = np.zeros(self.solid_num, dtype=np.float32)
        self.init_solid_positions_wp = wp.array(initial_solid_positions, dtype=wp.vec2)
        self.init_solid_angles_wp = wp.array(initial_solid_angles, dtype=wp.float32)

        # Get or set default device
        self.device = wp.get_device("cuda:0")

        # Load MuJoCo model
        mujoco_model = mujoco.MjModel.from_xml_path(
            xml_path
        )  # Save original model for geometry extraction
        self.mujoco_model = mujoco_model  # Save original model
        self.model = mjw.put_model(mujoco_model)  # Warp model
        self.data = mjw.make_data(
            mujoco_model, nworld
        )  # Warp data with nworld batches
        
        # Initialize single LBM solver managing all worlds
        self.solver = LBM_Solver(nx=nx, ny=ny, solid_num=self.solid_num, nworld=nworld)

        # Forward kinematics to initialize state
        mjw.forward(self.model, self.data)

        # Pre-create Warp array for body_ids and solid_ids (avoid repeated creation and list operations)
        self.body_ids_list = [cfg["body_id"] for cfg in solid_config]
        self.solid_ids_list = [cfg["solid_id"] for cfg in solid_config]
        self.body_ids_wp = wp.array(self.body_ids_list, dtype=wp.int32)
        self.solid_ids_wp = wp.array(self.solid_ids_list, dtype=wp.int32)

        # Pre-allocate position and quaternion buffers as 3D arrays for all worlds
        n_bodies = len(solid_config)
        self.positions_buffer = wp.zeros((nworld, n_bodies, 3), dtype=wp.float32)
        self.quaternions_buffer = wp.zeros((nworld, n_bodies, 4), dtype=wp.float32)

        # Use CUDA Graph to accelerate single MuJoCo step (capture once, replay in loop)
        self.mujoco_single_step_graph = None  # Single-step CUDA Graph object
        self.graph_initialized = False  # Whether Graph is initialized

        # Episode counter (one per world)
        self.current_steps = np.zeros(nworld, dtype=np.int32)

        # Rendering related
        self.frames = []  # Store rendered frames (for video saving)
        self.video_path = "lbm_episode.mp4"  # Video save path

        # Rendering mode control
        self.real_time_rendering = (
            False  # True: real-time display, False: collect frames for video
        )

        # Video rendering parameters (customizable)
        self.video_fps = 20  # Video frame rate
        self.video_dpi = 300  # Video resolution
        self.video_cmap = "magma"  # Velocity field colormap
        self.vorticity_cmap = "RdBu_r"  # Vorticity colormap (red-white-blue)
        self.video_vmin = 0.0  # Velocity field minimum
        self.video_vmax = 0.05  # Velocity field maximum
        self.video_interpolation = "antialiased"  # Interpolation method
        self.video_figsize = (8, 14)  # Figure size (width, height)

        # Real-time visualization related
        self.fig = None
        self.ax = None
        self.img_plot = None
        self.cbar = None
        self.info_text = None
        self.render_initialized = False

        # Define action space: MuJoCo controller input
        # Default uses all available control inputs
        n_ctrl = self.model.nu  # Control input dimension
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.nworld, n_ctrl), dtype=np.float32
        )
        self.action_scale = 1.0

        # Define observation space
        self.observation_space = self._create_observation_space()

    def _create_observation_space(self) -> spaces.Space:
        """
        Create observation space (default returns [obs_before, obs_after] = 2× (qpos + qvel))

        Returns:
            gym.spaces.Space: Observation space definition
        """
        # Default observation: [obs_before_action, obs_after_action]
        nq = self.model.nq  # Generalized coordinate dimension
        nv = self.model.nv  # Generalized velocity dimension
        obs_dim = 2 * (nq + nv)  # 2× for temporal stacking

        return spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.nworld, obs_dim), dtype=np.float32
        )

    def _get_obs(self) -> np.ndarray:
        """
        Get current observation (default returns qpos and qvel)

        Returns:
            np.ndarray: Observation array with shape (nworld, obs_dim)
        """
        obs = np.zeros((self.nworld, self.model.nq + self.model.nv), dtype=np.float32)
        qpos_all = self.data.qpos.numpy()
        qvel_all = self.data.qvel.numpy()
        for world_idx in range(self.nworld):
            obs[world_idx] = np.concatenate(
                [
                    qpos_all[world_idx],
                    qvel_all[world_idx],
                ]
            )

        return obs

    def _get_info(self) -> Dict[str, Any]:
        """
        Get additional information

        Returns:
            Dict[str, Any]: Additional information dictionary
        """
        return {}

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Reset environment

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            observation: Initial observation
        """
        if seed is not None:
            np.random.seed(seed)

        # Reset MuJoCo state using mjw.make_data for proper state reset
        self.data = mjw.make_data(
            self.mujoco_model, self.nworld
        )  # Warp data with nworld batches
        mjw.forward(self.model, self.data)
        
        # Reset LBM solver
        self._reset_lbm_solver()

        # Reset step counters for all worlds
        self.current_steps = np.zeros(self.nworld, dtype=np.int32)

        # Reset CUDA Graph (need to recapture after state change)
        self.graph_initialized = False
        self.mujoco_single_step_graph = None

        # Clear rendering frame buffer
        self.frames = []

        observation = self._get_obs()
        observation = np.concatenate([observation, observation], axis=1)

        return observation

    def _reset_lbm_solver(self):
        """Reset LBM solver and create rigid bodies"""
        # Create temporary MuJoCo data for geometry extraction
        temp_data = mujoco.MjData(self.mujoco_model)
        mujoco.mj_forward(self.mujoco_model, temp_data)

        # Create new solver managing all worlds
        self.solver = LBM_Solver(
            nx=self.nx, ny=self.ny, solid_num=self.solid_num, nworld=self.nworld
        )

        # Create rigid bodies according to configuration
        for config in self.solid_config:
            solid_id = config["solid_id"]
            body_or_geom_name = config["body_or_geom_name"]
            lbm_position = config["lbm_position"]
            is_body = config.get("is_body", True)  # Default to body if not specified

            self.solver.create_solid_from_mujoco(
                solid_id=solid_id,
                model=self.mujoco_model,  # Use original MuJoCo model
                data=temp_data,  # Use temporary data
                body_or_geom_name=body_or_geom_name,
                lbm_position=lbm_position,
                lbm_scale=self.lbm_scale,
                n_samples=int(config.get("n_samples", 20)),
                is_body=is_body,
            )


        # Finalize mappings and create Warp arrays
        solid_ids_list = [cfg["solid_id"] for cfg in self.solid_config]
        self.solver.finalize_mappings(solid_ids_list)
        self._precompute_initial_solid_boundary()

    def _precompute_initial_solid_boundary(self) -> None:
        """Build transformed solid boundary at reset time with zero initial boundary velocity."""
        wp.launch(
            precompute_transformed_segments,
            dim=(
                self.solver.nworld,
                self.solver.flows[0].n_objects,
                self.solver.flows[0].max_segments_per_object,
            ),
            inputs=[self.solver.flows_wp],
            device=self.solver.device,
        )
        wp.synchronize()

    def partial_reset(self, reset_mask: np.ndarray) -> np.ndarray:

        """
        Reset only specific worlds indicated by reset_mask.
        This is called when some environments terminate during training.

        Args:
            reset_mask: Boolean array of shape (nworld,) where True indicates world needs reset

        Returns:
            np.ndarray: New observations for all worlds (reset worlds get new initial obs)
        """
        from .lbm_func import (
            ResetSingleWorldFlow,
            ResetSingleWorldSolidTransform,
            ResetSingleWorldForces,
        )

        if not np.any(reset_mask):
            # No worlds need reset, just return current observation
            return self._get_obs()

        # Convert reset_mask to int32 for Warp kernel
        reset_mask_int = reset_mask.astype(np.int32)
        reset_mask_wp = wp.array(reset_mask_int, dtype=wp.int32)

        # Reset MuJoCo states for indicated worlds using mjw.reset_data
        mjw.reset_data(self.model, self.data, wp.array(reset_mask_int, dtype=wp.bool))
        mjw.forward(self.model, self.data)

        # Reset LBM flow field for indicated worlds
        wp.launch(
            ResetSingleWorldFlow,
            dim=(self.nworld, self.nx, self.ny),
            inputs=[self.solver.flows_wp, reset_mask_wp],
        )

        # Reset solid transformed positions for indicated worlds
        max_segments = self.solver.flows[0].max_segments_per_object
        wp.launch(
            ResetSingleWorldSolidTransform,
            dim=(self.nworld, self.solid_num, max_segments),
            inputs=[self.solver.flows_wp, reset_mask_wp],
        )

        # Reset solid pose (LBM space) for indicated worlds
        for world_idx in range(self.nworld):
            if reset_mask[world_idx]:
                flow = self.solver.flows[world_idx]
                wp.copy(flow.solid_position, self.init_solid_positions_wp)
                wp.copy(flow.solid_angle, self.init_solid_angles_wp)

        # Reset solid forces for indicated worlds
        wp.launch(
            ResetSingleWorldForces,
            dim=self.nworld,
            inputs=[self.solver.flows_wp, reset_mask_wp],
        )

        # Reset step counters for indicated worlds
        for world_idx in range(self.nworld):
            if reset_mask[world_idx]:
                self.current_steps[world_idx] = 0

        # Return updated observation
        observation = self._get_obs()
        observation = np.concatenate([observation, observation], axis=1)
        return observation

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Execute one step of environment interaction

        Args:
            action: Action vector (nworld, action_dim)

        Returns:
            observation: New observation (shape depends on nworld)
            reward: Reward (array of shape (nworld,))
            done: Whether done (array of shape (nworld,))
            info: Additional information
        """
        # Handle action shape - expect (nworld, action_dim)
        action = (
            np.clip(action, self.action_space.low, self.action_space.high)
            * self.action_scale
        )
        wp.copy(self.data.ctrl, wp.array(action, dtype=wp.float32))

        # Get observation BEFORE action execution (for temporal stacking)
        observation_before = self._get_obs()

        # Execute physics simulation step (including LBM-MuJoCo coupling)
        self._simulation_step()

        # Update step counts
        self.current_steps += 1

        # Get observation AFTER action execution
        observation_after = self._get_obs()

        # Stack [obs_before, obs_after] for temporal information
        observation = np.concatenate((observation_before, observation_after), axis=1)

        # Check numerical stability ONCE before computing reward and termination
        # This must be done before nan_to_num conversion
        instability_mask = (
            self._check_numerical_stability()
            if hasattr(self, "enable_stability_check") and self.enable_stability_check
            else None
        )

        # Now safe to convert NaN/Inf for network input
        if np.any(np.isnan(observation)) or np.any(np.isinf(observation)):
            observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)

        # Check termination condition (pass instability mask to avoid recomputation)
        terminated = self._is_terminated(instability_mask)
        truncated = np.array(self.current_steps >= self.max_episode_steps)

        # Compute reward (pass instability mask to avoid recomputation)
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
        Execute one complete physics simulation step
        Including bidirectional coupling between LBM fluid simulation and MuJoCo rigid body dynamics
        """
        # Perform coupled simulation over per_frame_steps sub-steps
        n_bodies = len(self.solid_config)

        for _ in range(self.per_frame_steps):
            # 1. Extract rigid body states from MuJoCo for ALL worlds (2D kernel launch)
            xipos_full = self.data.xipos  # (nworld, nbody, 3) - body COM position
            xquat_full = self.data.xquat  # (nworld, nbody, 4)

            # Use 2D Warp kernel to extract data for all worlds in parallel
            wp.launch(
                extract_body_states,
                dim=(self.nworld, n_bodies),
                inputs=[
                    xipos_full,
                    xquat_full,
                    self.body_ids_wp,
                    self.positions_buffer,
                    self.quaternions_buffer,
                ],
            )

            # 2. Update rigid body positions in all LBM solvers using 2D kernel
            wp.launch(
                convert_and_update_solid_batch_2d,
                dim=(self.nworld, n_bodies),
                inputs=[
                    self.solver.flows_wp,
                    self.solid_ids_wp,
                    self.positions_buffer,
                    self.quaternions_buffer,
                    self.solver.mujoco_origins_wp,
                    self.solver.lbm_origins_wp,
                    self.solver.scales_wp,
                ],
            )

            # 3. LBM fluid solver step for all worlds
            self.solver.step()

            # 4. Get fluid forces and torques from all LBM solvers using 2D kernel
            # Pre-allocate 3D arrays for forces and torques on first call
            if not hasattr(self, "forces_buffer"):
                self.forces_buffer = wp.zeros(
                    (self.nworld, n_bodies, 3), dtype=wp.float32
                )
                self.torques_buffer = wp.zeros(
                    (self.nworld, n_bodies, 3), dtype=wp.float32
                )

            wp.launch(
                extract_forces_torques_batch,
                dim=(self.nworld, n_bodies),
                inputs=[
                    self.solver.flows_wp,
                    self.solid_ids_wp,
                    self.solver.scales_wp,
                    self.forces_buffer,
                    self.torques_buffer,
                ],
            )

            # 5. Apply fluid forces to MuJoCo using 2D kernels
            self.data.xfrc_applied.zero_()

            wp.launch(
                fill_xfrc_kernel,
                dim=(self.nworld, n_bodies),
                inputs=[
                    self.data.xfrc_applied,
                    self.body_ids_wp,
                    self.forces_buffer,
                    self.torques_buffer,
                ],
            )

            self.data.qfrc_applied.zero_()
            # Use xfrc_accumulate to convert spatial forces to generalized forces
            mjw.xfrc_accumulate(self.model, self.data, self.data.qfrc_applied)

            # 6. MuJoCo step (all worlds in parallel)
            if not self.graph_initialized:
                with wp.ScopedCapture() as capture:
                    mjw.step(self.model, self.data)
                self.graph_initialized = True
                self.mujoco_single_step_graph = capture.graph
            else:
                wp.capture_launch(self.mujoco_single_step_graph)
            wp.synchronize()

    def _compute_reward(self, instability_mask: Optional[np.ndarray] = None) -> float:
        """
        Compute reward function

        Args:
            instability_mask: Pre-computed instability mask (optional, for subclass use)

        Returns:
            float: Reward value
        """
        NotImplementedError("Please implement reward function in subclass")

    def _is_terminated(self, instability_mask: Optional[np.ndarray] = None) -> bool:
        """
        Check whether to terminate

        Args:
            instability_mask: Pre-computed instability mask (optional, for subclass use)

        Returns:
            bool: Whether to terminate
        """
        NotImplementedError("Please implement termination condition in subclass")

    def render(self):
        """
        Render environment

        Based on render flags:
        - render_solid_boundary=True: render solid boundaries
        - render_vorticity=True: render vorticity field
        - Default: render velocity field magnitude

        Returns:
            Depending on render_mode:
            - "rgb_array": returns image array
            - "human": real-time display or collect frames
        """
        if self.render_mode is None:
            return None

        from .lbm_func import get_vorticity_img, get_u_img, get_solid_boundary_img

        # Always render the first world (world_idx=0)
        solver = self.solver
        flow = solver.flows[0]

        if self.render_solid_boundary:
            # Compute solid boundaries, result saved to u_img.
            wp.launch(
                get_solid_boundary_img, dim=(flow.nx, flow.ny), inputs=[flow, 0.0]
            )
            wp.synchronize()
        elif self.render_vorticity:
            # Compute vorticity field, result saved to u_img
            wp.launch(get_vorticity_img, dim=(flow.nx, flow.ny), inputs=[flow])
            wp.synchronize()
        else:
            # Compute velocity field magnitude, result saved to u_img
            wp.launch(get_u_img, dim=(flow.nx, flow.ny), inputs=[flow])
            wp.synchronize()

        # Read data uniformly from u_img
        img = flow.u_img.numpy().T  # shape: (ny, nx)

        if self.render_mode == "rgb_array":
            return img
        elif self.render_mode == "human":
            if self.real_time_rendering:
                self._render_realtime(img)
            else:
                # Collect frames mode
                self.frames.append(img.copy())
            return None

        return None

    def _render_realtime(self, img):
        """
        Display flow field image in real-time

        Based on render flags:
        - render_solid_boundary=True: solid boundary (binary map, gray/bone cmap)
        - render_vorticity=True: vorticity field (red-white-blue, symmetric range)
        - Default: velocity field (magma, non-negative range)
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if not self.render_initialized:
            # First render: initialize matplotlib window
            plt.ion()  # Enable interactive mode
            self.fig, self.ax = plt.subplots(figsize=(8, 14))

            # Set colormap and range based on rendering type
            if self.render_solid_boundary:
                self.img_plot = self.ax.imshow(
                    img,
                    origin="lower",
                    interpolation="nearest",  # sharpened for boundaries
                    vmin=0.0,
                    vmax=1.0,
                    cmap="binary",
                )
                title = "Solid Boundaries"
                cbar_label = "Occupancy"
            elif self.render_vorticity:
                # Vorticity field: symmetric range
                vmax_abs = np.max(np.abs(img))
                self.img_plot = self.ax.imshow(
                    img,
                    origin="lower",
                    interpolation=self.video_interpolation,
                    vmin=-vmax_abs,
                    vmax=vmax_abs,
                    cmap=self.vorticity_cmap,
                )
                title = "Vorticity"
                cbar_label = "ω (1/s)"
            else:
                # Velocity field: non-negative range
                self.img_plot = self.ax.imshow(
                    img,
                    origin="lower",
                    interpolation=self.video_interpolation,
                    vmin=self.video_vmin,
                    vmax=self.video_vmax,
                    cmap=self.video_cmap,
                )
                title = "Velocity Magnitude"
                cbar_label = "|u| (m/s)"

            self.ax.set_title(title, fontsize=14, fontweight="bold")
            self.ax.axis("off")
            self.cbar = plt.colorbar(
                self.img_plot, ax=self.ax, fraction=0.046, pad=0.04
            )
            self.cbar.set_label(cbar_label, rotation=270, labelpad=20)

            # Add information text box
            self.info_text = self.ax.text(
                0.02,
                0.98,
                "",
                transform=self.ax.transAxes,
                fontsize=11,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
            )

            plt.tight_layout()
            self.render_initialized = True
        else:
            # Update image data
            self.img_plot.set_data(img)

            # Update range/text based on mode
            if self.render_solid_boundary:
                # Fixed range [0, 1] is usually fine for binary mask
                info_str = f"Step: {self.current_steps[0]}"
            elif self.render_vorticity:
                vmax_abs = np.max(np.abs(img))
                self.img_plot.set_clim(-vmax_abs, vmax_abs)
                w_max = np.max(np.abs(img))
                w_mean = np.mean(img)
                info_str = (
                    f"Step: {self.current_steps[0]}\n"
                    f"ω_max: {w_max:.2f} 1/s\n"
                    f"ω_mean: {w_mean:.2f} 1/s"
                )
            else:
                # velocity field
                v_max = np.max(img)
                v_mean = np.mean(img)
                info_str = (
                    f"Step: {self.current_steps[0]}\n"
                    f"V_max: {v_max:.4f} m/s\n"
                    f"V_mean: {v_mean:.4f} m/s"
                )
            self.info_text.set_text(info_str)

        # Refresh display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)  # Brief pause to update display

    def save_video(self, filepath: Optional[str] = None):
        """
        Save collected frames to video file

        Based on render flags type of video saved

        Args:
            filepath: Video save path (optional), defaults to 'lbm_episode.mp4'

        Note:
            - Only effective when render_mode='human' and real_time_rendering=False
        """
        if self.real_time_rendering:
            print("Currently in real-time rendering mode, cannot save video.")
            print("To save video, set: env.real_time_rendering = False")
            print("Then re-run episode and call env.save_video()")
            return

        if filepath is not None:
            self.video_path = filepath

        # Check if there is frame data
        if len(self.frames) == 0:
            print("No frame data collected, cannot save video.")
            print("Ensure render_mode='human' and env.render() has been called")
            return

        if self.render_mode == "human":
            import os
            import imageio
            import numpy as np
            import matplotlib.pyplot as plt

            num_frames = len(self.frames)
            if self.render_solid_boundary:
                field_type = "solid boundary"
            else:
                field_type = (
                    "vorticity field" if self.render_vorticity else "velocity field"
                )
            print(
                f"\nSaving {field_type} video to {self.video_path}, {num_frames} frames total..."
            )

            # Ensure output directory exists
            output_dir = os.path.dirname(self.video_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Calculate value range
            all_values = np.concatenate([frame.flatten() for frame in self.frames])

            if self.render_solid_boundary:
                vmin, vmax = 0.0, 1.0
                cmap = plt.get_cmap("binary")
            elif self.render_vorticity:
                # Vorticity field: symmetric range
                vmax_abs = np.max(np.abs(all_values)) * 0.3
                vmin, vmax = -vmax_abs, vmax_abs
                cmap = plt.get_cmap(self.vorticity_cmap)
            else:
                # Velocity field: non-negative range
                vmin = np.min(all_values) * 0.6
                vmax = np.max(all_values) * 0.6
                cmap = plt.get_cmap(self.video_cmap)

            # Prepare video writer
            writer = imageio.get_writer(
                self.video_path,
                fps=self.video_fps,
                codec="libx264",
                quality=8,
                pixelformat="yuv420p",
                macro_block_size=1,
            )

            # Batch process frames
            for i, frame in enumerate(self.frames):
                # Normalize
                if vmax > vmin:
                    if self.render_solid_boundary:
                        # Simple clamp for solid boundaries
                        normalized = np.clip(frame, 0, 1)
                    elif self.render_vorticity:
                        # Vorticity field: symmetric normalization to [0, 1]
                        normalized = np.clip((frame + vmax_abs) / (2 * vmax_abs), 0, 1)
                    else:
                        # Velocity field: linear normalization to [0, 1]
                        normalized = np.clip((frame - vmin) / (vmax - vmin), 0, 1)
                else:
                    if self.render_solid_boundary:
                        normalized = np.zeros_like(frame)
                    else:
                        normalized = (
                            np.zeros_like(frame)
                            if not self.render_vorticity
                            else np.ones_like(frame) * 0.5
                        )

                # Apply colormap
                colored = cmap(normalized)
                rgb = (colored[:, :, :3] * 255).astype(np.uint8)
                rgb = np.flipud(rgb)  # Flip y-axis

                # Write frame
                writer.append_data(rgb)

            writer.close()
            print(f"Video saved to {self.video_path}")

            # Clear frame buffer
            self.frames = []

    def close(self):
        """
        Close environment and release resources
        """
        # Close matplotlib window (if in real-time rendering mode)
        if self.render_initialized and self.fig is not None:
            import matplotlib.pyplot as plt

            plt.close(self.fig)
            self.render_initialized = False
            self.fig = None
            self.ax = None
            self.img_plot = None
            self.cbar = None
            self.info_text = None
