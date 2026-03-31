import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import torch
from typing import Optional, Tuple, List
from phy import quaternion_to_rotation_matrix

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False


class QuadrotorRenderer:
    """Real-time 3D visualization for vectorized quadrotor environments."""
    
    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        target_fps: int = 100,
        camera_distance: float = 50.0,
        show_trails: bool = True,
        trail_length: int = 100,
        max_rendered_quads: int = 16
    ):
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.camera_distance = camera_distance
        self.show_trails = show_trails
        self.trail_length = trail_length
        self.max_rendered = max_rendered_quads
        
        # camera state
        self.cam_yaw = 45.0
        self.cam_pitch = 30.0
        self.cam_target = np.array([0.0, 0.0, 10.0])
        
        # trail buffers
        self.trails = {}
        self._last_pos = {}
        
        # colors
        self.colors = [
            (0.2, 0.6, 1.0), (1.0, 0.4, 0.2), (0.3, 0.9, 0.3), (0.9, 0.2, 0.6),
            (1.0, 0.8, 0.1), (0.5, 0.2, 0.9), (0.1, 0.8, 0.8), (0.9, 0.5, 0.1),
        ]
        
        self._init_pygame()
        self._init_opengl()
        
        # recording
        self._recording = False
        self._frames: List[np.ndarray] = []
        self._record_path: Optional[str] = None
        
    def _init_pygame(self):
        pygame.init()
        pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("HestiaFlight Visualizer")
        self.clock = pygame.time.Clock()
        
    def _init_opengl(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        glLightfv(GL_LIGHT0, GL_POSITION, (50, 50, 100, 1))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.3, 0.3, 0.3, 1))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.8, 0.8, 0.8, 1))
        
        glClearColor(0.1, 0.1, 0.15, 1.0)
        self._setup_projection()
        
    def _setup_projection(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60, self.width / self.height, 0.1, 1000.0)
        glMatrixMode(GL_MODELVIEW)
        
    def _update_camera(self):
        glLoadIdentity()
        
        rad_yaw = np.radians(self.cam_yaw)
        rad_pitch = np.radians(self.cam_pitch)
        
        cam_x = self.cam_target[0] + self.camera_distance * np.cos(rad_pitch) * np.sin(rad_yaw)
        cam_y = self.cam_target[1] + self.camera_distance * np.cos(rad_pitch) * np.cos(rad_yaw)
        cam_z = self.cam_target[2] + self.camera_distance * np.sin(rad_pitch)
        
        gluLookAt(cam_x, cam_y, cam_z,
                  self.cam_target[0], self.cam_target[1], self.cam_target[2],
                  0, 0, 1)
    
    def _draw_ground(self, size: float = 100, grid_spacing: float = 10):
        glDisable(GL_LIGHTING)
        glColor3f(0.3, 0.3, 0.35)
        glBegin(GL_LINES)
        for i in np.arange(-size, size + grid_spacing, grid_spacing):
            glVertex3f(i, -size, 0)
            glVertex3f(i, size, 0)
            glVertex3f(-size, i, 0)
            glVertex3f(size, i, 0)
        glEnd()
        glEnable(GL_LIGHTING)
        
    def _draw_axes(self, length: float = 5):
        glDisable(GL_LIGHTING)
        glLineWidth(2)
        glBegin(GL_LINES)
        glColor3f(1, 0, 0); glVertex3f(0, 0, 0); glVertex3f(length, 0, 0)
        glColor3f(0, 1, 0); glVertex3f(0, 0, 0); glVertex3f(0, length, 0)
        glColor3f(0, 0, 1); glVertex3f(0, 0, 0); glVertex3f(0, 0, length)
        glEnd()
        glLineWidth(1)
        glEnable(GL_LIGHTING)
        
    def _draw_quadrotor(self, pos: np.ndarray, rot_matrix: np.ndarray, color: Tuple[float, float, float], arm_length: float = 1.0):
        glPushMatrix()
        glTranslatef(*pos)
        
        # apply rotation (OpenGL expects column-major 4x4)
        rot_4x4 = np.eye(4)
        rot_4x4[:3, :3] = rot_matrix
        glMultMatrixf(rot_4x4.T.flatten())
        
        # body
        glColor3f(*color)
        glPushMatrix()
        glScalef(1.0, 1.0, 0.3)
        quad = gluNewQuadric()
        gluSphere(quad, arm_length * 0.6, 16, 8)
        gluDeleteQuadric(quad)
        glPopMatrix()
        
        # arms and motors
        arm_positions = [
            (arm_length, arm_length, 0),
            (arm_length, -arm_length, 0),
            (-arm_length, arm_length, 0),
            (-arm_length, -arm_length, 0)
        ]
        
        for i, arm_pos in enumerate(arm_positions):
            # arm
            glColor3f(0.4, 0.4, 0.4)
            glBegin(GL_LINES)
            glVertex3f(0, 0, 0)
            glVertex3f(*arm_pos)
            glEnd()
            
            # motor
            glPushMatrix()
            glTranslatef(*arm_pos)
            motor_color = (0.8, 0.2, 0.2) if i < 2 else (0.2, 0.2, 0.8)
            glColor3f(*motor_color)
            quad = gluNewQuadric()
            gluCylinder(quad, arm_length * 0.15, arm_length * 0.15, arm_length * 0.1, 8, 1)
            gluDeleteQuadric(quad)
            glPopMatrix()
        
        # direction indicator (front)
        glColor3f(1, 1, 0)
        glBegin(GL_TRIANGLES)
        glVertex3f(arm_length * 1.2, 0, 0)
        glVertex3f(arm_length * 0.8, arm_length * 0.15, 0)
        glVertex3f(arm_length * 0.8, -arm_length * 0.15, 0)
        glEnd()
        
        glPopMatrix()
        
    def _draw_trail(self, trail: list, color: Tuple[float, float, float]):
        if len(trail) < 2:
            return
        glDisable(GL_LIGHTING)
        glColor4f(*color, 0.5)
        glBegin(GL_LINE_STRIP)
        for p in trail:
            glVertex3f(*p)
        glEnd()
        glEnable(GL_LIGHTING)
        
    def _draw_waypoint(self, pos: np.ndarray, color: Tuple[float, float, float] = (0.2, 0.9, 0.2), size: float = 1.0):
        glPushMatrix()
        glTranslatef(*pos)
        glColor4f(*color, 0.6)
        quad = gluNewQuadric()
        gluSphere(quad, size, 12, 6)
        gluDeleteQuadric(quad)
        glPopMatrix()
        
    def _handle_input(self) -> bool:
        """Handle input. Returns False if should quit."""
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                return False
            elif event.type == MOUSEMOTION and pygame.mouse.get_pressed()[0]:
                self.cam_yaw += event.rel[0] * 0.5
                self.cam_pitch = np.clip(self.cam_pitch + event.rel[1] * 0.3, -89, 89)
            elif event.type == MOUSEWHEEL:
                self.camera_distance = np.clip(self.camera_distance - event.y * 5, 5, 200)
                
        keys = pygame.key.get_pressed()
        move_speed = 1.0
        if keys[K_w]: self.cam_target[1] += move_speed
        if keys[K_s]: self.cam_target[1] -= move_speed
        if keys[K_a]: self.cam_target[0] -= move_speed
        if keys[K_d]: self.cam_target[0] += move_speed
        if keys[K_q]: self.cam_target[2] += move_speed
        if keys[K_e]: self.cam_target[2] -= move_speed
        
        return True
    
    def render(
        self,
        positions: torch.Tensor,
        quaternions: torch.Tensor,
        waypoints: Optional[torch.Tensor] = None,
        current_wp_idx: Optional[torch.Tensor] = None
    ) -> bool:
        """
        Render current state. Returns False if window closed.
        
        Args:
            positions: [N, 3] quadrotor positions
            quaternions: [N, 4] quadrotor orientations (w, x, y, z)
            waypoints: Optional [N, M, 3] waypoint positions
            current_wp_idx: Optional [N] current waypoint index per env
        """
        if not self._handle_input():
            return False
            
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self._update_camera()
        
        self._draw_ground()
        self._draw_axes()
        
        
        pos_np = positions[:self.max_rendered].cpu().numpy()
        quat_np = quaternions[:self.max_rendered].cpu().numpy()
        rot_matrices = quaternion_to_rotation_matrix(quaternions[:self.max_rendered]).cpu().numpy()
        
        n_quads = min(len(pos_np), self.max_rendered)
        
        # update trails
        if self.show_trails:
            for i in range(n_quads):
                if i not in self.trails:
                    self.trails[i] = []
                    self._last_pos = {}
                
                # detect teleport at reset - clear trail if position jumped
                if i in self._last_pos:
                    dist = np.linalg.norm(pos_np[i] - self._last_pos[i])
                    if dist > 10.0:  # teleport threshold
                        self.trails[i] = []
                
                self._last_pos[i] = pos_np[i].copy()
                self.trails[i].append(pos_np[i].copy())
                if len(self.trails[i]) > self.trail_length:
                    self.trails[i].pop(0)
        
        # draw waypoints
        if waypoints is not None:
            wp_np = waypoints[:self.max_rendered].cpu().numpy()
            idx_np = current_wp_idx[:self.max_rendered].cpu().numpy() if current_wp_idx is not None else np.zeros(n_quads, dtype=int)
            
            for i in range(min(n_quads, 4)):  # limit waypoint rendering
                color = self.colors[i % len(self.colors)]
                for j, wp in enumerate(wp_np[i]):
                    if np.any(wp != 0):
                        alpha = 1.0 if j == idx_np[i] else 0.3
                        size = 1.5 if j == idx_np[i] else 0.8
                        self._draw_waypoint(wp, (*color, alpha)[:3], size)
        
        # draw quads and trails
        for i in range(n_quads):
            color = self.colors[i % len(self.colors)]
            if self.show_trails and i in self.trails:
                self._draw_trail(self.trails[i], color)
            self._draw_quadrotor(pos_np[i], rot_matrices[i], color)
        
        # auto-follow first quad (with teleport handling)
        if n_quads > 0:
            target = pos_np[0]
            dist_to_target = np.linalg.norm(target - self.cam_target)
            if dist_to_target > 20.0:  # teleport - snap camera
                self.cam_target = target.copy()
            else:  # smooth follow
                self.cam_target = 0.95 * self.cam_target + 0.05 * target
        
        # capture frame if recording
        if self._recording:
            self._capture_frame()
        
        pygame.display.flip()
        self.clock.tick(self.target_fps)
        return True
    
    def close(self):
        pygame.quit()
    
    def start_recording(self, path: str = "output.gif"):
        """Start recording frames. Supports .gif and .mp4"""
        if not HAS_IMAGEIO:
            print("Recording requires imageio: pip install imageio imageio-ffmpeg")
            return
        self._recording = True
        self._frames = []
        self._record_path = path
        print(f"Recording started -> {path}")
    
    def stop_recording(self) -> Optional[str]:
        """Stop recording and save to file."""
        if not self._recording or not self._frames:
            return None
        
        self._recording = False
        path = self._record_path
        
        print(f"Saving {len(self._frames)} frames to {path}...")
        
        if path.endswith('.gif'):
            imageio.mimsave(path, self._frames, fps=24, loop=0)
        elif path.endswith('.mp4'):
            imageio.mimsave(path, self._frames, fps=60, codec='libx264')
        else:
            # default to gif
            path = path + '.gif'
            imageio.mimsave(path, self._frames, fps=30, loop=0)
        
        print(f"Saved: {path}")
        self._frames = []
        return path
    
    def _capture_frame(self):
        """Capture current OpenGL framebuffer."""
        pixels = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
        frame = np.frombuffer(pixels, dtype=np.uint8).reshape(self.height, self.width, 3)
        frame = np.flipud(frame)  # OpenGL is bottom-up
        self._frames.append(frame)


if __name__ == "__main__":
    from vec_env import VecEnv
    from config import TrainingConfig, SpawnConfig
    from example_train import Points
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_envs = 8
    
    cfg = TrainingConfig(device=device, max_episode_steps=2048)
    cfg.domain_randomization.enabled = False
    
    spawn_cfg = SpawnConfig(spawn_radius_mean=5.0, spawn_radius_std=2.0)
    manager = Points(num_envs, device)
    
    env = VecEnv(
        stage_names="demo",
        num_envs=num_envs,
        trajectory_manager=manager,
        config=cfg,
        spawn_config=spawn_cfg
    )
    
    renderer = QuadrotorRenderer(max_rendered_quads=num_envs, show_trails=True)
    
    print("Controls: WASD=pan, QE=up/down, Mouse=orbit, Scroll=zoom, ESC=quit")
    print("Recording: R=start/stop recording")
    
    running = True
    frame_count = 0
    while running:
        # handle recording toggle
        for event in pygame.event.get(KEYDOWN):
            if event.key == K_r:
                if renderer._recording:
                    renderer.stop_recording()
                else:
                    renderer.start_recording("hestiaflight_demo.gif")
        
        # random actions for demo
        actions = torch.randn(num_envs, 4, device=device) * 0.3
        actions[:, 0] = 0.0  # hover thrust
        
        obs, rewards, term, trunc, infos = env.step(actions)
        
        running = renderer.render(
            env.pos,
            env.quat,
            env.env_traj_pos if not cfg.use_waypoint_mode else env.env_waypoints,
            env.target_idx if not cfg.use_waypoint_mode else env.current_waypoint_idx
        )
        
        frame_count += 1
        
        # auto-stop recording after 300 frames (~10 sec at 30fps)
        if renderer._recording and len(renderer._frames) >= 3000:
            renderer.stop_recording()
    
    # save if still recording
    if renderer._recording:
        renderer.stop_recording()
    
    renderer.close()