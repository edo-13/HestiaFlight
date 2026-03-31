import torch
from typing import Dict, Optional, Tuple, Final

# ============================================================================
# Constants
# ============================================================================

DEFAULT_MASS: Final[float] = 1.5
DEFAULT_ARM_LENGTH: Final[float] = 0.25
DEFAULT_THRUST_TO_WEIGHT: Final[float] = 2.5
DEFAULT_IXX: Final[float] = 0.0347563
DEFAULT_IYY: Final[float] = 0.0458929
DEFAULT_IZZ: Final[float] = 0.0977
DEFAULT_DRAG_COEF: Final[float] = 0.03
DEFAULT_GRAVITY: Final[float] = 9.81

# control gains
KP_ROLL: Final[float] = 5.0
KP_PITCH: Final[float] = 5.0
KP_YAW: Final[float] = 5.0

# physical limits
MAX_BODY_RATE: Final[float] = 5.0  # rad/s
DAMPING_COEF: Final[float] = 0.1

# motor dynamics (racing quad defaults)
DEFAULT_MOTOR_TAU: Final[float] = 0.015  # 15ms - racing quad

# second-order motor defaults (derived from tau)
# omega_n ≈ 2/tau gives similar settling time to first-order
# zeta = 0.85 is slightly underdamped
DEFAULT_MOTOR_OMEGA_N: Final[float] = 2.0 / DEFAULT_MOTOR_TAU  # ~133 rad/s
DEFAULT_MOTOR_ZETA: Final[float] = 0.85


# quat operations

@torch.jit.script
def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions using Hamilton product."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    result = torch.stack([w, x, y, z], dim=-1)
    return result / (torch.norm(result, dim=-1, keepdim=True) + 1e-8)


@torch.jit.script
def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to rotation matrix (optimized)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    # sums are lighter than multiplying x2
    x2, y2, z2 = x + x, y + y, z + z
    xx, xy, xz = x * x2, x * y2, x * z2
    yy, yz, zz = y * y2, y * z2, z * z2
    wx, wy, wz = w * x2, w * y2, w * z2
    
    # assignment to preallocated tensor
    batch_shape = q.shape[:-1]
    R = torch.empty(batch_shape + (3, 3), device=q.device, dtype=q.dtype)
    
    R[..., 0, 0] = 1.0 - (yy + zz)
    R[..., 0, 1] = xy - wz
    R[..., 0, 2] = xz + wy
    R[..., 1, 0] = xy + wz
    R[..., 1, 1] = 1.0 - (xx + zz)
    R[..., 1, 2] = yz - wx
    R[..., 2, 0] = xz - wy
    R[..., 2, 1] = yz + wx
    R[..., 2, 2] = 1.0 - (xx + yy)
    
    return R


@torch.jit.script
def normalize_quaternion(q: torch.Tensor) -> torch.Tensor:
    """Normalize quaternion to unit length."""
    return q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)


@torch.jit.script
def quaternion_derivative(q: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """Compute quaternion time derivative."""
    batch_size = q.shape[0]
    device = q.device
    
    omega_quat = torch.cat([
        torch.zeros(batch_size, 1, device=device),
        omega
    ], dim=-1)
    
    return 0.5 * quaternion_multiply(q, omega_quat)


@torch.jit.script
def integrate_quaternion_exp(q: torch.Tensor, omega: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """
    Quaternion integration assuming constant angular velocity.
    
    Uses quaternion exponential: q_new = q ⊗ exp(omega·dt/2)
    
    This is the exact solution to dq/dt = 0.5 * q ⊗ omega when omega is constant
    
    Args:
        q: Quaternion [batch, 4] as [w, x, y, z]
        omega: Angular velocity [batch, 3] in body frame (rad/s)
        dt: Time step [batch] or scalar
        
    Returns:
        Integrated quaternion [batch, 4], normalized
    """
    if dt.ndim == 0:
        dt = dt.unsqueeze(0)
    dt = dt.view(-1, 1)
    
    # theta = |omega| * dt
    omega_norm = torch.norm(omega, dim=-1, keepdim=True)
    half_angle = 0.5 * omega_norm * dt
    
    # Taylor expansion to avoid division by zero for small angles
    # sinc(x) = sin(x)/x -> 1 - x²/6 for small x
    small_angle = omega_norm < 1e-6
    
    # sin(theta/2) / |ω| = sin(theta/2) / (theta/dt) = dt * sin(theta/2) / θ
    # small angles -> 0.5 * dt * (1 - theta²/24) where theta = |omega|*dt
    sinc_half = torch.where(
        small_angle,
        0.5 * dt * (1.0 - half_angle * half_angle / 6.0),
        torch.sin(half_angle) / (omega_norm + 1e-8)
    )
    cos_half = torch.cos(half_angle)
    
    # delta quaternion: [cos(theta/2), sin(theta/2) * omega]
    # where omega^ = omega / |omega|, so sin(theta/2) * omega^ = sin(theta/2) / |theta| * omega = sinc_half * omega
    delta_q = torch.cat([cos_half, sinc_half * omega], dim=-1)
    
    return quaternion_multiply(q, delta_q)


@torch.jit.script
def apply_motor_lag(
    commanded: torch.Tensor,
    actual: torch.Tensor,
    tau: torch.Tensor,
    dt: torch.Tensor
) -> torch.Tensor:
    """
    Apply first-order exponential motor response (LEGACY - kept for API compat).
    
    Model: d(actual)/dt = (commanded - actual) / tau
    Solution: actual_new = actual + (commanded - actual) * (1 - exp(-dt/tau))
    
    Args:
        commanded: Commanded action [batch, 4]
        actual: Current motor state [batch, 4]
        tau: Motor time constant [batch]
        dt: Time step [batch]
        
    Returns:
        New motor state [batch, 4]
    """
    alpha = 1.0 - torch.exp(-dt / (tau + 1e-8))
    alpha = alpha.view(-1, 1)
    
    return actual + alpha * (commanded - actual)


@torch.jit.script
def apply_motor_lag_second_order(
    commanded: torch.Tensor,
    actual: torch.Tensor,
    velocity: torch.Tensor,
    omega_n: torch.Tensor,
    zeta: torch.Tensor,
    dt: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply second-order motor dynamics with configurable damping.
    
    Models combined ESC + motor + rotor inertia dynamics as:
        x'' + 2zeta*omega_n*x' + omega_n²*x = omega_n²*u
    
    This captures:
        - Rotor inertia (mass of spinning propeller)
        - ESC control loop dynamics  
        - Realistic overshoot/settling behavior
    
    Args:
        commanded: Commanded action [batch, 4]
        actual: Current motor output [batch, 4]
        velocity: Current motor velocity (dx/dt) [batch, 4]
        omega_n: Natural frequency [batch] (rad/s, ~2/tau for similar response)
        zeta: Damping ratio [batch] (0.7-1.0 typical, 1.0 = critically damped)
        dt: Time step [batch]
        
    Returns:
        Tuple of (new_actual, new_velocity)
    """
    omega_n_v = omega_n.view(-1, 1)
    zeta_v = zeta.view(-1, 1)
    dt_v = dt.view(-1, 1)
    
    # acceleration from second-order ODE
    error = commanded - actual
    accel = omega_n_v * omega_n_v * error - 2.0 * zeta_v * omega_n_v * velocity
    
    # Euler integration
    new_velocity = velocity + accel * dt_v
    new_actual = actual + new_velocity * dt_v
    
    # clamp to valid range
    new_actual = new_actual.clamp(-1.0, 1.0)
    
    # zero velocity at bounds to prevent windup
    at_min = new_actual <= -1.0
    at_max = new_actual >= 1.0
    new_velocity = torch.where(
        (at_min & (new_velocity < 0)) | (at_max & (new_velocity > 0)),
        torch.zeros_like(new_velocity),
        new_velocity
    )
    
    return new_actual, new_velocity


#quadrotor dynamics


class QuadrotorDynamics:
    """
    Stateless 6-DOF quadrotor dynamics simulator.
    All physics parameters are passed as tensors for batched heterogeneous simulation.
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)
    
    @torch.jit.export
    def step(
        self,
        state: Dict[str, torch.Tensor],
        action: torch.Tensor,
        dt: torch.Tensor,
        mass: torch.Tensor,
        thrust_to_weight: torch.Tensor,
        drag_coef: torch.Tensor,
        inertia_scale: torch.Tensor,
        wind_velocity: Optional[torch.Tensor] = None,
        gravity: float = DEFAULT_GRAVITY
    ) -> Dict[str, torch.Tensor]:
        """
        Step the dynamics forward by dt.
        
        Args:
            state: Dict with position, velocity, quaternion, angular_velocity
            action: [batch, 4] - thrust + body rates (AFTER motor lag applied)
            dt: [batch] time step
            mass, thrust_to_weight, drag_coef, inertia_scale: Physics params
            wind_velocity: Optional [batch, 3]
            gravity: Gravitational acceleration
            
        Returns:
            Updated state dict
        """
        batch_size = state['position'].shape[0]
        device = state['position'].device
        
        pos = state['position']
        vel = state['velocity']
        quat = state['quaternion']
        omega = state['angular_velocity']
        
        # reshaping
        mass_v = mass.view(-1, 1)
        drag_coef_v = drag_coef.view(-1, 1)
        thrust_to_weight_v = thrust_to_weight.view(-1, 1)
        inertia_scale_v = inertia_scale.view(-1)
        
        # linear dynamics
        
        thrust_cmd = (action[:, 0] + 1.0) * 0.5  # mapping action from RL agent: [-1,1] -> [0,1]
        body_rate_cmd = action[:, 1:].clamp(-1, 1)
        
        max_thrust = mass_v.squeeze(-1) * gravity * thrust_to_weight_v.squeeze(-1)
        
        # thrust vector
        thrust_mag = thrust_cmd * max_thrust
        thrust_body = torch.zeros(batch_size, 3, device=device, dtype=pos.dtype)
        thrust_body[:, 2] = thrust_mag
        
        R = quaternion_to_rotation_matrix(quat)
        thrust_world = torch.bmm(R, thrust_body.unsqueeze(-1)).squeeze(-1)
        
        # Gravity vector
        gravity_force = torch.zeros(batch_size, 3, device=device, dtype=pos.dtype)
        gravity_force[:, 2] = -mass_v.squeeze(-1) * gravity
        
        # drag: F_drag = -c * v * |v|
        vel_norm = torch.norm(vel, dim=-1, keepdim=True)
        drag = -drag_coef_v * vel * vel_norm
        
        # wind force and corrected physics
        wind_force = torch.zeros_like(vel)
        if wind_velocity is not None:
            relative_vel = wind_velocity - vel
            rel_vel_norm = torch.norm(relative_vel, dim=-1, keepdim=True)
            # F = 0.5 * rho * Cd * A * v^2
            # assuming drag_coef incorporates (0.5 * Cd * A) and caller folds in air_density
            wind_force = drag_coef_v * relative_vel * rel_vel_norm
        
        total_force = thrust_world + gravity_force + drag + wind_force
        linear_accel = total_force / mass_v
        
        # time step handling
        if dt.ndim == 0:
            dt_v = dt
        else:
            dt_v = dt.view(-1, 1)
            
        new_vel = vel + linear_accel * dt_v
        new_pos = pos + new_vel * dt_v
        
        # rotation al dynamics
        
        desired_omega = body_rate_cmd * MAX_BODY_RATE
        omega_error = desired_omega - omega
        
        # torque from P-controller
        torque_x = DEFAULT_IXX * KP_ROLL * omega_error[:, 0]
        torque_y = DEFAULT_IYY * KP_PITCH * omega_error[:, 1]
        torque_z = DEFAULT_IZZ * KP_YAW * omega_error[:, 2]
        
        # damping
        torque_x = torque_x - DAMPING_COEF * omega[:, 0]
        torque_y = torque_y - DAMPING_COEF * omega[:, 1]
        torque_z = torque_z - DAMPING_COEF * omega[:, 2]
        
        # I * omega for gyroscopic term
        I_omega_x = DEFAULT_IXX * omega[:, 0]
        I_omega_y = DEFAULT_IYY * omega[:, 1]
        I_omega_z = DEFAULT_IZZ * omega[:, 2]
        
        # omega x (I * omega) - expanded cross product
        gyro_x = omega[:, 1] * I_omega_z - omega[:, 2] * I_omega_y
        gyro_y = omega[:, 2] * I_omega_x - omega[:, 0] * I_omega_z
        gyro_z = omega[:, 0] * I_omega_y - omega[:, 1] * I_omega_x
        
        # angular acceleration: I^-1 * (torque - omega x I*omega)
        angular_accel = torch.stack([
            (torque_x - gyro_x) / (DEFAULT_IXX * inertia_scale_v),
            (torque_y - gyro_y) / (DEFAULT_IYY * inertia_scale_v),
            (torque_z - gyro_z) / (DEFAULT_IZZ * inertia_scale_v)
        ], dim=-1)
        
        new_omega = omega + angular_accel * dt_v
        new_quat = integrate_quaternion_exp(
            quat, new_omega, 
            dt if dt.ndim == 1 else dt.expand(batch_size)
        )
        
        return {
            'position': new_pos,
            'velocity': new_vel,
            'quaternion': new_quat,
            'angular_velocity': new_omega,
            'acceleration': linear_accel
        }


# batched env handling

class BatchedPhysicsEnv:
    """
    Manages parallel quadrotor simulations with:
    - Motor lag (second-order dynamics)
    - Control delays
    - Domain randomization
    - Environmental effects
    """
    
    def __init__(
        self,
        num_envs: int,
        dt: float = 0.01,
        device: str = "cuda",
        max_delay_steps: int = 10
    ):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.max_delay_steps = max_delay_steps
        
        self.dynamics = QuadrotorDynamics(device=device)
        
        # physics parameters
        self.mass = torch.full((num_envs,), DEFAULT_MASS, device=self.device)
        self.thrust_to_weight = torch.full((num_envs,), DEFAULT_THRUST_TO_WEIGHT, device=self.device)
        self.drag_coef = torch.full((num_envs,), DEFAULT_DRAG_COEF, device=self.device)
        self.inertia_scale = torch.ones(num_envs, device=self.device)
        self.dt = torch.full((num_envs,), dt, device=self.device)
        
        # motor dynamics - second order
        self.motor_omega_n = torch.full((num_envs,), DEFAULT_MOTOR_OMEGA_N, device=self.device)
        self.motor_zeta = torch.full((num_envs,), DEFAULT_MOTOR_ZETA, device=self.device)
        self.motor_state = torch.zeros((num_envs, 4), device=self.device)
        self.motor_velocity = torch.zeros((num_envs, 4), device=self.device)
        
        # legacy tau - kept for API compatibility, used to derive omega_n if set
        self.motor_tau = torch.full((num_envs,), DEFAULT_MOTOR_TAU, device=self.device)
        
        # environmental parameters
        self.wind_velocity = torch.zeros((num_envs, 3), device=self.device)
        self.wind_turbulence = torch.zeros(num_envs, device=self.device)
        self.air_density = torch.ones(num_envs, device=self.device)
        self.gravity = torch.full((num_envs,), DEFAULT_GRAVITY, device=self.device)
        
        # control delay buffer
        self.delay_buffer = torch.zeros(
            (max_delay_steps, num_envs, 4),
            device=self.device
        )
        self.current_delays = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.buffer_idx = 0
        
        # state tracking
        self.last_accel = torch.zeros((num_envs, 3), device=self.device)
        
        # preallocated buffers for step() to avoid repeated allocations
        self._wind_buffer = torch.zeros((num_envs, 3), device=self.device)
        self._effective_drag = torch.zeros(num_envs, device=self.device)
    
    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset specified environments."""
        if len(env_ids) > 0:
            self.delay_buffer[:, env_ids] = 0
            self.last_accel[env_ids] = 0
            self.motor_state[env_ids] = 0
            self.motor_velocity[env_ids] = 0 
    
    def apply_randomization(
        self,
        env_ids: torch.Tensor,
        params: Optional[Dict[str, torch.Tensor]] = None
    ) -> None:
        """Apply domain randomization parameters."""
        if len(env_ids) == 0:
            return
        
        if params is None:
            self.mass[env_ids] = DEFAULT_MASS
            self.thrust_to_weight[env_ids] = DEFAULT_THRUST_TO_WEIGHT
            self.drag_coef[env_ids] = DEFAULT_DRAG_COEF
            self.inertia_scale[env_ids] = 1.0
            self.wind_velocity[env_ids] = 0.0
            self.wind_turbulence[env_ids] = 0.0
            self.air_density[env_ids] = 1.0
            self.current_delays[env_ids] = 0
            self.dt[env_ids] = 0.01
            self.motor_tau[env_ids] = DEFAULT_MOTOR_TAU
            self.motor_omega_n[env_ids] = DEFAULT_MOTOR_OMEGA_N
            self.motor_zeta[env_ids] = DEFAULT_MOTOR_ZETA
            return
        
        p = params['physics']
        self.mass[env_ids] = p['mass']
        self.thrust_to_weight[env_ids] = p['thrust_to_weight']
        self.drag_coef[env_ids] = p['drag_coefficient']
        self.inertia_scale[env_ids] = p['inertia_scale']
        
        # motor parameters
        if 'motor_tau' in p:
            self.motor_tau[env_ids] = p['motor_tau']
            # derive second-order params from tau if not given
            if 'motor_omega_n' not in p:
                self.motor_omega_n[env_ids] = 2.0 / p['motor_tau']
            if 'motor_zeta' not in p:
                self.motor_zeta[env_ids] = DEFAULT_MOTOR_ZETA
        
        if 'motor_omega_n' in p:
            self.motor_omega_n[env_ids] = p['motor_omega_n']
        if 'motor_zeta' in p:
            self.motor_zeta[env_ids] = p['motor_zeta']
        
        t = params['time']
        self.dt[env_ids] = t['dt']
        
        e = params['environment']
        self.wind_velocity[env_ids] = e['wind_velocity']
        self.wind_turbulence[env_ids] = e['wind_turbulence']
        self.air_density[env_ids] = e['air_density_ratio']
        
        delay_seconds = params['sensors']['control_delay']
        delay_steps = (delay_seconds / self.dt[env_ids]).long()
        self.current_delays[env_ids] = torch.clamp(
            delay_steps, 0, self.max_delay_steps - 1
        )
    
    def step(
        self,
        state: Dict[str, torch.Tensor],
        action: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Step all environments forward."""
        
        # store commanded action in delay buffer
        self.delay_buffer[self.buffer_idx] = action
        
        # retrieve delayed action
        read_indices = (self.buffer_idx - self.current_delays) % self.max_delay_steps
        env_indices = torch.arange(self.num_envs, device=self.device)
        delayed_action = self.delay_buffer[read_indices, env_indices]
        
        self.motor_state, self.motor_velocity = apply_motor_lag_second_order(
            delayed_action,
            self.motor_state,
            self.motor_velocity,
            self.motor_omega_n,
            self.motor_zeta,
            self.dt
        )
        
        # advance buffer
        self.buffer_idx = (self.buffer_idx + 1) % self.max_delay_steps
        
        # compute wind with turbulence (using preallocated buffer)
        self._wind_buffer.copy_(self.wind_velocity)
        turbulent_mask = self.wind_turbulence > 0
        if turbulent_mask.any():
            n_turbulent = turbulent_mask.sum()
            noise = torch.randn(n_turbulent, 3, device=self.device)
            noise *= self.wind_turbulence[turbulent_mask].unsqueeze(1)
            self._wind_buffer[turbulent_mask] += noise
        
        # fold air density into effective drag coefficient
        # F_drag is proportianl to: ρ * Cd * A * v²
        self._effective_drag = self.drag_coef * self.air_density
        
        # step dynamics with motor output
        next_state = self.dynamics.step(
            state=state,
            action=self.motor_state,
            dt=self.dt,
            mass=self.mass,
            thrust_to_weight=self.thrust_to_weight,
            drag_coef=self._effective_drag,
            inertia_scale=self.inertia_scale,
            wind_velocity=self._wind_buffer,
            gravity=DEFAULT_GRAVITY
        )
        
        self.last_accel = next_state['acceleration']
        
        return next_state
    
    def get_motor_state(self) -> torch.Tensor:
        """Get current motor state (for observation if needed)."""
        return self.motor_state.clone()
    
    def get_motor_velocity(self) -> torch.Tensor:
        """Get current motor velocity state (for debugging/observation)."""
        return self.motor_velocity.clone()
    
    # compatibility alias
    def apply_randomization_batch(
        self,
        env_ids: torch.Tensor,
        params: Optional[Dict[str, torch.Tensor]] = None
    ) -> None:
        """Alias for apply_randomization()."""
        self.apply_randomization(env_ids, params)