import torch
from typing import Dict, Optional, Tuple, Final, NamedTuple

from config import PhysicsConstants

# Constants for JIT

_C = PhysicsConstants()

DEFAULT_MASS = _C.mass
DEFAULT_ARM_LENGTH = _C.arm_length
DEFAULT_THRUST_TO_WEIGHT = _C.thrust_to_weight
DEFAULT_IXX = _C.ixx
DEFAULT_IYY = _C.iyy
DEFAULT_IZZ = _C.izz
DEFAULT_GRAVITY = _C.gravity
DEFAULT_DRAG_XY = _C.drag_xy
DEFAULT_DRAG_Z = _C.drag_z
KP_ROLL = _C.kp_roll
KP_PITCH = _C.kp_pitch
KP_YAW = _C.kp_yaw
MAX_BODY_RATE = _C.max_body_rate
DAMPING_COEF = _C.damping
DEFAULT_YAW_TORQUE_COEF = _C.yaw_torque_coef
DEFAULT_MOTOR_TAU = _C.motor_tau
DEFAULT_MOTOR_OMEGA_N = _C.motor_omega_n
DEFAULT_MOTOR_ZETA = _C.motor_zeta
DEFAULT_ROTOR_RADIUS = _C.rotor_radius
GROUND_EFFECT_CEILING = _C.ground_effect_ceiling
INTEGRATOR_EULER = _C.integrator_euler
INTEGRATOR_SEMI_IMPLICIT = _C.integrator_semi_implicit
INTEGRATOR_RK4 = _C.integrator_rk4

class QuadConst(NamedTuple):
    """All physical constants needed inside @torch.jit.script functions."""
    ixx: float = 0.0347563
    iyy: float = 0.0458929
    izz: float = 0.0977
    kp_roll: float = 5.0
    kp_pitch: float = 5.0
    kp_yaw: float = 5.0
    max_body_rate: float = 5.0
    damping: float = 0.1


# Module-level default instance (constructed once, passed everywhere)
QUAD_CONST = QuadConst()


# Quaternion operations

@torch.jit.script
def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    result = torch.stack([w, x, y, z], dim=-1)
    return result / (torch.norm(result, dim=-1, keepdim=True) + 1e-8)


@torch.jit.script
def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    x2, y2, z2 = x + x, y + y, z + z
    xx, xy, xz = x * x2, x * y2, x * z2
    yy, yz, zz = y * y2, y * z2, z * z2
    wx, wy, wz = w * x2, w * y2, w * z2

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
def quaternion_rotate_vector(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qv = q[..., 1:4]
    qw = q[..., 0:1]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + qw * t + torch.cross(qv, t, dim=-1)


@torch.jit.script
def quaternion_rotate_vector_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qv = -q[..., 1:4]
    qw = q[..., 0:1]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + qw * t + torch.cross(qv, t, dim=-1)


@torch.jit.script
def integrate_quaternion_exp(
    q: torch.Tensor, omega: torch.Tensor, dt: torch.Tensor
) -> torch.Tensor:
    if dt.ndim == 0:
        dt = dt.unsqueeze(0)
    dt = dt.view(-1, 1)

    omega_norm = torch.norm(omega, dim=-1, keepdim=True)
    half_angle = 0.5 * omega_norm * dt
    small_angle = omega_norm < 1e-6

    sinc_half = torch.where(
        small_angle,
        0.5 * dt * (1.0 - half_angle * half_angle / 6.0),
        torch.sin(half_angle) / (omega_norm + 1e-8),
    )
    cos_half = torch.cos(half_angle)
    delta_q = torch.cat([cos_half, sinc_half * omega], dim=-1)
    return quaternion_multiply(q, delta_q)

# Motor dynamics

@torch.jit.script
def _motor_accel(
    commanded: torch.Tensor,
    actual: torch.Tensor,
    velocity: torch.Tensor,
    omega_n: torch.Tensor,
    zeta: torch.Tensor,
) -> torch.Tensor:
    omega_n_v = omega_n.view(-1, 1)
    zeta_v = zeta.view(-1, 1)
    error = commanded - actual
    return omega_n_v * omega_n_v * error - 2.0 * zeta_v * omega_n_v * velocity


@torch.jit.script
def _clamp_motor(
    actual: torch.Tensor, velocity: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    actual = actual.clamp(-1.0, 1.0)
    at_min = actual <= -1.0
    at_max = actual >= 1.0
    velocity = torch.where(
        (at_min & (velocity < 0)) | (at_max & (velocity > 0)),
        torch.zeros_like(velocity),
        velocity,
    )
    return actual, velocity


@torch.jit.script
def step_motor_euler(
    commanded: torch.Tensor,
    actual: torch.Tensor,
    velocity: torch.Tensor,
    omega_n: torch.Tensor,
    zeta: torch.Tensor,
    dt: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dt_v = dt.view(-1, 1)
    accel = _motor_accel(commanded, actual, velocity, omega_n, zeta)
    new_vel = velocity + accel * dt_v
    new_act = actual + new_vel * dt_v
    return _clamp_motor(new_act, new_vel)


@torch.jit.script
def step_motor_rk4(
    commanded: torch.Tensor,
    actual: torch.Tensor,
    velocity: torch.Tensor,
    omega_n: torch.Tensor,
    zeta: torch.Tensor,
    dt: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dt_v = dt.view(-1, 1)
    h = dt_v

    k1_v = velocity
    k1_a = _motor_accel(commanded, actual, velocity, omega_n, zeta)

    a2 = actual + 0.5 * h * k1_v
    v2 = velocity + 0.5 * h * k1_a
    k2_v = v2
    k2_a = _motor_accel(commanded, a2, v2, omega_n, zeta)

    a3 = actual + 0.5 * h * k2_v
    v3 = velocity + 0.5 * h * k2_a
    k3_v = v3
    k3_a = _motor_accel(commanded, a3, v3, omega_n, zeta)

    a4 = actual + h * k3_v
    v4 = velocity + h * k3_a
    k4_v = v4
    k4_a = _motor_accel(commanded, a4, v4, omega_n, zeta)

    new_act = actual + (h / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
    new_vel = velocity + (h / 6.0) * (k1_a + 2.0 * k2_a + 2.0 * k3_a + k4_a)
    return _clamp_motor(new_act, new_vel)


# Force and torque computation


@torch.jit.script
def compute_wrench(
    pos: torch.Tensor,
    vel: torch.Tensor,
    quat: torch.Tensor,
    omega: torch.Tensor,
    action: torch.Tensor,
    mass: torch.Tensor,
    thrust_to_weight: torch.Tensor,
    drag_coef_xy: torch.Tensor,
    drag_coef_z: torch.Tensor,
    inertia_scale: torch.Tensor,
    yaw_torque_coef: float,
    wind_velocity: Optional[torch.Tensor],
    gravity: float,
    ground_effect_enabled: bool,
    rotor_radius: float,
    C: QuadConst,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute total force (world frame) and total torque (body frame).
    All physical constants come from the QuadConst named tuple C.
    """
    batch_size = pos.shape[0]
    device = pos.device
    dtype = pos.dtype

    # Thrust
    thrust_cmd = (action[:, 0] + 1.0) * 0.5
    body_rate_cmd = action[:, 1:].clamp(-1, 1)

    max_thrust = mass * gravity * thrust_to_weight
    thrust_mag = thrust_cmd * max_thrust

    # Ground effect (Cheeseman-Bennett)
    if ground_effect_enabled:
        z = pos[:, 2].clamp(min=0.01)
        z_ratio = z / (4.0 * rotor_radius + 1e-6)
        inv_term = (1.0 / (4.0 * z_ratio + 1e-3))
        ge_denom = (1.0 - inv_term * inv_term).clamp(min=1.0, max=1.5)
        ge_factor = 1.0 / ge_denom
        thrust_mag = thrust_mag * ge_factor

    thrust_body = torch.zeros(batch_size, 3, device=device, dtype=dtype)
    thrust_body[:, 2] = thrust_mag
    thrust_world = quaternion_rotate_vector(quat, thrust_body)

    # Gravity
    gravity_force = torch.zeros(batch_size, 3, device=device, dtype=dtype)
    gravity_force[:, 2] = -mass * gravity

    # Per axis drag
    vel_body = quaternion_rotate_vector_inverse(quat, vel)
    drag_body = torch.zeros_like(vel_body)
    cd_xy = drag_coef_xy.view(-1, 1)
    cd_z = drag_coef_z.view(-1, 1)
    drag_body[:, 0] = -cd_xy.squeeze(1) * vel_body[:, 0] * vel_body[:, 0].abs()
    drag_body[:, 1] = -cd_xy.squeeze(1) * vel_body[:, 1] * vel_body[:, 1].abs()
    drag_body[:, 2] = -cd_z.squeeze(1) * vel_body[:, 2] * vel_body[:, 2].abs()
    drag_world = quaternion_rotate_vector(quat, drag_body)

    # Wind
    wind_force = torch.zeros_like(vel)
    if wind_velocity is not None:
        relative_vel = wind_velocity - vel
        rel_body = quaternion_rotate_vector_inverse(quat, relative_vel)
        wf_body = torch.zeros_like(rel_body)
        wf_body[:, 0] = cd_xy.squeeze(1) * rel_body[:, 0] * rel_body[:, 0].abs()
        wf_body[:, 1] = cd_xy.squeeze(1) * rel_body[:, 1] * rel_body[:, 1].abs()
        wf_body[:, 2] = cd_z.squeeze(1) * rel_body[:, 2] * rel_body[:, 2].abs()
        wind_force = quaternion_rotate_vector(quat, wf_body)

    force_world = thrust_world + gravity_force + drag_world + wind_force

    # body frame torques
    desired_omega = body_rate_cmd * C.max_body_rate
    omega_error = desired_omega - omega

    torque = torch.stack([
        C.ixx * C.kp_roll * omega_error[:, 0] - C.damping * omega[:, 0],
        C.iyy * C.kp_pitch * omega_error[:, 1] - C.damping * omega[:, 1],
        C.izz * C.kp_yaw * omega_error[:, 2] - C.damping * omega[:, 2],
    ], dim=-1)

    # Rotor reactive torque (yaw coupling)
    torque[:, 2] = torque[:, 2] - yaw_torque_coef * thrust_mag * omega[:, 2].sign()

    # Gyroscopic: τ_gyro = ω × (I·ω)
    I_omega = torch.stack([
        C.ixx * omega[:, 0],
        C.iyy * omega[:, 1],
        C.izz * omega[:, 2],
    ], dim=-1)
    gyro = torch.cross(omega, I_omega, dim=-1)
    torque_body = torque - gyro

    return force_world, torque_body


# Angular acceleration from torque


@torch.jit.script
def angular_accel_from_torque(
    torque_body: torch.Tensor,
    inertia_scale: torch.Tensor,
    C: QuadConst,
) -> torch.Tensor:
    s = inertia_scale.view(-1)
    return torch.stack([
        torque_body[:, 0] / (C.ixx * s),
        torque_body[:, 1] / (C.iyy * s),
        torque_body[:, 2] / (C.izz * s),
    ], dim=-1)


# Integrators


@torch.jit.script
def _step_euler(
    pos: torch.Tensor,
    vel: torch.Tensor,
    quat: torch.Tensor,
    omega: torch.Tensor,
    action: torch.Tensor,
    dt: torch.Tensor,
    mass: torch.Tensor,
    thrust_to_weight: torch.Tensor,
    drag_coef_xy: torch.Tensor,
    drag_coef_z: torch.Tensor,
    inertia_scale: torch.Tensor,
    yaw_torque_coef: float,
    wind_velocity: Optional[torch.Tensor],
    gravity: float,
    ground_effect_enabled: bool,
    rotor_radius: float,
    C: QuadConst,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = pos.shape[0]
    dt_v = dt.view(-1, 1) if dt.ndim > 0 else dt

    force, torque = compute_wrench(
        pos, vel, quat, omega, action,
        mass, thrust_to_weight, drag_coef_xy, drag_coef_z,
        inertia_scale, yaw_torque_coef, wind_velocity, gravity,
        ground_effect_enabled, rotor_radius, C,
    )
    lin_accel = force / mass.view(-1, 1)
    ang_accel = angular_accel_from_torque(torque, inertia_scale, C)

    new_vel = vel + lin_accel * dt_v
    new_pos = pos + vel * dt_v
    new_omega = omega + ang_accel * dt_v
    new_quat = integrate_quaternion_exp(
        quat, new_omega, dt if dt.ndim == 1 else dt.expand(batch_size)
    )
    return new_pos, new_vel, new_quat, new_omega, lin_accel


@torch.jit.script
def _step_semi_implicit(
    pos: torch.Tensor,
    vel: torch.Tensor,
    quat: torch.Tensor,
    omega: torch.Tensor,
    action: torch.Tensor,
    dt: torch.Tensor,
    mass: torch.Tensor,
    thrust_to_weight: torch.Tensor,
    drag_coef_xy: torch.Tensor,
    drag_coef_z: torch.Tensor,
    inertia_scale: torch.Tensor,
    yaw_torque_coef: float,
    wind_velocity: Optional[torch.Tensor],
    gravity: float,
    ground_effect_enabled: bool,
    rotor_radius: float,
    C: QuadConst,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = pos.shape[0]
    dt_v = dt.view(-1, 1) if dt.ndim > 0 else dt

    force, torque = compute_wrench(
        pos, vel, quat, omega, action,
        mass, thrust_to_weight, drag_coef_xy, drag_coef_z,
        inertia_scale, yaw_torque_coef, wind_velocity, gravity,
        ground_effect_enabled, rotor_radius, C,
    )
    lin_accel = force / mass.view(-1, 1)
    ang_accel = angular_accel_from_torque(torque, inertia_scale, C)

    new_vel = vel + lin_accel * dt_v
    new_pos = pos + new_vel * dt_v  # symplectic: new velocity
    new_omega = omega + ang_accel * dt_v
    new_quat = integrate_quaternion_exp(
        quat, new_omega, dt if dt.ndim == 1 else dt.expand(batch_size)
    )
    return new_pos, new_vel, new_quat, new_omega, lin_accel


@torch.jit.script
def _step_rk4(
    pos: torch.Tensor,
    vel: torch.Tensor,
    quat: torch.Tensor,
    omega: torch.Tensor,
    action: torch.Tensor,
    dt: torch.Tensor,
    mass: torch.Tensor,
    thrust_to_weight: torch.Tensor,
    drag_coef_xy: torch.Tensor,
    drag_coef_z: torch.Tensor,
    inertia_scale: torch.Tensor,
    yaw_torque_coef: float,
    wind_velocity: Optional[torch.Tensor],
    gravity: float,
    ground_effect_enabled: bool,
    rotor_radius: float,
    C: QuadConst,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = pos.shape[0]
    dt_v = dt.view(-1, 1) if dt.ndim > 0 else dt
    h = dt_v
    mass_v = mass.view(-1, 1)

    # k1
    f1, tau1 = compute_wrench(
        pos, vel, quat, omega, action,
        mass, thrust_to_weight, drag_coef_xy, drag_coef_z,
        inertia_scale, yaw_torque_coef, wind_velocity, gravity,
        ground_effect_enabled, rotor_radius, C,
    )
    k1_dp = vel
    k1_dv = f1 / mass_v
    k1_dw = angular_accel_from_torque(tau1, inertia_scale, C)

    # k2 (midpoint)
    p2 = pos + 0.5 * h * k1_dp
    v2 = vel + 0.5 * h * k1_dv
    w2 = omega + 0.5 * h * k1_dw
    q2 = integrate_quaternion_exp(
        quat, w2, 0.5 * dt if dt.ndim == 1 else (0.5 * dt).expand(batch_size)
    )
    f2, tau2 = compute_wrench(
        p2, v2, q2, w2, action,
        mass, thrust_to_weight, drag_coef_xy, drag_coef_z,
        inertia_scale, yaw_torque_coef, wind_velocity, gravity,
        ground_effect_enabled, rotor_radius, C,
    )
    k2_dp = v2
    k2_dv = f2 / mass_v
    k2_dw = angular_accel_from_torque(tau2, inertia_scale, C)

    # k3 (midpoint with k2)
    p3 = pos + 0.5 * h * k2_dp
    v3 = vel + 0.5 * h * k2_dv
    w3 = omega + 0.5 * h * k2_dw
    q3 = integrate_quaternion_exp(
        quat, w3, 0.5 * dt if dt.ndim == 1 else (0.5 * dt).expand(batch_size)
    )
    f3, tau3 = compute_wrench(
        p3, v3, q3, w3, action,
        mass, thrust_to_weight, drag_coef_xy, drag_coef_z,
        inertia_scale, yaw_torque_coef, wind_velocity, gravity,
        ground_effect_enabled, rotor_radius, C,
    )
    k3_dp = v3
    k3_dv = f3 / mass_v
    k3_dw = angular_accel_from_torque(tau3, inertia_scale, C)

    # k4 (endpoint)
    p4 = pos + h * k3_dp
    v4 = vel + h * k3_dv
    w4 = omega + h * k3_dw
    q4 = integrate_quaternion_exp(
        quat, w4, dt if dt.ndim == 1 else dt.expand(batch_size)
    )
    f4, tau4 = compute_wrench(
        p4, v4, q4, w4, action,
        mass, thrust_to_weight, drag_coef_xy, drag_coef_z,
        inertia_scale, yaw_torque_coef, wind_velocity, gravity,
        ground_effect_enabled, rotor_radius, C,
    )
    k4_dp = v4
    k4_dv = f4 / mass_v
    k4_dw = angular_accel_from_torque(tau4, inertia_scale, C)

    # weighted average
    sixth_h = h / 6.0
    new_pos = pos + sixth_h * (k1_dp + 2.0 * k2_dp + 2.0 * k3_dp + k4_dp)
    new_vel = vel + sixth_h * (k1_dv + 2.0 * k2_dv + 2.0 * k3_dv + k4_dv)
    new_omega = omega + sixth_h * (k1_dw + 2.0 * k2_dw + 2.0 * k3_dw + k4_dw)

    new_quat = integrate_quaternion_exp(
        quat, new_omega, dt if dt.ndim == 1 else dt.expand(batch_size)
    )

    lin_accel = (new_vel - vel) / (dt_v + 1e-8)
    return new_pos, new_vel, new_quat, new_omega, lin_accel


# QuadrotorDynamics


class QuadrotorDynamics:
    """
    Stateless 6-DOF quadrotor dynamics with selectable integrator.
    Physical constants are passed via QuadConst named tuple.
    """

    def __init__(self, device: str = "cuda", constants: QuadConst = QUAD_CONST):
        self.device = torch.device(device)
        self.C = constants

    @torch.jit.export
    def step(
        self,
        state: Dict[str, torch.Tensor],
        action: torch.Tensor,
        dt: torch.Tensor,
        mass: torch.Tensor,
        thrust_to_weight: torch.Tensor,
        drag_coef_xy: torch.Tensor,
        drag_coef_z: torch.Tensor,
        inertia_scale: torch.Tensor,
        yaw_torque_coef: float = DEFAULT_YAW_TORQUE_COEF,
        wind_velocity: Optional[torch.Tensor] = None,
        gravity: float = DEFAULT_GRAVITY,
        integrator: int = INTEGRATOR_SEMI_IMPLICIT,
        ground_effect_enabled: bool = False,
        rotor_radius: float = DEFAULT_ROTOR_RADIUS,
    ) -> Dict[str, torch.Tensor]:

        pos = state["position"]
        vel = state["velocity"]
        quat = state["quaternion"]
        omega = state["angular_velocity"]

        C = self.C
        args = (
            pos, vel, quat, omega, action, dt,
            mass, thrust_to_weight, drag_coef_xy, drag_coef_z,
            inertia_scale, yaw_torque_coef, wind_velocity, gravity,
            ground_effect_enabled, rotor_radius, C,
        )

        if integrator == INTEGRATOR_RK4:
            new_pos, new_vel, new_quat, new_omega, accel = _step_rk4(*args)
        elif integrator == INTEGRATOR_SEMI_IMPLICIT:
            new_pos, new_vel, new_quat, new_omega, accel = _step_semi_implicit(*args)
        else:
            new_pos, new_vel, new_quat, new_omega, accel = _step_euler(*args)

        return {
            "position": new_pos,
            "velocity": new_vel,
            "quaternion": new_quat,
            "angular_velocity": new_omega,
            "acceleration": accel,
        }


# Batched Physics

class BatchedPhysicsEnv:
    """
    Manages parallel quadrotor simulations with:
    - Selectable integrator (Euler / semi-implicit / RK4)
    - Substep support
    - Second-order motor dynamics (matched to chosen integrator)
    - Control delay pipeline
    - Per-axis drag
    - Domain randomization
    """

    def __init__(
        self,
        num_envs: int,
        dt: float = 0.01,
        device: str = "cuda",
        max_delay_steps: int = 10,
        integrator: int = INTEGRATOR_SEMI_IMPLICIT,
        substeps: int = 1,
        ground_effect: bool = False,
        constants: QuadConst = QUAD_CONST,
    ):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.max_delay_steps = max_delay_steps
        self.integrator = integrator
        self.substeps = max(1, substeps)
        self.ground_effect = ground_effect
        self.C = constants

        self.dynamics = QuadrotorDynamics(device=device, constants=constants)

        # Physics parameters
        self.mass = torch.full((num_envs,), DEFAULT_MASS, device=self.device)
        self.thrust_to_weight = torch.full(
            (num_envs,), DEFAULT_THRUST_TO_WEIGHT, device=self.device
        )
        self.drag_coef_xy = torch.full((num_envs,), DEFAULT_DRAG_XY, device=self.device)
        self.drag_coef_z = torch.full((num_envs,), DEFAULT_DRAG_Z, device=self.device)
        self.inertia_scale = torch.ones(num_envs, device=self.device)
        self.dt = torch.full((num_envs,), dt, device=self.device)
        self.yaw_torque_coef: float = DEFAULT_YAW_TORQUE_COEF
        self.rotor_radius: float = DEFAULT_ROTOR_RADIUS

        # Motor dynamics
        self.motor_omega_n = torch.full(
            (num_envs,), DEFAULT_MOTOR_OMEGA_N, device=self.device
        )
        self.motor_zeta = torch.full(
            (num_envs,), DEFAULT_MOTOR_ZETA, device=self.device
        )
        self.motor_state = torch.zeros((num_envs, 4), device=self.device)
        self.motor_velocity = torch.zeros((num_envs, 4), device=self.device)
        self.motor_tau = torch.full(
            (num_envs,), DEFAULT_MOTOR_TAU, device=self.device
        )

        # Environment
        self.wind_velocity = torch.zeros((num_envs, 3), device=self.device)
        self.wind_turbulence = torch.zeros(num_envs, device=self.device)
        self.air_density = torch.ones(num_envs, device=self.device)
        self.gravity = torch.full((num_envs,), DEFAULT_GRAVITY, device=self.device)

        self.pos_noise_std = torch.zeros(num_envs, device=self.device)
        self.vel_noise_std = torch.zeros(num_envs, device=self.device)
        self.att_noise_std = torch.zeros(num_envs, device=self.device)
        self.gyro_noise_std = torch.zeros(num_envs, device=self.device)
        self.accel_noise_std = torch.zeros(num_envs, device=self.device)

        # Delay buffer
        self.delay_buffer = torch.zeros(
            (max_delay_steps, num_envs, 4), device=self.device
        )
        self.current_delays = torch.zeros(
            num_envs, dtype=torch.long, device=self.device
        )
        self.buffer_idx = 0

        # Tracking
        self.last_accel = torch.zeros((num_envs, 3), device=self.device)

        # Preallocated
        self._wind_buffer = torch.zeros((num_envs, 3), device=self.device)

    # Legacy accessors

    @property
    def drag_coef(self) -> torch.Tensor:
        """Legacy accessor: returns lateral drag for backward compat."""
        return self.drag_coef_xy

    # Reser and randomization

    def reset(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        self.delay_buffer[:, env_ids] = 0
        self.last_accel[env_ids] = 0
        self.motor_state[env_ids] = 0
        self.motor_velocity[env_ids] = 0
        self.pos_noise_std[env_ids] = 0.0
        self.vel_noise_std[env_ids] = 0.0
        self.att_noise_std[env_ids] = 0.0
        self.gyro_noise_std[env_ids] = 0.0
        self.accel_noise_std[env_ids] = 0.0

    def apply_randomization(
        self,
        env_ids: torch.Tensor,
        params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        if len(env_ids) == 0:
            return

        if params is None:
            self.mass[env_ids] = DEFAULT_MASS
            self.thrust_to_weight[env_ids] = DEFAULT_THRUST_TO_WEIGHT
            self.drag_coef_xy[env_ids] = DEFAULT_DRAG_XY
            self.drag_coef_z[env_ids] = DEFAULT_DRAG_Z
            self.inertia_scale[env_ids] = 1.0
            self.wind_velocity[env_ids] = 0.0
            self.wind_turbulence[env_ids] = 0.0
            self.air_density[env_ids] = 1.0
            self.current_delays[env_ids] = 0
            self.dt[env_ids] = 0.01
            self.motor_tau[env_ids] = DEFAULT_MOTOR_TAU
            self.motor_omega_n[env_ids] = DEFAULT_MOTOR_OMEGA_N
            self.motor_zeta[env_ids] = DEFAULT_MOTOR_ZETA
            self.pos_noise_std[env_ids] = 0.0
            self.vel_noise_std[env_ids] = 0.0
            self.att_noise_std[env_ids] = 0.0
            self.gyro_noise_std[env_ids] = 0.0
            self.accel_noise_std[env_ids] = 0.0
            return

        p = params["physics"]
        self.mass[env_ids] = p["mass"]
        self.thrust_to_weight[env_ids] = p["thrust_to_weight"]
        self.inertia_scale[env_ids] = p["inertia_scale"]

        if "drag_coef_xy" in p:
            self.drag_coef_xy[env_ids] = p["drag_coef_xy"]
            self.drag_coef_z[env_ids] = p["drag_coef_z"]
        elif "drag_coefficient" in p:
            self.drag_coef_xy[env_ids] = p["drag_coefficient"]
            self.drag_coef_z[env_ids] = p["drag_coefficient"]

        if "motor_tau" in p:
            self.motor_tau[env_ids] = p["motor_tau"]
            if "motor_omega_n" not in p:
                self.motor_omega_n[env_ids] = 2.0 / p["motor_tau"]
            if "motor_zeta" not in p:
                self.motor_zeta[env_ids] = DEFAULT_MOTOR_ZETA
        if "motor_omega_n" in p:
            self.motor_omega_n[env_ids] = p["motor_omega_n"]
        if "motor_zeta" in p:
            self.motor_zeta[env_ids] = p["motor_zeta"]

        if "yaw_torque_coef" in p:
            self.yaw_torque_coef = float(p["yaw_torque_coef"])

        t = params["time"]
        self.dt[env_ids] = t["dt"]

        e = params["environment"]
        self.wind_velocity[env_ids] = e["wind_velocity"]
        self.wind_turbulence[env_ids] = e["wind_turbulence"]
        self.air_density[env_ids] = e["air_density_ratio"]

        delay_seconds = params["sensors"]["control_delay"]
        delay_steps = (delay_seconds / self.dt[env_ids]).long()
        self.current_delays[env_ids] = torch.clamp(
            delay_steps, 0, self.max_delay_steps - 1
        )

        s = params["sensors"]
        self.pos_noise_std[env_ids] = s["position_noise_std"]
        self.vel_noise_std[env_ids] = s["velocity_noise_std"]
        self.att_noise_std[env_ids] = s["attitude_noise_std"]
        self.gyro_noise_std[env_ids] = s["gyro_noise_std"]
        self.accel_noise_std[env_ids] = s["accel_noise_std"]

    # Alias
    apply_randomization_batch = apply_randomization

    # Step

    def step(
        self,
        state: Dict[str, torch.Tensor],
        action: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Step all environments forward (with substeps)."""

        # Delay pipeline
        self.delay_buffer[self.buffer_idx] = action
        read_indices = (self.buffer_idx - self.current_delays) % self.max_delay_steps
        env_indices = torch.arange(self.num_envs, device=self.device)
        delayed_action = self.delay_buffer[read_indices, env_indices]
        self.buffer_idx = (self.buffer_idx + 1) % self.max_delay_steps

        # Substep loop
        sub_dt = self.dt / self.substeps
        cur = state

        for _ in range(self.substeps):
            # Motor dynamics — match integrator
            if self.integrator == INTEGRATOR_RK4:
                self.motor_state, self.motor_velocity = step_motor_rk4(
                    delayed_action, self.motor_state, self.motor_velocity,
                    self.motor_omega_n, self.motor_zeta, sub_dt,
                )
            else:
                self.motor_state, self.motor_velocity = step_motor_euler(
                    delayed_action, self.motor_state, self.motor_velocity,
                    self.motor_omega_n, self.motor_zeta, sub_dt,
                )

            # Wind with turbulence
            self._wind_buffer.copy_(self.wind_velocity)
            turbulent_mask = self.wind_turbulence > 0
            if turbulent_mask.any():
                n = turbulent_mask.sum()
                noise = torch.randn(n, 3, device=self.device)
                noise *= self.wind_turbulence[turbulent_mask].unsqueeze(1)
                self._wind_buffer[turbulent_mask] += noise

            # Effective drag (fold in air density)
            eff_drag_xy = self.drag_coef_xy * self.air_density
            eff_drag_z = self.drag_coef_z * self.air_density

            cur = self.dynamics.step(
                state=cur,
                action=self.motor_state,
                dt=sub_dt,
                mass=self.mass,
                thrust_to_weight=self.thrust_to_weight,
                drag_coef_xy=eff_drag_xy,
                drag_coef_z=eff_drag_z,
                inertia_scale=self.inertia_scale,
                yaw_torque_coef=self.yaw_torque_coef,
                wind_velocity=self._wind_buffer,
                gravity=DEFAULT_GRAVITY,
                integrator=self.integrator,
                ground_effect_enabled=self.ground_effect,
                rotor_radius=self.rotor_radius,
            )

        self.last_accel = cur["acceleration"]

        return cur

    # Accessors

    def get_motor_state(self) -> torch.Tensor:
        return self.motor_state.clone()

    def get_motor_velocity(self) -> torch.Tensor:
        return self.motor_velocity.clone()