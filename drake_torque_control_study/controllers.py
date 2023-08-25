import dataclasses as dc
from textwrap import indent

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv

from pydrake.common.value import Value
from pydrake.math import RigidTransform
from pydrake.multibody.math import (
    SpatialAcceleration,
    SpatialForce,
    SpatialVelocity,
)
from pydrake.multibody.plant import ExternallyAppliedSpatialForce
from pydrake.multibody.tree import JacobianWrtVariable, ModelInstanceIndex
from pydrake.solvers import (
    ClpSolver,
    CommonSolverOption,
    GurobiSolver,
    MathematicalProgram,
    MosekSolver,
    OsqpSolver,
    ScsSolver,
    SnoptSolver,
    SolverOptions,
)
from pydrake.systems.framework import LeafSystem

from drake_torque_control_study.geometry import se3_vector_minus
from drake_torque_control_study.limits import PlantLimits, VectorLimits
from drake_torque_control_study.systems import declare_simple_init
from drake_torque_control_study.multibody_extras import calc_velocity_jacobian

from drake_torque_control_study.acceleration_bounds import compute_acceleration_bounds

SHOULD_STOP = False


class BaseController(LeafSystem):
    def __init__(self, plant, frame_W, frame_G):
        super().__init__()
        self.plant = plant
        self.frame_W = frame_W
        self.frame_G = frame_G
        self.context = plant.CreateDefaultContext()
        self.num_q = plant.num_positions()
        self.num_x = 2 * self.num_q
        assert plant.num_velocities() == self.num_q
        self.state_input = self.DeclareVectorInputPort("state", self.num_x)
        self.torques_output = self.DeclareVectorOutputPort(
            "torques",
            size=self.num_q,
            calc=self.calc_torques,
        )
        self.get_init_state = declare_simple_init(
            self,
            self.on_init,
        )
        self.check_limits = True
        self.nominal_limits = PlantLimits.from_plant(plant)
        # Will be set externally.
        self.traj = None

    def on_init(self, sys_context, init):
        x = self.state_input.Eval(sys_context)
        self.plant.SetPositionsAndVelocities(self.context, x)
        q = self.plant.GetPositions(self.context)
        init.q = q

    def calc_torques(self, sys_context, output):
        x = self.state_input.Eval(sys_context)
        t = sys_context.get_time()

        tol = 1e-4
        self.plant.SetPositionsAndVelocities(self.context, x)
        if self.check_limits:
            q = self.plant.GetPositions(self.context)
            v = self.plant.GetVelocities(self.context)
            self.nominal_limits.assert_values_within_limits(q=q, v=v, tol=tol)

        init = self.get_init_state(sys_context)
        q0 = init.q
        pose_actual = calc_spatial_values(
            self.plant, self.context, self.frame_W, self.frame_G
        )
        pose_desired = self.traj(t)
        tau = self.calc_control(t, pose_actual, pose_desired, q0)

        if self.check_limits:
            self.nominal_limits.assert_values_within_limits(u=tau, tol=tol)

        output.set_value(tau)

    def calc_control(self, t, pose_actual, pose_desired, q0):
        raise NotImplementedError()

    def show_plots(self):
        pass


@dc.dataclass
class Gains:
    kp: np.ndarray
    kd: np.ndarray

    @staticmethod
    def via_damping_ratio(kp, *, ratio=1.0):
        kd = ratio * 2 * np.sqrt(kp)
        return Gains(kp, kd)

    def __iter__(self):
        as_tuple = (self.kp, self.kd)
        return iter(as_tuple)


@dc.dataclass
class OscGains:
    task: Gains
    posture: Gains

    @staticmethod
    def via_damping_ratio(kp_t, kp_p, *, ratio_t=1.0, ratio_p=1.0):
        return OscGains(
            Gains.via_damping_ratio(kp_t, ratio=ratio_t),
            Gains.via_damping_ratio(kp_p, ratio=ratio_p),
        )

    def __iter__(self):
        as_tuple = (self.task, self.posture)
        return iter(as_tuple)


def calc_spatial_values(plant, context, frame_W, frame_G):
    X = plant.CalcRelativeTransform(context, frame_W, frame_G)
    J, Jdot_v = calc_velocity_jacobian(
        plant, context, frame_W, frame_G, include_bias=True
    )
    v = plant.GetVelocities(context)
    V = J @ v
    return X, V, J, Jdot_v


def calc_dynamics(plant, context):
    M = plant.CalcMassMatrix(context)
    C = plant.CalcBiasTerm(context)
    tau_g = plant.CalcGravityGeneralizedForces(context)
    return M, C, tau_g


def reproject_mass(Minv, Jt):
    _, num_dof = Jt.shape
    I_dyn = np.eye(num_dof)
    # Maps from task forces to task accelerations.
    Mtinv = Jt @ Minv @ Jt.T
    # Maps from task accelerations to task forces.
    Mt = np.linalg.inv(Mtinv)

    # # HACK
    # Jtpinv = np.linalg.pinv(Jt)
    # M = np.linalg.inv(Minv)
    # Mt = Jtpinv.T @ M @ Jtpinv
    # Mtinv = np.linalg.inv(Mt)

    # Maps from task accelerations to generalized accelerations.
    # Transpose maps from generalized forces to task forces.
    Jtbar = Minv @ Jt.T @ Mt
    # Generalized force nullspace.
    Nt_T = I_dyn - Jt.T @ Jtbar.T
    return Mt, Mtinv, Jt, Jtbar, Nt_T


def calc_null(J, Jpinv=None):
    n = J.shape[1]
    I = np.eye(n)
    if Jpinv is None:
        Jpinv = np.linalg.pinv(J)
    N = I - Jpinv @ J
    return N


class DiffIkAndId(BaseController):
    """Open/closed loop diff ik + inverse dynamics."""
    def __init__(self, plant, frame_W, frame_G, dt, gains_p):
        super().__init__(plant, frame_W, frame_G)
        self.dt = dt
        self.solver = OsqpSolver()
        self.solver_options = SolverOptions()
        self.gains_p = gains_p

        # Hacky state.
        self.should_save = False
        self.open_loop = False
        self.context_integ = None

    def calc_control(self, t, pose_actual, pose_desired, q0):
        if self.open_loop and self.context_integ is None:
            self.context_integ = self.plant.CreateDefaultContext()
            self.plant.SetPositions(self.context_integ, q0)

        q = self.plant.GetPositions(self.context)
        v = self.plant.GetVelocities(self.context)

        if self.open_loop:
            q_integ = self.plant.GetPositions(self.context_integ)
        else:
            q_integ = q

        # Compute desired joint velocity from diff ik.

        # Compute error in SE(3).
        if self.open_loop:
            pose_actual = calc_spatial_values(
                self.plant, self.context_integ, self.frame_W, self.frame_G
            )

        kp_ti = 10.0
        kp_pi = 10.0

        # Spatial feedback (integrated, so "ti").
        X, _, Jt, _ = pose_actual
        X_des, _, _ = pose_desired
        e_ti = se3_vector_minus(X, X_des)
        ed_ti_c = -kp_ti * e_ti
        # Posture feedback (integrated, so "pi").
        # Drive towards v = ed_pi_c = -k*e_pi in null-space Nt.
        Nt = calc_null(Jt)
        e_pi = q_integ - q0
        ed_pi_c = -kp_pi * e_pi
        # # hack
        # ed_pi_c *= -1

        direct_solve = True

        if direct_solve:
            Jtpinv = np.linalg.pinv(Jt)
            v_integ = Jtpinv @ ed_ti_c + Nt @ ed_pi_c
        else:
            # Formulate optimization.
            prog = MathematicalProgram()
            num_v = self.plant.num_velocities()
            num_t = 6
            v_next = prog.NewContinuousVariables(num_v, "v_next")
            alpha = prog.NewContinuousVariables(1, "alpha")
            # Scaling.
            weight = 100.0
            prog.AddLinearCost([-weight], alpha)
            # weight_s = np.sqrt(weight)
            # prog.Add2NormSquaredCost([weight_s], [weight_s * 1.0], alpha)
            prog.AddBoundingBoxConstraint([0.0], [1.0], alpha)
            # Jt*v_next = alpha*ed_ti_c
            prog.AddLinearEqualityConstraint(
                np.hstack([Jt, -ed_ti_c.reshape((-1, 1))]),
                np.zeros(num_t),
                np.hstack([v_next, alpha]),
            )
            # Null-space via cost as |Nt (v - ed_pi_c)|^2
            prog.Add2NormSquaredCost(Nt, Nt @ ed_pi_c, v_next)
            # prog.AddLinearEqualityConstraint(Nt, Nt @ ed_pi_c, v_next)
            # Solve.
            result = solve_or_die(self.solver, self.solver_options, prog)
            v_integ = result.GetSolution(v_next)

        # Internal integration.
        q_integ = q_integ + self.dt * v_integ
        if self.open_loop and self.should_save:
            self.plant.SetPositions(self.context_integ, q_integ)

        # Do basic ID, but closing loop on actual (not integrated).
        M, C, tau_g = calc_dynamics(self.plant, self.context)
        H = C - tau_g
        e_p = q - q_integ
        ed_p = v - v_integ
        kp_p, kd_p = self.gains_p
        edd_p_c = -kp_p * e_p - kd_p * ed_p
        u = M @ edd_p_c + H
        return u


class ResolvedAcc(BaseController):
    """I think resolved accel."""
    def __init__(self, plant, frame_W, frame_G, gains):
        super().__init__(plant, frame_W, frame_G)
        self.gains = gains

    def calc_control(self, t, pose_actual, pose_desired, q0):
        M, C, tau_g = calc_dynamics(self.plant, self.context)
        H = C - tau_g
        q = self.plant.GetPositions(self.context)
        v = self.plant.GetVelocities(self.context)

        (kp_t, kd_t) = self.gains.task
        (kp_p, kd_p) = self.gains.posture

        # Compute spatial feedback.
        X, V, Jt, Jtdot_v = pose_actual
        X_des, V_des, A_des = pose_desired
        V_des = V_des.get_coeffs()
        A_des = A_des.get_coeffs()
        e_t = se3_vector_minus(X, X_des)
        ed_t = V - V_des
        edd_t_c = A_des - kp_t * e_t - kd_t * ed_t

        # Compute posture feedback.
        e_p = q - q0
        ed_p = v
        edd_p_c = -kp_p * e_p - kd_p * ed_p

        Jtpinv = np.linalg.pinv(Jt)
        Nt = calc_null(Jt, Jtpinv)

        # Sum up tasks and cancel gravity + Coriolis terms.
        vd_c = Jtpinv @ edd_t_c + Nt @ edd_p_c
        tau = H + M @ vd_c
        return tau


def vec_dot_norm(a, b, *, tol=1e-8):
    n = np.linalg.norm(a) * np.linalg.norm(b)
    if n <= tol:
        return 0.0
    else:
        # arcos of this value gives angle.
        return a.dot(b) / n


class Osc(BaseController):
    """Explicit OSC."""
    def __init__(self, plant, frame_W, frame_G, gains):
        super().__init__(plant, frame_W, frame_G)
        self.gains = gains

        self.should_save = False
        self.ts = []
        self.e_p_dirs = []
        self.e_ps = []
        self.e_ps_null_acc = []
        self.e_ps_null_force = []

    def calc_control(self, t, pose_actual, pose_desired, q0):
        M, C, tau_g = calc_dynamics(self.plant, self.context)
        H = C - tau_g
        Minv = inv(M)
        q = self.plant.GetPositions(self.context)
        v = self.plant.GetVelocities(self.context)

        (kp_t, kd_t) = self.gains.task
        (kp_p, kd_p) = self.gains.posture

        # Compute spatial feedback.
        X, V, Jt, Jtdot_v = pose_actual
        Mt, _, _, _, Nt_T = reproject_mass(Minv, Jt)
        Nt = Nt_T.T
        X_des, V_des, A_des = pose_desired
        V_des = V_des.get_coeffs()
        A_des = A_des.get_coeffs()
        e_t = se3_vector_minus(X, X_des)
        ed_t = V - V_des
        edd_t_c = A_des - kp_t * e_t - kd_t * ed_t
        # edd_t_c = -kp_t * e_t - kd_t * ed_t
        Ft = Mt @ (edd_t_c - Jtdot_v)

        Nt_kin = calc_null(Jt)

        # Compute posture feedback.
        e_p = q - q0
        e_p_orig = e_p.copy()
        # # TODO(eric.cousineau): Find more prinicipled setup?
        e_p_dir = vec_dot_norm(e_p, Nt @ e_p)
        # e_p *= e_p_dir  # seems ok
        # e_p = Nt_kin @ e_p  # decent, but more null-space error
        # e_p = Nt @ e_p  # doesn't help failing case
        # e_p_dir = vec_dot_norm(Nt_kin @ e_p, Nt @ e_p)  # very... discrete?
        # F_e_p = M @ e_p
        # e_p_dir = vec_dot_norm(F_e_p, Nt_T @ F_e_p)
        # e_p *= np.sign(e_p_dir)
        # e_p *= np.sign(e_p * (Nt @ e_p))

        ed_p = v
        edd_p_c = -kp_p * e_p - kd_p * ed_p
        Fp = M @ edd_p_c

        # Sum up tasks and cancel gravity + Coriolis terms.
        tau = H + Jt.T @ Ft + Nt_T @ Fp

        if self.should_save:
            self.ts.append(t)
            self.e_p_dirs.append(e_p_dir)
            self.e_ps.append(e_p_orig)
            self.e_ps_null_acc.append(Nt @ e_p)
            self.e_ps_null_force.append(Nt_T @ M @ e_p)

        return tau

    def show_plots(self):
        ts = np.array(self.ts)
        e_p_dirs = np.array(self.e_p_dirs)
        e_ps = np.array(self.e_ps)
        e_ps_null_acc = np.array(self.e_ps_null_acc)
        e_ps_null_force = np.array(self.e_ps_null_force)
        e_p_mags = np.linalg.norm(e_ps, axis=-1)

        _, axs = plt.subplots(num=1, nrows=3)
        plt.sca(axs[0])
        plt.plot(ts, e_ps)
        legend_for(e_ps)
        plt.sca(axs[1])
        plt.plot(ts, e_ps_null_acc)
        plt.sca(axs[2])
        plt.plot(ts, e_ps_null_force)
        plt.tight_layout()

        _, axs = plt.subplots(num=2, nrows=3)
        plt.sca(axs[0])
        plt.plot(ts, e_p_mags)
        plt.sca(axs[1])
        plt.plot(ts, e_p_dirs)
        plt.sca(axs[2])
        plt.plot(ts, e_p_mags * e_p_dirs)
        plt.tight_layout()

        plt.show()


def make_osqp_solver_and_options(use_dairlab_settings=False):
    solver = OsqpSolver()
    solver_id = solver.solver_id()
    solver_options = SolverOptions()
    # https://osqp.org/docs/interfaces/solver_settings.html#solver-settings
    solver_options_dict = dict(
        # See https://github.com/RobotLocomotion/drake/issues/18711
        adaptive_rho=0,
    )
    if use_dairlab_settings:
        # https://github.com/DAIRLab/dairlib/blob/0da42bc2/examples/Cassie/osc_run/osc_running_qp_settings.yaml
        solver_options_dict.update(
            # rho=3.0,
            # rho=0.5,
            # sigma=1.0,
            # alpha=1.9,
            # sigma=100,  # er, int values messes things up?
            # rho=0.1,
            # sigma=1e-6,
            # max_iter=250,
            # max_iter=500,
            # max_iter=1000,
            # max_iter=2000,
            # max_iter=10000,
            # eps_abs=1e-3,
            # eps_rel=1e-4,
            # eps_abs=5e-4,
            # eps_rel=5e-4,
            # eps_abs=1e-5,
            # eps_rel=1e-5,
            # eps_abs=1e-6,
            # eps_rel=1e-6,
            # eps_prim_inf=1e-5,
            # eps_dual_inf=1e-5,
            # polish=1,
            # polish_refine_iter=1,
            # scaled_termination=1,
            # scaling=1,
        )
    for name, value in solver_options_dict.items():
        solver_options.SetOption(solver_id, name, value)
    return solver, solver_options


def make_clp_solver_and_options():
    solver = ClpSolver()
    solver_options = SolverOptions()
    return solver, solver_options


def make_gurobi_solver_and_options():
    solver = GurobiSolver()
    solver_options = SolverOptions()
    return solver, solver_options


def make_mosek_solver_and_options():
    solver = MosekSolver()
    solver_options = SolverOptions()
    return solver, solver_options


def make_snopt_solver_and_options():
    solver = SnoptSolver()
    solver_options = SolverOptions()
    return solver, solver_options

def make_scs_solver_and_options():
    solver = ScsSolver()
    solver_options = SolverOptions()
    return solver, solver_options


def solve_or_die(solver, solver_options, prog, *, x0=None):
    result = solver.Solve(
        prog, solver_options=solver_options, initial_guess=x0
    )
    if not result.is_success():
        solver_options.SetOption(
            CommonSolverOption.kPrintToConsole, True
        )
        solver.Solve(prog, solver_options=solver_options)
        print("\n".join(result.GetInfeasibleConstraintNames(prog)))
        print(result.get_solution_result())
        raise RuntimeError("Bad solution")
    return result


def vd_limits_from_tau(u_limits, Minv, H):
    num_u = len(H)
    if not u_limits.isfinite():
        return VectorLimits(
            lower=-np.inf * np.ones(num_u),
            upper=np.inf * np.ones(num_u),
        )
    u_min, u_max = u_limits
    vd_tau_limits = VectorLimits(
        lower=Minv @ (u_min - H),
        upper=Minv @ (u_max - H),
    )
    # TODO(eric.cousineau): Is this even right? How to handle sign flip?
    vd_tau_limits = vd_tau_limits.make_valid()
    # assert vd_tau_limits.is_valid()
    return vd_tau_limits


def add_plant_limits_to_qp(
    *,
    plant_limits,
    dt,
    q,
    v,
    prog,
    Au,
    bu,
    u_vars,
    Avd,
    bvd,
    vd_vars,
):
    num_v = len(v)
    Iv = np.eye(num_v)

    # Rescale gains.
    # TODO(eric.cousineau): Have alpha functions modulate gains closer
    # to boundary.
    # TODO(eric.cousineau): Make this configurable.
    # WARNING: These are sensitive to primary tracking gains :(
    q_dt_scale = 25
    v_dt_scale = 10

    # CBF-esque formulation.
    # WARNING: This is not a certificate-based barrier function, thus it
    # may lead to non-forward invariant behavior (e.g. one time-step is
    # feasible, the next time-step is not). Note that gain tuning is
    # something that will happen here.

    # N.B. Nominal CBFs (c*vd >= b) are lower bounds. For CBFs where c=-1,
    # we can pose those as upper bounds (vd <= -b).

    # Goal: h >= 0 for all admissible states
    # hdd = c*vd >= -k_1*h -k_2*hd = b

    if plant_limits.q.any_finite():
        q_min, q_max = plant_limits.q

        # Gains corresponding to naive formulation.
        aq_1 = lambda x: x
        aq_2 = aq_1
        kq_1 = 2 / (dt * dt)
        kq_2 = 2 / dt
        # Rescale.
        kq_1 /= q_dt_scale**2
        kq_2 /= v_dt_scale

        # q_min
        h_q_min = q - q_min
        hd_q_min = v
        c_q_min = 1
        b_q_min = -kq_1 * aq_1(h_q_min) - kq_2 * aq_2(hd_q_min)
        # q_max
        h_q_max = q_max - q
        hd_q_max = -v
        c_q_max = -1
        b_q_max = -kq_1 * aq_1(h_q_max) - kq_2 * aq_2(hd_q_max)

        prog.AddLinearConstraint(
            Avd,
            b_q_min - bvd,
            -b_q_max - bvd,
            vd_vars,
        ).evaluator().set_description("pos cbf ish")

    if plant_limits.v.any_finite():
        v_min, v_max = plant_limits.v

        # Gains corresponding to naive formulation.
        av_1 = lambda x: x
        kv_1 = 1 / dt
        # Rescale.
        kv_1 /= v_dt_scale

        # v_min
        h_v_min = v - v_min
        c_v_min = 1
        b_v_min = -kv_1 * av_1(h_v_min)
        # v_max
        h_v_max = v_max - v
        c_v_max = -1
        b_v_max = -kv_1 * av_1(h_v_max)

        prog.AddLinearConstraint(
            Avd,
            b_v_min - bvd,
            -b_v_max - bvd,
            vd_vars,
        ).evaluator().set_description("vel cbf ish")

    if plant_limits.vd.any_finite():
        vd_min, vd_max = plant_limits.vd
        prog.AddLinearConstraint(
            Avd,
            vd_min - bvd,
            vd_max - bvd,
            vd_vars,
        ).evaluator().set_description("accel")

    if plant_limits.u.any_finite():
        u_min, u_max = plant_limits.u
        prog.AddLinearConstraint(
            Au,
            u_min - bu,
            u_max - bu,
            u_vars,
        ).evaluator().set_description("torque")


class QpWithCosts(BaseController):
    def __init__(
        self,
        plant,
        frame_W,
        frame_G,
        *,
        gains,
        plant_limits,
        acceleration_bounds_dt,
        posture_weight,
        split_costs=None,
        use_torque_weights=False,
    ):
        super().__init__(plant, frame_W, frame_G)
        self.gains = gains
        self.plant_limits = plant_limits
        self.solver, self.solver_options = make_osqp_solver_and_options()
        # self.solver, self.solver_options = make_snopt_solver_and_options()
        self.acceleration_bounds_dt = acceleration_bounds_dt
        self.posture_weight = posture_weight
        self.split_costs = split_costs
        self.use_torque_weights = use_torque_weights

    def calc_control(self, t, pose_actual, pose_desired, q0):
        M, C, tau_g = calc_dynamics(self.plant, self.context)
        Minv = inv(M)

        q = self.plant.GetPositions(self.context)
        v = self.plant.GetVelocities(self.context)

        # Base QP formulation.
        Iv = np.eye(self.num_q)
        zv = np.zeros(self.num_q)
        prog = MathematicalProgram()

        vd_star = prog.NewContinuousVariables(self.num_q, "vd_star")
        u_star = prog.NewContinuousVariables(self.num_q, "u_star")

        # Dynamics constraint.
        dyn_vars = np.concatenate([vd_star, u_star])
        dyn_A = np.hstack([M, -Iv])
        dyn_b = -C + tau_g
        prog.AddLinearEqualityConstraint(
            dyn_A, dyn_b, dyn_vars
        ).evaluator().set_description("dyn")

        # Add limits.
        add_plant_limits_to_qp(
            plant_limits=self.plant_limits,
            dt=self.acceleration_bounds_dt,
            q=q,
            v=v,
            prog=prog,
            vd_vars=vd_star,
            Avd=np.eye(self.num_q),
            bvd=np.zeros(self.num_q),
            u_vars=u_star,
            Au=np.eye(self.num_q),
            bu=np.zeros(self.num_q),
        )

        # Compute spatial feedback.
        gains_t = self.gains.task
        X, V, Jt, Jtdot_v = pose_actual
        X_des, V_des, A_des = pose_desired
        V_des = V_des.get_coeffs()
        A_des = A_des.get_coeffs()
        e = se3_vector_minus(X, X_des)
        ed = V - V_des
        edd_c = A_des - gains_t.kp * e - gains_t.kd * ed

        Mt, Mtinv, Jt, Jtbar, Nt_T = reproject_mass(Minv, Jt)

        # Drive towards desired tracking, |(J*vdot + Jdot*v) - (edd_c)|^2
        task_A = Jt
        task_b = -Jtdot_v + edd_c

        num_t = 6
        It = np.eye(num_t)
        if self.use_torque_weights:
            # task_proj = Jt.T @ Mt
            task_proj = Mt
        else:
            task_proj = It
        task_A = task_proj @ task_A
        task_b = task_proj @ task_b
        # add_2norm_row_decoupled(prog, task_A, task_b, vd_star)
        if self.split_costs is None:
            prog.Add2NormSquaredCost(task_A, task_b, vd_star)
        else:
            slices = [slice(0, 3), slice(3, 6)]
            for weight_i, slice_i in zip(self.split_costs, slices):
                prog.Add2NormSquaredCost(
                    weight_i * task_A[slice_i],
                    weight_i* task_b[slice_i],
                    vd_star,
                )

        # Compute posture feedback.
        gains_p = self.gains.posture
        e = q - q0
        ed = v
        edd_c = -gains_p.kp * e - gains_p.kd * ed
        # Same as above, but lower weight.
        weight = self.posture_weight
        if self.use_torque_weights:
            task_proj = weight * Nt_T
        else:
            task_proj = weight * Iv
        task_A = task_proj
        task_b = task_proj @ edd_c
        prog.Add2NormSquaredCost(task_A, task_b, vd_star)
        # add_2norm_row_decoupled(prog, task_A, task_b, vd_star)

        # Solve.
        result = solve_or_die(self.solver, self.solver_options, prog)
        tau = result.GetSolution(u_star)

        return tau


def add_2norm_decoupled(prog, a, b, x):
    Q = 2 * np.diag(a * a)
    b = -2 * a * b
    c = np.sum(b * b)
    return prog.AddQuadraticCost(Q, b, c, x)


def add_2norm_row_decoupled(prog, A, b, x):
    for i in range(A.shape[0]):
        prog.Add2NormSquaredCost(A[i:i + 1, :], b[i:i + 1], x)


from pydrake.autodiffutils import (
    AutoDiffXd,
    ExtractGradient,
    ExtractValue,
    InitializeAutoDiff,
)


@np.vectorize
def ad_value(x):
    return x.value()


def ad_grad(x):
    xf = x.reshape(-1)
    nderiv = max(len(xi.derivatives()) for xi in xf)
    shape = x.shape + (nderiv,)
    J = np.zeros(shape)
    Jf = J.reshape((-1, nderiv))
    for xi, Ji in zip(xf, Jf):
        d = xi.derivatives()
        if len(d) > 0:
            Ji[:] = d
    return J


def calc_manip_index(
    plant,
    context,
    frame_W,
    frame_G,
    plant_ad,
    context_ad,
):
    # Based on:
    # https://github.com/vincekurtz/passivity_cbf_demo/blob/f27d8dc4/controller.py#L656-L659
    q = plant.GetPositions(context)
    v = plant.GetVelocities(context)
    q_ad = InitializeAutoDiff(q)
    plant_ad.SetPositions(context_ad, q_ad)
    frame_W_ad = plant_ad.get_frame(frame_W.index())
    frame_G_ad = plant_ad.get_frame(frame_G.index())
    J_ad = calc_velocity_jacobian(
        plant_ad, context_ad, frame_W_ad, frame_G_ad
    )
    J = ad_value(J_ad)
    Jpinv = np.linalg.pinv(J)
    _, s, _ = np.linalg.svd(J)
    mu = np.prod(s)
    dJ_dq = ad_grad(J_ad)
    num_q = len(q)
    Jmu = np.zeros(num_q)
    for i in range(num_q):
        Jmu[i] = np.trace(dJ_dq[:, :, i] @ Jpinv)
    return mu, Jmu


class QpWithDirConstraint(BaseController):
    def __init__(
        self,
        plant,
        frame_W,
        frame_G,
        *,
        gains,
        plant_limits,
        acceleration_bounds_dt,
    ):
        super().__init__(plant, frame_W, frame_G)
        self.gains = gains
        self.plant_limits = plant_limits

        # Can be a bit imprecise, but w/ tuning can improve.
        self.solver, self.solver_options = make_osqp_solver_and_options()

        # Best, it seems like?
        # self.solver, self.solver_options = make_snopt_solver_and_options()

        # self.solver, self.solver_options = make_clp_solver_and_options()

        # self.solver, self.solver_options = make_gurobi_solver_and_options()

        # self.solver, self.solver_options = make_scs_solver_and_options()

        # Infeasible for implicit=True. Good for implicit=False.
        # self.solver, self.solver_options = make_mosek_solver_and_options()

        self.acceleration_bounds_dt = acceleration_bounds_dt

        self.prev_sol = None

        self.should_save = False
        self.ts = []
        self.qs = []
        self.vs = []
        self.us = []
        self.edd_ts = []
        self.s_ts = []
        self.r_ts = []
        self.edd_ps = []
        self.edd_ps_null = []
        self.s_ps = []
        # self.r_ps = []
        self.limits_infos = []
        self.sigmas = []
        self.prev_dir = None
        self.Jmu_prev = None

    def calc_control(self, t, pose_actual, pose_desired, q0):
        q = self.plant.GetPositions(self.context)
        v = self.plant.GetVelocities(self.context)
        num_v = len(v)
        M, C, tau_g = calc_dynamics(self.plant, self.context)
        Minv = inv(M)
        H = C - tau_g

        # Base QP formulation.
        Iv = np.eye(self.num_q)
        zv = np.zeros(self.num_q)
        prog = MathematicalProgram()

        X, V, Jt, Jtdot_v = pose_actual
        Mt, Mtinv, Jt, Jtbar, Nt_T = reproject_mass(Minv, Jt)

        # Compute spatial feedback.
        kp_t, kd_t = self.gains.task
        num_t = 6
        It = np.eye(num_t)
        X_des, V_des, A_des = pose_desired
        V_des = V_des.get_coeffs()
        A_des = A_des.get_coeffs()
        e_t = se3_vector_minus(X, X_des)
        ed_t = V - V_des
        edd_t_c = A_des - kp_t * e_t - kd_t * ed_t

        # Compute posture feedback.
        kp_p, kd_p = self.gains.posture
        e_p = q - q0
        ed_p = v
        edd_p_c = -kp_p * e_p - kd_p * ed_p

        # *very* sloppy looking
        scale_A_t = np.eye(num_t)
        # # better, but may need relaxation
        # scale_A_t = np.ones((num_t, 1))
        # can seem "loose" towards end of traj for rotation
        # (small feedback -> scale a lot). relaxing only necessary for
        # implicit version.
        # scale_A_t = np.array([
        #     [1, 1, 1, 0, 0, 0],
        #     [0, 0, 0, 1, 1, 1],
        # ]).T

        num_scales_t = scale_A_t.shape[1]
        scale_vars_t = prog.NewContinuousVariables(num_scales_t, "scale_t")

        scale_A_p = np.ones((num_v, 1))
        # scale_A_p = np.eye(num_v)
        num_scales_p = scale_A_p.shape[1]
        scale_vars_p = prog.NewContinuousVariables(num_scales_p, "scale_p")

        proj_t = Jt.T @ Mt
        proj_p = Nt_T @ M

        u_vars = scale_vars_t
        Au_t = proj_t @ np.diag(edd_t_c) @ scale_A_t
        bu_t = -proj_t @ Jtdot_v

        u_vars = np.concatenate([u_vars, scale_vars_p])
        Au_p = proj_p @ np.diag(edd_p_c) @ scale_A_p
        bu_p = np.zeros(num_v)

        Au = np.hstack([Au_t, Au_p])
        bu = H + bu_t + bu_p

        vd_vars = u_vars
        Avd = Minv @ Au
        bvd = Minv @ (bu - H)

        # Add limits.
        add_plant_limits_to_qp(
            plant_limits=self.plant_limits,
            dt=self.acceleration_bounds_dt,
            q=q,
            v=v,
            prog=prog,
            vd_vars=vd_vars,
            Avd=Avd,
            bvd=bvd,
            u_vars=u_vars,
            Au=Au,
            bu=bu,
        )

        # Optimize towards scale=1.
        desired_scales_t = np.ones(num_scales_t)
        desired_scales_p = np.ones(num_scales_p)
        add_2norm_decoupled(
            prog,
            np.ones(num_scales_t),
            desired_scales_t,
            scale_vars_t,
        )
        add_2norm_decoupled(
            prog,
            np.ones(num_scales_p),
            desired_scales_p,
            scale_vars_p,
        )

        # Solve.
        result = solve_or_die(
            self.solver, self.solver_options, prog, x0=self.prev_sol
        )

        tol = 1e-5

        infeas = result.GetInfeasibleConstraintNames(prog, tol=tol)
        infeas_text = "\n" + indent("\n".join(infeas), "  ")
        assert len(infeas) == 0, infeas_text
        self.prev_sol = result.get_x_val()

        u_mul = result.GetSolution(u_vars)
        tau = Au @ u_mul + bu

        # tau = self.plant_limits.u.saturate(tau)

        # import pdb; pdb.set_trace()

        edd_c_p_null = Minv @ Nt_T @ M @ edd_p_c
        _, sigmas, _ = np.linalg.svd(Jt)
        scale_t = result.GetSolution(scale_vars_t)
        scale_p = result.GetSolution(scale_vars_p)

        if self.should_save:
            self.ts.append(t)
            self.qs.append(q)
            self.vs.append(v)
            self.us.append(tau)
            self.edd_ts.append(edd_t_c)
            self.s_ts.append(scale_t)
            # self.r_ts.append(relax_t)
            self.edd_ps.append(edd_p_c)
            self.edd_ps_null.append(edd_c_p_null)
            self.s_ps.append(scale_p)
            # self.limits_infos.append(limit_info)
            self.sigmas.append(sigmas)

        return tau

    def show_plots(self):
        ts = np.array(self.ts)
        qs = np.array(self.qs)
        vs = np.array(self.vs)
        us = np.array(self.us)
        edd_ts = np.array(self.edd_ts)
        s_ts = np.array(self.s_ts)
        r_ts = np.array(self.r_ts)
        edd_ps = np.array(self.edd_ps)
        edd_ps_null = np.array(self.edd_ps_null)
        s_ps = np.array(self.s_ps)
        sigmas = np.array(self.sigmas)

        # sub = slice(3, 4)
        sub = slice(None, None)
        sel = (slice(None, None), sub)

        def plot_lim(limits):
            lower, upper = limits
            lower = lower[sub]
            upper = upper[sub]
            ts_lim = ts[[0, -1]]
            reset_color_cycle()
            plt.plot(ts_lim, [lower, lower], ":")
            reset_color_cycle()
            plt.plot(ts_lim, [upper, upper], ":")

        _, axs = plt.subplots(num=1, nrows=3)
        plt.sca(axs[0])
        plt.plot(ts, qs[sel])
        legend_for(qs[sel])
        plot_lim(self.plant_limits.q)
        plt.title("q")
        plt.sca(axs[1])
        plt.plot(ts, vs[sel])
        legend_for(vs[sel])
        plot_lim(self.plant_limits.v)
        plt.title("v")
        plt.sca(axs[2])
        plt.plot(ts, us[sel])
        legend_for(us[sel])
        plot_lim(self.plant_limits.u)
        plt.title("u")
        plt.tight_layout()

        # plt.show()
        # return  # HACK

        _, axs = plt.subplots(num=2, nrows=3)
        plt.sca(axs[0])
        plt.plot(ts, edd_ts)
        legend_for(edd_ts)
        plt.title("edd_t_c")
        plt.sca(axs[1])
        plt.plot(ts, edd_ps)
        legend_for(edd_ps)
        plt.title("edd_p_c")
        plt.sca(axs[2])
        plt.plot(ts, edd_ps_null)
        legend_for(edd_ps_null)
        plt.title("edd_p_c null")
        plt.tight_layout()

        _, axs = plt.subplots(num=3, nrows=3)
        plt.sca(axs[0])
        plt.plot(ts, s_ts)
        plt.ylim(-5, 5)
        legend_for(s_ts)
        plt.title("s_t")
        plt.sca(axs[1])
        if len(r_ts) > 0:
            plt.plot(ts, r_ts)
            legend_for(r_ts)
        plt.title("r_t")
        plt.sca(axs[2])
        if len(s_ps) > 0:
            plt.plot(ts, s_ps)
            plt.ylim(-5, 5)
            legend_for(s_ps)
        plt.title("s_p")
        plt.tight_layout()

        _, axs = plt.subplots(num=4, nrows=2)
        plt.sca(axs[0])
        plt.plot(ts, sigmas)
        legend_for(sigmas)
        plt.title("singular values")
        plt.sca(axs[1])
        manips = np.prod(sigmas, axis=-1)
        plt.plot(ts, manips)
        plt.title("manip index")
        plt.tight_layout()

        plt.show()


def legend_for(xs):
    n = xs.shape[1]
    labels = [f"{i}" for i in range(1, n + 1)]
    plt.legend(labels, loc="center left", bbox_to_anchor=(1, 0.5))


def reset_color_cycle():
    # https://stackoverflow.com/a/39116381/7829525
    plt.gca().set_prop_cycle(None)
