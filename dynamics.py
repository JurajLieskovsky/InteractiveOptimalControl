import scipy
import numpy as np
import matplotlib.pyplot as plt

g = 9.81
mass = 1
moi = 1
arm = 0.1


# nonlinear continuous-time dynamics
def f(t, x, u):
    return np.array(
        [
            x[3],
            x[4],
            x[5],
            -np.sin(x[2]) * (u[0] + u[1]) / mass,
            np.cos(x[2]) * (u[0] + u[1]) / mass - g,
            arm * (u[0] - u[1]) / moi,
        ]
    )


# equilibrium input
u_eq = mass * g / 2 * np.ones(2)

# linearized continuous-time dynamics
A = np.array(
    [
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, -g, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
)

B = np.array(
    [
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [1 / mass, 1 / mass],
        [arm / moi, -arm / moi],
    ]
)


class LQR:
    def __init__(self, x_eq, u_eq, q, r, dt):
        self.x_eq = x_eq
        self.u_eq = u_eq

        dtA = np.eye(6) + dt * A
        dtB = dt * B

        P = scipy.linalg.solve_discrete_are(dtA, dtB, q, r)

        self.K = np.linalg.solve(r + dtB.T @ P @ dtB, dtB.T @ P @ dtA)

    def input(self, x, _):
        return self.u_eq - self.K @ (x - self.x_eq)


def input_saturation(u, u_min, u_max):
    if u < u_min:
        return u_min
    elif u > u_max:
        return u_max
    else:
        return u


def simulate(nstep, timestep, x0, controller, u_min=-np.inf, u_max=np.inf):
    solver = scipy.integrate.ode(f)
    solver.set_integrator("dopri5")
    solver.set_initial_value(x0)

    ts = np.zeros(nstep + 1)
    xs = [np.zeros(6) for _ in range(nstep + 1)]
    us = [np.zeros(2) for _ in range(nstep + 1)]

    ts[0] = 0.0
    xs[0] = solver.y

    for k in range(nstep):
        u = np.array(
            [input_saturation(input, u_min, u_max) for input in controller(solver.y, k)]
        )
        solver.set_f_params(u)
        solver.integrate(solver.t + timestep)

        us[k] = u
        xs[k + 1] = solver.y
        ts[k + 1] = solver.t

    us[nstep] = us[nstep - 1]

    return ts, xs, us


def plot_trajectory(xs):
    fig, ax = plt.subplots()

    ax.plot([x[0] for x in xs], [x[1] for x in xs])

    ax.set_xlabel(r"$x$ [m]")
    ax.set_ylabel(r"$y$ [m]")

    return fig, ax


def plot_states_and_inputs(ts, xs, us):
    fig, ax = plt.subplots(3)

    state_labels = [
        r"$x$ [m]",
        r"$y$ [m]",
        r"$\theta$ [rad]",
        r"$\dot{x}$ [m/s]",
        r"$\dot{y}$ [m/s]",
        r"$\dot{\theta}$ [rad/s]",
    ]
    input_labels = [
        r"$u_L$ [N]",
        r"$u_R$ [N]",
    ]

    for i, lbl in enumerate(state_labels[:3]):
        ax[0].plot(ts, [x[i] for x in xs], label=lbl)

    for i, lbl in enumerate(state_labels[3:]):
        ax[1].plot(ts, [x[i + 3] for x in xs], label=lbl)

    for i, lbl in enumerate(input_labels):
        ax[2].step(ts, [u[i] for u in us], where="post", label=lbl)

    ax[2].set_xlabel("$t$ [s]")

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()

    return fig, ax
