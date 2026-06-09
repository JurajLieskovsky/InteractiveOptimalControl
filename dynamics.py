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


def lqr_gain(Q, R, dt):
    dtA = np.eye(6) + dt * A
    dtB = dt * B

    P = scipy.linalg.solve_discrete_are(dtA, dtB, Q, R)
    K = np.linalg.solve(R + dtB.T @ P @ dtB, dtB.T @ P @ dtA)

    return K


def simulate(nstep, timestep, x0, controller):
    solver = scipy.integrate.ode(f)
    solver.set_integrator("dopri5")
    solver.set_initial_value(x0)

    ts = np.zeros(nstep + 1)
    xs = [np.zeros(6) for _ in range(nstep + 1)]
    us = [np.zeros(2) for _ in range(nstep + 1)]

    ts[0] = 0.0
    xs[0] = solver.y

    for k in range(nstep):
        u = controller(solver.y, k)
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

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    return fig, ax


def plot_states_and_inputs(ts, xs, us):
    fig, ax = plt.subplots(3)

    for i in range(3):
        ax[0].plot(ts, [x[i] for x in xs], label=f"x{i}")

    for i in range(3, 6):
        ax[1].plot(ts, [x[i] for x in xs], label=f"x{i}")

    for i in range(2):
        ax[2].step(ts, [u[i] for u in us], where="post", label=f"u{i}")

    ax[2].set_xlabel("t [s]")
    ax[2].set_ylabel("[N]")

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()

    return fig, ax
