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

def ct_lqr_gain():
    Q = 1e1 * np.identity(6)
    R = np.identity(2)

    P = scipy.linalg.solve_continuous_are(A, B, Q, R)
    # K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    K = np.linalg.solve(R, B.T @ P)

    return K


def simulate(nstep, timestep, x0, controller):
    solver = scipy.integrate.ode(f)
    solver.set_integrator("dopri5")
    solver.set_initial_value(x0)

    xs = [np.zeros(6) for _ in range(nstep + 1)]
    us = [np.zeros(2) for _ in range(nstep + 1)]

    xs[0] = solver.y

    for k in range(nstep):
        u = controller(solver.y, k)
        solver.set_f_params(u)
        solver.integrate(solver.t + timestep)

        us[k] = u
        xs[k + 1] = solver.y

    us[nstep] = us[nstep - 1]

    return xs, us


def plot_trajectory(xs):
    fig, ax = plt.subplots()
    ax.plot([x[0] for x in xs], [x[1] for x in xs])

    return fig, ax
