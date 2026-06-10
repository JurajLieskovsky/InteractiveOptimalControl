import jax
import jax.numpy as np

g = 9.81
mass = 1
moi = 1
arm = 0.1


# equilibrium input
u_eq = mass * g / 2 * np.ones(2)


# nonlinear continuous-time dynamics
def f(_, x, u):
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


# differentiation function
def df(t, x, u):
    return jax.jacobian(f, argnums=(1, 2))(t, x, u)
