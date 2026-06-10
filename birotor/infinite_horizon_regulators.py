import scipy
import numpy as np
import cvxpy as cp

from . import dynamics

class _LQR:
    def __init__(self, x_eq, u_eq, q, r, dt):
        self.x_eq = x_eq
        self.u_eq = u_eq

        self.q = q
        self.r = r

        self.A = np.eye(6) + dt * dynamics.A
        self.B = dt * dynamics.B

        self.P = scipy.linalg.solve_discrete_are(self.A, self.B, self.q, self.r)

    def running_cost(self, x, u, _):
        return x.T @ self.q @ x + u.T @ self.r @ u

    def final_cost(self, x):
        return x.T @ self.P @ x


class LQR(_LQR):
    def __init__(self, x_eq, u_eq, q, r, dt):
        super().__init__(x_eq, u_eq, q, r, dt)

        self.K = np.linalg.solve(
            r + self.B.T @ self.P @ self.B, self.B.T @ self.P @ self.A
        )

    def input(self, x, _):
        return self.u_eq - self.K @ (x - self.x_eq)


class MPC(_LQR):
    def __init__(self, M, x_eq, u_eq, q, r, dt, u_min=-np.inf, u_max=np.inf):
        super().__init__(x_eq, u_eq, q, r, dt)

        self.x = cp.Variable((6, M + 1))
        self.u = cp.Variable((2, M))

        self.x_init = cp.Parameter(6)

        constraints = [
            self.x[:, 1:] == self.A @ self.x[:, :-1] + self.B @ self.u,
            self.x[:, 0] == self.x_init,
            self.u >= u_min - u_eq[:, np.newaxis],
            self.u <= u_max - u_eq[:, np.newaxis],
        ]

        LQ = np.linalg.cholesky(self.q)
        LR = np.linalg.cholesky(self.r)

        objective = cp.Minimize(
            cp.sum_squares(LQ.T @ self.x[:, :-1])
            + cp.sum_squares(LR.T @ self.u)
            + cp.quad_form(self.x[:, 10], self.P)
        )

        self.problem = cp.Problem(objective, constraints)

    def input(self, x, _):
        self.x_init.value = x - self.x_eq
        self.problem.solve(solver=cp.OSQP, warm_starting=True, polish=True)

        return self.u_eq + self.u.value[:, 0]  # ty:ignore[not-subscriptable]


