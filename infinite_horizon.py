import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import dynamics as sim

    return mo, np, sim


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Infinite Horizon Regulation""")


@app.cell
def _(mo, np):
    state_weights = np.array([10, 10, 10, 1, 1, 1])
    input_weights = np.array([1, 1])

    alpha_step = 1e-4
    alpha = mo.ui.slider(
        start=alpha_step,
        stop=1 - alpha_step,
        step=alpha_step,
        value=0.5,
        debounce=True,
        label="alpha",
    )

    timestep = mo.ui.number(value=1e-2, debounce=True, label="timestep")
    nstep = mo.ui.number(
        start=1, step=1, value=500, debounce=True, label="number of steps"
    )

    initial_state = mo.ui.matrix(
        np.array([3, 4, 0, 0, 0, 0]),
        min_value=-5,
        max_value=5,
        step=0.1,
        debounce=True,
        label="Initial state",
    )

    target_position = mo.ui.matrix(
        np.zeros(2),
        min_value=-5,
        max_value=5,
        step=0.1,
        debounce=True,
        label="Target position",
    )

    mo.vstack(
        [
            mo.hstack([timestep, mo.md("s")], justify="start"),
            nstep,
            mo.md(f"diag(Q) = alpha * Q0, where Q0 = {state_weights.tolist()}"),
            mo.md(f"diag(R) = (1 - alpha) * R0, where  R0 = {input_weights.tolist()}"),
            alpha,
            mo.hstack([initial_state, target_position], justify="start"),
        ]
    )

    return initial_state, input_weights, state_weights, target_position


@app.cell
def _(
    initial_state,
    input_weights,
    np,
    nstep,
    sim,
    state_weights,
    target_position,
    alpha,
    timestep,
):
    x0 = np.array(initial_state.value)
    xt = np.array([target_position.value[0], target_position.value[1], 0, 0, 0, 0])

    q = alpha.value * np.diag(state_weights)
    r = (1 - alpha.value) * np.diag(input_weights)
    dt = timestep.value

    lqr_controller = sim.LQR(xt, sim.u_eq, q, r, dt)

    ts, xs, us = sim.simulate(nstep.value, timestep.value, x0, lqr_controller.input)

    return ts, us, xs, x0, xt


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## States and Inputs""")


@app.cell
def plot(sim, ts, us, xs):
    fig2, ax2 = sim.plot_states_and_inputs(ts, xs, us)

    fig2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Trajectory""")


@app.cell
def _(sim, x0, xs, xt):

    fig1, ax1 = sim.plot_trajectory(xs)

    ax1.scatter(x0[0], x0[1], label="initial")
    ax1.scatter(xt[0], xt[1], label="target")

    ax1.legend()

    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-5, 5)

    fig1
    return


if __name__ == "__main__":
    app.run()
