import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import dynamics as sim

    return mo, np, sim


@app.cell
def _(mo, np):
    initial_state = mo.ui.matrix(
        np.zeros(6),
        min_value=-5,
        max_value=5,
        step=0.1,
        debounce=True,
        label="initial state",
    )

    target_position = mo.ui.matrix(
        np.zeros(2),
        min_value=-5,
        max_value=5,
        step=0.1,
        debounce=True,
        label="target position",
    )

    mo.hstack([initial_state, target_position])
    return initial_state, target_position

@app.cell
def _(mo):
    timestep = mo.ui.number(value=1e-2, debounce=True, label="timestep")
    nstep = mo.ui.number(start=1, step=1, value=1000, debounce=True, label="number of steps")

    mo.vstack([timestep, nstep])
    return nstep, timestep
    
@app.cell
def _(np, sim, initial_state, target_position, nstep, timestep):
    x0 = np.array(initial_state.value)
    xt = np.array([target_position.value[0], target_position.value[1], 0, 0, 0, 0])

    K = sim.ct_lqr_gain()

    xs, _ = sim.simulate(
        nstep.value,
        timestep.value,
        x0,
        lambda x, _: sim.u_eq - K @ (x - xt),
    )

    fig, ax = sim.plot_trajectory(xs)

    ax.scatter(x0[0], x0[1], label="initial")
    ax.scatter(xt[0], xt[1], label="target")

    ax.legend()

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)

    fig
    return


if __name__ == "__main__":
    app.run()
