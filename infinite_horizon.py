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
def _(mo):
    target_x = mo.ui.number(
        start=-10, stop=10, step=0.1, value=0, debounce=True, label="xt"
    )
    target_y = mo.ui.number(
        start=-10, stop=10, step=0.1, value=0, debounce=True, label="yt"
    )

    mo.vstack([target_x, target_y])
    return target_x, target_y


@app.cell
def _(np, sim, target_x, target_y):

    x_eq = np.array([target_x.value, target_y.value, 0, 0, 0, 0.0])

    K = sim.ct_lqr_gain()

    xs, _ = sim.simulate(
        500,
        1e-2,
        np.zeros(6),
        lambda x, _: sim.u_eq - K @ (x - x_eq),
    )

    fig, ax = sim.plot_trajectory(xs)

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    fig
    return


if __name__ == "__main__":
    app.run()
