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
        label=r"$\alpha$",
    )

    timestep = mo.ui.number(value=1e-2, debounce=True, label=r"$\Delta t$ [s]")
    nstep = mo.ui.number(start=1, step=1, value=500, debounce=True, label=r"$N$")

    saturation_checkbox = mo.ui.checkbox(label=r"$u_{\min \& \max}$ [N]")
    saturation_slider = mo.ui.range_slider(start=-10, stop=10, step=0.5, value=[0, 10])

    initial_state = mo.ui.matrix(
        np.array([3, 4, 0, 0, 0, 0]),
        min_value=-5,
        max_value=5,
        step=0.5,
        debounce=True,
        label=r"$x_0$",
    )

    target_position = mo.ui.matrix(
        np.zeros(2),
        min_value=-5,
        max_value=5,
        step=0.5,
        debounce=True,
        label=r"$q_t$",
    )

    return (
        timestep,
        nstep,
        alpha,
        initial_state,
        input_weights,
        state_weights,
        target_position,
        saturation_checkbox,
        saturation_slider,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Settings""")


@app.cell()
def _(
    mo,
    nstep,
    timestep,
    alpha,
    initial_state,
    input_weights,
    state_weights,
    target_position,
    saturation_checkbox,
    saturation_slider,
):
        cond_saturation_slider = saturation_slider if saturation_checkbox.value else mo.Html(
            '<div style="opacity: 0.3; pointer-events: none; user-select: none;">'
            + saturation_slider._repr_html_()
            + "</div>"
        )

        mo.accordion(
        {
            "Number of steps and timestep": mo.vstack([nstep, timestep]),
            "Input saturation":
                    mo.hstack(
                        [saturation_checkbox, cond_saturation_slider],
                        justify="start",
                    ),
            "Cost weights" : mo.vstack(
                [
                    mo.md(
                        r"$Q = \alpha \, \operatorname{diag}(Q_0)$, where $Q_0$ = "
                        + f"{state_weights.tolist()}"
                    ),
                    mo.md(
                        r"$R = (1 - \alpha) \, \operatorname{diag}(R_0)$, where $R_0$ = "
                        + f"{input_weights.tolist()}"
                    ),
                    alpha,
                ]
            ),
            "Initial state and target position": mo.hstack(
                [initial_state, target_position], justify="center"
            ),
        },
        multiple=True,
    )


@app.cell
def _(
    np,
    initial_state,
    input_weights,
    nstep,
    sim,
    state_weights,
    target_position,
    alpha,
    timestep,
    saturation_checkbox,
    saturation_slider,
):
    x0 = np.array(initial_state.value)
    xt = np.array([target_position.value[0], target_position.value[1], 0, 0, 0, 0])

    q = alpha.value * np.diag(state_weights)
    r = (1 - alpha.value) * np.diag(input_weights)
    dt = timestep.value

    if saturation_checkbox.value:
        u_min = saturation_slider.value[0]
        u_max = saturation_slider.value[1]
    else:
        u_min = -np.inf
        u_max = np.inf

    lqr_controller = sim.LQR(xt, sim.u_eq, q, r, dt)

    ts, xs, us = sim.simulate(
        nstep.value,
        dt,
        x0,
        lqr_controller.input,
        u_min=u_min,
        u_max=u_max,
    )

    return ts, us, xs, x0, xt


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## States and Inputs""")


@app.cell
def plot(sim, ts, us, xs):
    fig2, _ = sim.plot_states_and_inputs(ts, xs, us)
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
