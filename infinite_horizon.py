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
    mo.md("""
    # Infinite Horizon Regulation
    """)
    return


@app.cell(hide_code=True)
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
        label=r"$\alpha =$ ",
    )

    timestep = mo.ui.number(value=1e-2, debounce=True, label=r"$\Delta t$ [s]")
    nstep = mo.ui.number(start=1, step=1, value=500, debounce=True, label=r"$N$")

    saturation_checkbox = mo.ui.checkbox(
        value=True, label=r"$u_{\min \& \max}$ [N] $=$ "
    )
    saturation_slider = mo.ui.range_slider(
        start=-10, stop=10, step=0.5, value=[0, 8], debounce=True
    )

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
        alpha,
        initial_state,
        input_weights,
        nstep,
        saturation_checkbox,
        saturation_slider,
        state_weights,
        target_position,
        timestep,
    )


@app.cell(hide_code=True)
def _(mo, nstep):
    controller_dropdown = mo.ui.dropdown(options=["MPC", "LQR"], value="MPC")
    mpc_horizon = mo.ui.number(
        start=1,
        stop=nstep.value,
        step=1,
        value=20,
        debounce=True,
        label=r"prediction horizon - $M =$ ",
    )

    return (
        controller_dropdown,
        mpc_horizon,
    )


@app.cell(hide_code=True)
def _(
    alpha,
    initial_state,
    input_weights,
    mo,
    nstep,
    saturation_checkbox,
    saturation_slider,
    state_weights,
    target_position,
    timestep,
    controller_dropdown,
    mpc_horizon,
):
    cond_saturation_slider = (
        saturation_slider
        if saturation_checkbox.value
        else mo.Html(
            '<div style="opacity: 0.3; pointer-events: none; user-select: none;">'
            + saturation_slider._repr_html_()
            + "</div>"
        )
    )

    cond_mpc_horizon = mpc_horizon if controller_dropdown.value == "MPC" else mo.md("")

    mo.accordion(
        {
            "Number of steps and timestep length": mo.vstack([nstep, timestep]),
            "Input saturation": mo.hstack(
                [saturation_checkbox, cond_saturation_slider],
                justify="start",
            ),
            "Controller": mo.hstack(
                [controller_dropdown, cond_mpc_horizon],
                justify="start",
            ),
            "Cost weights": mo.vstack(
                [
                    mo.md(
                        r"$Q = \alpha \, \operatorname{diag}(Q_0)$, where $Q_0 =$ "
                        + f"{state_weights.tolist()}"
                    ),
                    mo.md(
                        r"$R = (1 - \alpha) \, \operatorname{diag}(R_0)$, where $R_0 =$ "
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
    return


@app.cell
def _(
    alpha,
    initial_state,
    input_weights,
    np,
    nstep,
    saturation_checkbox,
    saturation_slider,
    sim,
    state_weights,
    target_position,
    timestep,
    controller_dropdown,
    mpc_horizon,
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

    if controller_dropdown.value == "MPC":
        controller = sim.MPC(mpc_horizon.value, xt, sim.u_eq, q, r, dt, u_min, u_max)
    elif controller_dropdown.value == "LQR":
        controller = sim.LQR(xt, sim.u_eq, q, r, dt)

    ts, xs, us, cs = sim.simulate(
        nstep.value,
        dt,
        x0,
        controller.input,
        controller.running_cost,
        controller.final_cost,
        u_min=u_min,
        u_max=u_max,
    )

    cs *= dt

    return ts, us, xs, x0, xt, cs


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## States and Inputs
    """)
    return


@app.cell(hide_code=True)
def plot(np, sim, ts, us, xs, cs):
    fig2, _ = sim.plot_states_and_inputs(ts, xs, us)
    fig2.suptitle(
        f"Cost = {np.sum(cs):.2f} ({np.sum(cs[:-1]):.2f} + {np.sum(cs[-1]):.2f})"
    )
    fig2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Trajectory
    """)
    return


@app.cell(hide_code=True)
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
