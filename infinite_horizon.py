import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import birotor

    return birotor, mo, np


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
    nstep = mo.ui.number(start=1, step=1, value=600, debounce=True, label=r"$N$")

    pos_checkbox = mo.ui.checkbox(value=True, label="soft position constraints")
    pos_penalty = mo.ui.number(
        start=0,
        step=100,
        stop=10000,
        value=1000,
        label=r"- violation penalty - $\rho =$ ",
        debounce=True,
    )

    saturation_checkbox = mo.ui.checkbox(
        value=True, label=r"$u_{\min \& \max}$ [N] $=$ "
    )
    saturation_slider = mo.ui.range_slider(
        start=-15, stop=15, step=1, value=[0, 8], debounce=True
    )

    initial_state = mo.ui.matrix(
        np.array([3.5, 4, 0, 0, 0, 0]),
        min_value=[-5, 0, -2*np.pi, -100, -100, -100],
        max_value=[5, 5, 2*np.pi, 100, 100, 100],
        step=0.5,
        debounce=True,
        label=r"$\vec{x}_0$",
    )

    target_position = mo.ui.matrix(
        np.array([0.0, 0.25]),
        min_value=[-5, 0],
        max_value=[5, 5],
        step=0.25,
        debounce=True,
        label=r"$[x_t, y_t]$",
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
        pos_checkbox,
        pos_penalty,
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

    return controller_dropdown, mpc_horizon


@app.cell(hide_code=True)
def _(
    alpha,
    controller_dropdown,
    initial_state,
    input_weights,
    mo,
    mpc_horizon,
    nstep,
    saturation_checkbox,
    saturation_slider,
    state_weights,
    target_position,
    timestep,
    pos_checkbox,
    pos_penalty,
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

    if pos_checkbox.value:
        cond_height = mo.vstack(
            [
                pos_checkbox,
                pos_penalty,
            ]
        )
    else:
        cond_height = pos_checkbox

    cond_mpc = (
        mo.vstack([mpc_horizon, cond_height])
        if controller_dropdown.value == "MPC"
        else mo.md("")
    )

    mo.accordion(
        {
            "Number of steps and timestep length": mo.vstack([nstep, timestep]),
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
            "Input saturation": mo.hstack(
                [saturation_checkbox, cond_saturation_slider],
                justify="start",
            ),
            "Initial state and target position": mo.hstack(
                [initial_state, target_position], justify="center"
            ),
            "Controller": mo.hstack(
                [controller_dropdown, cond_mpc],
                justify="start",
            ),
        },
        multiple=True,
    )
    return


@app.cell
def _(
    alpha,
    birotor,
    controller_dropdown,
    initial_state,
    input_weights,
    mpc_horizon,
    np,
    nstep,
    saturation_checkbox,
    saturation_slider,
    state_weights,
    target_position,
    timestep,
    pos_checkbox,
    pos_penalty,
):
    x0 = np.array(initial_state.value)
    xt = np.array([target_position.value[0], target_position.value[1], 0, 0, 0, 0.0])

    q = alpha.value * np.diag(state_weights)
    r = (1 - alpha.value) * np.diag(input_weights)
    dt = timestep.value

    if saturation_checkbox.value:
        u_min = saturation_slider.value[0]
        u_max = saturation_slider.value[1]
    else:
        u_min = -np.inf
        u_max = np.inf

    if pos_checkbox.value:
        pos_min = np.array([-5.0, 0])
        pos_max = np.array([5.0, 5])
    else:
        pos_min = -np.inf * np.ones(2)
        pos_max = np.inf * np.ones(2)

    if controller_dropdown.value == "MPC":
        controller = birotor.infinite_horizon_regulators.MPC(
            mpc_horizon.value,
            xt,
            birotor.dynamics.u_eq,
            q,
            r,
            dt,
            u_min,
            u_max,
            pos_min,
            pos_max,
            pos_penalty.value,
        )
    elif controller_dropdown.value == "LQR":
        controller = birotor.infinite_horizon_regulators.LQR(
            xt, birotor.dynamics.u_eq, q, r, dt
        )

    ts, xs, us, cs = birotor.simulation.simulate(
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
    return cs, ts, us, x0, xs, xt



@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Trajectory
    """)
    return


@app.cell(hide_code=True)
def _(birotor, x0, xs, xt):

    fig1, ax1 = birotor.simulation.plot_trajectory(xs)

    ax1.scatter(x0[0], x0[1], label="initial")
    ax1.scatter(xt[0], xt[1], label="target")

    ax1.legend()

    ax1.set_xlim(-5.5, 5.5)
    ax1.set_ylim(-0.5, 5.5)

    fig1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## States and Inputs
    """)
    return


@app.cell(hide_code=True)
def plot(birotor, ts, us, xs):
    fig2, _ = birotor.simulation.plot_states_and_inputs(ts, xs, us)
    fig2
    return

if __name__ == "__main__":
    app.run()
