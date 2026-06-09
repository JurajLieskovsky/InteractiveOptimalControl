import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import wigglystuff as ws
    import matplotlib.pyplot as plt

    return mo, ws, plt

@app.cell
def _(mo):
    get_x, set_x = mo.state(0)
    get_y, set_y = mo.state(0)
    return get_x, set_x, get_y, set_y

@app.cell
def _(ws, mo):
    target_state = mo.ui.anywidget(
        ws.Slider2D(
            width=160,
            height=160,
            x_bounds=(-2.0, 2.0),
            y_bounds=(-1.0, 1.5),
            debounce=True
        )
    )

    button = mo.ui.run_button()

    mo.hstack([target_state, button])

    return button, target_state

@app.cell
def _(mo, target_state, button, set_x, set_y):
    mo.stop(not button.value)
    set_x(target_state.x)
    set_y(target_state.y)


@app.cell
def _(plt, get_x, get_y):

    x_ref = get_x()
    y_ref = get_y()

    fig, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    ax.scatter(x_ref, y_ref)
    ax.quiver(
        0,
        0,
        x_ref,
        y_ref,
        angles="xy",
        scale_units="xy",
        scale=2,
    )
    ax.grid(True, alpha=0.3)

    fig
    return (fig,)


if __name__ == "__main__":
    app.run()
