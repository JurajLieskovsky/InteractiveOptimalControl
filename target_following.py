import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from wigglystuff import ChartPuck, Slider2D

    return Slider2D, ChartPuck, mo, np, plt


@app.cell
def _(mo):
    get_x, set_x = mo.state(0)
    get_y, set_y = mo.state(0)
    return get_x, set_x, get_y, set_y


@app.cell
def _(mo):
    refresh = mo.ui.refresh(default_interval=0.1)
    refresh
    return (refresh,)


@app.cell
def _(refresh, widget, get_x, set_x, get_y, set_y):
    refresh
    set_x(get_x() + 0.01 * (widget.x - get_x()))
    set_y(get_y() + 0.01 * (widget.y - get_y()))
    return


@app.cell
def _(Slider2D, mo):
    widget = mo.ui.anywidget(
        Slider2D(
            width=160,
            height=160,
            x_bounds=(-2.0, 2.0),
            y_bounds=(-1.0, 1.5),
            debounce=True
        )
    )
    return widget


@app.cell
def _(widget):
    widget
    return


@app.cell
def _(plt, widget, get_x, get_y):
    x_ref, y_ref = widget.x, widget.y

    _, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    ax.scatter(x_ref, y_ref)
    ax.scatter(get_x(), get_y())
    ax.quiver(
        get_x(),
        get_y(),
        x_ref - get_x(),
        y_ref - get_y(),
        angles="xy",
        scale_units="xy",
        scale=2,
    )
    ax.grid(True, alpha=0.3)

    ax


if __name__ == "__main__":
    app.run()
