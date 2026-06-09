import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from wigglystuff import ChartPuck

    return ChartPuck, mo, np


@app.cell
def _(ChartPuck):
    def draw_with_crosshairs(ax, widget):
        x, y = widget.x[0], widget.y[0]
        ax.quiver(0, 0, x, y, angles="xy", scale_units="xy", scale=1)
        ax.set_title(f"Position: ({x:.2f}, {y:.2f})")
        ax.grid(True, alpha=0.3)

    puck = ChartPuck.from_callback(
        draw_fn=draw_with_crosshairs,
        x_bounds=(-3, 3),
        y_bounds=(-3, 3),
        figsize=(6, 6),
        x=0,
        y=0,
        puck_color="#4caf50",
        throttle=100,
    )
    return (puck,)


@app.cell
def _(mo, puck):
    widget = mo.ui.anywidget(puck)
    return (widget,)


@app.cell
def _(widget):
    widget
    return


if __name__ == "__main__":
    app.run()
