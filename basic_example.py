import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from wigglystuff import ChartPuck

    return ChartPuck, mo, np


@app.cell
def _(np):
    np.random.seed(42)
    dynamic_data_x = np.random.randn(50)
    dynamic_data_y = np.random.randn(50)
    return dynamic_data_x, dynamic_data_y


@app.cell
def _(ChartPuck, dynamic_data_x, dynamic_data_y):
    def draw_with_crosshairs(ax, widget):
        x, y = widget.x[0], widget.y[0]
        ax.scatter(dynamic_data_x, dynamic_data_y, alpha=0.6)
        ax.axvline(x, color="red", linestyle="--", alpha=0.7)
        ax.axhline(y, color="red", linestyle="--", alpha=0.7)
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
