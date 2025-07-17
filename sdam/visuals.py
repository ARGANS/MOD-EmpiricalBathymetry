import matplotlib.pyplot as plt
import numpy as np


class Visualisation:
    def _plot_calibration(self, train_gdf, test_gdf, metrics) -> plt.Figure:
        """Plots regression using training data and optionally test data.

        Args:
            train_gdf (gpd.GeoDataFrame): Training (calibration) data.
            test_gdf (gpd.GeoDataFrame, optional): Testing data. Defaults to None.
            m0 (float): Intercept of the regression equation.
            m1 (float): Slope of the regression equation.
            metric (float): RMSE of the regression.

        Returns:
            plt.Figure: The resulting matplotlib figure.
        """

        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.style"] = "normal"

        m0 = metrics["m0"]
        m1 = metrics["m1"]

        x_train = train_gdf["z"]
        y_train = train_gdf["logratio"]

        # Combine train and test for plotting bounds
        if test_gdf is not None and not test_gdf.empty:
            x_test = test_gdf["z"]
            y_test = test_gdf["logratio"]
            x_min = min(x_train.min(), x_test.min())
            x_max = max(x_train.max(), x_test.max())
        else:
            x_test = y_test = None
            x_min = x_train.min()
            x_max = x_train.max()

        # Generate regression line
        x_reg = np.linspace(x_min, x_max, 100)
        y_reg = (x_reg - m0) / m1

        fig, axs = plt.subplots()

        axs.scatter(
            x_train, y_train, color="gray", s=0.5, marker="x", label="Training Data"
        )
        if x_test is not None:
            axs.scatter(
                x_test,
                y_test,
                color="forestgreen",
                s=1.75,
                marker="x",
                label="Test Data",
            )
        axs.plot(x_reg, y_reg, color="red", label="Regression")

        axs.set_xlabel("Depth (m)")
        axs.set_ylabel("Log Ratio")
        axs.set_title(
            f"Bathymetry Calibration Report\n(Z = {m1:.2f} * LogRatio + {m0:.2f})"
        )
        axs.legend()

        # Annotate stats
        for i, (k, v) in enumerate(metrics.items()):
            axs.text(
                0.05,
                0.95 - (i * 0.05),
                f"{k}: {v:.3f}",
                transform=axs.transAxes,
                verticalalignment="top",
            )

        return fig
