
import matplotlib.pyplot as plt  
import numpy as np
import geopandas as gpd


def plot_regression(insitu_gdf:gpd.GeoDataFrame, 
                    m0: float, 
                    m1: float, 
                    metric: float) -> plt.Figure:
    """Plots the calibration regression on the calibration data, returns a plot object 
       (to be saved or displayed later)

    Args:
        insitu_gdf (gpd.GeoDataFrame): GeoDataFrame containing the calibration data
        m0 (float): Intecept of the regression equation
        m1 (float): Slope of the regression
        metric (float): Final RMSE of the regression

    Returns:
        plt.Figure: The plot as an object (to be saved or displayed later)
    """

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.style"] = "normal"
    
    x = insitu_gdf['Z']
    y = insitu_gdf['logratio']

    # Technically we fit LogRatio as X and Depth as Y
    # But a better visualization is to plot Depth as X and LogRatio as Y
    # So we need to invert the regression equation
    x_reg = np.linspace(x.min(), x.max(), 10)
    y_reg = (x_reg - m0) / m1 # Invert linear equation so Depth is on the X

    fig, axs = plt.subplots()

    # Plot the data and the regression
    axs.scatter(x, y, color='navy', s=0.5,marker='x',label='Calibration Data')
    axs.plot(x_reg, y_reg,color='red',label=f'Regression')
    axs.set_xlabel('Depth (m)')
    axs.set_ylabel('Log Ratio')
    axs.set_title(f'Bathymetry Calibration Report\n( Z = {m1:.2f} * LogRatio + {m0:.2f} )')
    axs.legend()

    # Add statistics to the plot
    axs.text(0.05, 0.95, f'm0: {m0:.3f}', transform=axs.transAxes, verticalalignment='top')
    axs.text(0.05, 0.90, f'm1: {m1:.3f}', transform=axs.transAxes, verticalalignment='top')
    axs.text(0.05, 0.85, f'RMSE: {metric:.3f}', transform=axs.transAxes, verticalalignment='top')
    axs.text(0.05, 0.80, f'Nb: {len(insitu_gdf)}', transform=axs.transAxes, verticalalignment='top')

    return fig