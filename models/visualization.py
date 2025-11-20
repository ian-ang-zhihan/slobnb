import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_feature_importance(importance_df, model_name="Model", top_n=None, 
                           save_path=None, figsize=(10, 6)):
    """
    Create a horizontal bar chart showing feature importances.
    
    Parameters:
    -----------
    importance_df : pandas.DataFrame
        DataFrame with 'feature' and 'importance' columns
        (from analyze_feature_importance function)
    model_name : str, default="Model"
        Name of the model (for plot title)
    top_n : int or None
        Show only top N most important features (None = show all)
    save_path : str or None
        Path to save the figure (None = don't save, just display)
    figsize : tuple, default=(10, 6)
        Figure size in inches (width, height)
        
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    # Prepare data
    plot_df = importance_df.copy()
    
    # Limit to top N if specified
    if top_n:
        plot_df = plot_df.head(top_n)
    
    # Sort by importance (ascending for horizontal bar chart)
    # This puts the most important feature at the TOP
    plot_df = plot_df.sort_values('importance', ascending=True)
    
    # Convert to percentage for easier interpretation
    plot_df['importance_pct'] = plot_df['importance'] * 100
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create horizontal bar chart
    bars = ax.barh(plot_df['feature'], plot_df['importance_pct'], 
                   color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add value labels on the bars
    # This shows the exact percentage on each bar
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        ax.text(row['importance_pct'] + 0.5, i, f"{row['importance_pct']:.1f}%", 
                va='center', fontsize=9)
    
    # Labels and title
    ax.set_xlabel('Importance (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
    
    title = f'Feature Importance - {model_name}'
    if top_n:
        title += f' (Top {top_n})'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Grid for easier reading (only on x-axis)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)  # Put grid behind bars
    
    # Tight layout to prevent label cutoff
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFeature importance plot saved to: {save_path}")
    
    return fig, ax


def plot_actual_vs_predicted(y_true, y_pred, model_name="Model", 
                             save_path=None, figsize=(10, 8)):
    """
    Create a scatter plot of actual vs predicted values.
    
    Parameters:
    -----------
    y_true : array-like
        Actual prices (ground truth)
    y_pred : array-like
        Predicted prices (model output)
    model_name : str, default="Model"
        Name of the model (for plot title)
    save_path : str or None
        Path to save the figure (None = don't save, just display)
    figsize : tuple, default=(10, 8)
        Figure size in inches (width, height)
        
    Returns:
    --------
    fig, ax : matplotlib figure and axes
        The created plot objects
    """
    # Convert to numpy arrays for easier handling
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate R² for display on plot
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot of actual vs predicted
    ax.scatter(y_true, y_pred, alpha=0.5, s=30, color='steelblue', 
               edgecolors='black', linewidth=0.3, label='Predictions')
    
    # Add perfect prediction line (y = x)
    # This is the "ideal" line where predicted = actual
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
            'r--', linewidth=2, label='Perfect Prediction (y=x)', alpha=0.8)
    
    # Add R² text box
    # This shows how well the model explains variance
    textstr = f'R² = {r2:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    # Labels and title
    ax.set_xlabel('Actual Price ($)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Price ($)', fontsize=12, fontweight='bold')
    ax.set_title(f'Actual vs Predicted Prices - {model_name}', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add legend
    ax.legend(loc='lower right', fontsize=10)
    
    # Add grid for easier reading
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Equal aspect ratio makes the diagonal line look like 45 degrees
    # (Only if the ranges are similar)
    # ax.set_aspect('equal', adjustable='box')
    
    # Tight layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Actual vs predicted plot saved to: {save_path}")
    
    return fig, ax


def plot_residuals(y_true, y_pred, model_name="Model", 
                  save_path=None, figsize=(10, 8)):
    """
    Create a residual plot to check for systematic errors.
    
    Parameters:
    -----------
    y_true : array-like
        Actual prices (ground truth)
    y_pred : array-like
        Predicted prices (model output)
    model_name : str, default="Model"
        Name of the model (for plot title)
    save_path : str or None
        Path to save the figure (None = don't save, just display)
    figsize : tuple, default=(10, 8)
        Figure size in inches (width, height)
        
    Returns:
    --------
    fig, ax : matplotlib figure and axes
        The created plot objects
    """
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate residuals
    residuals = y_true - y_pred
    
    # Calculate some statistics for display
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot of residuals vs predicted values
    ax.scatter(y_pred, residuals, alpha=0.5, s=30, color='steelblue',
               edgecolors='black', linewidth=0.3)
    
    # Add horizontal line at y=0 (zero error line)
    # This is the "ideal" - all points should be near this line
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, 
               label='Zero Error Line', alpha=0.8)
    
    # Add mean residual line (should be very close to zero)
    if abs(mean_residual) > 1:  # Only show if not essentially zero
        ax.axhline(y=mean_residual, color='orange', linestyle=':', linewidth=2,
                   label=f'Mean Residual = ${mean_residual:.2f}', alpha=0.8)
    
    # Add ±1 standard deviation lines (optional, helps see spread)
    # About 68% of residuals should fall within these lines
    ax.axhline(y=std_residual, color='gray', linestyle=':', linewidth=1.5,
               label=f'±1 Std Dev (${std_residual:.2f})', alpha=0.5)
    ax.axhline(y=-std_residual, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    
    # Labels and title
    ax.set_xlabel('Predicted Price ($)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Residual (Actual - Predicted) ($)', fontsize=12, fontweight='bold')
    ax.set_title(f'Residual Plot - {model_name}', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add statistics text box
    textstr = f'Mean: ${mean_residual:.2f}\nStd Dev: ${std_residual:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=10)
    
    # Add grid
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Residual plot saved to: {save_path}")
    
    return fig, ax


def create_all_visualizations(model, importance_df, y_true, y_pred, 
                              model_name="Model", save_dir=None, show=True):
    """
    Create all three visualizations in one call.
    
    This is a convenience function that creates all visualizations
    and optionally saves them to a directory.
    
    Parameters:
    -----------
    model : trained model object
        The trained model (only used if importance_df is None)
    importance_df : pandas.DataFrame or None
        Feature importance dataframe (for Random Forest)
        If None, feature importance plot will be skipped
    y_true : array-like
        Actual prices
    y_pred : array-like
        Predicted prices
    model_name : str, default="Model"
        Name of the model (for plot titles and filenames)
    save_dir : str or None
        Directory to save plots (None = don't save)
        Will create directory if it doesn't exist
    show : bool, default=True
        Whether to display plots (plt.show())
        Set to False if running in script mode
        
    Returns:
    --------
    figures : dict
        Dictionary with keys 'importance', 'actual_vs_pred', 'residuals'
        Values are (fig, ax) tuples for each plot
    """
    print("\nCreating visualizations...")
    
    figures = {}
    
    # Create save directory if specified
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving plots to: {save_path}")
    
    # 1. Feature Importance Plot (if importance_df provided)
    if importance_df is not None:
        print("\n[1/3] Creating feature importance plot...")
        save_file = Path(save_dir) / f"{model_name.lower().replace(' ', '_')}_feature_importance.png" if save_dir else None
        fig, ax = plot_feature_importance(importance_df, model_name=model_name, 
                                         save_path=save_file)
        figures['importance'] = (fig, ax)
    else:
        print("\n[1/3] Skipping feature importance plot (not applicable for this model)")
    
    # 2. Actual vs Predicted Plot
    print("\n[2/3] Creating actual vs predicted plot...")
    save_file = Path(save_dir) / f"{model_name.lower().replace(' ', '_')}_actual_vs_predicted.png" if save_dir else None
    fig, ax = plot_actual_vs_predicted(y_true, y_pred, model_name=model_name,
                                       save_path=save_file)
    figures['actual_vs_pred'] = (fig, ax)
    
    # 3. Residual Plot
    print("\n[3/3] Creating residual plot...")
    save_file = Path(save_dir) / f"{model_name.lower().replace(' ', '_')}_residuals.png" if save_dir else None
    fig, ax = plot_residuals(y_true, y_pred, model_name=model_name,
                            save_path=save_file)
    figures['residuals'] = (fig, ax)
    
    print("\nAll visualizations created!")
    
    if save_dir:
        print(f"\nPlots saved to: {save_dir}/")
    
    if show:
        print("\nDisplaying plots... (close windows to continue)")
        plt.show()
    
    return figures
