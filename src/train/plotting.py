"""Plotting and visualization functions for training metrics."""

import os
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.optimize as optimize


def _log_func(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Logarithmic function for curve fitting.

    Args:
        x: Epoch numbers (must be > 0).
        a: Log slope parameter.
        b: Offset parameter.

    Returns:
        Array of a * ln(x) + b.
    """
    return a * np.log(x) + b


def _generate_forecast_plot(
    run_identifier: str,
    current_seed: int,
    fold_num_display: int,
    current_epoch_num_display: int,
    fold_plot_dir: str,
    all_drugs_to_train: List[str],
    external_dataset_configs: List[Dict],
    actual_epochs_plotted_xaxis: List[int],
    avg_val_spearmans_plot_data: List[float],
    avg_val_pearsons_plot_data: List[float],
    processed_gdsc_spearman_avg_plots_map: Dict[str, List[float]],
    processed_gdsc_pearson_avg_plots_map: Dict[str, List[float]],
    enable_plotting: bool,
    max_forecast_epoch: int = 50,
    min_points_for_fit: int = 4
) -> None:
    """Generate forecast plot using logarithmic curve fitting.

    Args:
        run_identifier: Unique identifier for the run.
        current_seed: Random seed used.
        fold_num_display: Current fold number.
        current_epoch_num_display: Current epoch number.
        fold_plot_dir: Directory to save plots.
        all_drugs_to_train: List of drug names.
        external_dataset_configs: Configuration for external datasets.
        actual_epochs_plotted_xaxis: Epoch numbers for x-axis.
        avg_val_spearmans_plot_data: Internal validation Spearman scores.
        avg_val_pearsons_plot_data: Internal validation Pearson scores.
        processed_gdsc_spearman_avg_plots_map: GDSC Spearman scores by dataset.
        processed_gdsc_pearson_avg_plots_map: GDSC Pearson scores by dataset.
        enable_plotting: Whether to generate plot.
        max_forecast_epoch: Maximum epoch to forecast to.
        min_points_for_fit: Minimum points needed for curve fitting.
    """
    if not enable_plotting:
        return

    fig, ax1 = plt.subplots(figsize=(20, 10))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Spearman Correlation (Forecast)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim([0, 0.8])
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(max(1, max_forecast_epoch // 20)))
    ax1.grid(axis='y', linestyle=':', alpha=0.6, color='gray')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Pearson Correlation (Forecast)', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_ylim([0, 0.8])
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax2.grid(axis='y', linestyle=':', alpha=0.6, color='gray')

    plot_lines_ax1_fc = []
    plot_labels_ax1_fc = []
    plot_lines_ax2_fc = []
    plot_labels_ax2_fc = []

    forecast_epochs_x_axis = np.arange(1, max_forecast_epoch + 1)
    curve_fit_bounds = ([0, -0.2], [0.2, 0.8])

    valid_x_spearman = []
    valid_y_spearman = []
    for i, s in enumerate(avg_val_spearmans_plot_data):
        if pd.notna(s):
            valid_x_spearman.append(actual_epochs_plotted_xaxis[i])
            valid_y_spearman.append(s)

    if valid_x_spearman:
        line, = ax1.plot(valid_x_spearman, valid_y_spearman, color='deepskyblue', marker='o', linestyle='-', label='Avg Internal Val Spearman')
        plot_lines_ax1_fc.append(line)
        plot_labels_ax1_fc.append('Avg Internal Val Spearman')
        if len(valid_x_spearman) >= min_points_for_fit:
            b_initial = np.clip(np.nanmean(valid_y_spearman) if valid_y_spearman else 0.1, curve_fit_bounds[0][1], curve_fit_bounds[1][1])
            try:
                popt, _ = optimize.curve_fit(_log_func, valid_x_spearman, valid_y_spearman, p0=[0.01, b_initial], bounds=curve_fit_bounds, loss='soft_l1', maxfev=5000)
                forecast_y = _log_func(forecast_epochs_x_axis, *popt)
                line_fc, = ax1.plot(forecast_epochs_x_axis, forecast_y, color='deepskyblue', linestyle=':', alpha=0.7, label='Avg Internal Val Spearman (Forecast)')
                plot_lines_ax1_fc.append(line_fc)
                plot_labels_ax1_fc.append('Avg Internal Val Spearman (Forecast)')
            except:
                pass

    valid_x_pearson = []
    valid_y_pearson = []
    for i, p in enumerate(avg_val_pearsons_plot_data):
        if pd.notna(p):
            valid_x_pearson.append(actual_epochs_plotted_xaxis[i])
            valid_y_pearson.append(p)

    if valid_x_pearson:
        line, = ax2.plot(valid_x_pearson, valid_y_pearson, color='salmon', marker='o', linestyle='--', label='Avg Internal Val Pearson')
        plot_lines_ax2_fc.append(line)
        plot_labels_ax2_fc.append('Avg Internal Val Pearson')
        if len(valid_x_pearson) >= min_points_for_fit:
            b_initial = np.clip(np.nanmean(valid_y_pearson) if valid_y_pearson else 0.1, curve_fit_bounds[0][1], curve_fit_bounds[1][1])
            try:
                popt, _ = optimize.curve_fit(_log_func, valid_x_pearson, valid_y_pearson, p0=[0.01, b_initial], bounds=curve_fit_bounds, loss='soft_l1', maxfev=5000)
                forecast_y = _log_func(forecast_epochs_x_axis, *popt)
                line_fc, = ax2.plot(forecast_epochs_x_axis, forecast_y, color='salmon', linestyle=':', alpha=0.7, label='Avg Internal Val Pearson (Forecast)')
                plot_lines_ax2_fc.append(line_fc)
                plot_labels_ax2_fc.append('Avg Internal Val Pearson (Forecast)')
            except:
                pass

    gdsc_base_colors = ['green', 'purple', 'orange', 'brown']
    gdsc_markers = ['^', 's', 'p', '*']

    if processed_gdsc_spearman_avg_plots_map:
        for idx, gdsc_config_item in enumerate(external_dataset_configs):
            gdsc_dataset_name = gdsc_config_item['name']
            gdsc_spearman_y_data = processed_gdsc_spearman_avg_plots_map.get(gdsc_dataset_name, [])

            valid_x_gdsc_sp = []
            valid_y_gdsc_sp = []
            for i, s in enumerate(gdsc_spearman_y_data):
                if pd.notna(s):
                    valid_x_gdsc_sp.append(actual_epochs_plotted_xaxis[i])
                    valid_y_gdsc_sp.append(s)

            if valid_x_gdsc_sp:
                color = gdsc_base_colors[idx % len(gdsc_base_colors)]
                marker = gdsc_markers[idx % len(gdsc_markers)]
                label_orig = f'Avg {gdsc_dataset_name} Spearman'
                line, = ax1.plot(valid_x_gdsc_sp, valid_y_gdsc_sp, color=color, marker=marker, linestyle='-', label=label_orig)
                plot_lines_ax1_fc.append(line)
                plot_labels_ax1_fc.append(label_orig)

                if len(valid_x_gdsc_sp) >= min_points_for_fit:
                    b_initial = np.clip(np.nanmean(valid_y_gdsc_sp) if valid_y_gdsc_sp else 0.1, curve_fit_bounds[0][1], curve_fit_bounds[1][1])
                    try:
                        popt, _ = optimize.curve_fit(_log_func, valid_x_gdsc_sp, valid_y_gdsc_sp, p0=[0.01, b_initial], bounds=curve_fit_bounds, loss='soft_l1', maxfev=5000)
                        forecast_y = _log_func(forecast_epochs_x_axis, *popt)
                        line_fc, = ax1.plot(forecast_epochs_x_axis, forecast_y, color=color, linestyle=':', alpha=0.7, label=f'{label_orig} (Forecast)')
                        plot_lines_ax1_fc.append(line_fc)
                        plot_labels_ax1_fc.append(f'{label_orig} (Forecast)')
                    except:
                        pass

    if processed_gdsc_pearson_avg_plots_map:
        for idx, gdsc_config_item in enumerate(external_dataset_configs):
            gdsc_dataset_name = gdsc_config_item['name']
            gdsc_pearson_y_data = processed_gdsc_pearson_avg_plots_map.get(gdsc_dataset_name, [])

            valid_x_gdsc_p = []
            valid_y_gdsc_p = []
            for i, s in enumerate(gdsc_pearson_y_data):
                if pd.notna(s):
                    valid_x_gdsc_p.append(actual_epochs_plotted_xaxis[i])
                    valid_y_gdsc_p.append(s)

            if valid_x_gdsc_p:
                color = gdsc_base_colors[idx % len(gdsc_base_colors)]
                marker = gdsc_markers[idx % len(gdsc_markers)]
                label_orig = f'Avg {gdsc_dataset_name} Pearson'
                line, = ax2.plot(valid_x_gdsc_p, valid_y_gdsc_p, color=color, marker=marker, linestyle='--', label=label_orig)
                plot_lines_ax2_fc.append(line)
                plot_labels_ax2_fc.append(label_orig)

                if len(valid_x_gdsc_p) >= min_points_for_fit:
                    b_initial = np.clip(np.nanmean(valid_y_gdsc_p) if valid_y_gdsc_p else 0.1, curve_fit_bounds[0][1], curve_fit_bounds[1][1])
                    try:
                        popt, _ = optimize.curve_fit(_log_func, valid_x_gdsc_p, valid_y_gdsc_p, p0=[0.01, b_initial], bounds=curve_fit_bounds, loss='soft_l1', maxfev=5000)
                        forecast_y = _log_func(forecast_epochs_x_axis, *popt)
                        line_fc, = ax2.plot(forecast_epochs_x_axis, forecast_y, color=color, linestyle=':', alpha=0.7, label=f'{label_orig} (Forecast)')
                        plot_lines_ax2_fc.append(line_fc)
                        plot_labels_ax2_fc.append(f'{label_orig} (Forecast)')
                    except:
                        pass

    if plot_lines_ax1_fc or plot_lines_ax2_fc:
        ax1.legend(plot_lines_ax1_fc + plot_lines_ax2_fc, plot_labels_ax1_fc + plot_labels_ax2_fc, loc='center left', bbox_to_anchor=(1.12, 0.5), fontsize='small')

    fig.tight_layout(rect=[0, 0, 0.88, 1])
    plt.title(f'Run {run_identifier}-S{current_seed}-Fold {fold_num_display}-Epoch {current_epoch_num_display}: Overall Forecast (Log Fit to {max_forecast_epoch} Epochs)', fontsize=10)
    forecast_plot_filename = os.path.join(fold_plot_dir, f"fold_{fold_num_display}_overall_forecast.png")
    plt.savefig(forecast_plot_filename)
    plt.close(fig)


def _generate_loss_plot(
    run_identifier: str,
    current_seed: int,
    fold_num_display: int,
    current_epoch_num_display: int,
    fold_plot_dir: str,
    actual_epochs_plotted_xaxis: List[int],
    train_losses_plot: List[float],
    val_losses_plot: List[float]
) -> None:
    """Generate loss plot showing training and validation losses.

    Args:
        run_identifier: Unique identifier for the run.
        current_seed: Random seed used.
        fold_num_display: Current fold number.
        current_epoch_num_display: Current epoch number.
        fold_plot_dir: Directory to save plots.
        actual_epochs_plotted_xaxis: Epoch numbers for x-axis.
        train_losses_plot: Training losses per epoch.
        val_losses_plot: Validation losses per epoch.
    """
    fig, ax = plt.subplots(figsize=(15, 8))

    valid_train_epochs = []
    valid_train_losses = []
    for i, loss in enumerate(train_losses_plot):
        if pd.notna(loss):
            valid_train_epochs.append(actual_epochs_plotted_xaxis[i])
            valid_train_losses.append(loss)

    valid_val_epochs = []
    valid_val_losses = []
    if val_losses_plot:
        for i, loss in enumerate(val_losses_plot):
            if pd.notna(loss):
                valid_val_epochs.append(actual_epochs_plotted_xaxis[i])
                valid_val_losses.append(loss)

    if valid_train_losses:
        ax.plot(valid_train_epochs, valid_train_losses, color='blue', marker='o', linestyle='-', linewidth=2, markersize=6, label='Training Loss', alpha=0.8)

    if valid_val_losses:
        ax.plot(valid_val_epochs, valid_val_losses, color='red', marker='s', linestyle='--', linewidth=2, markersize=6, label='Validation Loss', alpha=0.8)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'Training & Validation Loss - Run {run_identifier}-S{current_seed}-F{fold_num_display}-E{current_epoch_num_display}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=8))

    if valid_train_losses or valid_val_losses:
        ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

    all_losses = valid_train_losses + valid_val_losses
    if all_losses:
        min_loss = min(all_losses)
        max_loss = max(all_losses)
        loss_range = max_loss - min_loss
        y_margin = max(0.1 * loss_range, 0.01)
        ax.set_ylim(max(0, min_loss - y_margin), max_loss + y_margin)

    if valid_train_losses and valid_val_losses:
        latest_common_epoch = min(max(valid_train_epochs), max(valid_val_epochs))
        latest_train_loss = None
        latest_val_loss = None

        if latest_common_epoch in valid_train_epochs:
            idx = valid_train_epochs.index(latest_common_epoch)
            latest_train_loss = valid_train_losses[idx]

        if latest_common_epoch in valid_val_epochs:
            idx = valid_val_epochs.index(latest_common_epoch)
            latest_val_loss = valid_val_losses[idx]

        if latest_train_loss is not None and latest_val_loss is not None:
            loss_diff = latest_val_loss - latest_train_loss
            overfitting_status = "Overfitting" if loss_diff > 0.1 else "Good Fit" if loss_diff < 0.05 else "Slight Overfitting"

            textstr = f'Latest (Epoch {latest_common_epoch}):\n'
            textstr += f'Train Loss: {latest_train_loss:.4f}\n'
            textstr += f'Val Loss: {latest_val_loss:.4f}\n'
            textstr += f'Difference: {loss_diff:.4f}\n'
            textstr += f'Status: {overfitting_status}'

            props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9, verticalalignment='top', bbox=props)

    plt.tight_layout()
    loss_plot_filename = os.path.join(fold_plot_dir, f"fold_{fold_num_display}_loss_plot.png")
    plt.savefig(loss_plot_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    Loss plot saved: {loss_plot_filename}")


def _generate_and_save_epoch_plots(
    run_identifier: str,
    current_seed: int,
    fold_num_display: int,
    current_epoch_num_display: int,
    run_checkpoint_dir: str,
    all_drugs_to_train: List[str],
    external_dataset_configs: List[Dict],
    epochs_range_to_plot: range,
    train_losses_current_fold: List[float],
    val_losses_current_fold: List[float],
    avg_val_spearmans_current_fold: List[float],
    per_drug_val_spearmans_current_fold: List[Dict],
    gdsc_epoch_spearmans_current_fold: List[Dict],
    avg_val_pearsons_current_fold: List[float],
    per_drug_val_pearsons_current_fold: List[Dict],
    gdsc_epoch_pearsons_current_fold: List[Dict],
    enable_plotting: bool
) -> None:
    """Generate and save comprehensive epoch plots and metrics.

    Args:
        run_identifier: Unique identifier for the run.
        current_seed: Random seed used.
        fold_num_display: Current fold number.
        current_epoch_num_display: Current epoch number.
        run_checkpoint_dir: Directory for checkpoints.
        all_drugs_to_train: List of drug names.
        external_dataset_configs: Configuration for external datasets.
        epochs_range_to_plot: Range of epochs to plot.
        train_losses_current_fold: Training losses for all epochs.
        val_losses_current_fold: Validation losses for all epochs.
        avg_val_spearmans_current_fold: Average validation Spearman scores.
        per_drug_val_spearmans_current_fold: Per-drug validation Spearman scores.
        gdsc_epoch_spearmans_current_fold: GDSC Spearman scores per epoch.
        avg_val_pearsons_current_fold: Average validation Pearson scores.
        per_drug_val_pearsons_current_fold: Per-drug validation Pearson scores.
        gdsc_epoch_pearsons_current_fold: GDSC Pearson scores per epoch.
        enable_plotting: Whether to generate plots.
    """
    if not enable_plotting:
        return

    plots_base_dir = os.path.join(run_checkpoint_dir, "epoch_plots")
    os.makedirs(plots_base_dir, exist_ok=True)
    fold_plot_dir = os.path.join(plots_base_dir, f"fold_{fold_num_display}")
    os.makedirs(fold_plot_dir, exist_ok=True)

    num_epochs_to_plot = current_epoch_num_display
    actual_epochs_plotted_xaxis = list(epochs_range_to_plot)

    train_losses_plot = train_losses_current_fold[:num_epochs_to_plot]
    val_losses_plot = val_losses_current_fold[:num_epochs_to_plot] if val_losses_current_fold else []
    avg_val_spearmans_plot = avg_val_spearmans_current_fold[:num_epochs_to_plot]
    per_drug_val_spearmans_epochs_plot = per_drug_val_spearmans_current_fold[:num_epochs_to_plot]
    gdsc_epoch_metrics_plot = gdsc_epoch_spearmans_current_fold[:num_epochs_to_plot]
    avg_val_pearsons_plot = avg_val_pearsons_current_fold[:num_epochs_to_plot]
    per_drug_val_pearsons_epochs_plot = per_drug_val_pearsons_current_fold[:num_epochs_to_plot]
    gdsc_epoch_pearson_metrics_plot = gdsc_epoch_pearsons_current_fold[:num_epochs_to_plot]

    if not train_losses_plot or all(np.isnan(tl) for tl in train_losses_plot):
        return

    _generate_loss_plot(
        run_identifier=run_identifier,
        current_seed=current_seed,
        fold_num_display=fold_num_display,
        current_epoch_num_display=current_epoch_num_display,
        fold_plot_dir=fold_plot_dir,
        actual_epochs_plotted_xaxis=actual_epochs_plotted_xaxis,
        train_losses_plot=train_losses_plot,
        val_losses_plot=val_losses_plot
    )

    csv_base_dir = os.path.join(fold_plot_dir, "metrics_csv")
    os.makedirs(csv_base_dir, exist_ok=True)

    overall_csv_data = {'Epoch': actual_epochs_plotted_xaxis}
    overall_csv_data['Train Loss'] = train_losses_plot
    overall_csv_data['Val Loss'] = val_losses_plot if val_losses_plot else [np.nan] * len(train_losses_plot)
    overall_csv_data['Avg Internal Val Spearman'] = avg_val_spearmans_plot
    overall_csv_data['Avg Internal Val Pearson'] = avg_val_pearsons_plot

    if gdsc_epoch_metrics_plot:
        for gdsc_config_item in external_dataset_configs:
            gdsc_dataset_name = gdsc_config_item['name']
            avg_gdsc_scores = []
            for epoch_idx in range(num_epochs_to_plot):
                epoch_data = gdsc_epoch_metrics_plot[epoch_idx] if epoch_idx < len(gdsc_epoch_metrics_plot) else {}
                scores = []
                if epoch_data and gdsc_dataset_name in epoch_data:
                    drug_scores_map = epoch_data[gdsc_dataset_name]
                    for drug_name in all_drugs_to_train:
                        score = drug_scores_map.get(drug_name, np.nan)
                        if pd.notna(score):
                            scores.append(score)
                avg_gdsc_scores.append(np.nanmean(scores) if scores else np.nan)
            overall_csv_data[f'Avg {gdsc_dataset_name} Spearman'] = avg_gdsc_scores

    if gdsc_epoch_pearson_metrics_plot:
        for gdsc_config_item in external_dataset_configs:
            gdsc_dataset_name = gdsc_config_item['name']
            avg_gdsc_pearson = []
            for epoch_idx in range(num_epochs_to_plot):
                epoch_data = gdsc_epoch_pearson_metrics_plot[epoch_idx] if epoch_idx < len(gdsc_epoch_pearson_metrics_plot) else {}
                scores = []
                if epoch_data and gdsc_dataset_name in epoch_data:
                    drug_scores_map = epoch_data[gdsc_dataset_name]
                    for drug_name in all_drugs_to_train:
                        score = drug_scores_map.get(drug_name, np.nan)
                        if pd.notna(score):
                            scores.append(score)
                avg_gdsc_pearson.append(np.nanmean(scores) if scores else np.nan)
            overall_csv_data[f'Avg {gdsc_dataset_name} Pearson'] = avg_gdsc_pearson

    df_overall = pd.DataFrame(overall_csv_data)
    overall_csv_filename = os.path.join(csv_base_dir, f"fold_{fold_num_display}_metrics_overall.csv")
    df_overall.to_csv(overall_csv_filename, index=False)

    all_drugs_in_fold = set()
    if per_drug_val_spearmans_epochs_plot:
        for epoch_drug_scores in per_drug_val_spearmans_epochs_plot:
            if epoch_drug_scores:
                all_drugs_in_fold.update(epoch_drug_scores.keys())
    if per_drug_val_pearsons_epochs_plot:
        for epoch_drug_scores in per_drug_val_pearsons_epochs_plot:
            if epoch_drug_scores:
                all_drugs_in_fold.update(epoch_drug_scores.keys())

    for drug_name in sorted(list(all_drugs_in_fold)):
        drug_sheet_data = {'Epoch': actual_epochs_plotted_xaxis}
        drug_spearman_scores = []
        drug_pearson_scores = []
        for epoch_data in per_drug_val_spearmans_epochs_plot:
            drug_spearman_scores.append(epoch_data.get(drug_name, np.nan) if epoch_data else np.nan)
        for epoch_data in per_drug_val_pearsons_epochs_plot:
            drug_pearson_scores.append(epoch_data.get(drug_name, np.nan) if epoch_data else np.nan)
        drug_sheet_data['Internal Val Spearman'] = drug_spearman_scores
        drug_sheet_data['Internal Val Pearson'] = drug_pearson_scores

        if gdsc_epoch_metrics_plot:
            for gdsc_config_item in external_dataset_configs:
                gdsc_dataset_name = gdsc_config_item['name']
                drug_gdsc_scores = []
                for epoch_idx in range(num_epochs_to_plot):
                    epoch_gdsc_data = gdsc_epoch_metrics_plot[epoch_idx] if epoch_idx < len(gdsc_epoch_metrics_plot) else {}
                    score_val = np.nan
                    if epoch_gdsc_data and gdsc_dataset_name in epoch_gdsc_data and drug_name in epoch_gdsc_data[gdsc_dataset_name]:
                        score_val = epoch_gdsc_data[gdsc_dataset_name].get(drug_name, np.nan)
                    drug_gdsc_scores.append(score_val if pd.notna(score_val) else np.nan)
                drug_sheet_data[f'{gdsc_dataset_name} Spearman'] = drug_gdsc_scores

        if gdsc_epoch_pearson_metrics_plot:
            for gdsc_config_item in external_dataset_configs:
                gdsc_dataset_name = gdsc_config_item['name']
                drug_gdsc_pearson = []
                for epoch_idx in range(num_epochs_to_plot):
                    epoch_gdsc_data = gdsc_epoch_pearson_metrics_plot[epoch_idx] if epoch_idx < len(gdsc_epoch_pearson_metrics_plot) else {}
                    score_val = np.nan
                    if epoch_gdsc_data and gdsc_dataset_name in epoch_gdsc_data and drug_name in epoch_gdsc_data[gdsc_dataset_name]:
                        score_val = epoch_gdsc_data[gdsc_dataset_name].get(drug_name, np.nan)
                    drug_gdsc_pearson.append(score_val if pd.notna(score_val) else np.nan)
                drug_sheet_data[f'{gdsc_dataset_name} Pearson'] = drug_gdsc_pearson

        safe_drug_name = "".join(c if c.isalnum() else "_" for c in drug_name)
        drug_csv_filename = os.path.join(csv_base_dir, f"fold_{fold_num_display}_metrics_{safe_drug_name}.csv")
        df_drug = pd.DataFrame(drug_sheet_data)
        df_drug.to_csv(drug_csv_filename, index=False)

    fig, ax1 = plt.subplots(figsize=(20, 10))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Spearman Correlation', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim([0, 0.8])
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax1.grid(axis='y', linestyle=':', alpha=0.6, color='gray')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Pearson Correlation', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_ylim([0, 0.8])
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax2.grid(axis='y', linestyle=':', alpha=0.6, color='gray')

    plot_lines_ax1 = []
    plot_labels_ax1 = []
    plot_lines_ax2 = []
    plot_labels_ax2 = []

    valid_epochs_internal_spearman = []
    valid_avg_internal_spearmans = []
    for i, s in enumerate(avg_val_spearmans_plot):
        if pd.notna(s):
            valid_epochs_internal_spearman.append(epochs_range_to_plot[i])
            valid_avg_internal_spearmans.append(s)
    if valid_avg_internal_spearmans:
        line, = ax1.plot(valid_epochs_internal_spearman, valid_avg_internal_spearmans, color='deepskyblue', marker='o', linestyle='-', label='Avg Internal Val Spearman')
        plot_lines_ax1.append(line)
        plot_labels_ax1.append('Avg Internal Val Spearman')

    valid_epochs_internal_pearson = []
    valid_avg_internal_pearsons = []
    for i, p in enumerate(avg_val_pearsons_plot):
        if pd.notna(p):
            valid_epochs_internal_pearson.append(epochs_range_to_plot[i])
            valid_avg_internal_pearsons.append(p)
    if valid_avg_internal_pearsons:
        line, = ax2.plot(valid_epochs_internal_pearson, valid_avg_internal_pearsons, color='salmon', marker='o', linestyle='--', label='Avg Internal Val Pearson')
        plot_lines_ax2.append(line)
        plot_labels_ax2.append('Avg Internal Val Pearson')

    gdsc_base_colors = ['green', 'purple', 'orange', 'brown']
    gdsc_markers = ['^', 's', 'p', '*']

    if gdsc_epoch_metrics_plot:
        for idx, gdsc_config_item in enumerate(external_dataset_configs):
            gdsc_dataset_name = gdsc_config_item['name']
            avg_gdsc_scores = []
            for epoch_data in gdsc_epoch_metrics_plot:
                scores = []
                if epoch_data and gdsc_dataset_name in epoch_data:
                    drug_scores_map = epoch_data[gdsc_dataset_name]
                    for drug_name in all_drugs_to_train:
                        score = drug_scores_map.get(drug_name, np.nan)
                        if pd.notna(score):
                            scores.append(score)
                avg_gdsc_scores.append(np.nanmean(scores) if scores else np.nan)

            valid_epochs_gdsc = []
            valid_avg_gdsc_scores = []
            for i, s in enumerate(avg_gdsc_scores):
                if pd.notna(s):
                    valid_epochs_gdsc.append(epochs_range_to_plot[i])
                    valid_avg_gdsc_scores.append(s)
            if valid_avg_gdsc_scores:
                line, = ax1.plot(valid_epochs_gdsc, valid_avg_gdsc_scores, color=gdsc_base_colors[idx % len(gdsc_base_colors)], marker=gdsc_markers[idx % len(gdsc_markers)], linestyle='-', label=f'Avg {gdsc_dataset_name} Spearman')
                plot_lines_ax1.append(line)
                plot_labels_ax1.append(f'Avg {gdsc_dataset_name} Spearman')

    if gdsc_epoch_pearson_metrics_plot:
        for idx, gdsc_config_item in enumerate(external_dataset_configs):
            gdsc_dataset_name = gdsc_config_item['name']
            avg_gdsc_pearson = []
            for epoch_data in gdsc_epoch_pearson_metrics_plot:
                scores = []
                if epoch_data and gdsc_dataset_name in epoch_data:
                    drug_scores_map = epoch_data[gdsc_dataset_name]
                    for drug_name in all_drugs_to_train:
                        score = drug_scores_map.get(drug_name, np.nan)
                        if pd.notna(score):
                            scores.append(score)
                avg_gdsc_pearson.append(np.nanmean(scores) if scores else np.nan)

            valid_epochs_gdsc_p = []
            valid_avg_gdsc_scores_p = []
            for i, s in enumerate(avg_gdsc_pearson):
                if pd.notna(s):
                    valid_epochs_gdsc_p.append(epochs_range_to_plot[i])
                    valid_avg_gdsc_scores_p.append(s)
            if valid_avg_gdsc_scores_p:
                line, = ax2.plot(valid_epochs_gdsc_p, valid_avg_gdsc_scores_p, color=gdsc_base_colors[idx % len(gdsc_base_colors)], marker=gdsc_markers[idx % len(gdsc_markers)], linestyle='--', label=f'Avg {gdsc_dataset_name} Pearson')
                plot_lines_ax2.append(line)
                plot_labels_ax2.append(f'Avg {gdsc_dataset_name} Pearson')

    if plot_lines_ax1 or plot_lines_ax2:
        ax1.legend(plot_lines_ax1 + plot_lines_ax2, plot_labels_ax1 + plot_labels_ax2, loc='center left', bbox_to_anchor=(1.12, 0.5), fontsize='small')

    fig.tight_layout(rect=[0, 0, 0.88, 1])
    plt.title(f'Run {run_identifier}-S{current_seed}-Fold {fold_num_display}-Epoch {current_epoch_num_display}: Overall Spearman & Pearson', fontsize=10)
    plot_filename = os.path.join(fold_plot_dir, f"fold_{fold_num_display}_overall_corr_metrics.png")
    plt.savefig(plot_filename)
    plt.close(fig)

    for drug_name in sorted(list(all_drugs_in_fold)):
        drug_spearman_scores = []
        drug_pearson_scores = []
        for epoch_data in per_drug_val_spearmans_epochs_plot:
            drug_spearman_scores.append(epoch_data.get(drug_name, np.nan) if epoch_data else np.nan)
        for epoch_data in per_drug_val_pearsons_epochs_plot:
            drug_pearson_scores.append(epoch_data.get(drug_name, np.nan) if epoch_data else np.nan)
        has_internal_spearman_data = not all(np.isnan(s) for s in drug_spearman_scores)
        has_internal_pearson_data = not all(np.isnan(p) for p in drug_pearson_scores)

        if has_internal_spearman_data or has_internal_pearson_data:
            fig_drug, ax1_drug = plt.subplots(figsize=(20, 10))
            ax1_drug.set_xlabel('Epoch')
            ax1_drug.set_ylabel(f'{drug_name} Spearman Correlation', color='tab:blue')
            ax1_drug.tick_params(axis='y', labelcolor='tab:blue')
            ax1_drug.set_ylim([0, 0.8])

            ax2_drug = ax1_drug.twinx()
            ax2_drug.set_ylabel(f'{drug_name} Pearson Correlation', color='tab:red')
            ax2_drug.tick_params(axis='y', labelcolor='tab:red')
            ax2_drug.set_ylim([0, 0.8])

            plot_lines_ax1_drug = []
            plot_labels_ax1_drug = []
            plot_lines_ax2_drug = []
            plot_labels_ax2_drug = []

            valid_epochs_drug_sp = []
            valid_scores_drug_sp = []
            for i, s in enumerate(drug_spearman_scores):
                if pd.notna(s):
                    valid_epochs_drug_sp.append(actual_epochs_plotted_xaxis[i])
                    valid_scores_drug_sp.append(s)
            if valid_scores_drug_sp:
                line, = ax1_drug.plot(valid_epochs_drug_sp, valid_scores_drug_sp, color='deepskyblue', marker='o', linestyle='-', label='Internal Val Spearman')
                plot_lines_ax1_drug.append(line)
                plot_labels_ax1_drug.append('Internal Val Spearman')

            valid_epochs_drug_p = []
            valid_scores_drug_p = []
            for i, s in enumerate(drug_pearson_scores):
                if pd.notna(s):
                    valid_epochs_drug_p.append(actual_epochs_plotted_xaxis[i])
                    valid_scores_drug_p.append(s)
            if valid_scores_drug_p:
                line, = ax2_drug.plot(valid_epochs_drug_p, valid_scores_drug_p, color='salmon', marker='o', linestyle='--', label='Internal Val Pearson')
                plot_lines_ax2_drug.append(line)
                plot_labels_ax2_drug.append('Internal Val Pearson')

            if gdsc_epoch_metrics_plot:
                for idx, gdsc_config_item in enumerate(external_dataset_configs):
                    gdsc_dataset_name = gdsc_config_item['name']
                    gdsc_scores_drug = []
                    for epoch_data in gdsc_epoch_metrics_plot:
                        gdsc_scores_drug.append(epoch_data.get(gdsc_dataset_name, {}).get(drug_name, np.nan))

                    valid_epochs_gdsc_sp = []
                    valid_scores_gdsc_sp = []
                    for i, s in enumerate(gdsc_scores_drug):
                        if pd.notna(s):
                            valid_epochs_gdsc_sp.append(actual_epochs_plotted_xaxis[i])
                            valid_scores_gdsc_sp.append(s)
                    if valid_scores_gdsc_sp:
                        line, = ax1_drug.plot(valid_epochs_gdsc_sp, valid_scores_gdsc_sp, color=gdsc_base_colors[idx % len(gdsc_base_colors)], marker=gdsc_markers[idx % len(gdsc_markers)], linestyle='-', label=f'{gdsc_dataset_name} Spearman')
                        plot_lines_ax1_drug.append(line)
                        plot_labels_ax1_drug.append(f'{gdsc_dataset_name} Spearman')

            if gdsc_epoch_pearson_metrics_plot:
                for idx, gdsc_config_item in enumerate(external_dataset_configs):
                    gdsc_dataset_name = gdsc_config_item['name']
                    gdsc_scores_drug_p = []
                    for epoch_data in gdsc_epoch_pearson_metrics_plot:
                        gdsc_scores_drug_p.append(epoch_data.get(gdsc_dataset_name, {}).get(drug_name, np.nan))

                    valid_epochs_gdsc_p = []
                    valid_scores_gdsc_p = []
                    for i, s in enumerate(gdsc_scores_drug_p):
                        if pd.notna(s):
                            valid_epochs_gdsc_p.append(actual_epochs_plotted_xaxis[i])
                            valid_scores_gdsc_p.append(s)
                    if valid_scores_gdsc_p:
                        line, = ax2_drug.plot(valid_epochs_gdsc_p, valid_scores_gdsc_p, color=gdsc_base_colors[idx % len(gdsc_base_colors)], marker=gdsc_markers[idx % len(gdsc_markers)], linestyle='--', label=f'{gdsc_dataset_name} Pearson')
                        plot_lines_ax2_drug.append(line)
                        plot_labels_ax2_drug.append(f'{gdsc_dataset_name} Pearson')

            if plot_lines_ax1_drug or plot_lines_ax2_drug:
                ax1_drug.legend(plot_lines_ax1_drug + plot_lines_ax2_drug, plot_labels_ax1_drug + plot_labels_ax2_drug, loc='center left', bbox_to_anchor=(1.12, 0.5), fontsize='small')

            fig_drug.tight_layout(rect=[0, 0, 0.88, 1])
            plt.title(f'Run {run_identifier}-S{current_seed}-F{fold_num_display}-E{current_epoch_num_display}: Metrics for {drug_name}', fontsize=10)
            safe_drug_name = "".join(c if c.isalnum() else "_" for c in drug_name)
            drug_plot_filename = os.path.join(fold_plot_dir, f"fold_{fold_num_display}_metrics_plot_{safe_drug_name}.png")
            plt.savefig(drug_plot_filename)
            plt.close(fig_drug)

    processed_gdsc_spearman_avg_plots_map = {}
    processed_gdsc_pearson_avg_plots_map = {}

    if gdsc_epoch_metrics_plot:
        for gdsc_config_item in external_dataset_configs:
            gdsc_dataset_name = gdsc_config_item['name']
            avg_gdsc_scores = []
            for epoch_data in gdsc_epoch_metrics_plot:
                scores = []
                if epoch_data and gdsc_dataset_name in epoch_data:
                    drug_scores_map = epoch_data[gdsc_dataset_name]
                    for drug_name in all_drugs_to_train:
                        score = drug_scores_map.get(drug_name, np.nan)
                        if pd.notna(score):
                            scores.append(score)
                avg_gdsc_scores.append(np.nanmean(scores) if scores else np.nan)
            processed_gdsc_spearman_avg_plots_map[gdsc_dataset_name] = avg_gdsc_scores

    if gdsc_epoch_pearson_metrics_plot:
        for gdsc_config_item in external_dataset_configs:
            gdsc_dataset_name = gdsc_config_item['name']
            avg_gdsc_pearson = []
            for epoch_data in gdsc_epoch_pearson_metrics_plot:
                scores = []
                if epoch_data and gdsc_dataset_name in epoch_data:
                    drug_scores_map = epoch_data[gdsc_dataset_name]
                    for drug_name in all_drugs_to_train:
                        score = drug_scores_map.get(drug_name, np.nan)
                        if pd.notna(score):
                            scores.append(score)
                avg_gdsc_pearson.append(np.nanmean(scores) if scores else np.nan)
            processed_gdsc_pearson_avg_plots_map[gdsc_dataset_name] = avg_gdsc_pearson

    _generate_forecast_plot(
        run_identifier=run_identifier,
        current_seed=current_seed,
        fold_num_display=fold_num_display,
        current_epoch_num_display=current_epoch_num_display,
        fold_plot_dir=fold_plot_dir,
        all_drugs_to_train=all_drugs_to_train,
        external_dataset_configs=external_dataset_configs,
        actual_epochs_plotted_xaxis=actual_epochs_plotted_xaxis,
        avg_val_spearmans_plot_data=avg_val_spearmans_plot,
        avg_val_pearsons_plot_data=avg_val_pearsons_plot,
        processed_gdsc_spearman_avg_plots_map=processed_gdsc_spearman_avg_plots_map,
        processed_gdsc_pearson_avg_plots_map=processed_gdsc_pearson_avg_plots_map,
        enable_plotting=enable_plotting
    )
