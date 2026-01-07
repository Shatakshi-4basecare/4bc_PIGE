"""Plotting functions for pathway knockout interpretability analysis."""

import os
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ptitprince as pt
import seaborn as sns
from scipy.stats import mannwhitneyu, spearmanr


def _append_to_summary(summary_file_path: Optional[str], content: str) -> None:
    """Append content to summary file.

    Args:
        summary_file_path: Path to summary file.
        content: Content to append.
    """
    if summary_file_path:
        os.makedirs(os.path.dirname(summary_file_path), exist_ok=True)
        with open(summary_file_path, 'a') as f:
            f.write(content + '\n\n')


def _add_pvalue_bracket(ax, x1: float, x2: float, y: float, h: float, text: str) -> None:
    """Draw significance bracket on plot.

    Args:
        ax: Matplotlib axis.
        x1: Start x position.
        x2: End x position.
        y: Y position.
        h: Bracket height.
        text: Text to display.
    """
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c='k')
    ax.text((x1 + x2) / 2, y + h, text, ha='center', va='bottom')


def create_heatmap(
    data_df: pd.DataFrame,
    title: str,
    output_path: str,
    go_map: Optional[Dict],
    col_colors: Optional[pd.DataFrame] = None,
    y_label: str = "Pathways",
    cell_line_name_map: Optional[Dict[str, str]] = None,
    x_axis_label: str = "Cell Lines (Sorted by AAC)",
    actual_aac: Optional[pd.Series] = None,
    predicted_aac: Optional[pd.Series] = None,
    summary_file_path: Optional[str] = None,
) -> None:
    """Generate and save heatmap from knockout score data.

    Args:
        data_df: Data to plot.
        title: Plot title.
        output_path: Output file path.
        go_map: Mapping from pathway IDs to names.
        col_colors: Column colors for heatmap.
        y_label: Y-axis label.
        cell_line_name_map: Mapping from cell line IDs to names.
        x_axis_label: X-axis label.
        actual_aac: Actual AAC values.
        predicted_aac: Predicted AAC values.
        summary_file_path: Path to save summary text.
    """
    if data_df.empty:
        print(f"Data for heatmap '{title}' is empty. Skipping plot.")
        return

    if 'gdsc0_true_test' in title:
        title = title.replace('gdsc0_true_test', 'GDSC2 Leave Pair Out Cell Lines')

    plot_data = data_df.copy()

    if actual_aac is not None and len(actual_aac) > 40:
        common_cell_lines = actual_aac.index.intersection(plot_data.columns)
        if len(common_cell_lines) > 40:
            common_aac = actual_aac[common_cell_lines]
            sorted_cell_lines = common_aac.sort_values().index
            top_20 = sorted_cell_lines[-20:][::-1]
            bottom_20 = sorted_cell_lines[:20][::-1]
            selected_cell_lines = list(top_20) + list(bottom_20)

            plot_data = plot_data[selected_cell_lines]
            if predicted_aac is not None:
                predicted_aac = predicted_aac[selected_cell_lines]
            actual_aac = actual_aac[selected_cell_lines]

    top_10_entities = plot_data.index[:10].tolist()
    if go_map:
        top_10_entities = [go_map.get(entity, entity) for entity in top_10_entities]
    top_10_str = '\n'.join(top_10_entities)
    summary_content = (
        f"--- Heatmap Summary ---\n"
        f"File: {output_path}\n"
        f"Title: {title}\n"
        f"Top 10 Entities:\n{top_10_str}"
    )
    _append_to_summary(summary_file_path, summary_content)

    p = Path(output_path)
    csv_dir = p.parent / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f"{p.stem}.csv"

    csv_data_to_save = plot_data.copy()

    if actual_aac is not None and actual_aac.all() != 0.0:
        csv_data_to_save.loc['actual_aac'] = actual_aac.reindex(csv_data_to_save.columns)
    if predicted_aac is not None:
        csv_data_to_save.loc['predicted_aac'] = predicted_aac.reindex(csv_data_to_save.columns)

    if go_map:
        mapped_index = [go_map.get(idx, idx) if idx not in ['actual_aac', 'predicted_aac'] else idx for idx in csv_data_to_save.index]
        csv_data_to_save.index = mapped_index

    if cell_line_name_map:
        csv_data_to_save.columns = [cell_line_name_map.get(col, col) for col in csv_data_to_save.columns]

    csv_data_to_save = csv_data_to_save.round(6)
    csv_data_to_save.to_csv(csv_path)
    print(f"Heatmap data saved to: {csv_path}")

    if go_map:
        pathway_names = [go_map.get(idx, idx) for idx in data_df.index]
        plot_data.index = pathway_names

    if cell_line_name_map:
        plot_data.columns = [cell_line_name_map.get(col, col) for col in plot_data.columns]
        if col_colors is not None:
            col_colors.index = [cell_line_name_map.get(idx, idx) for idx in col_colors.index]

    if plot_data.size == 0:
        vmin, vmax = None, None
    else:
        v_limit = np.percentile(np.abs(plot_data.values), 95)
        if v_limit < 1e-6:
            vmin, vmax = None, None
        else:
            vmin, vmax = -v_limit, v_limit

    width_multiplier = 0.35
    fig_width = max(10, plot_data.shape[1] * width_multiplier)
    fig_height = max(8, plot_data.shape[0] * 0.3)

    if plot_data.shape[0] * plot_data.shape[1] > 10000:
        dpi = 150
        print(f"Large plot detected ({plot_data.shape[0]}x{plot_data.shape[1]}), using DPI=150")
    elif plot_data.shape[0] * plot_data.shape[1] > 5000:
        dpi = 175
        print(f"Medium-large plot detected ({plot_data.shape[0]}x{plot_data.shape[1]}), using DPI=175")
    else:
        dpi = 200
        print(f"Small plot detected ({plot_data.shape[0]}x{plot_data.shape[1]}), using DPI=200")

    g = sns.clustermap(
        plot_data,
        cmap='coolwarm_r',
        standard_scale=None,
        row_cluster=False,
        col_cluster=False,
        linewidths=.5,
        mask=None,
        fmt='s',
        col_colors=col_colors,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'label': 'Causal Importance Score'},
        figsize=(fig_width, fig_height)
    )

    g.ax_heatmap.set_facecolor('white')

    g.fig.suptitle(title, fontsize=16, y=0.85)
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=8)
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=9)
    g.ax_heatmap.set_xlabel(x_axis_label, fontsize=12)
    g.ax_heatmap.set_ylabel(y_label, fontsize=12)

    g.fig.canvas.draw()
    if hasattr(g, 'ax_cbar') and g.ax_cbar is not None:
        heatmap_pos = g.ax_heatmap.get_position()
        cbar_width = 0.015
        cbar_padding = 0.08
        new_cbar_pos = [
            heatmap_pos.x0 - cbar_width - cbar_padding,
            heatmap_pos.y0,
            cbar_width,
            heatmap_pos.height
        ]
        g.ax_cbar.set_position(new_cbar_pos)

    g.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(g.fig)
    print(f"Heatmap saved to: {output_path}")


def create_scatter_plot(
    data_df: pd.DataFrame,
    drug_name: str,
    target_pathway_name: str,
    output_path: str,
    x_col: str,
    x_label: str,
    summary_file_path: Optional[str] = None
) -> None:
    """Generate scatter plot of pathway importance vs. metric.

    Args:
        data_df: Data to plot.
        drug_name: Drug name.
        target_pathway_name: Target pathway name.
        output_path: Output file path.
        x_col: Column name for x-axis.
        x_label: Label for x-axis.
        summary_file_path: Path to save summary text.
    """
    if target_pathway_name not in data_df.columns or x_col not in data_df.columns:
        print(f"Required columns for scatter plot not found. Skipping plot for '{target_pathway_name}'.")
        return

    plt.figure(figsize=(10, 8))

    ax = sns.scatterplot(
        data=data_df,
        x=x_col,
        y=target_pathway_name
    )

    clean_df = data_df[[x_col, target_pathway_name]].dropna()
    if len(clean_df) > 2:
        corr, p_value = spearmanr(clean_df[x_col], clean_df[target_pathway_name])
        corr_text = f"Spearman = {corr:.3f}\np-value = {p_value:.3g}"

        summary_content = (
            f"--- Scatter Plot Summary ---\n"
            f"File: {output_path}\n"
            f"Drug: {drug_name}\n"
            f"Target: {target_pathway_name}\n"
            f"X-Axis: {x_label}\n"
            f"Spearman: {corr:.3f}\n"
            f"P-value: {p_value:.3g}"
        )
        _append_to_summary(summary_file_path, summary_content)

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, corr_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)

    plt.title(f"Impact of {target_pathway_name} Knockout on {drug_name} Efficacy", fontsize=16)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel("Causal Importance Score", fontsize=12)
    plt.grid(True)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Scatter plot saved to: {output_path}")


def create_box_plot(
    data_df: pd.DataFrame,
    drug_name: str,
    target_pathway_name: str,
    output_path: str,
    aac_threshold: float,
    sensitivity_col: str,
    sensitivity_label: str,
    summary_file_path: Optional[str] = None
) -> None:
    """Generate box plot of pathway importance grouped by sensitivity.

    Args:
        data_df: Data to plot.
        drug_name: Drug name.
        target_pathway_name: Target pathway name.
        output_path: Output file path.
        aac_threshold: AAC threshold for grouping.
        sensitivity_col: Column name for sensitivity metric.
        sensitivity_label: Label for sensitivity metric.
        summary_file_path: Path to save summary text.
    """
    if target_pathway_name not in data_df.columns or sensitivity_col not in data_df.columns:
        print(f"Required columns for box plot not found. Skipping plot for '{target_pathway_name}'.")
        return

    plot_data = data_df.copy()

    plot_data['response_group'] = np.where(plot_data[sensitivity_col] < aac_threshold, 'Resistant', 'Sensitive')
    resistant_label = f'Resistant ({sensitivity_label} < {aac_threshold})'
    sensitive_label = f'Sensitive ({sensitivity_label} >= {aac_threshold})'

    group_counts = plot_data['response_group'].value_counts()
    num_resistant = group_counts.get('Resistant', 0)
    num_sensitive = group_counts.get('Sensitive', 0)
    num_samples = len(plot_data)

    if num_resistant < 10 and num_samples >= 20:
        print(f"Resistant group size for '{target_pathway_name}' is {num_resistant}. Redefining as bottom 10.")
        bottom_10_indices = plot_data.nsmallest(10, sensitivity_col).index
        plot_data.loc[bottom_10_indices, 'response_group'] = 'Resistant'
        plot_data.loc[~plot_data.index.isin(bottom_10_indices), 'response_group'] = 'Sensitive'
        resistant_label = 'Resistant (Bottom 10)'
        sensitive_label = 'Sensitive'

    elif num_sensitive < 10 and num_samples >= 20:
        print(f"Sensitive group size for '{target_pathway_name}' is {num_sensitive}. Redefining as top 10.")
        top_10_indices = plot_data.nlargest(10, sensitivity_col).index
        plot_data.loc[top_10_indices, 'response_group'] = 'Sensitive'
        plot_data.loc[~plot_data.index.isin(top_10_indices), 'response_group'] = 'Resistant'
        sensitive_label = 'Sensitive (Top 10)'
        resistant_label = 'Resistant'

    if plot_data['response_group'].nunique() < 2:
        print(f"Could not form two distinct groups for box plot. Skipping plot for '{target_pathway_name}'.")
        return

    order = [resistant_label, sensitive_label]
    plot_data['response_group'] = plot_data['response_group'].map({'Resistant': resistant_label, 'Sensitive': sensitive_label})

    plt.figure(figsize=(8, 8))

    ax = sns.boxplot(
        data=plot_data,
        x='response_group',
        y=target_pathway_name,
        order=order,
        showfliers=False
    )

    sns.swarmplot(data=plot_data, x='response_group', y=target_pathway_name, color=".25", size=4, order=order, ax=ax)

    resistant_scores = plot_data[plot_data['response_group'] == resistant_label][target_pathway_name].dropna()
    sensitive_scores = plot_data[plot_data['response_group'] == sensitive_label][target_pathway_name].dropna()

    if len(resistant_scores) > 1 and len(sensitive_scores) > 1:
        stat, p_value = mannwhitneyu(resistant_scores, sensitive_scores, alternative='two-sided')

        summary_content = (
            f"--- Box Plot Summary ---\n"
            f"File: {output_path}\n"
            f"Drug: {drug_name}\n"
            f"Target: {target_pathway_name}\n"
            f"Groups: '{resistant_label}' vs. '{sensitive_label}'\n"
            f"Mann-Whitney U p-value: {p_value:.3g}"
        )
        _append_to_summary(summary_file_path, summary_content)

        p_val_text = f'p = {p_value:.2e}'
        data_vals = plot_data[target_pathway_name].dropna()
        if len(data_vals) > 0:
            data_min = float(np.nanmin(data_vals))
            data_max = float(np.nanmax(data_vals))
            data_range = data_max - data_min if data_max > data_min else (abs(data_max) if data_max != 0 else 1.0)
            y = data_max + 0.05 * data_range
            h = 0.02 * data_range
            _add_pvalue_bracket(ax, 0, 1, y, h, p_val_text)
            ax.set_ylim(top=y + h + 0.05 * data_range)

    plt.title(f"Impact of {target_pathway_name} Knockout by {drug_name} Sensitivity", fontsize=16)
    plt.xlabel(f"{sensitivity_label} Group", fontsize=12)
    plt.ylabel("Causal Importance Score", fontsize=12)
    ax.grid(False)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Box plot saved to: {output_path}")


def create_quartile_box_plot(
    data_df: pd.DataFrame,
    drug_name: str,
    target_pathway_name: str,
    output_path: str,
    sensitivity_col: str,
    sensitivity_label: str,
    summary_file_path: Optional[str] = None
) -> None:
    """Generate box plot comparing top and bottom quartiles.

    Args:
        data_df: Data to plot.
        drug_name: Drug name.
        target_pathway_name: Target pathway name.
        output_path: Output file path.
        sensitivity_col: Column name for sensitivity metric.
        sensitivity_label: Label for sensitivity metric.
        summary_file_path: Path to save summary text.
    """
    if target_pathway_name not in data_df.columns or sensitivity_col not in data_df.columns:
        print(f"Required columns for quartile box plot not found. Skipping plot for '{target_pathway_name}'.")
        return

    if len(data_df) < 20:
        print(f"Not enough samples ({len(data_df)}) to generate quartile box plot. Skipping.")
        return

    plot_data = data_df.copy()

    lower_quartile = plot_data[sensitivity_col].quantile(0.25)
    upper_quartile = plot_data[sensitivity_col].quantile(0.75)

    plot_data['response_group'] = np.nan
    plot_data.loc[plot_data[sensitivity_col] <= lower_quartile, 'response_group'] = 'Bottom Quartile (Resistant)'
    plot_data.loc[plot_data[sensitivity_col] >= upper_quartile, 'response_group'] = 'Top Quartile (Sensitive)'

    plot_data.dropna(subset=['response_group'], inplace=True)

    if plot_data['response_group'].nunique() < 2:
        print(f"Could not form two distinct quartile groups for box plot. Skipping plot for '{target_pathway_name}'.")
        return

    plt.figure(figsize=(8, 8))
    order = ['Bottom Quartile (Resistant)', 'Top Quartile (Sensitive)']
    palette = {
        'Bottom Quartile (Resistant)': '#D3D3D3',
        'Top Quartile (Sensitive)': '#006d77'
    }

    ax = sns.boxplot(
        data=plot_data,
        x='response_group',
        y=target_pathway_name,
        order=order,
        showfliers=False,
        palette=palette
    )

    sns.swarmplot(data=plot_data, x='response_group', y=target_pathway_name, color=".25", size=4, order=order, ax=ax)

    bottom_scores = plot_data[plot_data['response_group'] == 'Bottom Quartile (Resistant)'][target_pathway_name].dropna()
    top_scores = plot_data[plot_data['response_group'] == 'Top Quartile (Sensitive)'][target_pathway_name].dropna()

    if len(bottom_scores) > 1 and len(top_scores) > 1:
        stat, p_value = mannwhitneyu(bottom_scores, top_scores, alternative='two-sided')

        summary_content = (
            f"--- Quartile Box Plot Summary ---\n"
            f"File: {output_path}\n"
            f"Drug: {drug_name}\n"
            f"Target: {target_pathway_name}\n"
            f"Groups: Top vs. Bottom Quartile by {sensitivity_label}\n"
            f"Mann-Whitney U p-value: {p_value:.3g}"
        )
        _append_to_summary(summary_file_path, summary_content)

        p_val_text = f'p = {p_value:.2e}'
        data_vals = plot_data[target_pathway_name].dropna()
        if len(data_vals) > 0:
            data_min = float(np.nanmin(data_vals))
            data_max = float(np.nanmax(data_vals))
            data_range = data_max - data_min if data_max > data_min else (abs(data_max) if data_max != 0 else 1.0)
            y = data_max + 0.05 * data_range
            h = 0.02 * data_range
            _add_pvalue_bracket(ax, 0, 1, y, h, p_val_text)
            ax.set_ylim(top=y + h + 0.05 * data_range)

    plt.title(f"Impact of {target_pathway_name} Knockout by {drug_name} Sensitivity (Quartiles)", fontsize=16)
    plt.xlabel(f"{sensitivity_label} Group (Quartiles)", fontsize=12)
    plt.ylabel("Causal Importance Score", fontsize=12)
    ax.grid(False)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Quartile box plot saved to: {output_path}")


def create_top_bottom_responder_box_plot(
    data_df: pd.DataFrame,
    drug_name: str,
    target_entity_name: str,
    output_path: str,
    sensitivity_col: str,
    sensitivity_label: str,
    n: int = 10,
    summary_file_path: Optional[str] = None
) -> None:
    """Generate box plot comparing top N and bottom N responders.

    Args:
        data_df: Data to plot.
        drug_name: Drug name.
        target_entity_name: Target entity name.
        output_path: Output file path.
        sensitivity_col: Column name for sensitivity metric.
        sensitivity_label: Label for sensitivity metric.
        n: Number of top/bottom responders.
        summary_file_path: Path to save summary text.
    """
    if target_entity_name not in data_df.columns or sensitivity_col not in data_df.columns:
        print(f"Required columns for top/bottom responder box plot not found. Skipping plot for '{target_entity_name}'.")
        return

    plot_data = data_df.copy()
    plot_data = plot_data.dropna(subset=[target_entity_name, sensitivity_col])
    plot_data = plot_data.sort_values(by=sensitivity_col)

    if len(plot_data) < 2 * n:
        print(f"Not enough data points ({len(plot_data)}) for top/bottom {n} responder plot. Skipping.")
        return

    bottom_n = plot_data.head(n)
    top_n = plot_data.tail(n)

    bottom_label = f'Bottom {n} Responders (Resistant)'
    top_label = f'Top {n} Responders (Sensitive)'

    plot_df = pd.concat([bottom_n, top_n])
    plot_df['responder_group'] = [bottom_label] * n + [top_label] * n

    plt.figure(figsize=(8, 8))
    order = [bottom_label, top_label]

    ax = sns.boxplot(
        data=plot_df,
        x='responder_group',
        y=target_entity_name,
        order=order,
        showfliers=False
    )

    sns.swarmplot(
        data=plot_df,
        x='responder_group',
        y=target_entity_name,
        color=".25",
        size=4,
        order=order,
        ax=ax
    )

    bottom_scores = bottom_n[target_entity_name].dropna()
    top_scores = top_n[target_entity_name].dropna()

    if len(bottom_scores) > 1 and len(top_scores) > 1:
        stat, p_value = mannwhitneyu(bottom_scores, top_scores, alternative='two-sided')

        summary_content = (
            f"--- Top/Bottom Responder Box Plot Summary ---\n"
            f"File: {output_path}\n"
            f"Drug: {drug_name}\n"
            f"Target: {target_entity_name}\n"
            f"Groups: Top {n} vs. Bottom {n} Responders by {sensitivity_label}\n"
            f"Mann-Whitney U p-value: {p_value:.3g}"
        )
        _append_to_summary(summary_file_path, summary_content)

        p_val_text = f'p = {p_value:.2e}'
        data_vals = plot_df[target_entity_name].dropna()
        if len(data_vals) > 0:
            data_min = float(np.nanmin(data_vals))
            data_max = float(np.nanmax(data_vals))
            data_range = data_max - data_min if data_max > data_min else (abs(data_max) if data_max != 0 else 1.0)
            y = data_max + 0.05 * data_range
            h = 0.02 * data_range
            _add_pvalue_bracket(ax, 0, 1, y, h, p_val_text)
            ax.set_ylim(top=y + h + 0.05 * data_range)

    plt.title(f"'{target_entity_name}' Causal Importance\nin Top vs. Bottom {n} Responders for {drug_name}", fontsize=16)
    plt.xlabel(f"Responder Group based on {sensitivity_label}", fontsize=12)
    plt.ylabel("Causal Importance Score", fontsize=12)
    plt.xticks(rotation=0)
    ax.grid(False)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Top/bottom responder box plot saved to: {output_path}")


def _calculate_importance_p_value(all_scores_df: pd.DataFrame, target_id: str) -> Optional[float]:
    """Calculate p-value of entity importance vs. all others.

    Args:
        all_scores_df: DataFrame with all importance scores.
        target_id: Target entity ID.

    Returns:
        P-value or None if calculation fails.
    """
    target_scores = all_scores_df[all_scores_df['entity_id'] == target_id]['importance']
    other_scores = all_scores_df[all_scores_df['entity_id'] != target_id]['importance']

    if len(target_scores) < 1 or len(other_scores) < 1:
        return None

    _, p_value = mannwhitneyu(target_scores, other_scores, alternative='two-sided')
    return p_value


def create_raincloud_plot(
    drug_name: str,
    target_entity_name: str,
    original_aac_series: pd.Series,
    knockout_aac_series: pd.Series,
    output_path: str,
    summary_file_path: Optional[str] = None,
) -> None:
    """Generate raincloud plot of importance score distribution.

    Args:
        drug_name: Drug name.
        target_entity_name: Target entity name.
        original_aac_series: Original AAC values.
        knockout_aac_series: Knockout AAC values.
        output_path: Output file path.
        summary_file_path: Path to save summary text.
    """
    importance_scores = original_aac_series - knockout_aac_series
    plot_df = pd.DataFrame({'Importance Score': importance_scores})

    reference_line_val = 0.0

    summary_content = (
        f"--- Raincloud Plot Summary ---\n"
        f"File: {output_path}\n"
        f"Drug: {drug_name}\n"
        f"Target: {target_entity_name}\n"
        f"Reference Importance (null median): {reference_line_val:.4f}"
    )
    _append_to_summary(summary_file_path, summary_content)

    fig, ax = plt.subplots(figsize=(10, 6))
    pt.RainCloud(
        y='Importance Score',
        data=plot_df,
        ax=ax,
        orient='h',
        width_viol=.8,
        width_box=.4,
        move=.0,
        palette=['#1f77b4'],
        **{'box_' + k: v for k, v in {'showfliers': False}.items()}
    )

    ax.axvline(reference_line_val, color='red', linestyle='--', linewidth=2, label=f'Null Median ({reference_line_val:.3f})')

    ax.set_title(f'Distribution of Importance Scores for {target_entity_name}\nDrug: {drug_name}', fontsize=16)
    ax.set_xlabel('Causal Importance Score', fontsize=12)
    ax.set_ylabel('')
    ax.tick_params(axis='x', labelsize=10)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.set_yticks([])
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Raincloud plot of importance scores saved to: {output_path}")


def create_drug_gene_heatmap(
    data_df: pd.DataFrame,
    title: str,
    output_path: str,
    csv_output_path: Optional[str] = None,
    y_label: Optional[str] = None,
    summary_file_path: Optional[str] = None
) -> None:
    """Generate heatmap of mean gene importances across drugs.

    Args:
        data_df: Data to plot.
        title: Plot title.
        output_path: Output file path.
        csv_output_path: Path to save CSV data.
        y_label: Y-axis label.
        summary_file_path: Path to save summary text.
    """
    if data_df.empty:
        print(f"Data for gene heatmap '{title}' is empty. Skipping plot.")
        return
    if 'gdsc0_true_test' in title:
        title = title.replace('gdsc0_true_test', 'GDSC2 Leave Pair Out Cell Lines')
    create_drug_pathway_heatmap(data_df, title, output_path, go_map=None, csv_output_path=csv_output_path, y_label=y_label, is_gene_knockout=True, summary_file_path=summary_file_path)


def create_drug_pathway_heatmap(
    data_df: pd.DataFrame,
    title: str,
    output_path: str,
    go_map: Optional[Dict],
    csv_output_path: Optional[str] = None,
    y_label: Optional[str] = None,
    is_gene_knockout: bool = False,
    summary_file_path: Optional[str] = None
) -> None:
    """Generate heatmap of mean pathway importances across drugs.

    Args:
        data_df: Data to plot.
        title: Plot title.
        output_path: Output file path.
        go_map: Mapping from pathway IDs to names.
        csv_output_path: Path to save CSV data.
        y_label: Y-axis label.
        is_gene_knockout: Whether this is gene-level knockout.
        summary_file_path: Path to save summary text.
    """
    if data_df.empty:
        print(f"Data for heatmap '{title}' is empty. Skipping plot.")
        return
    if 'gdsc0_true_test' in title:
        title = title.replace('gdsc0_true_test', 'GDSC2 Leave Pair Out Cell Lines')

    plot_data = data_df.copy()

    plot_data['mean_abs_across_drugs'] = plot_data.abs().mean(axis=1)
    plot_data = plot_data.sort_values(by='mean_abs_across_drugs', ascending=False)
    plot_data = plot_data.drop(columns=['mean_abs_across_drugs'])
    plot_data = plot_data.head(50)

    top_10_entities = plot_data.index[:10].tolist()
    if go_map:
        top_10_entities = [go_map.get(entity, entity) for entity in top_10_entities]
    top_10_str = '\n'.join(top_10_entities)
    summary_content = (
        f"--- Cross-Drug Heatmap Summary ---\n"
        f"File: {output_path}\n"
        f"Title: {title}\n"
        f"Top 10 Entities:\n{top_10_str}"
    )
    _append_to_summary(summary_file_path, summary_content)

    if go_map:
        y_axis_labels = [go_map.get(idx, idx) for idx in plot_data.index]
        plot_data.index = y_axis_labels

    plot_data.index.name = "entity"

    if csv_output_path:
        csv_p = Path(csv_output_path)
        csv_p.parent.mkdir(parents=True, exist_ok=True)
        plot_data.to_csv(csv_p)
        print(f"Heatmap data saved to: {csv_p}")

    if plot_data.size == 0:
        vmin, vmax = None, None
    else:
        v_limit = np.percentile(np.abs(plot_data.values), 95)
        vmin, vmax = (-v_limit, v_limit) if v_limit > 1e-6 else (None, None)

    fig = plt.figure(figsize=(max(10, plot_data.shape[1] * 0.3), max(8, plot_data.shape[0] * 0.3)))

    gs = fig.add_gridspec(1, 2, width_ratios=[50, 1], wspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    cbar_ax = fig.add_subplot(gs[0, 1])

    sns.heatmap(
        plot_data,
        annot=False,
        cmap='coolwarm_r',
        linewidths=.5,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        cbar_ax=cbar_ax,
        cbar_kws={'label': 'Mean Causal Importance Score'}
    )

    fig.suptitle(title, fontsize=16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    ax.set_xlabel("Drugs", fontsize=12)

    if y_label is None:
        default_y_label = "Top 50 Pathways (Sorted by Mean Absolute Importance Across Drugs)"
        if is_gene_knockout:
            default_y_label = "Top 50 Genes (Sorted by Mean Absolute Importance Across Drugs)"
        y_label = default_y_label
    ax.set_ylabel(y_label, fontsize=12)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Summary heatmap saved to: {output_path}")


def generate_final_summary_heatmaps(
    all_drug_summaries: Dict,
    config: Dict,
    go_map: Optional[Dict],
    summary_file_path: Optional[str],
    output_dir: Path
) -> None:
    """Generate final summary heatmaps across all drugs.

    Args:
        all_drug_summaries: Dictionary of all drug summaries.
        config: Configuration dictionary.
        go_map: Mapping from pathway IDs to names.
        summary_file_path: Path to save summary text.
        output_dir: Output directory.
    """
    print("\n--- Generating Final Summary Heatmaps Across All Drugs ---")

    mean_importance_summaries = {k: v['mean_importance'] for k, v in all_drug_summaries.items()}
    mean_abs_importance_summaries = {k: v['mean_abs_importance'] for k, v in all_drug_summaries.items()}
    diff_importance_summaries = {k: v['differential_importance'] for k, v in all_drug_summaries.items()}
    spearman_corr_actual_summaries = {k: v.get('spearman_corr_vs_actual_aac') for k, v in all_drug_summaries.items()}

    csv_summary_dir = output_dir / "csv_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_summary_dir.mkdir(parents=True, exist_ok=True)

    is_gene_knockout = "gene" in config.get("knockout_target", "pathway")
    entity_name_plural = "Genes" if is_gene_knockout else "Pathways"
    entity_name_singular = "Gene" if is_gene_knockout else "Pathway"
    label_map = go_map if not is_gene_knockout else None

    y_label_heatmap = f"Top 50 {entity_name_plural} (Sorted by Mean Absolute Importance Across Drugs)"
    if is_gene_knockout and "gene_specific_knockout_y_label" in config:
        y_label_heatmap = config["gene_specific_knockout_y_label"]

    heatmap_func = create_drug_pathway_heatmap if not is_gene_knockout else create_drug_gene_heatmap
    heatmap_args = {
        "title": f"Mean {entity_name_singular} Importance Across All Drugs and Datasets",
        "output_path": output_dir / "all_drugs_summary_mean_importance_heatmap.png",
        "csv_output_path": csv_summary_dir / "all_drugs_summary_mean_importance.csv",
        "y_label": y_label_heatmap,
        "summary_file_path": summary_file_path
    }
    if not is_gene_knockout:
        heatmap_args["go_map"] = go_map

    mean_importance_df = pd.DataFrame(mean_importance_summaries).fillna(0)
    heatmap_func(data_df=mean_importance_df, **heatmap_args)

    mean_abs_importance_df = pd.DataFrame(mean_abs_importance_summaries).fillna(0)
    heatmap_args["title"] = f"Mean Absolute {entity_name_singular} Importance Across All Drugs and Datasets"
    heatmap_args["output_path"] = output_dir / "all_drugs_summary_mean_abs_importance_heatmap.png"
    heatmap_args["csv_output_path"] = csv_summary_dir / "all_drugs_summary_mean_abs_importance.csv"
    heatmap_func(data_df=mean_abs_importance_df, **heatmap_args)

    diff_importance_df = pd.DataFrame(diff_importance_summaries).fillna(0)
    heatmap_args["title"] = f"Differential {entity_name_singular} Importance Across All Drugs and Datasets"
    heatmap_args["output_path"] = output_dir / "all_drugs_summary_differential_importance_heatmap.png"
    heatmap_args["csv_output_path"] = csv_summary_dir / "all_drugs_summary_differential_importance.csv"
    heatmap_func(data_df=diff_importance_df, **heatmap_args)

    spearman_corr_actual_df = pd.DataFrame(spearman_corr_actual_summaries).fillna(0)
    if not spearman_corr_actual_df.empty:
        heatmap_args["title"] = f"Spearman Correlation vs. Actual AAC for {entity_name_singular} Importance"
        heatmap_args["output_path"] = output_dir / "all_drugs_summary_spearman_corr_actual_heatmap.png"
        heatmap_args["csv_output_path"] = csv_summary_dir / "all_drugs_summary_spearman_corr_actual.csv"
        heatmap_func(data_df=spearman_corr_actual_df, **heatmap_args)
