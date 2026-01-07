"""Metrics and evaluation functions for model training."""

from typing import Dict, List

import numpy as np
import pandas as pd


def select_best_epoch_from_metrics(
    fold_epoch_metrics: Dict,
    all_drugs_to_train: List[str],
    primary_metric_name: str = 'GDSC0_true_test',
    smoothing_window_size: int = 5,
    min_points_for_smoothing: int = 3
) -> int:
    """Select best epoch based on smoothed generalization score.

    Args:
        fold_epoch_metrics: Dictionary containing all metrics for a single fold.
        all_drugs_to_train: List of drugs used in training for averaging scores.
        primary_metric_name: Name of the GDSC dataset to use for selection.
        smoothing_window_size: Size of the rolling window for smoothing.
        min_points_for_smoothing: Minimum data points required to apply smoothing.

    Returns:
        Selected best epoch number. Returns -1 if selection fails.
    """
    gdsc_spearmans_by_epoch = fold_epoch_metrics.get("gdsc_epoch_spearmans", [])
    gdsc_pearsons_by_epoch = fold_epoch_metrics.get("gdsc_epoch_pearsons", [])

    if not gdsc_spearmans_by_epoch or not gdsc_pearsons_by_epoch:
        avg_val_scores = fold_epoch_metrics.get("avg_val_spearmans", [])
        if avg_val_scores:
            best_epoch = int(np.argmax(avg_val_scores) + 1)
            print(f"  Epoch selection: Using internal validation. Selected epoch {best_epoch}.")
            return best_epoch
        return -1

    primary_spearman_scores = []
    for epoch_data in gdsc_spearmans_by_epoch:
        if epoch_data and primary_metric_name in epoch_data:
            drug_scores_map = epoch_data[primary_metric_name]
            scores = [drug_scores_map.get(drug, np.nan) for drug in all_drugs_to_train]
            valid_scores = [s for s in scores if pd.notna(s)]
            primary_spearman_scores.append(np.mean(valid_scores) if valid_scores else np.nan)
        else:
            primary_spearman_scores.append(np.nan)

    primary_pearson_scores = []
    for epoch_data in gdsc_pearsons_by_epoch:
        if epoch_data and primary_metric_name in epoch_data:
            drug_scores_map = epoch_data[primary_metric_name]
            scores = [drug_scores_map.get(drug, np.nan) for drug in all_drugs_to_train]
            valid_scores = [s for s in scores if pd.notna(s)]
            primary_pearson_scores.append(np.mean(valid_scores) if valid_scores else np.nan)
        else:
            primary_pearson_scores.append(np.nan)

    df_scores = pd.DataFrame({
        'epoch': range(1, len(primary_spearman_scores) + 1),
        'spearman': primary_spearman_scores,
        'pearson': primary_pearson_scores
    }).dropna()

    if df_scores.empty:
        return -1

    df_scores['gen_score'] = (df_scores['spearman'] + df_scores['pearson']) / 2

    if len(df_scores) >= min_points_for_smoothing:
        df_scores['smooth_score'] = df_scores['gen_score'].rolling(
            window=smoothing_window_size,
            center=True,
            min_periods=1
        ).mean()
    else:
        df_scores['smooth_score'] = df_scores['gen_score']

    if df_scores.empty or 'smooth_score' not in df_scores.columns or df_scores['smooth_score'].isnull().all():
        return -1

    best_epoch_row = df_scores.loc[df_scores['smooth_score'].idxmax()]
    best_epoch = int(best_epoch_row['epoch'])

    print(f"  Best epoch: {best_epoch} (smoothed score: {best_epoch_row['smooth_score']:.4f})")

    return best_epoch
