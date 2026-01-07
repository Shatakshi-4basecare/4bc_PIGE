"""Core training and validation functions.

This module contains epoch-level training and validation logic for the SCM model.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import numpy as np
import pandas as pd


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    clip_grad_norm: float,
    log_prefix_epoch: str
) -> Tuple[float, bool]:
    """Train model for one epoch.

    Args:
        model: Neural network model.
        dataloader: Training data loader.
        optimizer: Optimizer instance.
        loss_fn: Loss function.
        device: Device for computation.
        clip_grad_norm: Gradient clipping threshold.
        log_prefix_epoch: Prefix for logging messages.

    Returns:
        Tuple of (average_loss, has_issues_flag).
    """
    model.train()
    total_loss = 0.0
    num_valid_batches = 0

    for batch_idx, batch_data in enumerate(dataloader):
        pafe_flat = batch_data['pafe_features_flat']
        edge_idx = batch_data['edge_index_batch']
        drug_fp = batch_data['drug_fingerprints']
        labels = batch_data['labels']

        if labels.ndim == 1:
            labels = labels.unsqueeze(1)
        if labels.shape[1] != 1:
            labels = labels.view(-1, 1)
        if labels.dtype != torch.float32:
            labels = labels.float()

        optimizer.zero_grad()
        predictions = model(pafe_flat, edge_idx, drug_fp)

        if predictions.dtype != torch.float32:
            predictions = predictions.float()

        loss = loss_fn(predictions, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        num_valid_batches += 1

    if num_valid_batches == 0:
        return float('nan'), True

    return (total_loss / num_valid_batches), False


def validate_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    log_prefix_epoch: str
) -> Tuple[float, float, float, Dict[str, float], Dict[str, float], float, float, bool]:
    """Validate model for one epoch.

    Args:
        model: Neural network model.
        dataloader: Validation data loader.
        loss_fn: Loss function.
        log_prefix_epoch: Prefix for logging messages.

    Returns:
        Tuple containing:
            - avg_loss: Average validation loss
            - avg_of_per_drug_spearman: Average of per-drug Spearman correlations
            - avg_of_per_drug_pearson: Average of per-drug Pearson correlations
            - per_drug_spearman: Dict mapping drug names to Spearman scores
            - per_drug_pearson: Dict mapping drug names to Pearson scores
            - combined_spearman_all_samples: Combined Spearman over all samples
            - combined_pearson_all_samples: Combined Pearson over all samples
            - epoch_has_issues: Flag indicating if epoch had issues
    """
    model.eval()
    total_loss = 0.0
    num_valid_batches = 0
    all_preds_device = []
    all_labels_device = []
    all_drug_names = []

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            pafe_flat = batch_data['pafe_features_flat']
            edge_idx = batch_data['edge_index_batch']
            drug_fp = batch_data['drug_fingerprints']
            labels = batch_data['labels']
            drug_names_batch = batch_data['drug_names']

            if labels.ndim == 1:
                labels = labels.unsqueeze(1)
            if labels.shape[1] != 1:
                labels = labels.view(-1, 1)
            if labels.dtype != torch.float32:
                labels = labels.float()

            predictions = model(pafe_flat, edge_idx, drug_fp)

            if predictions.dtype != torch.float32:
                predictions = predictions.float()

            loss = loss_fn(predictions, labels)

            total_loss += loss.item()
            all_preds_device.append(predictions)
            all_labels_device.append(labels)
            all_drug_names.extend(drug_names_batch)
            num_valid_batches += 1

    avg_loss = (total_loss / num_valid_batches) if num_valid_batches > 0 else float('nan')

    per_drug_spearman = {}
    per_drug_pearson = {}
    avg_of_per_drug_spearman = 0.0
    avg_of_per_drug_pearson = 0.0
    combined_spearman_all_samples = 0.0
    combined_pearson_all_samples = 0.0

    if all_preds_device and all_labels_device and all_drug_names:
        preds_cat_np = torch.cat(all_preds_device).squeeze().cpu().numpy()
        labels_cat_np = torch.cat(all_labels_device).squeeze().cpu().numpy()

        if (preds_cat_np.size > 1 and labels_cat_np.size > 1 and
            preds_cat_np.shape == labels_cat_np.shape and
            len(np.unique(preds_cat_np)) > 1 and len(np.unique(labels_cat_np)) > 1):
            df_corr_all_samples = pd.DataFrame({'preds': preds_cat_np, 'labels': labels_cat_np})
            combined_spearman_all_samples = df_corr_all_samples.corr(method='spearman').iloc[0, 1]
            combined_pearson_all_samples = df_corr_all_samples.corr(method='pearson').iloc[0, 1]
            if np.isnan(combined_spearman_all_samples):
                combined_spearman_all_samples = 0.0
            if np.isnan(combined_pearson_all_samples):
                combined_pearson_all_samples = 0.0

        df_results = pd.DataFrame({
            'preds': preds_cat_np,
            'labels': labels_cat_np,
            'drug_name': all_drug_names
        })

        for drug_name in df_results['drug_name'].unique():
            df_drug = df_results[df_results['drug_name'] == drug_name]
            if (len(df_drug) > 1 and len(df_drug['preds'].unique()) > 1 and
                len(df_drug['labels'].unique()) > 1):
                spearman_drug = df_drug[['preds', 'labels']].corr(method='spearman').iloc[0, 1]
                pearson_drug = df_drug[['preds', 'labels']].corr(method='pearson').iloc[0, 1]
                per_drug_spearman[drug_name] = spearman_drug if not np.isnan(spearman_drug) else 0.0
                per_drug_pearson[drug_name] = pearson_drug if not np.isnan(pearson_drug) else 0.0
            else:
                per_drug_spearman[drug_name] = 0.0
                per_drug_pearson[drug_name] = 0.0

        valid_spearman_scores = [s for s in per_drug_spearman.values() if pd.notna(s)]
        valid_pearson_scores = [p for p in per_drug_pearson.values() if pd.notna(p)]

        avg_of_per_drug_spearman = np.mean(valid_spearman_scores) if valid_spearman_scores else 0.0
        avg_of_per_drug_pearson = np.mean(valid_pearson_scores) if valid_pearson_scores else 0.0

        if np.isnan(avg_of_per_drug_spearman):
            avg_of_per_drug_spearman = 0.0
        if np.isnan(avg_of_per_drug_pearson):
            avg_of_per_drug_pearson = 0.0

    return (
        avg_loss,
        avg_of_per_drug_spearman,
        avg_of_per_drug_pearson,
        per_drug_spearman,
        per_drug_pearson,
        combined_spearman_all_samples,
        combined_pearson_all_samples,
        False
    )
