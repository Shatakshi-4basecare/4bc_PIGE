"""Analyzes ranking results from knockout experiments.

Provides multiple importance calculation methods for pathway and gene knockout analysis.
"""

import json
import itertools
from pathlib import Path
from typing import Dict, List, Callable, Optional, Tuple
from collections import defaultdict

import yaml
import numpy as np
import pandas as pd

config_gene_path = "src/model_interpretability/configs_ranking_analyzer/config_Fig_2a_gene.yaml"
config_pathway_path = "src/model_interpretability/configs_ranking_analyzer/config_Fig_2a_pathway.yaml"

def mean_importance(scores_df: pd.DataFrame, aac_scores: Optional[pd.Series] = None) -> Optional[pd.Series]:
    """Calculate importance as mean scores across all cell lines.

    Args:
        scores_df: DataFrame with cell lines as rows and entities as columns.
        aac_scores: Optional AAC scores (not used for this method).

    Returns:
        Series with mean importance scores for each entity.
    """
    return scores_df.mean()


def spearman_importance(scores_df: pd.DataFrame, aac_scores: Optional[pd.Series] = None) -> Optional[pd.Series]:
    """Calculate importance as Spearman correlation with AAC scores.

    Args:
        scores_df: DataFrame with cell lines as rows and entities as columns.
        aac_scores: AAC scores for each cell line.

    Returns:
        Series with Spearman correlation scores for each entity, or None if AAC scores not provided.
    """
    if aac_scores is None:
        return None
    return scores_df.corrwith(aac_scores, method='spearman')


def raw_causal_importance(scores_df: pd.DataFrame, aac_scores: Optional[pd.Series] = None) -> Optional[pd.Series]:
    """Calculate importance as sum of raw scores across all cell lines.

    Args:
        scores_df: DataFrame with cell lines as rows and entities as columns.
        aac_scores: Optional AAC scores (not used for this method).

    Returns:
        Series with summed raw scores for each entity.
    """
    return scores_df.sum()


def raw_abs_causal_importance(scores_df: pd.DataFrame, aac_scores: Optional[pd.Series] = None) -> Optional[pd.Series]:
    """Calculate importance as sum of absolute raw scores across all cell lines.

    Args:
        scores_df: DataFrame with cell lines as rows and entities as columns.
        aac_scores: Optional AAC scores (not used for this method).

    Returns:
        Series with summed absolute scores for each entity.
    """
    return scores_df.abs().sum()


def just_top_20_importance(scores_df: pd.DataFrame, aac_scores: Optional[pd.Series] = None) -> Optional[pd.Series]:
    """Calculate importance by summing absolute scores for top 20 sensitive cell lines.

    Focuses on entities most important in the most sensitive cell lines.

    Args:
        scores_df: DataFrame with cell lines as rows and entities as columns.
        aac_scores: AAC scores to determine top 20 sensitive cell lines.

    Returns:
        Series with importance scores based on top 20 cell lines.
    """
    if aac_scores is None:
        return scores_df.abs().sum()

    top_20_cell_lines = aac_scores.nlargest(20).index
    top_20_scores_df = scores_df.loc[top_20_cell_lines]
    return top_20_scores_df.abs().sum()


def percentage_contribution_importance(scores_df: pd.DataFrame, aac_scores: Optional[pd.Series] = None) -> Optional[pd.Series]:
    """Calculate importance as percentage contribution of each entity to total.

    Args:
        scores_df: DataFrame with cell lines as rows and entities as columns.
        aac_scores: Optional AAC scores (not used for this method).

    Returns:
        Series with percentage contribution scores for each entity.
    """
    entity_sums = scores_df[scores_df.abs() > 5.0603e-03].sum()
    total_sum = entity_sums.sum()
    if total_sum == 0:
        return pd.Series(0.0, index=scores_df.columns)
    return entity_sums / total_sum


def differential_importance(scores_df: pd.DataFrame, aac_scores: pd.Series) -> Optional[pd.Series]:
    """Calculate differential importance between top and bottom responders.

    Splits cell lines into top and bottom halves based on AAC scores and calculates
    the difference in mean importance scores between these groups.

    Args:
        scores_df: DataFrame with cell lines as rows and entities as columns.
        aac_scores: AAC scores for each cell line.

    Returns:
        Series with differential importance scores, or None if insufficient data.
    """
    if aac_scores is None:
        return None

    common_cell_lines = scores_df.index.intersection(aac_scores.index)
    if len(common_cell_lines) < 2:
        return None

    aligned_scores_df = scores_df.loc[common_cell_lines]
    aligned_aac_scores = aac_scores.loc[common_cell_lines]

    sorted_cells = aligned_aac_scores.sort_values(ascending=False).index
    num_cells = len(sorted_cells)
    split_size = num_cells // 2

    if split_size == 0:
        return None

    top_cells = sorted_cells[:split_size]
    bottom_cells = sorted_cells[-split_size:]

    mean_top_scores = aligned_scores_df.loc[top_cells].mean(axis=0)
    mean_bottom_scores = aligned_scores_df.loc[bottom_cells].mean(axis=0)

    return mean_top_scores - mean_bottom_scores


IMPORTANCE_METHODS: Dict[str, Callable[[pd.DataFrame, pd.Series], Optional[pd.Series]]] = {
    'raw_causal': raw_causal_importance,
    'raw_abs_causal': raw_abs_causal_importance,
    'just_top_20': just_top_20_importance,
    'differential': differential_importance,
    'percentage_contribution': percentage_contribution_importance,
    'spearman': spearman_importance,
}


def parse_gmt_for_counts(gmt_file_path: str) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Parse GMT file to get pathway and gene counts.

    Args:
        gmt_file_path: Path to the .gmt file.

    Returns:
        Tuple of (pathway_to_gene_count, gene_to_pathway_count) dictionaries.
    """
    pathway_to_gene_count = {}
    gene_to_pathway_count = defaultdict(int)

    with open(gmt_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            pathway_name = parts[1]
            genes = parts[2:]
            pathway_to_gene_count[pathway_name] = len(genes)
            for gene in genes:
                gene_to_pathway_count[gene] += 1

    print(f"Parsed GMT file: {gmt_file_path}")
    return pathway_to_gene_count, dict(gene_to_pathway_count)


def print_top_k_summary(top_k_df: pd.DataFrame, config: dict, title: str) -> Optional[Dict]:
    """Print per-drug and overall Top-K hit rate summaries.

    Args:
        top_k_df: DataFrame with top-k hit data for each drug.
        config: Configuration dictionary.
        title: Title for the summary section.

    Returns:
        Dictionary with summary statistics, or None if no data.
    """
    if top_k_df.empty:
        print(f"No Top-K results to display for {title}")
        return None

    print(f"\n--- Top-K Hit by Drug for {title} ---")
    print(top_k_df.set_index('drug').to_string())

    num_drugs_processed = len(top_k_df)
    summary_data = {'group': title, 'num_drugs': num_drugs_processed}

    print(f"\n--- Overall Top-K Summary for {title} ---")

    if num_drugs_processed > 0:
        print(f"Total drugs processed: {num_drugs_processed}")

        if config['knockout_target'] == 'gene':
            k_values = [1, 5, 10, 15, 20, 25, 50, 100, 200]
            percentages = [0.05, 0.25, 0.5, 0.75, 1, 1.25, 2.5, 5, 10]
        elif config['knockout_target'] == 'pathway':
            k_values = [1, 5, 10, 15, 20, 25, 50]
            percentages = [0.4, 2, 4, 6, 9, 11, 21]
        else:
            k_values = []
            percentages = []

        for i, k in enumerate(k_values):
            hit_col = f'hit_in_top_{k}'
            if hit_col in top_k_df.columns:
                total_hits = top_k_df[hit_col].sum()
                hit_rate = total_hits / num_drugs_processed

                summary_data[f'top_{k}_hits'] = total_hits
                summary_data[f'top_{k}_rate'] = hit_rate

                print(f"Top {k} Hit Rate ({percentages[i]}%): {hit_rate:.2%} ({total_hits}/{num_drugs_processed})")
    else:
        print("No drugs processed to calculate Top-K summary.")

    return summary_data


def main(config: dict) -> None:
    """Run the ranking analysis pipeline.

    Args:
        config: Configuration dictionary with analysis parameters.
    """
    output_dir = Path(config["output_dir"])
    entity_type = config["knockout_target"].replace('double_', '')
    knockout_type = 'double' if 'double' in config["knockout_target"] else 'single'
    knockout_base_dir = output_dir / f"{knockout_type.capitalize()}{entity_type.capitalize()}KO"

    with open(config["relevant_entities_file"], 'r') as f:
        relevant_entities_by_drug = json.load(f)

    with open("src/model_interpretability/relevant_entities/drug_classes.json", 'r') as f:
        drug_classes_with_spaces = json.load(f)

    drug_classes = {}
    for key, value in drug_classes_with_spaces.items():
        drug_classes[key] = [v.replace(' ', '_') for v in value]

    new_relevant_entities_by_drug = {}
    for drug_name in relevant_entities_by_drug:
        new_relevant_entities_by_drug[drug_name.replace(" ", "_")] = relevant_entities_by_drug[drug_name]

    relevant_entities_by_drug = new_relevant_entities_by_drug

    gmt_file = "data/input_data/all_pathway_genesets.gmt"
    pathway_to_gene_count, gene_to_pathway_count = parse_gmt_for_counts(gmt_file)

    if entity_type == 'pathway':
        normalization_factors = pd.Series(pathway_to_gene_count)
    elif entity_type == 'gene':
        normalization_factors = pd.Series(gene_to_pathway_count)
    else:
        normalization_factors = None

    model_df = pd.read_csv("data/input_data/Model.csv")
    cell_line_name_map = pd.Series(model_df.CellLineName.values, index=model_df.ModelID).to_dict()
    cell_line_disease_map = pd.Series(model_df.OncotreePrimaryDisease.values, index=model_df.ModelID).to_dict()

    drugs_to_process = list(relevant_entities_by_drug.keys())
    if config.get("drugs_to_process"):
        drugs_to_process = [d for d in config["drugs_to_process"] if d in drugs_to_process]

    if 'gsea_analysis_results' in config['output_dir']:
        drugs_to_process = [d.replace('-', '_') for d in drugs_to_process]
        relevant_entities_by_drug = {k.replace('-', '_'): v for k, v in relevant_entities_by_drug.items()}

    run_all = config.get("run_all", False)
    if run_all:
        methods_to_run = list(IMPORTANCE_METHODS.keys())
    else:
        methods_to_run = [config.get("importance_method", "raw_causal")]

    aac_column = config.get("aac_column", "actual_aac")
    print_top_n = config.get("print_top_n_entities")
    drugs_to_print = config.get("drugs_to_print_top_entities")
    print_by_cell_line_type = config.get("print_by_cell_line_type", False)
    top_k_drug_list = config.get("top_k_drug_list")
    save_top_20_to_csv = config.get("save_top_20_to_csv", False)
    save_per_drug_importance = config.get("save_per_drug_importance", False)

    default_per_drug_dir = (
        Path(config["output_dir"]) / f"{knockout_type.capitalize()}{entity_type.capitalize()}KO" /
        config.get("dataset_name", "ctrpv2") / "per_drug_importance"
    )
    per_drug_importance_dir = Path(config.get("per_drug_importance_output_dir", default_per_drug_dir))

    for_saving = []
    for importance_method in methods_to_run:
        print(f"\n\n{'='*20} RUNNING ANALYSIS FOR: {importance_method.upper()} {'='*20}")

        all_drug_contributions = {}
        top_20_entities_by_drug_for_file = {}
        disease_entity_scores = defaultdict(lambda: defaultdict(list))
        disease_cell_counts = defaultdict(set)
        overall_importance_scores = pd.Series(dtype='float64')

        importance_func = IMPORTANCE_METHODS.get(importance_method)
        if not importance_func:
            print(f"Unknown importance_method: {importance_method}. Skipping.")
            continue

        pair_level_data = []
        num_ranked_entities = 0
        ranks_by_drug = {}
        num_relevant_by_drug = {}
        cell_counts_by_drug = {}
        top_k_results = []

        for drug_name in drugs_to_process:
            dataset_name = config.get("dataset_name", "ctrpv2")
            score_file_path = knockout_base_dir / dataset_name / "raw_scores" / f"{drug_name}_{dataset_name}_knockout_raw_scores.csv"

            if not score_file_path.exists():
                score_file_path = knockout_base_dir / dataset_name / "raw_scores" / f"{drug_name.replace('_', ' ')}_{dataset_name}_knockout_raw_scores.csv"
                if not score_file_path.exists():
                    print(f"Score file not found for {drug_name}, skipping")
                    continue
                else:
                    print(f"Found score file with spaces: {score_file_path}")

            raw_data_df = pd.read_csv(score_file_path)
            if cell_line_disease_map:
                raw_data_df['disease_type'] = raw_data_df['cell_id'].map(cell_line_disease_map)

            if aac_column not in raw_data_df.columns or 'cell_id' not in raw_data_df.columns:
                print(f"Score file for {drug_name} does not contain required columns.")
                continue

            aac_scores = raw_data_df.set_index('cell_id')[aac_column]
            cell_counts_by_drug[drug_name] = len(aac_scores.index)

            score_cols = [col for col in raw_data_df.columns if col not in ['cell_id', 'predicted_aac', 'actual_aac', 'disease_type']]
            scores_df = raw_data_df.set_index('cell_id')[score_cols]

            print_random_sample_cell_lines = config.get("print_random_sample_cell_lines", 0)
            if print_random_sample_cell_lines > 0:
                random_seed = 42
                np.random.seed(random_seed)

                total_cell_lines = len(scores_df)
                num_to_sample = max(1, int(total_cell_lines * print_random_sample_cell_lines))
                num_to_sample = min(num_to_sample, total_cell_lines)

                sampled_cell_lines = np.random.choice(scores_df.index, size=num_to_sample, replace=False)

                scores_df_sampled = scores_df.loc[sampled_cell_lines]
                aac_scores_sampled = aac_scores.loc[sampled_cell_lines]

                print(f"\n--- Random Sample Analysis for {drug_name} ---")
                print(f"Total cell lines: {total_cell_lines}")
                print(f"Sampled cell lines: {num_to_sample} ({print_random_sample_cell_lines:.1%})")
                print(f"Random seed used: {random_seed}")

                importance_scores_sampled = importance_func(scores_df_sampled, aac_scores_sampled)
                if importance_scores_sampled is not None:
                    if normalization_factors is not None and not normalization_factors.empty and config.get("normalize_by_size", False):
                        common_entities = importance_scores_sampled.index.intersection(normalization_factors.index)

                        if len(common_entities) > 0:
                            aligned_norm_factors = normalization_factors.loc[common_entities]
                            aligned_norm_factors[aligned_norm_factors == 0] = 1
                            importance_scores_sampled.loc[common_entities] = importance_scores_sampled.loc[common_entities] / aligned_norm_factors

                    ranks_sampled = importance_scores_sampled.rank(ascending=False, method='min')

                    top_k_sampled = min(print_top_n or 20, len(ranks_sampled))
                    top_k_ranks_sampled = ranks_sampled.nsmallest(top_k_sampled)
                    top_k_df_sampled = top_k_ranks_sampled.to_frame(name='Rank')
                    top_k_df_sampled['Importance_Score'] = importance_scores_sampled.loc[top_k_df_sampled.index]

                    relevant_entities = relevant_entities_by_drug[drug_name]
                    styled_index_sampled = [
                        f"{entity}<------" if entity in relevant_entities else entity
                        for entity in top_k_df_sampled.index
                    ]

                    print_df_sampled = top_k_df_sampled.copy()
                    print_df_sampled.index = styled_index_sampled

                    print(f"--- Top {len(top_k_df_sampled)} {entity_type}s for {drug_name} (Random Sample) ---")
                    print(print_df_sampled.to_string())
                    print("-" * 20)

            if print_by_cell_line_type and 'disease_type' in raw_data_df.columns:
                unique_diseases = raw_data_df['disease_type'].dropna().unique()
                for disease in unique_diseases:
                    disease_cell_ids = raw_data_df[raw_data_df['disease_type'] == disease]['cell_id']
                    disease_cell_counts[disease].update(disease_cell_ids)

                    disease_scores_df = scores_df.loc[scores_df.index.intersection(disease_cell_ids)]
                    disease_aac_scores = aac_scores.loc[aac_scores.index.intersection(disease_cell_ids)]

                    if disease_scores_df.empty:
                        continue

                    importance_scores_disease_drug = importance_func(disease_scores_df, disease_aac_scores)

                    if importance_scores_disease_drug is not None:
                        valid_scores = importance_scores_disease_drug.dropna()
                        for entity, score in valid_scores.items():
                            if not np.isinf(score):
                                disease_entity_scores[disease][entity].append(score)

            importance_scores = importance_func(scores_df, aac_scores)

            if importance_method == 'total_percentage_contribution' and importance_scores is not None:
                all_drug_contributions[drug_name] = importance_scores

            if importance_scores is None:
                print(f"Could not calculate importance for {drug_name} using method '{importance_method}', skipping.")
                continue

            if normalization_factors is not None and not normalization_factors.empty and config.get("normalize_by_size", False):
                common_entities = importance_scores.index.intersection(normalization_factors.index)

                if len(common_entities) > 0:
                    aligned_norm_factors = normalization_factors.loc[common_entities]
                    aligned_norm_factors[aligned_norm_factors == 0] = 1
                    importance_scores.loc[common_entities] = importance_scores.loc[common_entities] / aligned_norm_factors

            if save_per_drug_importance:
                per_drug_importance_dir.mkdir(parents=True, exist_ok=True)
                cleaned_scores = importance_scores.replace([np.inf, -np.inf], np.nan).dropna()
                sorted_scores = cleaned_scores.sort_values(ascending=False)
                df_out = sorted_scores.reset_index()
                df_out.columns = ["biological_entity", "causal_importance_score"]
                out_fname = per_drug_importance_dir / f"{drug_name}_{importance_method}_{entity_type}_importance.csv"
                df_out.to_csv(out_fname, index=False)
                print(f"Saved per-drug importance to {out_fname}")

            overall_importance_scores = overall_importance_scores.add(importance_scores, fill_value=0)

            ranks = importance_scores.rank(ascending=False, method='min')

            ranks_by_drug[drug_name] = ranks

            relevant_entities = relevant_entities_by_drug[drug_name]

            if top_k_drug_list is None or drug_name in top_k_drug_list:
                top_1_hit = 1 if len(set(ranks.nsmallest(1).index).intersection(relevant_entities)) > 0 else 0
                top_5_hit = 1 if len(set(ranks.nsmallest(5).index).intersection(relevant_entities)) > 0 else 0
                top_10_hit = 1 if len(set(ranks.nsmallest(10).index).intersection(relevant_entities)) > 0 else 0
                top_15_hit = 1 if len(set(ranks.nsmallest(15).index).intersection(relevant_entities)) > 0 else 0
                top_20_hit = 1 if len(set(ranks.nsmallest(20).index).intersection(relevant_entities)) > 0 else 0
                top_25_hit = 1 if len(set(ranks.nsmallest(25).index).intersection(relevant_entities)) > 0 else 0
                top_50_hit = 1 if len(set(ranks.nsmallest(50).index).intersection(relevant_entities)) > 0 else 0
                top_100_hit = 1 if len(set(ranks.nsmallest(100).index).intersection(relevant_entities)) > 0 else 0
                top_200_hit = 1 if len(set(ranks.nsmallest(200).index).intersection(relevant_entities)) > 0 else 0

                if config.get("knockout_target") == "pathway":
                    top_k_results.append({
                        "drug": drug_name,
                        "hit_in_top_1": top_1_hit,
                        "hit_in_top_5": top_5_hit,
                        "hit_in_top_10": top_10_hit,
                        "hit_in_top_15": top_15_hit,
                        "hit_in_top_20": top_20_hit,
                        "hit_in_top_25": top_25_hit,
                        "hit_in_top_50": top_50_hit,
                    })
                else:
                    top_k_results.append({
                        "drug": drug_name,
                        "hit_in_top_1": top_1_hit,
                        "hit_in_top_5": top_5_hit,
                        "hit_in_top_10": top_10_hit,
                        "hit_in_top_15": top_15_hit,
                        "hit_in_top_20": top_20_hit,
                        "hit_in_top_25": top_25_hit,
                        "hit_in_top_50": top_50_hit,
                        "hit_in_top_100": top_100_hit,
                        "hit_in_top_200": top_200_hit,
                    })

            if print_top_n and print_top_n > 0:
                if drugs_to_print is None or drug_name in drugs_to_print:
                    top_n_ranks = ranks.nsmallest(print_top_n)
                    top_n_df = top_n_ranks.to_frame(name='Rank')
                    top_n_df['Importance_Score'] = importance_scores.loc[top_n_df.index]

                    styled_index = [
                        f"{entity}<------" if entity in relevant_entities else entity
                        for entity in top_n_df.index
                    ]

                    print_df = top_n_df.copy()
                    print_df.index = styled_index

                    print(f"\n--- Top {len(top_n_df)} {entity_type}s for {drug_name} ---")
                    print(print_df.to_string())
                    print("-" * 20)

                    if relevant_entities:
                        if config.get("knockout_target") == "pathway":
                            k_values_for_new_metric = [1, 5, 10, 20, 50]
                        else:
                            k_values_for_new_metric = [1, 5, 10, 20, 50, 100, 200]
                        num_cell_lines = len(scores_df)

                        ranks_per_cell_line = scores_df.rank(axis=1, ascending=False, method='min')

                        per_entity_top_k_hits = []

                        for entity in relevant_entities:
                            if entity in ranks_per_cell_line.columns:
                                entity_ranks = ranks_per_cell_line[entity]

                                hits_info = {'entity': entity}
                                for k in k_values_for_new_metric:
                                    top_k_count = (entity_ranks <= k).sum()
                                    pct = (top_k_count / num_cell_lines) * 100 if num_cell_lines > 0 else 0
                                    hits_info[f'top_{k}_pct'] = pct
                                per_entity_top_k_hits.append(hits_info)

                        if per_entity_top_k_hits:
                            print(f"\n--- Relevant Entity Top-K Hit Rate Across Cell Lines for {drug_name} ---")
                            hits_df = pd.DataFrame(per_entity_top_k_hits).set_index('entity')
                            hits_df = hits_df[[f'top_{k}_pct' for k in k_values_for_new_metric]]
                            hits_df.columns = [f"Top {k} Hit Rate (%)" for k in k_values_for_new_metric]
                            print(hits_df.to_string(float_format="%.2f"))
                            print("-" * 20)

            if num_ranked_entities == 0:
                num_ranked_entities = len(ranks)

            top_20_entities = ranks.nsmallest(20).index.tolist()
            top_20_entities_by_drug_for_file[drug_name] = top_20_entities

            for_saving.append({
                "drug": drug_name,
                "top_25_entities": ranks.nsmallest(25).index.tolist()
            })
            num_relevant_by_drug[drug_name] = len(relevant_entities)

            available_relevant_entities = [entity for entity in relevant_entities if entity in ranks.index]

            if not available_relevant_entities:
                drug_score = len(ranks)
            else:
                drug_score = ranks[available_relevant_entities].min()

            for cell_id in aac_scores.index:
                pair_level_data.append({
                    "drug": drug_name,
                    "cell_id": cell_id,
                    "score": drug_score,
                    "top_20_entities": top_20_entities
                })

        if importance_method == 'total_percentage_contribution' and all_drug_contributions:
            contribution_df = pd.DataFrame(all_drug_contributions)
            csv_filename = output_dir / f"total_percentage_contribution_{entity_type}.csv"
            contribution_df.to_csv(csv_filename)
            print(f"Saved total percentage contributions to {csv_filename}")

        results_df = pd.DataFrame(pair_level_data)

        if cell_line_disease_map:
            results_df['disease_type'] = results_df['cell_id'].map(cell_line_disease_map)

        drug_summary_df = results_df.groupby('drug').agg(
            mean_score=('score', 'mean'),
            num_samples=('cell_id', 'nunique')
        ).sort_values(by='mean_score', ascending=True)
        print("\n--- Summary by Drug (lower score is better) ---")
        print(drug_summary_df.to_string())

        top_20_drugs = drug_summary_df.index.tolist()
        print(f"\n--- Top 20 Drugs ---")
        print(top_20_drugs)

        cell_line_summary = results_df.groupby('cell_id')['score'].sum().sort_values(ascending=True)

        if cell_line_name_map:
            cell_line_summary.index = cell_line_summary.index.map(lambda x: cell_line_name_map.get(x, x))

        print("\n--- Summary by Cell Line ---")
        print(cell_line_summary.head(20).to_string())

        if print_by_cell_line_type and 'disease_type' in results_df.columns and results_df['disease_type'].notna().any():
            disease_summary_df = results_df.groupby('disease_type').agg(
                mean_score=('score', 'mean'),
                num_samples=('cell_id', 'nunique')
            ).sort_values(by='mean_score', ascending=True)
            print("\n--- Summary by Cell Line Type (OncotreePrimaryDisease) ---")
            print(disease_summary_df.to_string())

        if print_by_cell_line_type and disease_entity_scores:
            print(f"\n--- Top 10 {entity_type.capitalize()}s by Cell Line Type (Method: {importance_method.upper()}) ---")

            aggregated_disease_scores = defaultdict(dict)
            for disease, entity_scores_map in disease_entity_scores.items():
                for entity, scores_list in entity_scores_map.items():
                    if scores_list:
                        numeric_scores = [s for s in scores_list if isinstance(s, (int, float)) and not np.isnan(s) and not np.isinf(s)]
                        if numeric_scores:
                            aggregated_disease_scores[disease][entity] = np.mean(numeric_scores)

            disease_sample_counts = {disease: len(cells) for disease, cells in disease_cell_counts.items()}

            sorted_diseases = sorted(aggregated_disease_scores.keys(), key=lambda d: disease_sample_counts.get(d, 0), reverse=True)

            for disease in sorted_diseases:
                if not aggregated_disease_scores[disease]:
                    continue

                num_samples = disease_sample_counts.get(disease, 0)
                print(f"\n--- {disease} (Samples: {num_samples}) ---")

                disease_ranks_s = pd.Series(aggregated_disease_scores[disease])
                top_10 = disease_ranks_s.nlargest(100)

                top_10_df = top_10.to_frame(name='Aggregated_Importance_Score')
                print(top_10_df.to_string())
                print("-" * 20)

        if top_k_results:
            top_k_df = pd.DataFrame(top_k_results)
            per_drug_filename = output_dir / f"top_k_hits_per_drug_{importance_method}_{entity_type}.csv"
            top_k_df.to_csv(per_drug_filename, index=False)
            print(f"Saved per-drug top-k hits to {per_drug_filename}")

            all_summaries = []

            overall_summary = print_top_k_summary(top_k_df, config, "Overall")
            if overall_summary:
                all_summaries.append(overall_summary)

            print("\n" + "="*20 + " TOP-K ANALYSIS BY DRUG CLASS " + "="*20)
            for drug_class, drugs_in_class in drug_classes.items():
                class_top_k_df = top_k_df[top_k_df['drug'].isin(drugs_in_class)]
                class_summary = print_top_k_summary(class_top_k_df, config, drug_class)
                if class_summary:
                    all_summaries.append(class_summary)

            if all_summaries:
                summary_df = pd.DataFrame(all_summaries)
                summary_filename = output_dir / f"top_k_summary_by_group_{importance_method}_{entity_type}.csv"
                summary_df.to_csv(summary_filename, index=False)
                print(f"Saved top-k summary by group to {summary_filename}")

        total_score = results_df['score'].sum()

        best_possible_score = len(results_df)
        worst_possible_score = len(results_df) * num_ranked_entities if num_ranked_entities > 0 else float('inf')

        if worst_possible_score != best_possible_score:
            normalized_score = 1 - (total_score - best_possible_score) / (worst_possible_score - best_possible_score)
        else:
            normalized_score = 0.0

        top_entities_by_drug = {item['drug']: item['top_20_entities'] for item in pair_level_data}

        jaccard_scores = []
        processed_drugs = list(top_entities_by_drug.keys())
        if len(processed_drugs) > 1:
            for drug1, drug2 in itertools.combinations(processed_drugs, 2):
                set1 = set(top_entities_by_drug[drug1][:10])
                set2 = set(top_entities_by_drug[drug2][:10])

                union_size = len(set1.union(set2))
                if union_size == 0:
                    continue

                jaccard_index = len(set1.intersection(set2)) / union_size
                jaccard_scores.append(jaccard_index)

            if jaccard_scores:
                avg_similarity = np.mean(jaccard_scores)
                variability_index = 1 - avg_similarity
            else:
                variability_index = np.nan
        else:
            variability_index = np.nan

        if not overall_importance_scores.empty:
            print(f"\n--- Top 50 {entity_type.capitalize()}s Overall (Sum of Importance Scores Across All Drugs) ---")
            top_50_overall = overall_importance_scores.nlargest(40)
            top_50_df = top_50_overall.to_frame(name='Aggregated_Importance_Score')
            print(top_50_df.to_string())
            print(top_50_df.index.tolist())

        print(f"\n--- Final Score and Variability for {importance_method.upper()} ---")
        print(f"Total summed score: {total_score:.2f}")
        print(f"Best Possible Score: {best_possible_score:.2f}")
        print(f"Worst Possible Score: {worst_possible_score:.2f}")
        print(f"Normalized Score (1 is best, 0 is worst): {normalized_score:.4f}")
        print(f"Variability Index (1 is best, 0 is worst): {variability_index:.4f}")
        print(f"Final Score: {normalized_score * 10 + variability_index}")

    with open(f"ranks_{config['knockout_target']}.json", "w") as f:
        json.dump(for_saving, f, indent=4)

    output_filename = output_dir / f"top_20_entities_{importance_method}_{entity_type}.txt"
    with open(output_filename, "w") as f:
        for drug_name, entities in top_20_entities_by_drug_for_file.items():
            f.write(f"{drug_name}\n")
            for entity in entities:
                f.write(f"{entity}\n")
    print(f"Saved top 20 entities to {output_filename}")

    if save_top_20_to_csv:
        csv_data = []
        for drug_name, entities in top_20_entities_by_drug_for_file.items():
            row = {"drug": drug_name}
            for i, entity in enumerate(entities, 1):
                row[f"top_{i}_entity"] = entity
            csv_data.append(row)

        top_20_csv_df = pd.DataFrame(csv_data)
        csv_filename = output_dir / f"top_20_entities_{importance_method}_{entity_type}.csv"
        top_20_csv_df.to_csv(csv_filename, index=False)
        print(f"Saved top 20 entities to CSV: {csv_filename}")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    config_gene = load_config(config_gene_path)
    config_pathway = load_config(config_pathway_path)

    main(config_pathway)
    main(config_gene)
