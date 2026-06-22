"""
| **Scenario** | **Patient Biological State** | **AI Logic & Score** | **Output Status** | **Example Text (Pathway or Gene)** |
| --- | --- | --- | --- | --- |
| **1. The Futility Trap** | A required biological driver is actively **Downregulated**. | Fails instantly. Score: **-100** | `NOT RECOMMENDED` | FUTILITY: Required gene 'kras' is downregulated (Pat Z: -2.3 |
| **2. No Match** | No significant overlap between the patient's data and the drug's targets/shields. | Neutral. Score: **0** | `NO STRONG SIGNAL` | No Pathway Match / No Gene Match |
| **3. The Perfect Match** | Patient has **Upregulated** targets, and NO active shields blocking the drug. | Base +10 per target. Score: **> 0** | `TOP CANDIDATE` | TARGET MATCH: 'dna repair' (Pat NES: 1.8 |
| **4A. Successful Rescue** | Tumor has an **Upregulated** shield, *BUT* patient also has the SL partner upregulated. | +10 (Target), +20 (Rescue). Score: **> 0** | `COMBINATION RECOMMENDED` | RESCUE Shield 'egfr': Combine with drug hitting 'pik3ca' (Synergy: -0.002). |
| **4B. Manual Intervention** | Tumor has an **Upregulated** shield, and the AI cannot find an active SL partner. | +10 (Target), -50 (Unrescued). Score: **< 0** | `NOT RECOMMENDED` | MANUAL INTERVENTION REQUIRED TO TARGET SHIELD 'cell adhesion'. |
"""
#Version 6

import argparse
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description="PIGE Precision Oncology Engine - Multi-Column v6")
    parser.add_argument("--gsego", required=True, help="Path to patient's gseGO.BP.tsv file")
    parser.add_argument("--gene-tsv", required=True, help="Path to patient's gene differential expression TSV")
    parser.add_argument("--pige-base-dir", required=True, help="Base path containing the 4 KO subfolders")
    parser.add_argument("--depmap", required=True, help="Path to DepMap Model.csv")
    parser.add_argument("--cancer-type", default="Breast", help="Cancer type for DepMap filtering")
    parser.add_argument("--nes-thresh", type=float, default=1.5, help="NES threshold for pathways")
    parser.add_argument("--z-thresh", type=float, default=2.0, help="Z-score threshold for genes")
    parser.add_argument("--q-thresh", type=float, default=0.05, help="Q-value/FDR significance threshold")
    parser.add_argument("--importance-thresh", type=float, default=0.015, help="Macro/Micro importance threshold")
    parser.add_argument("--synergy-thresh", type=float, default=-0.0005, help="Double KO synthetic lethality threshold")
    parser.add_argument("--output-prefix", default="PIGE_Clinical_Report_v6", help="Prefix for the output CSV report file")
    return parser.parse_args()

def standardize_name(name):
    return str(name).strip().lower()

def load_patient_profile(gsego_path, gene_path, nes_thresh, z_thresh, q_thresh):
    print(f"[*] Loading Patient Profile (q <= {q_thresh})...")
    df_pathways = pd.read_csv(gsego_path, sep='\t')
    q_col_path = 'p.adjust' if 'p.adjust' in df_pathways.columns else 'qvalue'
    sig_paths = df_pathways[df_pathways[q_col_path] <= q_thresh]
    
    # Store tuples: {name: (NES, q_val)}
    up_paths = {standardize_name(row['Description']): (row['NES'], row[q_col_path]) for _, row in sig_paths[sig_paths['NES'] >= nes_thresh].iterrows()}
    down_paths = {standardize_name(row['Description']): (row['NES'], row[q_col_path]) for _, row in sig_paths[sig_paths['NES'] <= -nes_thresh].iterrows()}
    
    df_genes = pd.read_csv(gene_path, sep='\t')
    q_col_gene = 'p_adj' if 'p_adj' in df_genes.columns else ('q_value' if 'q_value' in df_genes.columns else 'qvalue')
    gene_col = 'gene_symbol' if 'gene_symbol' in df_genes.columns else 'gene'
    all_genes_in_data = set(df_genes[gene_col].dropna().apply(standardize_name))
    
    if 'z_score' in df_genes.columns and q_col_gene in df_genes.columns:
        sig_genes = df_genes[df_genes[q_col_gene] <= q_thresh]
        up_genes = {standardize_name(row[gene_col]): (row['z_score'], row[q_col_gene]) for _, row in sig_genes[sig_genes['z_score'] >= z_thresh].iterrows()}
        down_genes = {standardize_name(row[gene_col]): (row['z_score'], row[q_col_gene]) for _, row in sig_genes[sig_genes['z_score'] <= -z_thresh].iterrows()}
    else:
        up_genes, down_genes = {}, {}
    
    print(f"    -> Upregulated: {len(up_paths)} Pathways | {len(up_genes)} Genes")
    print(f"    -> Downregulated: {len(down_paths)} Pathways | {len(down_genes)} Genes")
    return up_paths, down_paths, up_genes, down_genes, all_genes_in_data

def get_tissue_specific_cell_lines(depmap_path, cancer_type):
    df_meta = pd.read_csv(depmap_path)
    return df_meta[df_meta['OncotreePrimaryDisease'].str.contains(cancer_type, na=False, case=False)]['ModelID'].tolist()

def load_pige_data(drug_name, folder_path, cancer_lines):
    folder = Path(folder_path)
    summary_files = list(folder.rglob(f"{drug_name}*summary*.csv"))
    raw_files = list(folder.rglob(f"{drug_name}*raw_scores*.csv"))
    df_macro = pd.read_csv(summary_files[0]) if summary_files else None
    if df_macro is not None:
        col = 'pathway_name' if 'pathway_name' in df_macro.columns else 'entity'
        df_macro['standard_name'] = df_macro[col].apply(standardize_name)
    micro_means = None
    if raw_files:
        df_raw = pd.read_csv(raw_files[0], index_col=0)
        df_sub = df_raw[df_raw.index.isin(cancer_lines)] if cancer_lines else df_raw
        if df_sub.empty: df_sub = df_raw
        micro_means = df_sub[[c for c in df_sub.columns if c not in ['predicted_aac', 'actual_aac']]].mean()
        micro_means.index = micro_means.index.map(standardize_name)
    return df_macro, micro_means

def find_patient_specific_rescue(shield_name, double_micro_means, up_genes, up_paths, synergy_thresh):
    if double_micro_means is None: return None
    best_partner, best_score, best_p_val, best_q_val = None, 0, 0, 0
    for pair, score in double_micro_means.items():
        parts = []
        if ' → ' in str(pair): parts = [p.strip() for p in str(pair).split(' → ')]
        elif ' <-> ' in str(pair): parts = [p.strip() for p in str(pair).split(' <-> ')]
        if len(parts) == 2:
            p_a, p_b = parts
            partner = p_b if shield_name == p_a else (p_a if shield_name == p_b else None)
            if partner:
                std_p = standardize_name(partner)
                if (std_p in up_genes or std_p in up_paths) and score < synergy_thresh:
                    if score < best_score:
                        best_score, best_partner = score, partner
                        best_p_val, best_q_val = up_genes[std_p] if std_p in up_genes else up_paths[std_p]
    return (best_partner, best_score, best_p_val, best_q_val) if best_partner else None

def get_all_lethal_partners(shield_name, double_micro_means, synergy_thresh):
    partners = []
    if double_micro_means is None: return partners
    for pair, score in double_micro_means.items():
        parts = []
        if ' → ' in str(pair): parts = [p.strip() for p in str(pair).split(' → ')]
        elif ' <-> ' in str(pair): parts = [p.strip() for p in str(pair).split(' <-> ')]
        if len(parts) == 2:
            p_a, p_b = parts
            partner = p_b if shield_name == p_a else (p_a if shield_name == p_b else None)
            if partner and score < synergy_thresh:
                partners.append((partner, score))
    return sorted(partners, key=lambda x: x[1])

def evaluate_drugs(pige_base_dir, cancer_lines, up_paths, down_paths, up_genes, down_genes, all_genes, imp_thresh, syn_thresh):
    print("[*] Evaluating Drugs (Detailed Multi-Column Mode)...")
    base_path = Path(pige_base_dir)
    drug_names = sorted(list(set([f.name.split('_gdsc0')[0] for f in (base_path / "SinglePathwayKO").rglob("*summary*.csv") if "all_drugs" not in f.name])))
    
    results = []
    for drug in drug_names:
        p_macro, p_micro = load_pige_data(drug, base_path / "SinglePathwayKO", cancer_lines)
        g_macro, g_micro = load_pige_data(drug, base_path / "SingleGeneKO", cancer_lines)
        _, dp_micro = load_pige_data(drug, base_path / "DoublePathwayKO", cancer_lines)
        _, dg_micro = load_pige_data(drug, base_path / "DoubleGeneKO", cancer_lines)

        if p_macro is None: continue
        status, score = "Pending", 0
        p_rationale, g_rationale, clinical_context, active_shields = [], [], [], []

        # 1. Futility Check (Independent for Genes and Pathways)
        for df, down_dict, label, rat_list in [(p_macro, down_paths, "pathway", p_rationale), (g_macro, down_genes, "gene", g_rationale)]:
            if df is not None:
                m = df[(df['standard_name'].isin(down_dict.keys())) & (df['mean_importance'] > imp_thresh)]
                if not m.empty:
                    status, score = "NOT RECOMMENDED", -100
                    row = m.iloc[0]
                    p_score, p_q = down_dict[row['standard_name']]
                    metric = "NES" if label == "pathway" else "Z-score"
                    rat_list.append(f"FUTILITY: Required {label} '{row['standard_name']}' is downregulated (Pat {metric}: {p_score:.2f}, q:{p_q:.4f} | Imp: {row['mean_importance']:.3f}).")

        # 2. Match Logic (Capture both levels simultaneously)
        for micro, up_dict, label, rat_list in [(p_micro, up_paths, "pathway", p_rationale), (g_micro, up_genes, "gene", g_rationale)]:
            if micro is not None:
                # TRACK STRONGEST SIGNAL (to ensure we report at least one gene)
                strongest_item, strongest_val, strongest_p_score, strongest_q = None, 0, 0, 0
                
                for item in up_dict.keys():
                    if item in micro.index:
                        imp, (p_val, q_val) = micro[item], up_dict[item]
                        if abs(imp) > strongest_val:
                            strongest_val = abs(imp)
                            strongest_item, strongest_p_score, strongest_q = item, p_val, q_val
                            
                        metric = "NES" if label == "pathway" else "Z-score"
                        if imp < -imp_thresh:
                            active_shields.append({'type': label, 'name': item, 'val': imp, 'p_val': p_val, 'q_val': q_val})
                            rat_list.append(f"SHIELD: '{item}' blocks efficacy (Pat {metric}: {p_val:.2f}, q:{q_val:.3e} | Imp: {imp:.3f}).")
                        elif imp > imp_thresh:
                            if status == "Pending": score += 10
                            rat_list.append(f"TARGET MATCH: {item} (Pat {metric}: {p_val:.2f}, q:{q_val:.3e} | PIGE: {imp:.3f})")
                
                # FALLBACK: If no gene met the threshold, report the single strongest one found
                if label == "gene" and not g_rationale and strongest_item:
                    metric = "Z-score"
                    sig_type = "Potential Target" if micro[strongest_item] > 0 else "Weak Shield"
                    g_rationale.append(f"STRONGEST GENE ({sig_type}): '{strongest_item}' (Pat {metric}: {strongest_p_score:.2f}, q:{strongest_q:.3e} | Imp: {micro[strongest_item]:.4f}).")

        # 3. Rescue Logic
        if active_shields:
            if status == "Pending":
                status = "COMBINATION RECOMMENDED"
                score -= 50
            
            unrescued = []
            for s in active_shields:
                res = find_patient_specific_rescue(s['name'], dp_micro if s['type']=='pathway' else dg_micro, up_genes, up_paths, syn_thresh)
                if res:
                    partner, syn_score, p_metric_val, q_val = res
                    metric = "Z-score" if standardize_name(partner) in up_genes else "NES"
                    clinical_context.append(f"RESCUE Shield '{s['name']}': Combine with drug hitting '{partner}' (Pat {metric}: {p_metric_val:.2f}, q:{q_val:.3e} | Synergy: {syn_score:.4f})")
                    if status != "NOT RECOMMENDED": score += 20
                else:
                    all_p = get_all_lethal_partners(s['name'], dp_micro if s['type']=='pathway' else dg_micro, syn_thresh)
                    top_p_str = "None"
                    if all_p:
                        top_p, top_s = all_p[0]
                        status_str = "NOT UPREGULATED" if standardize_name(top_p) in all_genes else "NOT IN DATA"
                        top_p_str = f"{top_p} (Syn: {top_s:.4f}, {status_str})"
                    
                    clinical_context.append(f"MANUAL INTERVENTION REQUIRED TO TARGET SHIELD '{s['name']}': Requires '{top_p_str}' which is missing.")
                    unrescued.append(s['name'])
            
            if unrescued:
                status, score = "NOT RECOMMENDED", score - 50

        if status == "Pending":
            status = "TOP CANDIDATE" if score > 0 else "NO STRONG SIGNAL"

        results.append({
            "Drug": drug, "Status": status, "Score": score,
            "Pathway_Rationale": " | ".join(p_rationale) if p_rationale else "No Pathway Match",
            "Gene_Rationale": " | ".join(g_rationale) if g_rationale else "No Gene Match",
            "Clinical_Context": " | ".join(clinical_context) if clinical_context else "Single Agent Protocol"
        })
    return pd.DataFrame(results)

def main():
    args = parse_args()
    up_p, down_p, up_g, down_g, all_g = load_patient_profile(args.gsego, args.gene_tsv, args.nes_thresh, args.z_thresh, args.q_thresh)
    cancer_lines = get_tissue_specific_cell_lines(args.depmap, args.cancer_type)
    df = evaluate_drugs(args.pige_base_dir, cancer_lines, up_p, down_p, up_g, down_g, all_g, args.importance_thresh, args.synergy_thresh)
    df = df.sort_values(by="Score", ascending=False)
    
    print("\n" + "="*140 + "\n  PIGE PRECISION ONCOLOGY REPORT (v6 - SEPARATED ANALYSIS)\n" + "="*140)
    pd.set_option('display.max_colwidth', 40)
    print(df[['Drug', 'Status', 'Score', 'Pathway_Rationale', 'Gene_Rationale']].to_string(index=False))
    
    df.to_csv(f"{args.output_prefix}.csv", index=False)
    print(f"\n[*] Detailed Report saved to {args.output_prefix}.csv")

if __name__ == "__main__":
    main()