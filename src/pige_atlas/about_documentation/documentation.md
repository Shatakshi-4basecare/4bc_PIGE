# **How to Interpret a PIGE Graph**

## **Introduction**

PIGE graphs visualize **causal mechanisms** of drug response in cancer cells. PIGE models the response-relevant information flow between pathways to predict drug sensitivity or resistance. We then extract the most important pathways and interactions to visualize in the graph.

## **Graph Elements**

Nodes (circles) \= Biological pathways (groups of functionally related genes)

Edges (arrows) \= Directional interactions between pathways

Color Strength \= Importance scores (strength of effect on drug response)

Why pathways? Cells respond to drugs through groups of genes working together in pathways (like "apoptosis" or "DNA repair") that communicate via crosstalk.

## **Reading the Graph**

### Step 1: Identify Sensitivity vs. Resistance

**Node Colors:**

* **BLUE \= Promotes sensitivity. So, this pathway helps the drug work.**
* **RED \= Promotes resistance. So, this pathway helps the cell resist.**

**Color Strength**: Stronger color \= stronger effect

**Example**: BLUE "apoptosis" \= pathway helps the drug kill cells. RED "DNA repair" \= pathway helps cells resist the drug.

### Step 2: Follow Information Flow

**Edge Colors:**

* BLUE \= Signal flow promoting sensitivity  
* RED \= Signal flow promoting resistance  
* Thickness of edge \= Strength of interaction

### Example\: EGFR Inhibitor in Lung Cancer

* **Large BLUE**: "EGFR signaling" → The EGFR signaling pathway helps the drug kill the cell.  
* **Large RED**: "PI3K/AKT signaling" → Bypasses EGFR blockade and causes resistance.  
* **BLUE edge**: EGFR → MAPK → Apoptosis → Drug working through this route. This axis helps the drug work.  
* **RED edge**: PI3K/AKT → mTOR → Parallel survival pathway stays active.

Therefore, combining EGFR \+ PI3K/AKT inhibitors might overcome resistance to the drug.

Edges also show which specific interactions matter most by showing how sensitivity and resistance information flows through the model’s wiring.

## **Common Patterns**

Edges also show which specific nodes matter most by showing how sensitivity and resistance information flows through the model’s wiring.

**Feedback Loops**: Negative or positive feedback loops (A → B and B → A).

**Critical Genes**: Genes that are critical for pathway crosstalk are included.

**Hub Node (Connected to Many Nodes)**: One pathway/gene with many connections. This integrates/broadcasts signals to multiple targets. Often good drug targets (disrupting affects multiple processes).

**Parallel Pathways**: Multiple independent mechanisms for sensitivity/resistance.

* BLUE parallel: Redundant cell death routes (drug works well)  
* RED parallel: Multiple resistance mechanisms (harder to overcome)

**Opposing Forces (BLUE and RED Connected)**: Sensitivity pathway activating resistance pathway (or vice versa). A tug-of-war determines the net outcome. This can help explain intermediate drug responses.

## **Clinical Translation**

1. **Biomarkers** (large BLUE pathways \= predict response)   
2. **Resistance mechanisms** (large RED pathways \= overcome with combination therapy) and thus combination strategies  
3. **Prioritize experiments** (top pathways/edges)

---

**For technical and implementation details, see the Methods section of the PIGE paper or the Github repository [here](https://github.com/charifbahloul/PIGE).**
