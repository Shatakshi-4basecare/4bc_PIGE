# About the PIGE Pan-Cancer Causal Graph Atlas

## What is the PIGE Graph Atlas?

The PIGE Pan-Cancer Causal Graph Atlas provides **mechanistic explanations** for drug response in cancer cell lines. PIGE shows:

- Which pathways drive drug sensitivity or resistance  
- How pathway interactions (crosstalk) amplify or suppress drug effects  
- Causal mechanisms that can guide combination therapy strategies

For each graph, PIGE asks the question: "What pathways, genes and interactions cause this specific cancer cell line to be sensitive or resistant to this drug?"

## Why PIGE?

**The Problem:** Genes don't act alone. A mutation or overexpression of a gene can have different effects depending on the cellular context. Single biomarkers are not enough to understand the complexity of drug response. The literature painstakingly documents hundreds of factors affecting drug sensitivity for a given drug at immense cost in time and money.

**The Solution:** PIGE addresses these limitations through four key capabilities:

- **Unbiased discovery from only multi-omics data**: PIGE requires no prior knowledge of gene function or drug mechanisms. This allows it to identify novel dependencies that the literature may have missed.  
- **Prioritization**: While the literature documents hundreds of factors affecting drug sensitivity for a drug, there is no ranking as to how important a given gene/pathway is. PIGE quantifies and ranks the relative importance of pathways and genes  
- **Cell line-specific insights for personalized medicine**: Unlike single-model studies, PIGE generates mechanistic explanations for hundreds of cancer cell lines because the same drug can work through different pathways depending on the cellular context. This is crucial for personalized medicine.  
- **Rapid characterization of novel compounds for drug repurposing and combination therapy discovery**: Traditional experimental approaches require systematic knockout screens to map drug dependencies. PIGE allows researchers to quickly screen the mechanistic dependencies of new compounds without experimental validation.

## Atlas Coverage

PIGE covers:

- **70 drugs** across a variety of drug classes.  
- **45,000+ graphs** across hundreds of cancer cell lines.

## How to Interpret the Graphs

PIGE graphs include the most important pathways/genes in the drug response for a given drug and cell line. The color of the nodes and edges indicates the direction and magnitude of the effect of the pathway/gene on drug response:

- **BLUE \= sensitivity** **(helps drug kill cell)**.  
- **RED \= resistance** **(stops drug from killing cell)**.  
- **Thickness of edge/color strength of node \= importance** **(thicker edge or stronger node \= stronger effect on drug response)**.

See [Documentation](http://documentation.html) for a more detailed interpretation guide.

## DISCLAIMER

**PIGE IS FOR RESEARCH USE ONLY. NOT FOR CLINICAL USE.**

The PIGE Graph Atlas has not been clinically validated and is not approved for clinical use. PIGE predictions are computational models based on cancer cell line data and should thus not be used to make patient treatment decisions.

## Data Sources & Acknowledgments

- **Drug Response**: GDSC2, CTRPv2, BeatAML  
- **Genomics**: DepMap Public 24Q4, BeatAML  
- **Pathways**: Gene Ontology, OmniPath

## License

**Research Use**: Freely available for academic and non-commercial research.   
**Raw Data Access**: Node and edge data in csv format are available upon request.

## Citation & Contact

**Citation**: \[Will be added when paper is published\] 
**Contact**: [Charif Bahloul](mailto:cbahl076@uottawa.ca) 
**GitHub Repository**: [https://github.com/charifbahloul/PIGE](https://github.com/charifbahloul/PIGE)

