"""Generate a brand-compliant PIGE graph index with enhanced search and filtering.

Follows PIGE Brand Guidelines for colors, typography, and visual design.
"""

import json
from pathlib import Path
from typing import Dict


def build_content_page_html(title: str, md_filename: str) -> str:
    """Build a simple page that loads and displays markdown content.

    Args:
        title: Page title to display in header and browser tab.
        md_filename: Filename of the markdown file to load and render.

    Returns:
        Complete HTML string for the content page.
    """
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title} - PIGE Graph Atlas</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {{
      --pige-blue: #1E5AA8;
      --pige-navy: #0D2137;
      --pige-white: #FFFFFF;
      --bg-primary: #FAFAFA;
      --bg-card: #FFFFFF;
      --border-light: #E0E0E0;
      --text-primary: #0D2137;
      --text-secondary: #666666;
      --hover-blue: #174a8f;
    }}

    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    html, body {{
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg-primary);
      color: var(--text-primary);
      line-height: 1.6;
    }}

    .header {{
      background: linear-gradient(135deg, var(--pige-navy) 0%, var(--pige-blue) 100%);
      color: var(--pige-white);
      padding: 48px 0;
      margin-bottom: 32px;
    }}

    .header-content {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 0 48px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 32px;
    }}

    .header h1 {{
      font-size: 28px;
      font-weight: 700;
      letter-spacing: -0.02em;
    }}

    .back-link {{
      color: var(--pige-white);
      text-decoration: none;
      font-size: 14px;
      font-weight: 500;
      padding: 8px 16px;
      border-radius: 4px;
      transition: background 0.2s;
      white-space: nowrap;
      border: 1px solid rgba(255, 255, 255, 0.3);
    }}

    .back-link:hover {{
      background: rgba(255, 255, 255, 0.15);
    }}

    .container {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 0 48px 64px 48px;
    }}

    .content {{
      background: var(--bg-card);
      border: 1px solid var(--border-light);
      border-radius: 8px;
      padding: 48px;
      line-height: 1.8;
    }}

    .content h1 {{
      font-size: 28px;
      font-weight: 700;
      margin-bottom: 24px;
      color: var(--text-primary);
    }}

    .content h2 {{
      font-size: 22px;
      font-weight: 600;
      margin-top: 32px;
      margin-bottom: 16px;
      color: var(--text-primary);
    }}

    .content h3 {{
      font-size: 18px;
      font-weight: 600;
      margin-top: 24px;
      margin-bottom: 12px;
      color: var(--text-primary);
    }}

    .content p {{
      margin-bottom: 16px;
    }}

    .content ul, .content ol {{
      margin-bottom: 16px;
      padding-left: 32px;
    }}

    .content li {{
      margin-bottom: 8px;
    }}

    .content code {{
      background: #F5F5F5;
      padding: 2px 6px;
      border-radius: 3px;
      font-family: 'Courier New', monospace;
      font-size: 13px;
    }}

    .content pre {{
      background: #F5F5F5;
      padding: 16px;
      border-radius: 4px;
      overflow-x: auto;
      margin-bottom: 16px;
    }}

    .content pre code {{
      background: none;
      padding: 0;
    }}

    @media (max-width: 768px) {{
      .header {{
        padding: 32px 0;
      }}

      .header h1 {{
        font-size: 24px;
      }}

      .container {{
        padding: 0 24px 48px 24px;
      }}

      .content {{
        padding: 32px 24px;
      }}
    }}
  </style>
</head>
<body>
  <div class="header">
    <div class="header-content">
      <h1>{title}</h1>
      <a href="index.html" class="back-link">← Back to Atlas</a>
    </div>
  </div>

  <div class="container">
    <div class="content" id="content">
      Loading...
    </div>
  </div>

  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <script>
    async function loadContent() {{
      try {{
        const response = await fetch('{md_filename}');
        const text = await response.text();

        if (text && text.trim()) {{
          document.getElementById('content').innerHTML = marked.parse(text);
        }} else {{
          document.getElementById('content').innerHTML = '<p>Content coming soon.</p>';
        }}
      }} catch (err) {{
        document.getElementById('content').innerHTML = '<p>Content coming soon.</p>';
      }}
    }}

    loadContent();
  </script>
</body>
</html>
"""
    return html


def build_index_html(data_filename: str, key_entities_filename: str) -> str:
    """Build the main index.html with PIGE brand guidelines styling.

    Includes drug class and disease type filtering, plus advanced search.

    Args:
        data_filename: Filename for the graphs index JSON data.
        key_entities_filename: Filename for the key entities JSON data.

    Returns:
        Complete HTML string for the index page.
    """
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>PIGE Pan-Cancer Causal Graph Atlas</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {{
      --pige-blue: #1E5AA8;
      --pige-navy: #0D2137;
      --pige-white: #FFFFFF;
      --bg-primary: #FAFAFA;
      --bg-card: #FFFFFF;
      --border-light: #E0E0E0;
      --border-medium: #CCCCCC;
      --text-primary: #0D2137;
      --text-secondary: #666666;
      --text-muted: #999999;
      --hover-blue: #174a8f;
      --focus-blue: #1E5AA8;
    }}

    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    html, body {{
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg-primary);
      color: var(--text-primary);
      line-height: 1.5;
    }}

    .container {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 0 24px 32px 24px;
    }}

    .header {{
      background: linear-gradient(135deg, var(--pige-navy) 0%, var(--pige-blue) 100%);
      color: var(--pige-white);
      padding: 48px 0;
      margin-bottom: 32px;
    }}

    .header-content {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 0 48px;
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 32px;
    }}

    .header-main {{
      flex: 1;
    }}

    .header h1 {{
      font-size: 32px;
      font-weight: 700;
      margin-bottom: 12px;
      letter-spacing: -0.02em;
    }}

    .header .subtitle {{
      font-size: 16px;
      opacity: 0.95;
      line-height: 1.6;
      max-width: 800px;
    }}

    .header-nav {{
      display: flex;
      gap: 16px;
      padding-top: 4px;
    }}

    .header-link {{
      color: var(--pige-white);
      text-decoration: none;
      font-size: 14px;
      font-weight: 500;
      padding: 6px 12px;
      border-radius: 4px;
      transition: background 0.2s;
      white-space: nowrap;
    }}

    .header-link:hover {{
      background: rgba(255, 255, 255, 0.15);
    }}

    .controls {{
      background: var(--bg-card);
      border: 1px solid var(--border-light);
      border-radius: 8px;
      padding: 24px;
      margin-bottom: 24px;
    }}

    .controls-title {{
      font-size: 14px;
      font-weight: 600;
      color: var(--text-secondary);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 16px;
    }}

    .search-row {{
      display: grid;
      grid-template-columns: 1fr minmax(140px, auto);
      gap: 12px;
      margin-bottom: 16px;
      align-items: center;
    }}

    .search-input {{
      width: 100%;
      padding: 12px 16px;
      font-size: 15px;
      font-family: inherit;
      border: 1px solid var(--border-medium);
      border-radius: 4px;
      background: var(--pige-white);
      color: var(--text-primary);
      transition: border-color 0.2s;
    }}

    .search-input:focus {{
      outline: none;
      border: 2px solid var(--focus-blue);
      padding: 11px 15px;
    }}

    .search-input::placeholder {{
      color: var(--text-muted);
    }}

    .advanced-search-btn {{
      padding: 12px 20px;
      font-size: 14px;
      font-weight: 500;
      background: var(--pige-white);
      border: 1px solid var(--border-medium);
      border-radius: 4px;
      color: var(--pige-blue);
      cursor: pointer;
      white-space: nowrap;
      transition: all 0.2s;
    }}

    .advanced-search-btn:hover {{
      background: #F8F9FA;
      border-color: var(--pige-blue);
    }}

    .advanced-search-btn.active {{
      background: var(--pige-blue);
      color: var(--pige-white);
      border-color: var(--pige-blue);
    }}

    .filters-row {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 12px;
    }}

    .select {{
      padding: 10px 12px;
      font-size: 14px;
      font-family: inherit;
      border: 1px solid var(--border-medium);
      border-radius: 4px;
      background: var(--pige-white);
      color: var(--text-primary);
      cursor: pointer;
      transition: border-color 0.2s;
    }}

    .select:focus {{
      outline: none;
      border: 2px solid var(--focus-blue);
      padding: 9px 11px;
    }}

    .advanced-panel {{
      margin-top: 12px;
      padding: 16px;
      background: #F8F9FA;
      border: 1px solid var(--border-light);
      border-radius: 6px;
      display: none;
    }}

    .advanced-panel.visible {{
      display: block;
    }}

    .advanced-row {{
      display: grid;
      grid-template-columns: 75px 140px 1fr;
      gap: 8px;
      align-items: center;
      margin-bottom: 8px;
    }}

    .advanced-row:last-child {{
      margin-bottom: 0;
    }}

    .logic-select {{
      padding: 8px 6px;
      font-size: 13px;
      border: 1px solid var(--border-medium);
      border-radius: 4px;
      background: var(--pige-white);
      height: 36px;
    }}

    .field-select {{
      padding: 8px 6px;
      font-size: 13px;
      border: 1px solid var(--border-medium);
      border-radius: 4px;
      background: var(--pige-white);
      height: 36px;
    }}

    .rule-input {{
      padding: 8px 12px;
      font-size: 14px;
      border: 1px solid var(--border-medium);
      border-radius: 4px;
      background: var(--pige-white);
      height: 36px;
    }}

    .rule-input:focus {{
      outline: none;
      border: 2px solid var(--focus-blue);
      padding: 7px 11px;
    }}

    .advanced-actions {{
      display: flex;
      gap: 8px;
      margin-top: 12px;
      align-items: center;
    }}

    .add-rule-btn {{
      background: var(--pige-white);
      color: var(--pige-blue);
      border: 1px solid var(--border-medium);
      padding: 8px 12px;
      border-radius: 4px;
      cursor: pointer;
      font-size: 13px;
      font-weight: 500;
      height: 36px;
      white-space: nowrap;
    }}

    .add-rule-btn:hover {{
      background: #F8F9FA;
      border-color: var(--pige-blue);
    }}

    .remove-btn {{
      background: var(--pige-white);
      color: #D9534F;
      border: 1px solid var(--border-medium);
      padding: 8px 12px;
      border-radius: 4px;
      cursor: pointer;
      font-size: 13px;
      font-weight: 500;
      height: 36px;
      white-space: nowrap;
    }}

    .remove-btn:hover {{
      background: #F8F9FA;
      border-color: #D9534F;
    }}

    .stats {{
      display: flex;
      gap: 24px;
      flex-wrap: wrap;
      padding-top: 16px;
      border-top: 1px solid var(--border-light);
      margin-top: 16px;
    }}

    .stat {{
      font-size: 13px;
      color: var(--text-secondary);
    }}

    .stat strong {{
      color: var(--pige-blue);
      font-weight: 600;
      font-size: 18px;
      margin-left: 4px;
    }}

    .results {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
      gap: 16px;
    }}

    .card {{
      background: var(--bg-card);
      border: 1px solid var(--border-light);
      border-radius: 8px;
      padding: 16px;
      transition: all 0.2s ease;
      text-decoration: none;
      color: inherit;
      display: flex;
      flex-direction: column;
      gap: 10px;
      content-visibility: auto;
      contain: layout style paint;
    }}

    .card:hover {{
      border-color: var(--pige-blue);
      box-shadow: 0 4px 12px rgba(30, 90, 168, 0.15);
      transform: translateY(-2px);
    }}

    .card-header {{
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
    }}

    .card-title {{
      font-size: 16px;
      font-weight: 600;
      color: var(--text-primary);
      line-height: 1.3;
    }}

    .card-drug {{
      color: var(--pige-blue);
      font-weight: 700;
    }}

    .card-meta {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      font-size: 12px;
    }}

    .badge {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 12px;
      font-size: 11px;
      font-weight: 500;
      border: 1px solid;
    }}

    .badge-default {{
      background: #F8F9FA;
      border-color: #E0E0E0;
      color: var(--text-secondary);
    }}

    .badge-average {{
      background: #E8F5E9;
      border-color: #81C784;
      color: #2E7D32;
    }}

    .badge-aac {{
      background: #E3F2FD;
      border-color: #64B5F6;
      color: var(--pige-blue);
    }}

    .empty {{
      text-align: center;
      padding: 64px 24px;
      color: var(--text-muted);
      font-size: 15px;
    }}

    .loading {{
      text-align: center;
      padding: 32px;
      color: var(--text-muted);
      font-size: 14px;
    }}

    .filter-active {{
      background: #E3F2FD !important;
      border-color: var(--pige-blue) !important;
      font-weight: 500;
    }}

    @media (max-width: 768px) {{
      .header {{
        padding: 32px 0;
      }}

      .header h1 {{
        font-size: 24px;
      }}

      .header-content {{
        flex-direction: column;
        gap: 16px;
        padding: 0 24px;
      }}

      .header-nav {{
        width: 100%;
        justify-content: flex-start;
      }}

      .container {{
        padding: 0 16px 32px 16px;
      }}

      .controls {{
        padding: 16px;
      }}

      .search-row {{
        grid-template-columns: 1fr;
      }}

      .filters-row {{
        grid-template-columns: 1fr;
      }}

      .select {{
        width: 100%;
        min-width: 0;
      }}

      .results {{
        grid-template-columns: 1fr;
      }}

      .advanced-row {{
        grid-template-columns: 1fr;
      }}

      .logic-select, .field-select, .rule-input {{
        width: 100%;
        min-width: 0;
      }}

      .advanced-actions {{
        flex-direction: column;
      }}

      .add-rule-btn, .remove-btn {{
        width: 100%;
      }}
    }}

    @media (max-width: 480px) {{
      .header h1 {{
        font-size: 20px;
      }}

      .header .subtitle {{
        font-size: 14px;
      }}

      .header-content {{
        padding: 0 16px;
      }}

      .container {{
        padding: 0 12px 24px 12px;
      }}

      .controls {{
        padding: 12px;
      }}

      .search-input {{
        font-size: 14px;
        padding: 10px 12px;
      }}

      .select {{
        font-size: 13px;
        padding: 8px 10px;
      }}

      .logic-select, .field-select {{
        font-size: 12px;
        padding: 6px 4px;
      }}

      .rule-input {{
        font-size: 13px;
        padding: 6px 10px;
      }}
    }}
  </style>
</head>
<body>
  <div class="header">
    <div class="header-content">
      <div class="header-main">
        <h1>PIGE Pan-Cancer Causal Graph Atlas</h1>
        <p class="subtitle">
          Explore causal pathway graphs revealing drug response mechanisms across cancer cell lines.
          Search by any criteria to discover mechanistic insights into the factors driving sensitivity and resistance.
        </p>
      </div>
      <nav class="header-nav">
        <a href="about.html" class="header-link">About</a>
        <a href="documentation.html" class="header-link">Interpretation Guide</a>
      </nav>
    </div>
  </div>

  <div class="container">
    <div class="controls">
      <div class="controls-title">Search & Filter</div>

      <div class="search-row">
        <input
          id="searchInput"
          type="search"
          class="search-input"
          placeholder="Search by drug, cell line, disease, drug class, pathway, or gene..."
          autocomplete="off"
        />
        <button class="advanced-search-btn" id="advancedToggle">Advanced Search</button>
      </div>

      <div class="filters-row">
        <select id="drugClassFilter" class="select">
          <option value="">All Drug Classes</option>
        </select>

        <select id="diseaseTypeFilter" class="select">
          <option value="">All Disease Types</option>
        </select>

        <select id="datasetFilter" class="select">
          <option value="">All Datasets</option>
        </select>
      </div>

      <div class="advanced-panel" id="advancedPanel">
        <div id="advancedRules"></div>
        <div class="advanced-actions">
          <button class="add-rule-btn" id="addRuleBtn">+ Add Filter Rule</button>
          <button class="remove-btn" id="clearRulesBtn" style="display: none;">Clear All</button>
        </div>
      </div>

      <div class="stats">
        <div class="stat">Drugs: <strong id="statDrugs">0</strong></div>
        <div class="stat">Cell Lines: <strong id="statCellLines">0</strong></div>
        <div class="stat">Total Graphs: <strong id="statTotal">0</strong></div>
      </div>
    </div>

    <div id="loading" class="loading">Loading graph data...</div>
    <div id="results" class="results" style="display:none"></div>
    <div id="empty" class="empty" style="display:none">
      No graphs match your search criteria. Try adjusting your filters.
    </div>
  </div>

  <script>
    const USE_CHUNKED_LOADING = true;
    const PREVIEW_URL = "index_preview.json";
    const REMAINING_URL = "index_remaining.json";
    const PATHWAY_GENE_URL = "pathway_gene_data.json";
    const KEY_ENTITIES_URL = {json.dumps(key_entities_filename)};

    let graphData = [];
    let keyEntities = null;
    let pathwayGeneData = null;
    let drugClassOptions = [];
    let diseaseTypeOptions = [];
    let datasetOptions = [];
    let advancedRules = [];
    let searchIndex = null;
    let currentSearchId = 0;

    const searchInput = document.getElementById('searchInput');
    const drugClassFilter = document.getElementById('drugClassFilter');
    const diseaseTypeFilter = document.getElementById('diseaseTypeFilter');
    const datasetFilter = document.getElementById('datasetFilter');
    const results = document.getElementById('results');
    const empty = document.getElementById('empty');
    const loading = document.getElementById('loading');
    const statDrugs = document.getElementById('statDrugs');
    const statCellLines = document.getElementById('statCellLines');
    const statTotal = document.getElementById('statTotal');
    const advancedToggle = document.getElementById('advancedToggle');
    const advancedPanel = document.getElementById('advancedPanel');
    const advancedRulesContainer = document.getElementById('advancedRules');
    const addRuleBtn = document.getElementById('addRuleBtn');
    const clearRulesBtn = document.getElementById('clearRulesBtn');

    function buildSearchIndex(data) {{
      const index = new Map();

      for (let i = 0; i < data.length; i++) {{
        const item = data[i];
        const text = item._searchText || '';

        const words = text.split(/\\s+/);
        for (const word of words) {{
          if (word.length > 0) {{
            if (!index.has(word)) {{
              index.set(word, new Set());
            }}
            index.get(word).add(i);
          }}
        }}
      }}

      return index;
    }}

    function animateCount(element, target) {{
      const duration = 1500;
      const start = 0;
      const startTime = performance.now();

      function update(currentTime) {{
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const current = Math.max(0, Math.floor(start + (target - start) * progress));

        element.textContent = current.toLocaleString();

        if (progress < 1) {{
          requestAnimationFrame(update);
        }} else {{
          element.textContent = Math.max(0, target).toLocaleString();
        }}
      }}

      requestAnimationFrame(update);
    }}

    advancedToggle.addEventListener('click', () => {{
      const isVisible = advancedPanel.classList.toggle('visible');
      advancedToggle.classList.toggle('active', isVisible);

      if (isVisible) {{
        // Load pathway/gene data on-demand (with sessionStorage caching)
        if (!pathwayGeneData) {{
          const cached = sessionStorage.getItem('pathwayGeneData');
          if (cached) {{
            try {{
              pathwayGeneData = JSON.parse(cached);
              console.log('Loaded pathway/gene data from cache');
              mergePathwayGeneData();
            }} catch (e) {{
              console.warn('Failed to parse cached pathway/gene data');
            }}
          }} else {{
            fetch(PATHWAY_GENE_URL)
              .then(r => r.json())
              .then(data => {{
                pathwayGeneData = data;
                sessionStorage.setItem('pathwayGeneData', JSON.stringify(data));
                console.log('Loaded pathway/gene data');
                mergePathwayGeneData();
              }})
              .catch(err => {{
                console.warn('Failed to load pathway/gene data:', err);
              }});
          }}
        }}

        if (advancedRules.length === 0) {{
          const ruleId = Date.now();
          advancedRules.push({{
            id: ruleId,
            logic: 'AND',
            field: 'drug',
            value: ''
          }});
          renderAdvancedRules();
        }}
      }}
    }});

    function mergePathwayGeneData() {{
      if (!pathwayGeneData) return;

      graphData.forEach(item => {{
        const key = `${{item.drug}}|${{item.cell_line}}`;
        const data = pathwayGeneData[key];
        if (data) {{
          item._pathwayText = data.pathways || '';
          item._geneText = data.genes || '';
        }}
      }});

      console.log('Merged pathway/gene data into graph entries');
    }}

    addRuleBtn.addEventListener('click', () => {{
      const ruleId = Date.now();
      advancedRules.push({{
        id: ruleId,
        logic: 'AND',
        field: 'drug',
        value: ''
      }});
      renderAdvancedRules();
    }});

    clearRulesBtn.addEventListener('click', () => {{
      advancedRules = [];
      renderAdvancedRules();
      render();
    }});

    function renderAdvancedRules() {{
      if (advancedRules.length === 0) {{
        advancedRulesContainer.innerHTML = '<p style="color: #999; font-size: 13px;">No advanced rules. Click "Add Filter Rule" to start.</p>';
        clearRulesBtn.style.display = 'none';
        return;
      }}

      clearRulesBtn.style.display = 'inline-block';

      advancedRulesContainer.innerHTML = advancedRules.map((rule, index) => `
        <div class="advanced-row">
          <select class="logic-select" data-rule-id="${{rule.id}}" data-field="logic">
            <option value="AND" ${{rule.logic === 'AND' ? 'selected' : ''}}>AND</option>
            <option value="OR" ${{rule.logic === 'OR' ? 'selected' : ''}}>OR</option>
          </select>
          <select class="field-select" data-rule-id="${{rule.id}}" data-field="field">
            <option value="drug" ${{rule.field === 'drug' ? 'selected' : ''}}>Drug</option>
            <option value="cell_line" ${{rule.field === 'cell_line' ? 'selected' : ''}}>Cell Line</option>
            <option value="disease_type" ${{rule.field === 'disease_type' ? 'selected' : ''}}>Disease</option>
            <option value="drug_class" ${{rule.field === 'drug_class' ? 'selected' : ''}}>Drug Class</option>
            <option value="pathway" ${{rule.field === 'pathway' ? 'selected' : ''}}>Pathway</option>
            <option value="gene" ${{rule.field === 'gene' ? 'selected' : ''}}>Gene</option>
          </select>
          <input type="text" class="rule-input" placeholder="Search value..." value="${{rule.value || ''}}" data-rule-id="${{rule.id}}" data-field="value" />
        </div>
      `).join('');

      advancedRulesContainer.querySelectorAll('[data-rule-id]').forEach(el => {{
        const ruleId = parseInt(el.dataset.ruleId);
        const field = el.dataset.field;

        el.addEventListener('change', (e) => {{
          const rule = advancedRules.find(r => r.id === ruleId);
          if (rule) {{
            rule[field] = e.target.value;
            render();
          }}
        }});

        el.addEventListener('input', (e) => {{
          const rule = advancedRules.find(r => r.id === ruleId);
          if (rule && field === 'value') {{
            rule.value = e.target.value;
            render();
          }}
        }});
      }});
    }}

    async function init() {{
      try {{
        // Load preview data first (contains first 5000 entries with top 500 validated)
        const previewResponse = await fetch(PREVIEW_URL);
        const previewData = await previewResponse.json();
        let data = previewData.entries || previewData;
        const metadata = previewData.metadata || null;

        // Process preview data for immediate display
        data.forEach(item => {{
          if (!item._searchText) {{
            item._searchText = [
              item.drug,
              item.name,
              item.cell_line,
              item.disease_type,
              item.drug_class,
              item.dataset
            ].join(' ').toLowerCase();
          }}
          if (!item._pathwayText) item._pathwayText = '';
          if (!item._geneText) item._geneText = '';
        }});

        graphData = data;

        // Build search index for preview data
        const indexStart = performance.now();
        searchIndex = buildSearchIndex(graphData);
        console.log(`Search index built in ${{(performance.now() - indexStart).toFixed(2)}}ms with ${{searchIndex.size}} unique words`);

        if (metadata) {{
          drugClassOptions = metadata.drug_classes || [];
          diseaseTypeOptions = metadata.disease_types || [];
          datasetOptions = metadata.datasets || [];
        }} else {{
          // Fallback to preview data if metadata not available
          drugClassOptions = [...new Set(graphData.map(d => d.drug_class))].filter(Boolean).sort();
          diseaseTypeOptions = [...new Set(graphData.map(d => d.disease_type))].filter(Boolean).sort();
          datasetOptions = [...new Set(graphData.map(d => d.dataset))].sort();
        }}

        populateSelect(drugClassFilter, drugClassOptions);
        populateSelect(diseaseTypeFilter, diseaseTypeOptions);
        populateSelect(datasetFilter, datasetOptions);

        // Start animating immediately to final counts from metadata
        if (metadata && metadata.stats) {{
          animateCount(statDrugs, metadata.stats.unique_drugs);
          animateCount(statCellLines, metadata.stats.unique_cell_lines);
          animateCount(statTotal, metadata.stats.total_graphs);
        }}

        loading.style.display = 'none';
        results.style.display = 'grid';

        render();

        // Silently load remaining data in background
        fetch(REMAINING_URL)
          .then(r => r.json())
          .then(remainingData => {{
            remainingData.forEach(item => {{
              if (!item._searchText) {{
                item._searchText = [
                  item.drug,
                  item.name,
                  item.cell_line,
                  item.disease_type,
                  item.drug_class,
                  item.dataset
                ].join(' ').toLowerCase();
              }}
              if (!item._pathwayText) item._pathwayText = '';
              if (!item._geneText) item._geneText = '';
            }});

            graphData = graphData.concat(remainingData);

            // Rebuild search index with full data
            searchIndex = buildSearchIndex(graphData);
            console.log(`Full search index built with ${{searchIndex.size}} unique words`);

            console.log(`Loaded all ${{graphData.length}} graphs`);
          }})
          .catch(err => {{
            console.warn('Could not load remaining data:', err);
          }});

      }} catch (err) {{
        loading.textContent = 'Error loading graph data: ' + err.message;
        console.error(err);
      }}
    }}

    function populateSelect(selectEl, options) {{
      const currentValue = selectEl.value;
      const allOption = selectEl.querySelector('option[value=""]');
      selectEl.innerHTML = '';
      selectEl.appendChild(allOption);

      options.forEach(opt => {{
        const option = document.createElement('option');
        option.value = opt;
        option.textContent = opt;
        selectEl.appendChild(option);
      }});

      selectEl.value = currentValue;
    }}

    function updateFilterVisuals() {{
      [drugClassFilter, diseaseTypeFilter, datasetFilter].forEach(select => {{
        if (select.value) {{
          select.classList.add('filter-active');
        }} else {{
          select.classList.remove('filter-active');
        }}
      }});
    }}

    function getMatchingIndices(query) {{
      if (!query || !searchIndex) return null;

      const words = query.split(/\\s+/).filter(w => w.length > 0);
      if (words.length === 0) return null;

      let resultSet = null;

      for (const word of words) {{
        let wordMatches = new Set();

        for (const [indexWord, indices] of searchIndex) {{
          if (indexWord.indexOf(word) !== -1) {{
            for (const idx of indices) {{
              wordMatches.add(idx);
            }}
          }}
        }}

        if (resultSet === null) {{
          resultSet = wordMatches;
        }} else {{
          resultSet = new Set([...resultSet].filter(idx => wordMatches.has(idx)));
          if (resultSet.size === 0) return resultSet;
        }}
      }}

      return resultSet;
    }}

    function matchesAdvancedRules(item) {{
      if (advancedRules.length === 0) return true;

      let hasAnd = false;
      let hasOr = false;
      let andPass = true;
      let orPass = false;

      for (let i = 0; i < advancedRules.length; i++) {{
        const rule = advancedRules[i];
        if (!rule.value) continue;

        const searchValue = rule.value.toLowerCase().trim();
        if (!searchValue) continue;

        let matches = false;
        let textToSearch = '';

        if (rule.field === 'pathway') {{
          textToSearch = item._pathwayText || '';
        }} else if (rule.field === 'gene') {{
          textToSearch = item._geneText || '';
        }} else {{
          textToSearch = (item[rule.field] || '').toLowerCase();
        }}

        const words = searchValue.split(/\\s+/);
        matches = true;
        for (const word of words) {{
          if (textToSearch.indexOf(word) === -1) {{
            matches = false;
            break;
          }}
        }}

        if (rule.logic === 'AND') {{
          hasAnd = true;
          if (!matches) {{
            andPass = false;
            if (!hasOr) return false;
          }}
        }} else {{
          hasOr = true;
          if (matches) {{
            orPass = true;
          }}
        }}
      }}

      return andPass && (!hasOr || orPass);
    }}

    function render() {{
      const searchId = ++currentSearchId;
      const query = searchInput.value.trim().toLowerCase();
      const drugClass = drugClassFilter.value;
      const diseaseType = diseaseTypeFilter.value;
      const dataset = datasetFilter.value;

      updateFilterVisuals();

      let candidateIndices = null;
      if (query && searchIndex) {{
        candidateIndices = getMatchingIndices(query);
        if (candidateIndices && candidateIndices.size === 0) {{
          results.style.display = 'none';
          empty.style.display = 'block';
          return;
        }}
      }}

      // Synchronous filtering for better performance
      const itemsToCheck = candidateIndices
        ? Array.from(candidateIndices).map(idx => graphData[idx])
        : graphData;

      const filtered = itemsToCheck.filter(item => {{
        if (drugClass && item.drug_class !== drugClass) return false;
        if (diseaseType && item.disease_type !== diseaseType) return false;
        if (dataset && item.dataset !== dataset) return false;
        if (!matchesAdvancedRules(item)) return false;
        return true;
      }});

      if (filtered.length === 0) {{
        results.style.display = 'none';
        empty.style.display = 'block';
      }} else {{
        results.style.display = 'grid';
        empty.style.display = 'none';

        // Efficient DOM rendering with DocumentFragment batching
        results.innerHTML = '';

        const BATCH_SIZE = 500;
        let currentBatch = 0;

        function renderBatch() {{
          if (searchId !== currentSearchId) return;

          const start = currentBatch * BATCH_SIZE;
          const end = Math.min(start + BATCH_SIZE, filtered.length);
          const batch = filtered.slice(start, end);

          const fragment = document.createDocumentFragment();
          batch.forEach(item => {{
            const div = document.createElement('div');
            div.innerHTML = renderCard(item);
            fragment.appendChild(div.firstElementChild);
          }});

          results.appendChild(fragment);

          currentBatch++;
          if (end < filtered.length) {{
            requestAnimationFrame(renderBatch);
          }}
        }}

        renderBatch();
      }}
    }}

    function renderCard(item) {{
      const avgBadge = item.average
        ? '<span class="badge badge-average">AVERAGE</span>'
        : '';

      const aacBadge = (typeof item.aac === 'number')
        ? `<span class="badge badge-aac">AAC ${{item.aac.toFixed(3)}}</span>`
        : '';

      const diseaseBadge = item.disease_type && !item.average
        ? `<span class="badge badge-default">${{item.disease_type}}</span>`
        : '';

      return `
        <a href="${{item.href}}" class="card" target="_blank" rel="noopener">
          <div class="card-header">${{item.dataset}} • ${{item.drug_class || 'Other'}}</div>
          <div class="card-title">
            ${{item.name}} - <span class="card-drug">${{item.drug}}</span>
          </div>
          <div class="card-meta">
            ${{avgBadge}}
            ${{diseaseBadge}}
            ${{aacBadge}}
          </div>
        </a>
      `;
    }}

    searchInput.addEventListener('input', render);
    drugClassFilter.addEventListener('change', render);
    diseaseTypeFilter.addEventListener('change', render);
    datasetFilter.addEventListener('change', render);

    init();
  </script>
</body>
</html>
"""
    return html


def generate_html_index(config: Dict) -> None:
    """Generate HTML index and content pages from configuration.

    Args:
        config: Configuration dictionary containing:
            - output_dir: Directory to write HTML files
            - data_json: Filename of the graphs index JSON
            - key_entities_json: Filename of the key entities JSON
    """
    output_dir = Path(config['output_dir']).resolve()
    data_json = config.get('data_json', 'graphs_index.json')
    key_entities_json = config.get('key_entities_json', 'key_entities.json')

    output_dir.mkdir(parents=True, exist_ok=True)

    html = build_index_html(data_json, key_entities_json)
    index_path = output_dir / "index.html"

    print(f"Writing index.html to {index_path}")
    index_path.write_text(html, encoding='utf-8')

    about_html = build_content_page_html("About", "about.md")
    about_path = output_dir / "about.html"
    print(f"Writing about.html to {about_path}")
    about_path.write_text(about_html, encoding='utf-8')

    docs_html = build_content_page_html("Documentation", "documentation.md")
    docs_path = output_dir / "documentation.html"
    print(f"Writing documentation.html to {docs_path}")
    docs_path.write_text(docs_html, encoding='utf-8')

    print("Done")
