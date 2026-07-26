# Ref-Cit-Analysis: Multi-Level DOI Citation Analysis Tool

**A comprehensive scientific tool for analyzing citation networks across three interconnected levels: References (Level I), Analyzed Papers (Level II), and Citing Works (Level III).**

---

## 📋 Overview

**Ref-Cit-Analysis** is a powerful, multi-level citation analysis tool designed for researchers, scientists, and academic professionals who need to understand the complete citation ecosystem surrounding a set of scholarly publications. By analyzing DOIs (Digital Object Identifiers) across three distinct levels, the tool provides a holistic view of academic influence, research impact, and scholarly networks.

The tool goes beyond simple citation counting by examining:
- **Level I (References):** The papers cited by your analyzed articles
- **Level II (Analyzed):** Your core set of publications under investigation
- **Level III (Citing Works):** The papers that cite your analyzed articles

This three-level approach reveals the full citation landscape, from foundational works through your research to its academic impact.

---

## 🎯 Key Features

### 🔍 Multi-Level Citation Analysis
- **Three interconnected levels** of citation relationships
- **Weighted counting** to account for citation frequency across levels
- **Cross-level citation detection** to identify potential self-citation cycles
- **Comprehensive metadata extraction** including author, affiliation, and publication data

### 📊 Advanced Analytics
- **Author collaboration analysis** with ORCID integration
- **Geographic distribution** and international collaboration metrics
- **Citation dynamics** with temporal trend analysis
- **First citation lag analysis** to measure research impact speed
- **H-index, g-index, i10-index, and i100-index** calculations

### 🏷️ Semantic Analysis
- **Title keyword extraction** with lemmatization and stopword filtering
- **Compound word detection** for scientific terminology
- **Topic clustering** across all three levels
- **Concept extraction** using OpenAlex taxonomy

### 🌐 Network Visualization
- **Multilevel relationship matrices** (authors, affiliations, journals, publishers)
- **Citation network heatmaps** showing publication-citation dynamics
- **Temporal relationship visualization** (time lags between levels)
- **Collaboration pattern analysis** (domestic vs. international)

### 📄 Comprehensive Reporting
- **Interactive HTML reports** with collapsible sections
- **Rich visualizations** including charts, tables, and heatmaps
- **Full reference lists** with citation weights
- **Detailed citation view** with complete bibliographic information
- **Downloadable self-contained reports** with embedded resources

---

## 🔬 Technical Architecture

### Data Sources
- **OpenAlex API** – Primary data source for publication metadata and citation relationships
- **ORCID API** – For author profile information and persistent identifiers
- **DOI Resolution** – Automatic normalization and validation of DOIs

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Ref-Cit-Analysis                         │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Level I     │  │  Level II    │  │  Level III   │     │
│  │  References  │  │  Analyzed    │  │  Citing      │     │
│  │  (Weighted)  │  │  DOIs        │  │  Works       │     │
│  │              │  │  (Unique)    │  │  (Weighted)  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│         │                  │                  │            │
│         └──────────────────┼──────────────────┘            │
│                            ▼                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Analysis Engine                        │   │
│  │  • Metadata Extraction                              │   │
│  │  • Author/ Affiliation Analysis                     │   │
│  │  • Geographic & Collaboration Analysis             │   │
│  │  • Citation Dynamics                                │   │
│  │  • Topic & Keyword Extraction                       │   │
│  │  • Temporal Relationship Analysis                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                            │                               │
│                            ▼                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         HTML Report Generator                       │   │
│  │  • Interactive Navigation                           │   │
│  │  • Dynamic Visualizations                           │   │
│  │  • Self-contained HTML                              │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack
- **Python 3.8+** – Core programming language
- **Streamlit** – Web interface for interactive data loading and configuration
- **Pandas & NumPy** – Data manipulation and numerical analysis
- **Matplotlib & Seaborn** – Static visualizations
- **Plotly** – Interactive charts and heatmaps
- **NLTK** – Natural language processing for title keyword extraction
- **BeautifulSoup & Requests** – Web scraping and API interaction
- **Tenacity** – Robust retry logic for API calls

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Internet connection (for API access)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/ref-cit-analysis.git
cd ref-cit-analysis
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install NLTK Data

```bash
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-eng'); nltk.download('stopwords'); nltk.download('punkt')"
```

### Step 5: Run the Application

```bash
streamlit run main.py
```

---

## 📁 Project Structure

```
ref-cit-analysis/
├── main.py                      # Main application entry point
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── icons/                       # Icon assets for HTML reports
│   ├── 01.png                   # Overview
│   ├── 02.png                   # References
│   ├── 03.png                   # Analyzed Articles
│   ├── 04.png                   # Citation Analysis
│   ├── 05.png                   # Citing Works
│   ├── 06.png                   # Topics Analysis
│   ├── 07.png                   # Detailed Citations
│   ├── 08.png                   # Multilevel Relationships
│   ├── 09.png                   # Title Keywords
│   └── 10.png                   # Temporal Relationships
│
├── logo.png                     # Application logo (optional)
│
├── cache_doi/                   # Cache directory for API responses
│   └── *.json                   # Cached metadata
│
└── output/                      # Generated reports
    └── doi_analysis_*.html      # Self-contained HTML reports
```

---

## 🚀 Usage Guide

### 1. Launch the Application

```bash
streamlit run main.py
```

### 2. Input DOIs

Enter DOIs in the text area. The tool supports multiple formats:
- Plain DOI: `10.1016/j.jechem.2024.02.047`
- URL format: `https://doi.org/10.1021/acs.chemrev.3c00123`
- Multiple DOIs separated by newlines, commas, or semicolons

**Note:** Maximum 50 DOIs per analysis session.

### 3. Run Analysis

Click **"Analyze DOI Network"** to start the analysis. The process includes:

1. **Stage 1:** Fetching Level II (analyzed DOIs)
2. **Stage 2:** Fetching Level I (references)
3. **Stage 3:** Fetching Level III (citing works)
4. **Stage 4:** Fetching metadata for all levels
5. **Stage 5:** Analyzing data and generating report

### 4. Explore Results

The HTML report provides comprehensive analysis across ten sections (detailed below).

---

## 📊 Report Sections: Complete Metric Descriptions

### 1. 📋 Overview

**Purpose:** Provides a high-level summary of all three citation levels with key performance indicators.

**Level I (References) Metrics:**

| Metric | Description |
|--------|-------------|
| **Total Items** | Number of unique reference DOIs cited by Level II papers |
| **Total Weighted** | Sum of citation frequencies (a paper cited multiple times counts multiple times) |
| **Total Citations** | Total citation count received by all reference papers combined |
| **Avg Citations** | Average number of citations per reference paper |
| **Unique Authors** | Total distinct authors across all reference papers |
| **Open Access** | Percentage of reference papers available in open access format |
| **H-index** | H-index calculated from reference papers (h papers with at least h citations) |
| **Active Years** | Number of distinct publication years represented in references |

**Level II (Analyzed) Metrics:**

| Metric | Description |
|--------|-------------|
| **Total Items** | Number of analyzed DOIs entered by the user |
| **Total Citations** | Total citation count received by all analyzed papers combined |
| **Avg Citations** | Average number of citations per analyzed paper |
| **Unique Authors** | Total distinct authors across all analyzed papers |
| **Open Access** | Percentage of analyzed papers available in open access format |
| **H-index** | H-index calculated from analyzed papers |
| **Active Years** | Number of distinct publication years represented in analyzed papers |
| **International Collaboration Rate** | Percentage of papers with authors from multiple countries |

**Level III (Citing Works) Metrics:**

| Metric | Description |
|--------|-------------|
| **Total Items** | Number of unique citing work DOIs that cite Level II papers |
| **Total Weighted** | Sum of citation frequencies (a work citing multiple Level II papers counts multiple times) |
| **Total Citations** | Total citation count received by all citing works combined |
| **Avg Citations** | Average number of citations per citing work |
| **Unique Authors** | Total distinct authors across all citing works |
| **Open Access** | Percentage of citing works available in open access format |
| **H-index** | H-index calculated from citing works |
| **Active Years** | Number of distinct publication years represented in citing works |

**Open Access Breakdown:**
- **Gold:** Published in fully open access journals
- **Hybrid:** Published in subscription journals with open access option
- **Green:** Self-archived in repositories
- **Bronze:** Free to read but not formally open access
- **Closed:** Paywalled content

**Cross-Level Citations:** Identifies when a Level II DOI appears in Level I or Level III, potentially indicating self-citation or citation cycles.

---

### 2. 📖 References (Level I)

**Purpose:** Detailed examination of all papers cited by the analyzed articles.

**Metrics Displayed:**

| Metric | Description |
|--------|-------------|
| **DOI** | Digital Object Identifier of the reference paper |
| **Title** | Full title of the reference paper |
| **Year** | Publication year |
| **Weighted Count** | Number of Level II papers that cite this reference (citation frequency) |
| **Journal** | Journal name where the reference was published |

**Interpretation:** Higher weighted counts indicate foundational papers that are frequently cited by the analyzed articles. These represent the intellectual heritage of your research.

---

### 3. 📄 Analyzed Articles (Level II)

**Purpose:** Comprehensive analysis of the user's input papers.

**Author Distribution:**

| Metric | Description |
|--------|-------------|
| **Author Count Categories** | Distribution of papers by number of authors (1, 2, 3-5, 6-7, 8-10, 11-15, 15+) |
| **Total** | Total number of analyzed papers |

**Author Analysis:**

| Metric | Description |
|--------|-------------|
| **Author Name** | Full name of the author |
| **ORCID** | Author's ORCID identifier (linked to ORCID.org) |
| **Affiliations** | Institutional affiliations of the author |
| **Countries** | Countries associated with the author's affiliations |
| **Publications Count** | Number of analyzed papers co-authored by this author |
| **Citations Count** | Total citations received by all papers co-authored by this author |

**Top Affiliations:**

| Metric | Description |
|--------|-------------|
| **Affiliation** | Name of the institution or organization |
| **Publications Count** | Number of analyzed papers affiliated with this institution |
| **ROR ID** | Research Organization Registry identifier (linked to colab.ws) |

**Geographic Analysis:**

| Metric | Description |
|--------|-------------|
| **Avg Countries per Publication** | Average number of distinct countries represented in affiliations per paper |
| **Min/Max Countries** | Minimum and maximum number of countries per paper |
| **Single-Country Papers** | Papers with authors from only one country |
| **Multi-Country Papers** | Papers with authors from two or more countries |
| **Country Rankings** | List of countries by number of unique works and author count |

**Collaboration Patterns:**
- **Domestic:** Collaborations within the same country
- **International:** Cross-border collaborations

---

### 4. 📈 Citation Analysis

**Purpose:** In-depth analysis of citation patterns and impact metrics.

**Citation Dynamics by Year:**

| Metric | Description |
|--------|-------------|
| **Publication Year** | Year when the cited paper was published |
| **Citation Year** | Year when the citation occurred |
| **Citations Count** | Number of citations received in that year |

**First Citation Analysis:**

| Metric | Description |
|--------|-------------|
| **Min Lag (days)** | Shortest time from publication to first citation |
| **Max Lag (days)** | Longest time from publication to first citation |
| **Avg Lag (days)** | Average time from publication to first citation |
| **Median Lag (days)** | Median time from publication to first citation |

**Cumulative Citations:**

| Metric | Description |
|--------|-------------|
| **Year** | Calendar year |
| **Citations** | Total cumulative citations up to that year |

**Interpretation:** Steep cumulative curves indicate rapid impact; gradual curves suggest slower recognition.

**Citation Network Heatmap:**
- **X-axis:** Citation years
- **Y-axis:** Publication years of analyzed papers
- **Cell Values:** Number of citations from publication year to citation year
- **Color Intensity:** Darker = more citations

**Most Cited Publications:**

| Metric | Description |
|--------|-------------|
| **Rank** | Ranking by total citations |
| **Title** | Paper title |
| **Year** | Publication year |
| **Citations** | Total citations received |
| **Citations/Year** | Average citations per year since publication |
| **Authors** | Author list |
| **DOI** | Linked to the full publication |

---

### 5. 📚 Citing Works Analysis (Level III)

**Purpose:** Examination of papers that cite the analyzed articles.

**Overview Metrics:**

| Metric | Description |
|--------|-------------|
| **Total Citing Works** | Weighted count of all citing works |
| **Unique** | Number of unique citing works |
| **Citing Works Weighted** | Works with weighted counts (how many Level II papers they cite) |

**Top Citing Authors:**

| Metric | Description |
|--------|-------------|
| **Author Name** | Name of the citing author |
| **ORCID** | Author's ORCID identifier |
| **Citations Count** | Weighted count (number of Level II papers this author cites) |

**Top Citing Affiliations:**

| Metric | Description |
|--------|-------------|
| **Affiliation** | Institution of the citing author |
| **Citations Count** | Weighted count of citations from this affiliation |
| **ROR ID** | Research Organization Registry identifier |

**Top Citing Countries:**

| Metric | Description |
|--------|-------------|
| **Country** | Country of the citing author |
| **Citations Count** | Weighted count of citations from this country |

**Top Citing Journals:**

| Metric | Description |
|--------|-------------|
| **Journal** | Journal where citing work was published |
| **Citations Count** | Weighted count of citations from this journal |

**Top Citing Publishers:**

| Metric | Description |
|--------|-------------|
| **Publisher** | Publisher of the citing work |
| **Citations Count** | Weighted count of citations from this publisher |

**Citing Works Weighted Count:**

| Metric | Description |
|--------|-------------|
| **DOI** | Citing work identifier |
| **Title** | Title of the citing work |
| **Year** | Publication year |
| **Weighted Count** | Number of Level II papers cited by this work |
| **Journal** | Journal where citing work was published |
| **Authors** | Author list |

**Interpretation:** Higher weighted counts indicate works that are directly building upon multiple analyzed papers.

---

### 6. 🏷️ Topics Analysis

**Purpose:** Thematic structure analysis across all three levels.

**Topics Table:**

| Metric | Description |
|--------|-------------|
| **Topic** | Topic name from OpenAlex taxonomy |
| **Count I (References)** | Weighted frequency in Level I |
| **Count II (Analyzed)** | Frequency in Level II |
| **Count III (Citing)** | Weighted frequency in Level III |
| **Norm I** | Normalized frequency in Level I (relative to total) |
| **Norm II** | Normalized frequency in Level II (relative to total) |
| **Norm III** | Normalized frequency in Level III (relative to total) |
| **Total Norm** | Sum of normalized frequencies across all levels |
| **First Year** | Earliest publication year for this topic |
| **Peak Year** | Year with most publications on this topic |

**Top Cited Categories:**

| Category | Description | Examples |
|----------|-------------|----------|
| **Topics** | High-level research areas | Materials Science, Chemistry, Physics |
| **Subtopics** | Specific research subfields | Nanomaterials, Catalysis, Polymers |
| **Fields** | Academic disciplines | Chemistry, Engineering, Biology |
| **Domains** | Broad scientific domains | Physical Sciences, Life Sciences |
| **Concepts** | Specific research concepts | Graphene, Nanoparticles, Doping |

**Interpretation:** High values in Level I indicate foundational concepts; high values in Level III indicate current research trends.

---

### 7. 📋 Detailed Citations

**Purpose:** Complete citation list with full bibliographic information.

**Structure:** Each analyzed paper is listed as a collapsible section containing:

**Paper Information:**
- Title (linked to DOI)
- Publication year
- Total citation count
- DOI

**Citation Details for Each Citing Work:**

| Metric | Description |
|--------|-------------|
| **Citing Title** | Title of the citing work |
| **Citing Journal** | Journal where citing work appeared |
| **Citing Year** | Publication year of citing work |
| **Citing Date** | Exact publication date |
| **Citation Lag** | Days between analyzed paper and citing work publication |
| **Authors** | List of citing work authors |
| **Countries** | Countries of citing work authors |
| **Topics** | Topics associated with the citing work |
| **DOI** | Direct link to citing work |

**Interpretation:** Shorter citation lags indicate rapid impact; patterns of citing countries reveal international reach.

---

### 8. 🔗 Multilevel Relationships

**Purpose:** Matrix-based analysis of entities appearing across all three levels.

**Author Matrix:**

| Metric | Description |
|--------|-------------|
| **Author Name** | Name of the author |
| **ORCID** | Author's ORCID identifier |
| **Count I** | Frequency in Level I (weighted) |
| **Count II** | Frequency in Level II |
| **Count III** | Frequency in Level III (weighted) |
| **Norm I/II/III** | Normalized frequencies across levels |
| **Total Norm** | Combined normalized presence across all levels |

**Affiliation Matrix:**

| Metric | Description |
|--------|-------------|
| **Affiliation** | Institution name |
| **Count I/II/III** | Frequency across levels |
| **Norm I/II/III** | Normalized frequencies |
| **Total Norm** | Combined normalized presence |

**Journal Matrix:**

| Metric | Description |
|--------|-------------|
| **Journal** | Publication venue |
| **Count I/II/III** | Frequency across levels |
| **Norm I/II/III** | Normalized frequencies |
| **Total Norm** | Combined normalized presence |

**Publisher Matrix:**

| Metric | Description |
|--------|-------------|
| **Publisher** | Publishing organization |
| **Count I/II/III** | Frequency across levels |
| **Norm I/II/III** | Normalized frequencies |
| **Total Norm** | Combined normalized presence |

**Interpretation:** Entities with high Total Norm are central to the citation network. Patterns across levels reveal:
- Authors who are active in all three levels (highly connected researchers)
- Affiliations that contribute across the citation chain
- Journals that serve as both sources and destinations
- Publishers that dominate the research ecosystem

---

### 9. 🔤 Title Keywords Analysis

**Purpose:** Extraction and normalization of key terms from titles across all levels.

**Process:**
1. **Lemmatization:** Reduces words to base form (e.g., "analyses" → "analysis")
2. **Compound Detection:** Identifies hyphenated scientific terms
3. **Stopword Filtering:** Removes common scientific terms
4. **Variant Tracking:** Groups different forms of the same word

**Keywords Table:**

| Metric | Description |
|--------|-------------|
| **Title Term (Lemma)** | Base form of the extracted keyword |
| **Variants** | All forms of this keyword found in titles |
| **Type** | Content, Scientific, or Compound |
| **Count I (Ref)** | Frequency in Level I titles |
| **Count II (Analyzed)** | Frequency in Level II titles |
| **Count III (Citing)** | Frequency in Level III titles |
| **Norm I/II/III** | Normalized frequencies across levels |
| **Total Norm** | Combined normalized presence |

**Interpretation:**
- **Content words:** Core research concepts
- **Scientific terms:** Technical vocabulary specific to the field
- **Compound words:** Multi-word technical terms
- High frequency across all levels indicates field-defining concepts
- Level-specific terms reveal research evolution

**Distribution Across Levels:**
- **Level I:** Foundational concepts and established terminology
- **Level II:** Current research focus and specific methodologies
- **Level III:** Emerging trends and new applications

---

### 10. ⏰ Temporal Relationships

**Purpose:** Analysis of time dynamics between citation levels.

**Reference → Analyzed Connections:**

| Metric | Description |
|--------|-------------|
| **Reference DOI** | Cited paper identifier |
| **Reference Date** | Publication date of reference paper |
| **Analyzed DOI** | Citing paper identifier |
| **Analyzed Date** | Publication date of analyzed paper |
| **Time Lag (days)** | Days between reference and analyzed publication |
| **Title (Ref)** | Title of the reference paper |
| **Title (Analyzed)** | Title of the analyzed paper |

**Analyzed → Citing Connections:**

| Metric | Description |
|--------|-------------|
| **Analyzed DOI** | Cited paper identifier |
| **Analyzed Date** | Publication date of analyzed paper |
| **Citing DOI** | Citing paper identifier |
| **Citing Date** | Publication date of citing work |
| **Time Lag (days)** | Days between analyzed and citing publication |
| **Title (Analyzed)** | Title of the analyzed paper |
| **Title (Citing)** | Title of the citing work |

**Temporal Lag Statistics:**

| Metric | Description |
|--------|-------------|
| **Min Lag (days)** | Shortest time between connected papers |
| **Max Lag (days)** | Longest time between connected papers |
| **Avg Lag (days)** | Average time between connected papers |
| **Median Lag (days)** | Median time between connected papers |
| **Total Connections** | Number of temporal connections |

**Interpretation:**
- **Reference → Analyzed:** How quickly researchers build upon prior work
  - Short lags: Fast-paced, rapidly evolving fields
  - Long lags: Papers that become relevant years later

- **Analyzed → Citing:** How quickly your research influences others
  - Short lags: Immediate impact, hot topics
  - Long lags: Steady influence over time

- **Patterns in lag distribution:**
  - Clustering around specific values may indicate field conventions
  - Outliers may indicate groundbreaking work that took time to recognize

---

## 📈 Example Output

### Sample Metrics

```
Level I (References):
- Total Items: 1,247
- Total Weighted: 3,891
- Total Citations: 45,829
- Unique Authors: 8,342
- H-index: 67
- Open Access: 43.2%

Level II (Analyzed):
- Total Items: 15
- Total Citations: 1,234
- Avg Citations: 82.3
- International Collaboration: 73.3%
- H-index: 12

Level III (Citing Works):
- Total Items: 1,893
- Total Weighted: 4,567
- Unique Authors: 12,456
- Open Access: 51.8%
```

### Sample Visualizations

The HTML report includes:
- **Publication activity dynamics** – Yearly publication trends
- **Citation heatmap** – Publication-to-citation year matrix
- **Author network** – Top collaborating authors
- **Geographic distribution** – Country-level collaboration maps
- **Topic clustering** – Thematic structure across levels

---

## 🔍 Advanced Features

### Title Keyword Extraction

The tool uses advanced NLP to extract meaningful keywords from titles:

- **Lemmatization** using NLTK WordNet
- **Irregular plural handling** (e.g., "analyses" → "analysis")
- **Compound word detection** (e.g., "electrochemical-impedance-spectroscopy")
- **Scientific stopword filtering** (removes common scientific terms)
- **Variant tracking** (normalizes different forms of the same word)

### Temporal Relationship Analysis

Understanding the time dynamics of citations:

- **Reference → Analyzed:** Time lag between cited and citing papers
- **Analyzed → Citing:** Time lag between analyzed and citing papers
- **First citation lag analysis:** Speed of research impact
- **Cumulative citation tracking:** Citation growth over time

### Multilevel Matrices

Visualizing relationships across all three levels:

- **Author Matrix:** How authors appear across Level I, II, and III
- **Affiliation Matrix:** Institutional presence across levels
- **Journal Matrix:** Publication venues across levels
- **Publisher Matrix:** Publishing patterns across levels

---

## 🛠️ Troubleshooting

### Common Issues

**1. "No DOIs found" error**
- Ensure DOIs are valid and correctly formatted
- Check for typos or extra spaces
- Verify DOIs exist in OpenAlex

**2. Slow API responses**
- Enable caching to reduce API calls
- Use smaller batch sizes
- Reduce concurrent requests

**3. Icons not displaying in HTML report**
- Ensure `icons/` folder contains all PNG files
- Check file permissions
- Verify the report is opened in a modern browser

**4. NLTK errors**
- Run the NLTK download command in installation steps
- Ensure NLTK data is installed in the correct location

**5. Memory issues with large datasets**
- Reduce the number of DOIs analyzed in one session
- Clear cache regularly

---

## 📚 Dependencies

### Core Dependencies
```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
requests>=2.31.0
aiohttp>=3.8.0
tenacity>=8.0.0
```

### Visualization Dependencies
```
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0
wordcloud>=1.9.0
```

### NLP Dependencies
```
nltk>=3.8.0
```

### Web & Utilities
```
beautifulsoup4>=4.12.0
lxml>=4.9.0
tqdm>=4.65.0
python-dotenv>=1.0.0
```

---

## 📝 License

This project is licensed under the MIT License – see the LICENSE file for details.

---

## 📧 Contact

**Author:** daM  
**Institution:** Chimica Techno Acta  
**Journal:** [chimicatechnoacta.ru](https://chimicatechnoacta.ru)  

---

## 📖 Citation

If you use this tool in your research, please cite:

```
daM. Ref-Cit-Analysis
Chimica Techno Acta. https://chimicatechnoacta.ru
```

---

**Made with ❤️ for the scientific community.**
