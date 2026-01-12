📚 Scientific Article Analyzer by DOI
Streamlit application for comprehensive analysis of scientific articles using DOI.

🚀 Quick Start
Enter DOI in text area (multiple DOI supported, separated by commas/semicolons)

Configure settings in sidebar:

Parallel threads (1-10)

Enable ROR analysis for organization data

Click "Process DOI" to start analysis

Download Excel report when complete

📋 Features
🔄 Smart Caching
Multi-level caching (memory + file-based)

24-hour cache expiration

Resume capability from interruptions

Cache statistics monitoring

📊 Multi-Level Analysis
Primary DOI: User-provided articles

Reference DOI: Articles cited by primary (up to 10k)

Citation DOI: Articles citing primary (up to 10k)

Failed DOI: Automatic retry system

⚡ Performance
Parallel processing (1-10 threads)

Adaptive request timing

Batch processing (50 DOI/batch)

Automatic rate limit handling

📑 Comprehensive Excel Export (25+ tabs)
Article Analysis:

Article_Analyzed: Primary article details

Article_ref: Reference articles

Article_citing: Citing articles

Frequency Statistics:

Author, Journal, Affiliation, Country frequencies

Citation metrics and analysis

Summary & Connections:

Author and Affiliation summaries

Temporal relationship analysis

Failed DOI tracking

Content Analysis:

Title keyword analysis with lemmatization

Terms and Topics analysis

⚙️ Technical Details
Supported APIs
Crossref API for basic metadata

OpenAlex API for citations and concepts

ROR API for organization data (optional)

Data Collection
Parallel API requests with retry logic

Data normalization (authors, countries, ORCIDs)

Relationship mapping (citations/references)

Smart error handling and recovery

Excel Report Features
Normalized statistics across sources

ROR organization integration

Lemmatized keyword extraction

Temporal publication analysis

Comprehensive metadata tracking

🔧 Installation
bash
# Install required packages
pip install streamlit pandas requests numpy networkx scikit-learn fuzzywuzzy openpyxl nltk joblib tqdm

# Run application
streamlit run app.py
📊 Key Metrics in Report
Citation velocity: Citations per year

Geographic diversity: Countries per author

Journal impact: Citation statistics per journal

Author productivity: Publication frequency

Concept coverage: Topic distribution

💡 Usage Tips
For Best Results
Start with 2-4 parallel threads

Enable ROR analysis for better organization data

Process 50-200 DOI at a time for optimal performance

Use resume feature for large analyses

Data Sources
Title Keywords analysis: https://rca-title-keywords.streamlit.app/

Terms and Concepts analysis: https://rca-terms-concepts.streamlit.app/

Article analysis: https://rca-analysis.streamlit.app/

🎯 Quick Reference
Component	Purpose
Cache	Reduces API calls, enables resume
Parallel Processing	Speeds up large analyses
Excel Export	Complete analysis in one file
ROR Integration	Organization identification
Title Analysis	Keyword extraction and lemmatization
📈 Performance Notes
Typical cache hit ratio: 60-80%

Processing speed: ~10-50 DOI/minute

Memory usage: ~50-200MB depending on cache size

File output: 1-10MB per 100 DOI analyzed

🔄 Workflow
Input: DOI list (10.xxx/xxx format)

Processing: Parallel API calls, data extraction

Analysis: Statistics, relationships, keywords

Output: Excel file with 25+ analysis tabs

Options: Resume, cache clear, ROR lookup

developed by @daM, @CTA, https://chimicatechnoacta.ru
