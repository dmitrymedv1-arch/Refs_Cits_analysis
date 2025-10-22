import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import time
from datetime import datetime
from collections import Counter
import re
import os
import tempfile
import nltk
from nltk.corpus import stopwords

# Настройка страницы Streamlit
st.set_page_config(
    page_title="Citation Analyzer",
    page_icon="📚",
    layout="wide"
)

# Download NLTK data if needed
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class Config:
    REQUEST_TIMEOUT = 30
    MAX_RETRIES = 3
    DELAY_BETWEEN_REQUESTS = 0.5

class FastAffiliationProcessor:
    def __init__(self):
        self.common_keywords = {
            'university', 'college', 'institute', 'school', 'department', 'faculty',
            'laboratory', 'center', 'centre', 'academy', 'research'
        }
        self.organization_cache = {}

    def extract_main_organization_fast(self, affiliation: str) -> str:
        if not affiliation or affiliation in ['Unknown', 'Error', '']:
            return "Unknown"

        if affiliation in self.organization_cache:
            return self.organization_cache[affiliation]

        clean_affiliation = affiliation.strip()
        clean_affiliation = re.sub(r'\S+@\S+', '', clean_affiliation)
        clean_affiliation = re.sub(r'\d{5,}(?:-\d{4})?', '', clean_affiliation)

        parts = re.split(r'[,;]', clean_affiliation)
        main_org_candidates = []

        for part in parts:
            part = part.strip()
            if not part or len(part) < 5:
                continue

            part_lower = part.lower()
            has_org_keyword = any(keyword in part_lower for keyword in self.common_keywords)

            if has_org_keyword:
                main_org_candidates.append(part)

        if main_org_candidates:
            main_org_candidates.sort(key=len, reverse=True)
            main_org = main_org_candidates[0]
        else:
            for part in parts:
                part = part.strip()
                if len(part) > 10:
                    main_org = part
                    break
            else:
                main_org = clean_affiliation

        main_org = re.sub(r'\s+', ' ', main_org).strip()
        result = main_org if main_org else "Unknown"
        self.organization_cache[affiliation] = result
        return result

class CitationAnalyzer:
    def __init__(self):
        self.crossref_cache = {}
        self.openalex_cache = {}
        self.fast_affiliation_processor = FastAffiliationProcessor()
        self.stop_words = set(stopwords.words('english'))

    def validate_doi(self, doi: str) -> bool:
        if not doi or not isinstance(doi, str):
            return False

        doi = self.normalize_doi(doi)
        doi_pattern = r'^10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+$'

        if not bool(re.match(doi_pattern, doi, re.IGNORECASE)):
            return False

        return True

    def normalize_doi(self, doi: str) -> str:
        if not doi or not isinstance(doi, str):
            return ""

        doi = doi.strip()
        prefixes = [
            'https://doi.org/', 'http://doi.org/', 'doi.org/',
            'doi:', 'DOI:', 'https://dx.doi.org/', 'http://dx.doi.org/',
        ]

        for prefix in prefixes:
            if doi.lower().startswith(prefix.lower()):
                doi = doi[len(prefix):]
                break

        doi = doi.split('?')[0].split('#')[0]
        return doi.strip().lower()

    def parse_doi_input(self, input_text: str, max_dois: int = 50) -> List[str]:
        if not input_text or not isinstance(input_text, str):
            st.error("Error: Input is empty or not a string")
            return []

        lines = input_text.strip().split('\n')
        dois = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            line = line.rstrip('.,;')
            doi_pattern = r'10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+'
            found_dois = re.findall(doi_pattern, line, re.IGNORECASE)

            if found_dois:
                dois.extend(found_dois)
            else:
                if 'doi.org/' in line.lower():
                    doi_part = line.lower().split('doi.org/')[-1]
                    doi_part = doi_part.split('?')[0].split('#')[0].strip()
                    if self.validate_doi(doi_part):
                        dois.append(doi_part)
                elif self.validate_doi(line):
                    dois.append(line)

        cleaned_dois = []
        for doi in dois:
            normalized_doi = self.normalize_doi(doi)
            if self.validate_doi(normalized_doi):
                cleaned_dois.append(normalized_doi)

        unique_dois = []
        seen = set()
        for doi in cleaned_dois:
            if doi not in seen:
                seen.add(doi)
                unique_dois.append(doi)

        unique_dois = unique_dois[:max_dois]

        if not unique_dois:
            st.error("Error: No valid DOIs found in the input.")
            st.info("Valid examples: 10.1234/abcd.1234, https://doi.org/10.1234/abcd.1234")
        else:
            st.success(f"Found {len(unique_dois)} valid DOI(s)")

        return unique_dois

    def get_crossref_data(self, doi: str) -> Dict:
        if doi in self.crossref_cache:
            return self.crossref_cache[doi]
        
        try:
            url = f"https://api.crossref.org/works/{doi}"
            response = requests.get(url, timeout=Config.REQUEST_TIMEOUT)
            
            if response.status_code == 200:
                data = response.json()['message']
                
                # Extract year
                year = None
                for key in ['published-print', 'published-online', 'issued']:
                    if key in data and 'date-parts' in data[key]:
                        date_parts = data[key]['date-parts'][0]
                        year = date_parts[0] if date_parts else None
                        break
                data['publication_year'] = year if year else 'Unknown'
                
                self.crossref_cache[doi] = data
                return data
            else:
                return {'publication_year': 'Unknown', 'title': ['Unknown']}
                
        except Exception as e:
            return {'publication_year': 'Unknown', 'title': ['Unknown']}

    def get_openalex_data(self, doi: str) -> Dict:
        if doi in self.openalex_cache:
            return self.openalex_cache[doi]
        
        try:
            openalex_url = f"https://api.openalex.org/works/https://doi.org/{doi}"
            response = requests.get(openalex_url, timeout=Config.REQUEST_TIMEOUT)
            
            if response.status_code == 200:
                data = response.json()
                self.openalex_cache[doi] = data
                return data
            else:
                return {}
                
        except Exception:
            return {}

    def get_combined_article_data(self, doi: str) -> Dict[str, Any]:
        try:
            crossref_data = self.get_crossref_data(doi)
            openalex_data = self.get_openalex_data(doi)

            # Get title
            title = 'Unknown'
            if openalex_data and openalex_data.get('title'):
                title = openalex_data['title']
            elif crossref_data.get('title'):
                title_list = crossref_data['title']
                if title_list:
                    title = title_list[0]

            # Get year
            year = 'Unknown'
            if openalex_data and openalex_data.get('publication_year'):
                year = str(openalex_data['publication_year'])
            elif crossref_data.get('publication_year') != 'Unknown':
                year = str(crossref_data['publication_year'])

            # Get authors
            authors = []
            if openalex_data:
                for author in openalex_data.get('authorships', []):
                    name = author.get('author', {}).get('display_name', 'Unknown')
                    if name != 'Unknown':
                        authors.append(name)

            if not authors and crossref_data.get('author'):
                for author in crossref_data['author']:
                    given = author.get('given', '')
                    family = author.get('family', '')
                    if given or family:
                        name = f"{given} {family}".strip()
                        authors.append(name)

            authors_str = ', '.join(authors) if authors else 'Unknown'
            author_count = len(authors) if authors else 0

            # Get journal info
            journal_info = self.get_journal_info_from_crossref(doi)

            # Get citations
            citation_count = crossref_data.get('is-referenced-by-count', 0)

            return {
                'doi': doi,
                'title': title,
                'year': year,
                'authors': authors_str,
                'author_count': author_count,
                'journal': journal_info['full_name'],
                'publisher': journal_info['publisher'],
                'citation_count': citation_count,
            }
            
        except Exception as e:
            return {
                'doi': doi,
                'title': 'Error',
                'year': 'Unknown',
                'authors': 'Error',
                'author_count': 0,
                'journal': 'Error',
                'publisher': 'Error',
                'citation_count': 0,
                'error': str(e)
            }

    def get_journal_info_from_crossref(self, doi: str) -> Dict[str, Any]:
        try:
            data = self.get_crossref_data(doi)
            container_title = data.get('container-title', [])
            full_name = container_title[0] if container_title else 'Unknown'
            return {
                'full_name': full_name,
                'publisher': data.get('publisher', 'Unknown'),
            }
        except:
            return {
                'full_name': 'Unknown',
                'publisher': 'Unknown',
            }

    def process_references_analysis(self, doi_list: List[str]) -> pd.DataFrame:
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, doi in enumerate(doi_list):
            status_text.text(f"Processing article {i+1}/{len(doi_list)}: {doi}")
            
            article_data = self.get_combined_article_data(doi)
            results.append(article_data)
            
            time.sleep(Config.DELAY_BETWEEN_REQUESTS)
            progress_bar.progress((i + 1) / len(doi_list))
        
        status_text.empty()
        progress_bar.empty()
        
        return pd.DataFrame(results)

    def analyze_titles(self, titles: List[str]) -> pd.DataFrame:
        content_words = []
        valid_titles = [t for t in titles if t not in ['Unknown', 'Error']]
        
        for title in valid_titles:
            if not title:
                continue
                
            text = title.lower()
            text = re.sub(r'[^a-zA-Z\s-]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            words = text.split()
            
            for word in words:
                if len(word) > 3 and word not in self.stop_words:
                    content_words.append(word)
        
        word_freq = Counter(content_words)
        return pd.DataFrame(word_freq.most_common(20), columns=['Word', 'Frequency'])

def main():
    st.title("📚 Simple Citation Analyzer")
    st.markdown("Analyze basic information for scientific articles using DOI identifiers")
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = CitationAnalyzer()
    
    analyzer = st.session_state.analyzer
    
    st.header("Basic Article Analysis")
    
    doi_input = st.text_area(
        "Enter DOIs for analysis",
        value="10.1038/s41586-023-06924-6\n10.1126/science.abl4471",
        placeholder="Enter one or more DOIs separated by new lines",
        height=100
    )
    
    if st.button("Analyze Articles", type="primary"):
        if doi_input:
            with st.spinner("Parsing DOIs..."):
                doi_list = analyzer.parse_doi_input(doi_input)
            
            if doi_list:
                st.success(f"Found {len(doi_list)} valid DOI(s)")
                
                with st.spinner("Processing articles..."):
                    try:
                        results_df = analyzer.process_references_analysis(doi_list)
                        
                        # Display results
                        st.header("Analysis Results")
                        
                        # Basic metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Articles Processed", len(results_df))
                        with col2:
                            avg_citations = results_df['citation_count'].mean()
                            st.metric("Avg Citations", f"{avg_citations:.1f}")
                        with col3:
                            success_rate = len(results_df[results_df['title'] != 'Error']) / len(results_df) * 100
                            st.metric("Success Rate", f"{success_rate:.1f}%")
                        
                        # Display data
                        st.subheader("Article Data")
                        display_cols = ['doi', 'title', 'authors', 'year', 'journal', 'citation_count']
                        available_cols = [col for col in display_cols if col in results_df.columns]
                        st.dataframe(results_df[available_cols])
                        
                        # Title word analysis
                        st.subheader("Title Word Frequency")
                        word_freq_df = analyzer.analyze_titles(results_df['title'].tolist())
                        st.dataframe(word_freq_df)
                        
                        # Download option
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name="citation_analysis_results.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"Error during processing: {str(e)}")
            else:
                st.error("No valid DOIs found. Please check the input format.")
        else:
            st.error("Please enter at least one DOI.")

if __name__ == "__main__":
    main()
