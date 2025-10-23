import streamlit as st
import pandas as pd
import requests
import time
from typing import List, Dict, Any
import re
from collections import Counter
import tempfile
import os

# Настройки
st.set_page_config(
    page_title="Refs/Cits Analysis",
    page_icon="📚",
    layout="wide"
)

class SimpleCitationAnalyzer:
    def __init__(self):
        self.request_delay = 0.5
        self.max_dois = 50
        
    def validate_doi(self, doi: str) -> bool:
        """Проверяет валидность DOI"""
        if not doi or not isinstance(doi, str):
            return False
            
        doi = self.normalize_doi(doi)
        doi_pattern = r'^10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+$'
        
        return bool(re.match(doi_pattern, doi, re.IGNORECASE))
    
    def normalize_doi(self, doi: str) -> str:
        """Нормализует DOI"""
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
    
    def parse_doi_input(self, input_text: str) -> List[str]:
        """Парсит ввод DOI"""
        if not input_text or not isinstance(input_text, str):
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
        
        # Очистка и удаление дубликатов
        cleaned_dois = []
        seen = set()
        for doi in dois:
            normalized_doi = self.normalize_doi(doi)
            if self.validate_doi(normalized_doi) and normalized_doi not in seen:
                seen.add(normalized_doi)
                cleaned_dois.append(normalized_doi)
                
        return cleaned_dois[:self.max_dois]
    
    def get_crossref_data(self, doi: str) -> Dict:
        """Получает данные из Crossref"""
        try:
            url = f"https://api.crossref.org/works/{doi}"
            headers = {'User-Agent': 'Streamlit-App/1.0'}
            
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 200:
                data = response.json()
                return data.get('message', {})
        except Exception as e:
            st.warning(f"Error fetching Crossref data for {doi}: {e}")
            
        return {}
    
    def get_openalex_data(self, doi: str) -> Dict:
        """Получает данные из OpenAlex"""
        try:
            url = f"https://api.openalex.org/works/https://doi.org/{doi}"
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            st.warning(f"Error fetching OpenAlex data for {doi}: {e}")
            
        return {}
    
    def extract_authors(self, crossref_data: Dict, openalex_data: Dict) -> str:
        """Извлекает авторов"""
        authors = []
        
        # Сначала пробуем OpenAlex
        if openalex_data and 'authorships' in openalex_data:
            for authorship in openalex_data['authorships']:
                author_name = authorship.get('author', {}).get('display_name', '')
                if author_name:
                    authors.append(author_name)
        
        # Если нет авторов из OpenAlex, пробуем Crossref
        if not authors and crossref_data and 'author' in crossref_data:
            for author in crossref_data['author']:
                given = author.get('given', '')
                family = author.get('family', '')
                if given or family:
                    authors.append(f"{given} {family}".strip())
        
        return ', '.join(authors) if authors else 'Unknown'
    
    def extract_journal_info(self, crossref_data: Dict) -> Dict:
        """Извлекает информацию о журнале"""
        try:
            container_title = crossref_data.get('container-title', [])
            short_container_title = crossref_data.get('short-container-title', [])
            
            full_name = container_title[0] if container_title else (
                short_container_title[0] if short_container_title else 'Unknown'
            )
            abbreviation = short_container_title[0] if short_container_title else (
                container_title[0] if container_title else 'Unknown'
            )
            
            return {
                'full_name': full_name,
                'abbreviation': abbreviation,
                'publisher': crossref_data.get('publisher', 'Unknown')
            }
        except:
            return {
                'full_name': 'Unknown',
                'abbreviation': 'Unknown',
                'publisher': 'Unknown'
            }
    
    def extract_year(self, crossref_data: Dict, openalex_data: Dict) -> str:
        """Извлекает год публикации"""
        # Пробуем OpenAlex
        if openalex_data and openalex_data.get('publication_year'):
            return str(openalex_data['publication_year'])
        
        # Пробуем Crossref
        for key in ['published-print', 'published-online', 'issued']:
            if key in crossref_data and 'date-parts' in crossref_data[key]:
                date_parts = crossref_data[key]['date-parts'][0]
                if date_parts and len(date_parts) > 0:
                    return str(date_parts[0])
        
        return 'Unknown'
    
    def get_article_data(self, doi: str) -> Dict[str, Any]:
        """Получает полные данные о статье"""
        with st.spinner(f"Fetching data for {doi}..."):
            crossref_data = self.get_crossref_data(doi)
            time.sleep(self.request_delay)
            
            openalex_data = self.get_openalex_data(doi)
            time.sleep(self.request_delay)
            
            title = 'Unknown'
            if openalex_data and openalex_data.get('title'):
                title = openalex_data['title']
            elif crossref_data.get('title'):
                title_list = crossref_data['title']
                if title_list:
                    title = title_list[0]
            
            authors = self.extract_authors(crossref_data, openalex_data)
            journal_info = self.extract_journal_info(crossref_data)
            year = self.extract_year(crossref_data, openalex_data)
            
            # Цитирования
            citation_count_crossref = crossref_data.get('is-referenced-by-count', 0)
            citation_count_openalex = openalex_data.get('cited_by_count', 0)
            
            return {
                'doi': doi,
                'title': title,
                'authors': authors,
                'year': year,
                'journal_full_name': journal_info['full_name'],
                'journal_abbreviation': journal_info['abbreviation'],
                'publisher': journal_info['publisher'],
                'citation_count_crossref': citation_count_crossref,
                'citation_count_openalex': citation_count_openalex
            }
    
    def get_references(self, doi: str) -> List[Dict]:
        """Получает ссылки на статьи"""
        try:
            crossref_data = self.get_crossref_data(doi)
            return crossref_data.get('reference', [])
        except:
            return []
    
    def get_citing_articles(self, doi: str) -> List[str]:
        """Получает цитирующие статьи"""
        citing_dois = []
        try:
            # Crossref
            crossref_data = self.get_crossref_data(doi)
            if 'is-referenced-by' in crossref_data:
                for ref in crossref_data['is-referenced-by']:
                    if isinstance(ref, dict) and 'DOI' in ref:
                        citing_dois.append(ref['DOI'])
            
            # OpenAlex
            openalex_data = self.get_openalex_data(doi)
            if openalex_data and 'cited_by_count' in openalex_data and openalex_data['cited_by_count'] > 0:
                work_id = openalex_data.get('id', '').split('/')[-1]
                if work_id:
                    citing_url = f"https://api.openalex.org/works?filter=cites:{work_id}&per-page=25"
                    response = requests.get(citing_url, timeout=30)
                    if response.status_code == 200:
                        citing_data = response.json()
                        for work in citing_data.get('results', []):
                            if work.get('doi'):
                                citing_dois.append(work['doi'])
            
        except Exception as e:
            st.warning(f"Error getting citing articles for {doi}: {e}")
            
        return list(set(citing_dois))
    
    def analyze_references(self, doi_list: List[str]) -> pd.DataFrame:
        """Анализирует ссылки на статьи"""
        all_references_data = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, doi in enumerate(doi_list):
            status_text.text(f"Processing source article {i+1}/{len(doi_list)}: {doi}")
            
            # Данные исходной статьи
            source_data = self.get_article_data(doi)
            source_data['type'] = 'source'
            source_data['source_doi'] = doi
            all_references_data.append(source_data)
            
            # Ссылки
            references = self.get_references(doi)
            for j, ref in enumerate(references[:10]):  # Ограничиваем для скорости
                ref_doi = ref.get('DOI')
                if ref_doi and self.validate_doi(ref_doi):
                    ref_data = self.get_article_data(ref_doi)
                    ref_data['type'] = 'reference'
                    ref_data['source_doi'] = doi
                    ref_data['position'] = j + 1
                    all_references_data.append(ref_data)
                    time.sleep(self.request_delay)
            
            progress_bar.progress((i + 1) / len(doi_list))
        
        status_text.text("Analysis complete!")
        return pd.DataFrame(all_references_data)
    
    def analyze_citations(self, doi_list: List[str]) -> pd.DataFrame:
        """Анализирует цитирующие статьи"""
        all_citations_data = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, doi in enumerate(doi_list):
            status_text.text(f"Processing source article {i+1}/{len(doi_list)}: {doi}")
            
            # Данные исходной статьи
            source_data = self.get_article_data(doi)
            source_data['type'] = 'source'
            source_data['source_doi'] = doi
            all_citations_data.append(source_data)
            
            # Цитирующие статьи
            citing_dois = self.get_citing_articles(doi)
            for citing_doi in citing_dois[:10]:  # Ограничиваем для скорости
                if self.validate_doi(citing_doi):
                    citing_data = self.get_article_data(citing_doi)
                    citing_data['type'] = 'citation'
                    citing_data['source_doi'] = doi
                    all_citations_data.append(citing_data)
                    time.sleep(self.request_delay)
            
            progress_bar.progress((i + 1) / len(doi_list))
        
        status_text.text("Analysis complete!")
        return pd.DataFrame(all_citations_data)

def main():
    st.title("📚 Refs/Cits Analysis")
    st.markdown("Анализ ссылок и цитирований научных статей")
    
    analyzer = SimpleCitationAnalyzer()
    
    # Боковая панель с настройками
    with st.sidebar:
        st.header("Настройки")
        st.info("""
        Введите DOI статей для анализа:
        - 10.1234/abcd.1234
        - https://doi.org/10.1234/abcd.1234  
        - doi:10.1234/abcd.1234
        """)
        
        max_dois = st.slider("Максимум DOI для анализа", 5, 50, 10)
        analyzer.max_dois = max_dois
        
        st.markdown("---")
        st.markdown("### Информация")
        st.markdown("""
        Этот инструмент анализирует:
        - **References**: Ссылки на статьи
        - **Citations**: Цитирующие статьи
        """)
    
    # Основной контент
    tab1, tab2, tab3 = st.tabs(["Ввод данных", "Анализ ссылок", "Анализ цитирований"])
    
    with tab1:
        st.header("Ввод DOI статей")
        
        input_method = st.radio("Способ ввода:", ["Текст", "Файл"])
        
        if input_method == "Текст":
            doi_input = st.text_area(
                "Введите DOI статей:",
                height=150,
                placeholder="Введите DOI через запятую, точку с запятой или с новой строки:\n10.1038/s41586-023-06924-6\n10.1126/science.abl8921"
            )
            
            if st.button("Проверить DOI", key="validate"):
                if doi_input:
                    dois = analyzer.parse_doi_input(doi_input)
                    if dois:
                        st.success(f"Найдено {len(dois)} валидных DOI:")
                        for doi in dois:
                            st.write(f"- {doi}")
                    else:
                        st.error("Не найдено валидных DOI. Проверьте формат.")
        
        else:
            uploaded_file = st.file_uploader("Загрузите файл с DOI", type=['txt', 'csv'])
            if uploaded_file:
                content = uploaded_file.getvalue().decode()
                dois = analyzer.parse_doi_input(content)
                if dois:
                    st.success(f"Найдено {len(dois)} валидных DOI в файле")
                    st.write("Обнаруженные DOI:")
                    for doi in dois[:10]:  # Показываем первые 10
                        st.write(f"- {doi}")
                    if len(dois) > 10:
                        st.write(f"... и еще {len(dois) - 10} DOI")
    
    with tab2:
        st.header("Анализ ссылок (References)")
        
        if 'doi_input' in locals() and doi_input:
            dois = analyzer.parse_doi_input(doi_input)
            
            if st.button("Запустить анализ ссылок", type="primary"):
                if not dois:
                    st.error("Сначала введите валидные DOI на вкладке 'Ввод данных'")
                else:
                    with st.spinner("Выполняется анализ ссылок..."):
                        references_df = analyzer.analyze_references(dois)
                    
                    if not references_df.empty:
                        st.success("Анализ завершен!")
                        
                        # Основная статистика
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            total_refs = len(references_df[references_df['type'] == 'reference'])
                            st.metric("Всего ссылок", total_refs)
                        with col2:
                            unique_journals = references_df['journal_abbreviation'].nunique()
                            st.metric("Уникальных журналов", unique_journals)
                        with col3:
                            avg_citations = references_df['citation_count_openalex'].mean()
                            st.metric("Средние цитирования", f"{avg_citations:.1f}")
                        
                        # Данные
                        st.subheader("Данные ссылок")
                        st.dataframe(references_df)
                        
                        # Экспорт
                        csv = references_df.to_csv(index=False)
                        st.download_button(
                            "Скачать CSV",
                            csv,
                            "references_analysis.csv",
                            "text/csv"
                        )
                    else:
                        st.error("Не удалось получить данные")
    
    with tab3:
        st.header("Анализ цитирований (Citations)")
        
        if 'doi_input' in locals() and doi_input:
            dois = analyzer.parse_doi_input(doi_input)
            
            if st.button("Запустить анализ цитирований", type="primary"):
                if not dois:
                    st.error("Сначала введите валидные DOI на вкладке 'Ввод данных'")
                else:
                    with st.spinner("Выполняется анализ цитирований..."):
                        citations_df = analyzer.analyze_citations(dois)
                    
                    if not citations_df.empty:
                        st.success("Анализ завершен!")
                        
                        # Основная статистика
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            total_cits = len(citations_df[citations_df['type'] == 'citation'])
                            st.metric("Всего цитирований", total_cits)
                        with col2:
                            citing_journals = citations_df[citations_df['type'] == 'citation']['journal_abbreviation'].nunique()
                            st.metric("Цитирующих журналов", citing_journals)
                        with col3:
                            recent_citations = len(citations_df[
                                (citations_df['type'] == 'citation') & 
                                (citations_df['year'] == '2024')
                            ])
                            st.metric("Цитирований 2024", recent_citations)
                        
                        # Данные
                        st.subheader("Данные цитирований")
                        st.dataframe(citations_df)
                        
                        # Экспорт
                        csv = citations_df.to_csv(index=False)
                        st.download_button(
                            "Скачать CSV",
                            csv,
                            "citations_analysis.csv",
                            "text/csv"
                        )
                    else:
                        st.error("Не удалось получить данные")

if __name__ == "__main__":
    main()