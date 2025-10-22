import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from tqdm import tqdm
import time
from datetime import datetime
import re
import json
import io
import os
import sys

# Добавляем текущую директорию в путь для импорта config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Настройка страницы Streamlit - все настройки теперь в коде, а не в config.toml
st.set_page_config(
    page_title="Advanced Citation Analyzer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

class CitationAnalyzer:
    def __init__(self):
        self.cache = {}
    
    def validate_doi(self, doi):
        """Проверяет валидность DOI"""
        if not doi or not isinstance(doi, str):
            return False
            
        doi = self.normalize_doi(doi)
        pattern = r'^10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+$'
        return bool(re.match(pattern, doi))
    
    def normalize_doi(self, doi):
        """Нормализует DOI"""
        if not doi:
            return ""
            
        doi = doi.strip()
        
        # Удаляем префиксы
        prefixes = [
            'https://doi.org/', 'http://doi.org/', 'doi.org/',
            'doi:', 'DOI:', 'https://dx.doi.org/', 'http://dx.doi.org/'
        ]
        
        for prefix in prefixes:
            if doi.lower().startswith(prefix.lower()):
                doi = doi[len(prefix):]
                break
                
        doi = doi.split('?')[0].split('#')[0]
        return doi.strip().lower()
    
    def parse_doi_input(self, input_text, max_dois=20):
        """Парсит ввод с DOI"""
        if not input_text:
            st.error("Please enter at least one DOI")
            return []
            
        lines = input_text.strip().split('\n')
        dois = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Ищем DOI в строке
            doi_pattern = r'10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+'
            matches = re.findall(doi_pattern, line, re.IGNORECASE)
            
            if matches:
                dois.extend(matches)
            elif self.validate_doi(line):
                dois.append(self.normalize_doi(line))
        
        # Убираем дубликаты и ограничиваем количество
        unique_dois = []
        seen = set()
        for doi in dois:
            normalized = self.normalize_doi(doi)
            if normalized not in seen and self.validate_doi(normalized):
                seen.add(normalized)
                unique_dois.append(normalized)
                
        unique_dois = unique_dois[:max_dois]
        
        if not unique_dois:
            st.error("No valid DOIs found. Examples: 10.1038/s41586-023-06924-6")
        else:
            st.success(f"Found {len(unique_dois)} valid DOI(s)")
            
        return unique_dois
    
    def get_article_data(self, doi):
        """Получает данные статьи из Crossref API"""
        if doi in self.cache:
            return self.cache[doi]
            
        try:
            url = f"https://api.crossref.org/works/{doi}"
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()['message']
                
                # Извлекаем основную информацию
                title = data.get('title', ['Unknown'])[0] if data.get('title') else 'Unknown'
                
                # Извлекаем год публикации
                year = 'Unknown'
                for key in ['published-print', 'published-online', 'issued']:
                    if key in data and data[key].get('date-parts'):
                        date_parts = data[key]['date-parts'][0]
                        if date_parts and len(date_parts) > 0:
                            year = str(date_parts[0])
                            break
                
                # Извлекаем авторов
                authors = []
                if data.get('author'):
                    for author in data['author']:
                        given = author.get('given', '')
                        family = author.get('family', '')
                        if given or family:
                            authors.append(f"{given} {family}".strip())
                
                authors_str = ', '.join(authors) if authors else 'Unknown'
                
                # Извлекаем журнал
                journal = data.get('container-title', ['Unknown'])[0] if data.get('container-title') else 'Unknown'
                
                # Извлекаем издателя
                publisher = data.get('publisher', 'Unknown')
                
                # Количество цитирований
                citation_count = data.get('is-referenced-by-count', 0)
                
                result = {
                    'doi': doi,
                    'title': title,
                    'year': year,
                    'authors': authors_str,
                    'journal': journal,
                    'publisher': publisher,
                    'citation_count': citation_count,
                    'author_count': len(authors),
                    'error': None
                }
                
                self.cache[doi] = result
                return result
                
            else:
                result = {
                    'doi': doi,
                    'title': 'Error',
                    'year': 'Unknown',
                    'authors': 'Error',
                    'journal': 'Error',
                    'publisher': 'Error',
                    'citation_count': 0,
                    'author_count': 0,
                    'error': f"HTTP {response.status_code}"
                }
                self.cache[doi] = result
                return result
                
        except Exception as e:
            result = {
                'doi': doi,
                'title': 'Error',
                'year': 'Unknown',
                'authors': 'Error',
                'journal': 'Error',
                'publisher': 'Error',
                'citation_count': 0,
                'author_count': 0,
                'error': str(e)
            }
            self.cache[doi] = result
            return result
    
    def analyze_articles(self, doi_list):
        """Анализирует список статей"""
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, doi in enumerate(doi_list):
            status_text.text(f"Processing {i+1}/{len(doi_list)}: {doi}")
            result = self.get_article_data(doi)
            results.append(result)
            
            # Задержка чтобы не перегружать API
            time.sleep(0.5)
            
            progress_bar.progress((i + 1) / len(doi_list))
        
        status_text.empty()
        progress_bar.empty()
        
        return results

def create_visualizations(df):
    """Создает визуализации для данных"""
    
    # Только успешные записи
    success_df = df[df['title'] != 'Error'].copy()
    
    if len(success_df) == 0:
        st.warning("No successful data to visualize")
        return
    
    # Преобразуем год в числовой формат
    success_df['year_num'] = pd.to_numeric(success_df['year'], errors='coerce')
    success_df = success_df.dropna(subset=['year_num'])
    
    if len(success_df) == 0:
        st.warning("No valid year data for visualization")
        return
    
    # Создаем вкладки для визуализаций
    tab1, tab2, tab3 = st.tabs(["📊 Citations Analysis", "📅 Year Distribution", "👥 Authors Analysis"])
    
    with tab1:
        # Анализ цитирований
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(success_df, x='citation_count', 
                              title='Distribution of Citation Counts',
                              labels={'citation_count': 'Citation Count'},
                              nbins=20,
                              color_discrete_sequence=['#1E88E5'])
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if len(success_df) > 1:
                fig = px.scatter(success_df, x='year_num', y='citation_count',
                                hover_data=['title'],
                                title='Citations vs Publication Year',
                                labels={'year_num': 'Publication Year', 'citation_count': 'Citation Count'},
                                color_discrete_sequence=['#1E88E5'])
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Распределение по годам
        year_counts = success_df['year_num'].value_counts().sort_index()
        fig = px.bar(x=year_counts.index, y=year_counts.values,
                    title='Publications by Year',
                    labels={'x': 'Year', 'y': 'Number of Publications'},
                    color_discrete_sequence=['#1E88E5'])
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Анализ авторов
        col1, col2 = st.columns(2)
        
        with col1:
            author_stats = success_df['author_count'].describe()
            st.metric("Average Authors per Paper", f"{author_stats['mean']:.1f}")
            st.metric("Max Authors", int(author_stats['max']))
            st.metric("Min Authors", int(author_stats['min']))
        
        with col2:
            fig = px.box(success_df, y='author_count', 
                        title='Distribution of Author Counts',
                        labels={'author_count': 'Number of Authors'},
                        color_discrete_sequence=['#1E88E5'])
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig, use_container_width=True)

def display_statistics(df):
    """Отображает статистику анализа"""
    successful = len(df[df['title'] != 'Error'])
    total_articles = len(df)
    
    if successful > 0:
        success_df = df[df['title'] != 'Error'].copy()
        
        st.subheader("📈 Detailed Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Success Rate", f"{successful/total_articles*100:.1f}%")
            st.metric("Total Citations", int(success_df['citation_count'].sum()))
            
        with col2:
            citation_mean = success_df['citation_count'].mean()
            citation_median = success_df['citation_count'].median()
            st.metric("Mean Citations", f"{citation_mean:.1f}")
            st.metric("Median Citations", f"{citation_median:.1f}")
            
        with col3:
            author_mean = success_df['author_count'].mean()
            author_max = success_df['author_count'].max()
            st.metric("Avg Authors", f"{author_mean:.1f}")
            st.metric("Max Authors", int(author_max))
        
        # Распределение по годам
        st.subheader("📅 Publications by Year")
        year_counts = success_df[success_df['year'] != 'Unknown']['year'].value_counts()
        if len(year_counts) > 0:
            st.dataframe(year_counts.sort_index(ascending=False))
        else:
            st.info("No year data available")

def main():
    st.title("📚 Advanced DOI Citation Analyzer")
    st.markdown("""
    Analyze scientific articles using DOI identifiers with comprehensive visualizations and statistics.
    Enter one or more DOIs to get detailed information about each publication.
    """)
    
    # Боковая панель с информацией
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This tool helps researchers analyze scientific publications by:
        
        - 📊 **Citation analysis**
        - 📅 **Publication trends**  
        - 👥 **Author statistics**
        - 📈 **Visual analytics**
        
        **Supported DOI formats:**
        - `10.1038/s41586-023-06924-6`
        - `https://doi.org/10.1038/s41586-023-06924-6`
        - `doi:10.1038/s41586-023-06924-6`
        """)
        
        st.header("⚙️ Settings")
        max_dois = st.slider("Maximum DOIs to process", 5, 50, 20)
        api_delay = st.slider("API delay (seconds)", 0.1, 2.0, 0.5)
    
    # Инициализация анализатора
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = CitationAnalyzer()
    
    analyzer = st.session_state.analyzer
    
    # Основной контент
    st.header("🔍 Input DOIs")
    
    doi_input = st.text_area(
        "Enter DOIs for analysis (one per line or separated by any punctuation)",
        value="10.1038/s41586-023-06924-6\n10.1126/science.abl4471\n10.1038/s41567-023-02076-6",
        height=150,
        help="Examples: 10.1038/s41586-023-06924-6, https://doi.org/10.1126/science.abl4471"
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        analyze_clicked = st.button("🚀 Analyze Articles", type="primary", use_container_width=True)
    
    with col2:
        if st.button("🔄 Clear Cache", use_container_width=True):
            st.session_state.analyzer = CitationAnalyzer()
            st.success("Cache cleared!")
    
    if analyze_clicked:
        if doi_input:
            with st.spinner("Parsing DOIs..."):
                doi_list = analyzer.parse_doi_input(doi_input, max_dois=max_dois)
            
            if doi_list:
                # Анализ статей
                with st.spinner("Fetching article data from Crossref..."):
                    results = analyzer.analyze_articles(doi_list)
                
                # Создаем DataFrame
                df = pd.DataFrame(results)
                
                # Показываем результаты
                st.header("📋 Analysis Results")
                
                # Основные метрики
                st.subheader("📊 Overview Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                total_articles = len(df)
                successful = len(df[df['title'] != 'Error'])
                success_rate = successful/total_articles*100 if total_articles > 0 else 0
                avg_citations = df[df['citation_count'] > 0]['citation_count'].mean()
                total_citations = df['citation_count'].sum()
                
                with col1:
                    st.metric("Total Articles", total_articles)
                with col2:
                    st.metric("Successful", successful, f"{success_rate:.1f}%")
                with col3:
                    st.metric("Avg Citations", f"{avg_citations:.1f}" if not np.isnan(avg_citations) else "0")
                with col4:
                    st.metric("Total Citations", int(total_citations))
                
                # Детальная таблица
                st.subheader("📄 Article Details")
                
                # Выбираем колонки для отображения
                display_columns = ['doi', 'title', 'year', 'authors', 'journal', 'citation_count']
                available_columns = [col for col in display_columns if col in df.columns]
                
                # Стилизуем таблицу
                styled_df = df[available_columns].style.format({
                    'citation_count': '{:.0f}'
                }).apply(lambda x: ['background: #e6f3ff' if x.name % 2 == 0 else '' for i in x], axis=0)
                
                st.dataframe(styled_df, use_container_width=True, height=400)
                
                # Визуализации
                if successful > 0:
                    st.header("📈 Visual Analytics")
                    create_visualizations(df)
                    
                    # Детальная статистика
                    display_statistics(df)
                
                # Опции для скачивания
                st.header("💾 Export Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # CSV download
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name="citation_analysis.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    # JSON download
                    json_data = df.to_json(orient='records', indent=2)
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_data,
                        file_name="citation_analysis.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
                with col3:
                    # Excel download
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False, sheet_name='Citation Analysis')
                    excel_data = output.getvalue()
                    st.download_button(
                        label="📥 Download Excel",
                        data=excel_data,
                        file_name="citation_analysis.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                
            else:
                st.error("No valid DOIs to analyze")
        else:
            st.error("Please enter at least one DOI")

if __name__ == "__main__":
    main()
