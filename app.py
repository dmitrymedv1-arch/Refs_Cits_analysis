import streamlit as st
import pandas as pd
import requests
import time
from typing import List, Dict, Any, Tuple
import re
from collections import Counter
import tempfile
import os
import concurrent.futures
from functools import lru_cache
import json

# Настройки
st.set_page_config(
    page_title="Refs/Cits Analysis",
    page_icon="📚",
    layout="wide"
)

class FastCitationAnalyzer:
    def __init__(self):
        self.request_delay = 0.2  # Уменьшили задержку
        self.max_dois = 50
        self.max_references_per_article = 20  # Ограничение для скорости
        self.max_citations_per_article = 20
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Academic-Analyzer/1.0 (mailto:example@university.edu)',
            'Accept': 'application/json'
        })
        
        # Кеши
        self._crossref_cache = {}
        self._openalex_cache = {}
        self._article_data_cache = {}

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
    
    def get_crossref_data_batch(self, dois: List[str]) -> Dict[str, Dict]:
        """Получает данные из Crossref для нескольких DOI сразу"""
        results = {}
        
        # Разбиваем на группы по 10 DOI для batch запроса
        for i in range(0, len(dois), 10):
            batch_dois = dois[i:i+10]
            
            # Crossref не поддерживает настоящие batch запросы, но можно попробовать
            # Вместо этого используем многопоточность для отдельных запросов
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_doi = {
                    executor.submit(self._get_single_crossref_data, doi): doi 
                    for doi in batch_dois
                }
                
                for future in concurrent.futures.as_completed(future_to_doi):
                    doi = future_to_doi[future]
                    try:
                        data = future.result()
                        results[doi] = data
                    except Exception as e:
                        st.warning(f"Error fetching Crossref data for {doi}: {e}")
                        results[doi] = {}
            
            time.sleep(self.request_delay)  # Небольшая пауза между батчами
        
        return results
    
    def _get_single_crossref_data(self, doi: str) -> Dict:
        """Получает данные для одного DOI из Crossref"""
        if doi in self._crossref_cache:
            return self._crossref_cache[doi]
            
        try:
            url = f"https://api.crossref.org/works/{doi}"
            response = self.session.get(url, timeout=15)
            if response.status_code == 200:
                data = response.json().get('message', {})
                self._crossref_cache[doi] = data
                return data
        except Exception as e:
            st.warning(f"Crossref error for {doi}: {e}")
            
        return {}
    
    def get_openalex_data_batch(self, dois: List[str]) -> Dict[str, Dict]:
        """Получает данные из OpenAlex для нескольких DOI сразу"""
        results = {}
        
        # OpenAlex поддерживает фильтрацию по нескольким DOI
        doi_filter = "|".join([f"https://doi.org/{doi}" for doi in dois])
        url = f"https://api.openalex.org/works?filter=doi:{doi_filter}&per-page=50"
        
        try:
            response = self.session.get(url, timeout=20)
            if response.status_code == 200:
                data = response.json()
                works = data.get('results', [])
                
                # Создаем маппинг DOI -> данные
                for work in works:
                    if work.get('doi'):
                        clean_doi = self.normalize_doi(work['doi'])
                        self._openalex_cache[clean_doi] = work
                        results[clean_doi] = work
            
            # Для DOI, которых нет в batch ответе, делаем индивидуальные запросы
            missing_dois = set(dois) - set(results.keys())
            if missing_dois:
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    future_to_doi = {
                        executor.submit(self._get_single_openalex_data, doi): doi 
                        for doi in missing_dois
                    }
                    
                    for future in concurrent.futures.as_completed(future_to_doi):
                        doi = future_to_doi[future]
                        try:
                            data = future.result()
                            results[doi] = data
                        except Exception:
                            results[doi] = {}
                            
        except Exception as e:
            st.warning(f"OpenAlex batch error: {e}")
            # Fallback к индивидуальным запросам
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_doi = {
                    executor.submit(self._get_single_openalex_data, doi): doi 
                    for doi in dois
                }
                
                for future in concurrent.futures.as_completed(future_to_doi):
                    doi = future_to_doi[future]
                    try:
                        data = future.result()
                        results[doi] = data
                    except Exception:
                        results[doi] = {}
        
        return results
    
    def _get_single_openalex_data(self, doi: str) -> Dict:
        """Получает данные для одного DOI из OpenAlex"""
        if doi in self._openalex_cache:
            return self._openalex_cache[doi]
            
        try:
            url = f"https://api.openalex.org/works/https://doi.org/{doi}"
            response = self.session.get(url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                self._openalex_cache[doi] = data
                return data
        except Exception as e:
            st.warning(f"OpenAlex error for {doi}: {e}")
            
        return {}
    
    def extract_authors(self, crossref_data: Dict, openalex_data: Dict) -> str:
        """Извлекает авторов"""
        authors = []
        
        # Сначала пробуем OpenAlex (обычно лучше структурированы)
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
    
    def get_article_data_batch(self, dois: List[str]) -> Dict[str, Dict]:
        """Получает данные о статьях для списка DOI"""
        # Проверяем кеш
        results = {}
        dois_to_fetch = []
        
        for doi in dois:
            if doi in self._article_data_cache:
                results[doi] = self._article_data_cache[doi]
            else:
                dois_to_fetch.append(doi)
        
        if not dois_to_fetch:
            return results
        
        # Параллельно получаем данные из Crossref и OpenAlex
        with st.spinner("Fetching article data from APIs..."):
            crossref_results = self.get_crossref_data_batch(dois_to_fetch)
            time.sleep(self.request_delay)
            openalex_results = self.get_openalex_data_batch(dois_to_fetch)
        
        # Объединяем данные
        for doi in dois_to_fetch:
            crossref_data = crossref_results.get(doi, {})
            openalex_data = openalex_results.get(doi, {})
            
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
            
            article_data = {
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
            
            results[doi] = article_data
            self._article_data_cache[doi] = article_data
        
        return results
    
    def get_references_batch(self, doi_list: List[str]) -> Dict[str, List[Dict]]:
        """Получает ссылки для списка DOI"""
        all_references = {}
        
        # Сначала получаем все данные Crossref
        crossref_data = self.get_crossref_data_batch(doi_list)
        
        for doi in doi_list:
            data = crossref_data.get(doi, {})
            references = data.get('reference', [])
            # Ограничиваем количество ссылок для скорости
            all_references[doi] = references[:self.max_references_per_article]
        
        return all_references
    
    def get_citing_articles_batch(self, doi_list: List[str]) -> Dict[str, List[str]]:
        """Получает цитирующие статьи для списка DOI"""
        all_citing_articles = {}
        
        # Получаем данные OpenAlex для поиска цитирований
        openalex_data = self.get_openalex_data_batch(doi_list)
        
        for doi in doi_list:
            citing_dois = []
            data = openalex_data.get(doi, {})
            
            # Используем OpenAlex для поиска цитирований
            if data and 'cited_by_count' in data and data['cited_by_count'] > 0:
                work_id = data.get('id', '').split('/')[-1]
                if work_id:
                    try:
                        citing_url = f"https://api.openalex.org/works?filter=cites:{work_id}&per-page={self.max_citations_per_article}"
                        response = self.session.get(citing_url, timeout=15)
                        if response.status_code == 200:
                            citing_data = response.json()
                            for work in citing_data.get('results', []):
                                if work.get('doi'):
                                    citing_dois.append(work['doi'])
                    except Exception as e:
                        st.warning(f"Error getting citations for {doi}: {e}")
            
            # Ограничиваем количество цитирований
            all_citing_articles[doi] = citing_dois[:self.max_citations_per_article]
        
        return all_citing_articles
    
    def analyze_references_fast(self, doi_list: List[str]) -> pd.DataFrame:
        """Быстрый анализ ссылок"""
        all_references_data = []
        
        # Прогресс бар
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Step 1/3: Getting source articles data...")
        # Получаем данные исходных статей
        source_articles_data = self.get_article_data_batch(doi_list)
        
        # Добавляем исходные статьи в результаты
        for doi, data in source_articles_data.items():
            data['type'] = 'source'
            data['source_doi'] = doi
            all_references_data.append(data)
        
        progress_bar.progress(33)
        
        status_text.text("Step 2/3: Collecting references...")
        # Получаем все ссылки
        all_references = self.get_references_batch(doi_list)
        
        # Собираем все DOI ссылок
        all_reference_dois = set()
        for references in all_references.values():
            for ref in references:
                ref_doi = ref.get('DOI')
                if ref_doi and self.validate_doi(ref_doi):
                    all_reference_dois.add(ref_doi)
        
        progress_bar.progress(66)
        
        status_text.text("Step 3/3: Getting references data...")
        # Получаем данные всех ссылок
        if all_reference_dois:
            reference_articles_data = self.get_article_data_batch(list(all_reference_dois))
            
            # Создаем записи для ссылок
            for source_doi, references in all_references.items():
                for i, ref in enumerate(references):
                    ref_doi = ref.get('DOI')
                    if ref_doi and self.validate_doi(ref_doi):
                        ref_data = reference_articles_data.get(ref_doi, {})
                        if ref_data:
                            ref_data['type'] = 'reference'
                            ref_data['source_doi'] = source_doi
                            ref_data['position'] = i + 1
                            all_references_data.append(ref_data)
        
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        
        return pd.DataFrame(all_references_data)
    
    def analyze_citations_fast(self, doi_list: List[str]) -> pd.DataFrame:
        """Быстрый анализ цитирований"""
        all_citations_data = []
        
        # Прогресс бар
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Step 1/3: Getting source articles data...")
        # Получаем данные исходных статей
        source_articles_data = self.get_article_data_batch(doi_list)
        
        # Добавляем исходные статьи в результаты
        for doi, data in source_articles_data.items():
            data['type'] = 'source'
            data['source_doi'] = doi
            all_citations_data.append(data)
        
        progress_bar.progress(33)
        
        status_text.text("Step 2/3: Collecting citations...")
        # Получаем все цитирующие статьи
        all_citing_articles = self.get_citing_articles_batch(doi_list)
        
        # Собираем все DOI цитирующих статей
        all_citing_dois = set()
        for citing_dois in all_citing_articles.values():
            all_citing_dois.update(citing_dois)
        
        progress_bar.progress(66)
        
        status_text.text("Step 3/3: Getting citations data...")
        # Получаем данные всех цитирующих статей
        if all_citing_dois:
            citing_articles_data = self.get_article_data_batch(list(all_citing_dois))
            
            # Создаем записи для цитирований
            for source_doi, citing_dois in all_citing_articles.items():
                for citing_doi in citing_dois:
                    if self.validate_doi(citing_doi):
                        citing_data = citing_articles_data.get(citing_doi, {})
                        if citing_data:
                            citing_data['type'] = 'citation'
                            citing_data['source_doi'] = source_doi
                            all_citations_data.append(citing_data)
        
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        
        return pd.DataFrame(all_citations_data)

def main():
    st.title("📚 Refs/Cits Analysis - Fast Version")
    st.markdown("⚡ Ускоренный анализ ссылок и цитирований научных статей")
    
    analyzer = FastCitationAnalyzer()
    
    # Боковая панель с настройками
    with st.sidebar:
        st.header("⚙️ Настройки")
        st.info("""
        Введите DOI статей для анализа:
        - 10.1234/abcd.1234
        - https://doi.org/10.1234/abcd.1234  
        - doi:10.1234/abcd.1234
        """)
        
        max_dois = st.slider("Максимум DOI для анализа", 5, 50, 15)
        analyzer.max_dois = max_dois
        
        max_refs = st.slider("Макс. ссылок на статью", 5, 50, 15)
        analyzer.max_references_per_article = max_refs
        
        max_cits = st.slider("Макс. цитирований на статью", 5, 50, 15)
        analyzer.max_citations_per_article = max_cits
        
        st.markdown("---")
        st.markdown("### 📊 Информация")
        st.markdown("""
        **Ускоренная версия:**
        - Пакетные запросы к API
        - Многопоточность
        - Кеширование данных
        - Оптимизированные лимиты
        """)
    
    # Основной контент
    tab1, tab2, tab3 = st.tabs(["📥 Ввод данных", "🔗 Анализ ссылок", "📈 Анализ цитирований"])
    
    with tab1:
        st.header("Ввод DOI статей")
        
        input_method = st.radio("Способ ввода:", ["📝 Текст", "📁 Файл"])
        
        if input_method == "📝 Текст":
            doi_input = st.text_area(
                "Введите DOI статей:",
                height=150,
                placeholder="Введите DOI через запятую, точку с запятой или с новой строки:\n10.1038/s41586-023-06924-6\n10.1126/science.abl8921\n10.1016/j.cell.2023.08.012"
            )
            
            if st.button("🔍 Проверить DOI", key="validate"):
                if doi_input:
                    dois = analyzer.parse_doi_input(doi_input)
                    if dois:
                        st.success(f"✅ Найдено {len(dois)} валидных DOI:")
                        for doi in dois:
                            st.write(f"- `{doi}`")
                        
                        # Сохраняем в session state
                        st.session_state.dois = dois
                    else:
                        st.error("❌ Не найдено валидных DOI. Проверьте формат.")
        
        else:
            uploaded_file = st.file_uploader("Загрузите файл с DOI", type=['txt', 'csv'])
            if uploaded_file:
                content = uploaded_file.getvalue().decode()
                dois = analyzer.parse_doi_input(content)
                if dois:
                    st.success(f"✅ Найдено {len(dois)} валидных DOI в файле")
                    st.write("Обнаруженные DOI:")
                    for doi in dois[:10]:
                        st.write(f"- `{doi}`")
                    if len(dois) > 10:
                        st.write(f"... и еще {len(dois) - 10} DOI")
                    
                    # Сохраняем в session state
                    st.session_state.dois = dois
    
    with tab2:
        st.header("🔗 Анализ ссылок (References)")
        
        if 'dois' in st.session_state and st.session_state.dois:
            dois = st.session_state.dois
            
            st.info(f"Будет проанализировано {len(dois)} статей (максимум {analyzer.max_references_per_article} ссылок на статью)")
            
            if st.button("🚀 Запустить анализ ссылок", type="primary"):
                if not dois:
                    st.error("❌ Сначала введите валидные DOI на вкладке 'Ввод данных'")
                else:
                    with st.spinner("⚡ Выполняется ускоренный анализ ссылок..."):
                        references_df = analyzer.analyze_references_fast(dois)
                    
                    if not references_df.empty:
                        st.success("✅ Анализ завершен!")
                        
                        # Основная статистика
                        st.subheader("📊 Статистика")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            total_refs = len(references_df[references_df['type'] == 'reference'])
                            st.metric("Всего ссылок", total_refs)
                        with col2:
                            unique_journals = references_df['journal_abbreviation'].nunique()
                            st.metric("Уникальных журналов", unique_journals)
                        with col3:
                            source_articles = len(references_df[references_df['type'] == 'source'])
                            st.metric("Исходных статей", source_articles)
                        with col4:
                            avg_citations = references_df['citation_count_openalex'].mean()
                            st.metric("Средние цитирования", f"{avg_citations:.1f}")
                        
                        # Визуализация по годам
                        st.subheader("📅 Распределение по годам")
                        year_counts = references_df[references_df['year'] != 'Unknown']['year'].value_counts()
                        if not year_counts.empty:
                            st.bar_chart(year_counts.head(10))
                        
                        # Данные
                        st.subheader("📋 Данные ссылок")
                        
                        # Фильтры
                        col1, col2 = st.columns(2)
                        with col1:
                            show_type = st.selectbox("Тип записей:", ["Все", "Только исходные", "Только ссылки"])
                        with col2:
                            show_columns = st.multiselect(
                                "Колонки для отображения:",
                                references_df.columns,
                                default=['doi', 'title', 'authors', 'year', 'journal_abbreviation', 'citation_count_openalex']
                            )
                        
                        # Применяем фильтры
                        filtered_df = references_df
                        if show_type == "Только исходные":
                            filtered_df = references_df[references_df['type'] == 'source']
                        elif show_type == "Только ссылки":
                            filtered_df = references_df[references_df['type'] == 'reference']
                        
                        if show_columns:
                            filtered_df = filtered_df[show_columns]
                        
                        st.dataframe(filtered_df, use_container_width=True)
                        
                        # Экспорт
                        st.subheader("💾 Экспорт данных")
                        csv = references_df.to_csv(index=False)
                        st.download_button(
                            "📥 Скачать CSV",
                            csv,
                            "references_analysis.csv",
                            "text/csv",
                            key='download_refs'
                        )
                    else:
                        st.error("❌ Не удалось получить данные")
        else:
            st.warning("⚠️ Сначала введите DOI на вкладке 'Ввод данных'")
    
    with tab3:
        st.header("📈 Анализ цитирований (Citations)")
        
        if 'dois' in st.session_state and st.session_state.dois:
            dois = st.session_state.dois
            
            st.info(f"Будет проанализировано {len(dois)} статей (максимум {analyzer.max_citations_per_article} цитирований на статью)")
            
            if st.button("🚀 Запустить анализ цитирований", type="primary"):
                if not dois:
                    st.error("❌ Сначала введите валидные DOI на вкладке 'Ввод данных'")
                else:
                    with st.spinner("⚡ Выполняется ускоренный анализ цитирований..."):
                        citations_df = analyzer.analyze_citations_fast(dois)
                    
                    if not citations_df.empty:
                        st.success("✅ Анализ завершен!")
                        
                        # Основная статистика
                        st.subheader("📊 Статистика")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            total_cits = len(citations_df[citations_df['type'] == 'citation'])
                            st.metric("Всего цитирований", total_cits)
                        with col2:
                            citing_journals = citations_df[citations_df['type'] == 'citation']['journal_abbreviation'].nunique()
                            st.metric("Цитирующих журналов", citing_journals)
                        with col3:
                            source_articles = len(citations_df[citations_df['type'] == 'source'])
                            st.metric("Исходных статей", source_articles)
                        with col4:
                            recent_year = str(pd.Timestamp.now().year)
                            recent_citations = len(citations_df[
                                (citations_df['type'] == 'citation') & 
                                (citations_df['year'] == recent_year)
                            ])
                            st.metric(f"Цитирований {recent_year}", recent_citations)
                        
                        # Визуализация по годам
                        st.subheader("📅 Распределение по годам")
                        year_counts = citations_df[citations_df['year'] != 'Unknown']['year'].value_counts()
                        if not year_counts.empty:
                            st.bar_chart(year_counts.head(10))
                        
                        # Данные
                        st.subheader("📋 Данные цитирований")
                        
                        # Фильтры
                        col1, col2 = st.columns(2)
                        with col1:
                            show_type = st.selectbox("Тип записей:", ["Все", "Только исходные", "Только цитирования"], key='cit_type')
                        with col2:
                            show_columns = st.multiselect(
                                "Колонки для отображения:",
                                citations_df.columns,
                                default=['doi', 'title', 'authors', 'year', 'journal_abbreviation', 'citation_count_openalex'],
                                key='cit_columns'
                            )
                        
                        # Применяем фильтры
                        filtered_df = citations_df
                        if show_type == "Только исходные":
                            filtered_df = citations_df[citations_df['type'] == 'source']
                        elif show_type == "Только цитирования":
                            filtered_df = citations_df[citations_df['type'] == 'citation']
                        
                        if show_columns:
                            filtered_df = filtered_df[show_columns]
                        
                        st.dataframe(filtered_df, use_container_width=True)
                        
                        # Экспорт
                        st.subheader("💾 Экспорт данных")
                        csv = citations_df.to_csv(index=False)
                        st.download_button(
                            "📥 Скачать CSV",
                            csv,
                            "citations_analysis.csv",
                            "text/csv",
                            key='download_cits'
                        )
                    else:
                        st.error("❌ Не удалось получить данные")
        else:
            st.warning("⚠️ Сначала введите DOI на вкладке 'Ввод данных'")

if __name__ == "__main__":
    main()
