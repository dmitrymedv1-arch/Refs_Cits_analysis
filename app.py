import streamlit as st
import pandas as pd
import requests
import time
from typing import List, Dict, Any, Tuple, Set
import re
from collections import Counter
import tempfile
import os
import concurrent.futures
from functools import lru_cache
import json
from datetime import datetime

# Настройки
st.set_page_config(
    page_title="Refs/Cits Analysis - Full Version",
    page_icon="📚",
    layout="wide"
)

# Попытка импорта NLTK с обработкой ошибок
try:
    import nltk
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        try:
            nltk.download('stopwords', quiet=True)
        except:
            st.warning("NLTK stopwords не доступны. Используем базовые стоп-слова.")
    
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    st.warning("NLTK не установлен. Используем упрощенную обработку текста.")

class Config:
    REQUEST_TIMEOUT = 30
    MAX_RETRIES = 3
    DELAY_BETWEEN_REQUESTS = 0.1

class FastAffiliationProcessor:
    """Быстрый процессор аффилиаций с группировкой похожих организаций"""

    def __init__(self):
        self.common_keywords = {
            'university', 'college', 'institute', 'school', 'department', 'faculty',
            'laboratory', 'center', 'centre', 'academy', 'universität', 'universitat',
            'université', 'universite', 'polytechnic', 'technical', 'technology',
            'research', 'science', 'sciences', 'studies', 'medical', 'hospital',
            'clinic', 'foundation', 'corporation', 'company', 'inc', 'ltd', 'corp'
        }
        self.organization_cache = {}
        self.country_keywords = {
            'usa', 'united states', 'us', 'u.s.', 'u.s.a.', 'america',
            'uk', 'united kingdom', 'great britain', 'england', 'scotland', 'wales',
            'germany', 'deutschland', 'france', 'french', 'italy', 'italian',
            'spain', 'spanish', 'china', 'chinese', 'japan', 'japanese',
            'russia', 'russian', 'india', 'indian', 'brazil', 'brazilian',
            'canada', 'canadian', 'australia', 'australian', 'korea', 'korean'
        }

    def extract_main_organization_fast(self, affiliation: str) -> str:
        """Быстрое извлечение основной организации из полной аффилиации"""
        if not affiliation or affiliation in ['Unknown', 'Error', '']:
            return "Unknown"

        if affiliation in self.organization_cache:
            return self.organization_cache[affiliation]

        clean_affiliation = affiliation.strip()
        clean_affiliation = re.sub(r'\S+@\S+', '', clean_affiliation)
        clean_affiliation = re.sub(r'\d{5,}(?:-\d{4})?', '', clean_affiliation)
        clean_affiliation = re.sub(r'p\.?o\.? box \d+', '', clean_affiliation, flags=re.IGNORECASE)
        clean_affiliation = re.sub(r'\b\d+\s+[a-zA-Z]+\s+[a-zA-Z]+\b', '', clean_affiliation)

        parts = re.split(r'[,;]', clean_affiliation)
        main_org_candidates = []

        for part in parts:
            part = part.strip()
            if not part or len(part) < 5:
                continue

            part_lower = part.lower()
            has_org_keyword = any(keyword in part_lower for keyword in self.common_keywords)
            has_country = any(country in part_lower for country in self.country_keywords)

            if has_org_keyword and not has_country:
                main_org_candidates.append(part)

        if main_org_candidates:
            main_org_candidates.sort(key=len, reverse=True)
            main_org = main_org_candidates[0]
        else:
            for part in parts:
                part = part.strip()
                if len(part) > 10 and not any(country in part.lower() for country in self.country_keywords):
                    main_org = part
                    break
            else:
                for part in parts:
                    part = part.strip()
                    if part:
                        main_org = part
                        break
                else:
                    main_org = clean_affiliation

        main_org = re.sub(r'\s+', ' ', main_org).strip()
        main_org = re.sub(r'^[^a-zA-Z0-9]*|[^a-zA-Z0-9]*$', '', main_org)

        result = main_org if main_org else "Unknown"
        self.organization_cache[affiliation] = result
        return result

class AltmetricProcessor:
    """Процессор для сбора альтметрических данных"""

    def __init__(self):
        self.altmetric_cache = {}

    def clean_doi(self, doi: str) -> str:
        """Очищает DOI от лишних символов"""
        if not doi or doi in ['Unknown', 'Error', '']:
            return None

        doi = doi.strip().lower()
        doi = re.sub(r'^(doi:)?\s*', '', doi)
        doi = re.sub(r'\s+', '', doi)
        if re.match(r'^10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+$', doi):
            return doi
        return None

    def get_altmetric_data(self, doi: str) -> Dict:
        """Получает данные из бесплатного API Altmetric по DOI"""
        clean_doi = self.clean_doi(doi)
        if not clean_doi:
            return None

        if clean_doi in self.altmetric_cache:
            return self.altmetric_cache[clean_doi]

        url = f"https://api.altmetric.com/v1/doi/{clean_doi}"
        try:
            response = requests.get(url, timeout=Config.REQUEST_TIMEOUT)
            if response.status_code == 200:
                data = response.json()
                self.altmetric_cache[clean_doi] = data
                return data
            else:
                self.altmetric_cache[clean_doi] = None
                return None
        except requests.exceptions.RequestException:
            self.altmetric_cache[clean_doi] = None
            return None

    def get_altmetric_metrics(self, doi: str) -> Dict[str, Any]:
        """Извлекает ключевые альтметрические показатели для DOI"""
        data = self.get_altmetric_data(doi)

        if not data:
            return {
                'altmetric_score': 0,
                'cited_by_posts_count': 0,
                'cited_by_tweeters_count': 0,
                'cited_by_feeds_count': 0,
                'cited_by_accounts_count': 0
            }

        return {
            'altmetric_score': data.get('score', 0),
            'cited_by_posts_count': data.get('cited_by_posts_count', 0),
            'cited_by_tweeters_count': data.get('cited_by_tweeters_count', 0),
            'cited_by_feeds_count': data.get('cited_by_feeds_count', 0),
            'cited_by_accounts_count': data.get('cited_by_accounts_count', 0)
        }

class FullCitationAnalyzer:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Academic-Analyzer/1.0 (mailto:research@university.edu)',
            'Accept': 'application/json'
        })
        
        # Кеши
        self._crossref_cache = {}
        self._openalex_cache = {}
        self._article_data_cache = {}
        self._references_cache = {}
        self._citations_cache = {}
        
        # Процессоры
        self.fast_affiliation_processor = FastAffiliationProcessor()
        self.altmetric_processor = AltmetricProcessor()
        
        # NLP компоненты (с обработкой отсутствия NLTK)
        if NLTK_AVAILABLE:
            self.stop_words = set(stopwords.words('english'))
            self.stemmer = PorterStemmer()
        else:
            # Базовые стоп-слова если NLTK недоступен
            self.stop_words = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", 
                "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 
                'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 
                'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 
                'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
                'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 
                'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 
                'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 
                'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 
                'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 
                'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 
                'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 
                'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 
                'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 
                'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', 
                "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 
                'wouldn', "wouldn't"
            }
            self.stemmer = None
        
        self.scientific_stopwords = {
            'using', 'based', 'study', 'studies', 'research', 'analysis',
            'effect', 'effects', 'properties', 'property', 'development',
            'application', 'applications', 'method', 'methods', 'approach',
            'review', 'investigation', 'characterization', 'evaluation',
            'performance', 'behavior', 'structure', 'synthesis', 'design',
            'fabrication', 'preparation', 'processing', 'measurement',
            'model', 'models', 'system', 'systems', 'technology', 'material',
            'materials', 'sample', 'samples', 'device', 'devices', 'film',
            'films', 'layer', 'layers', 'surface', 'surfaces', 'interface',
            'interfaces', 'nanoparticle', 'nanoparticles', 'nanostructure',
            'nanostructures', 'composite', 'composites', 'coating', 'coatings'
        }
        
        if NLTK_AVAILABLE and self.stemmer:
            self.scientific_stopwords_stemmed = {self.stemmer.stem(word) for word in self.scientific_stopwords}
        else:
            self.scientific_stopwords_stemmed = self.scientific_stopwords

    def validate_doi(self, doi: str) -> bool:
        """Проверяет валидность DOI"""
        if not doi or not isinstance(doi, str):
            return False
            
        doi = self.normalize_doi(doi)
        doi_pattern = r'^10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+$'
        
        if not bool(re.match(doi_pattern, doi, re.IGNORECASE)):
            return False

        if len(doi) < 10:
            return False

        if re.search(r'[^\w\.\-_;()/:]', doi):
            return False

        return True
    
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
                elif line.lower().startswith('doi:'):
                    doi_part = line[4:].strip()
                    if self.validate_doi(doi_part):
                        dois.append(doi_part)
                elif self.validate_doi(line):
                    dois.append(line)
        
        cleaned_dois = []
        seen = set()
        for doi in dois:
            normalized_doi = self.normalize_doi(doi)
            if self.validate_doi(normalized_doi) and normalized_doi not in seen:
                seen.add(normalized_doi)
                cleaned_dois.append(normalized_doi)
        
        return cleaned_dois

    def get_crossref_data_batch(self, dois: List[str], progress_bar=None, status_text=None) -> Dict[str, Dict]:
        """Получает данные из Crossref для нескольких DOI сразу"""
        results = {}
        total_dois = len(dois)
        
        if status_text:
            status_text.text("📡 Получение данных из Crossref...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_doi = {
                executor.submit(self._get_single_crossref_data, doi): doi 
                for doi in dois
            }
            
            for i, future in enumerate(concurrent.futures.as_completed(future_to_doi)):
                doi = future_to_doi[future]
                try:
                    data = future.result()
                    results[doi] = data
                except Exception as e:
                    results[doi] = {}
                
                if progress_bar and total_dois > 0:
                    progress_bar.progress((i + 1) / total_dois)
                
                time.sleep(Config.DELAY_BETWEEN_REQUESTS)
        
        return results

    def _get_single_crossref_data(self, doi: str) -> Dict:
        """Получает данные для одного DOI из Crossref"""
        if doi in self._crossref_cache:
            return self._crossref_cache[doi]
            
        for attempt in range(Config.MAX_RETRIES):
            try:
                url = f"https://api.crossref.org/works/{doi}"
                response = self.session.get(url, timeout=Config.REQUEST_TIMEOUT)
                
                if response.status_code == 200:
                    data = response.json().get('message', {})
                    
                    # Извлекаем аффилиации
                    affiliations, countries = self.extract_affiliations_from_crossref(data)
                    data['extracted_affiliations'] = affiliations
                    data['extracted_countries'] = countries
                    
                    self._crossref_cache[doi] = data
                    return data
                    
            except Exception as e:
                if attempt == Config.MAX_RETRIES - 1:
                    return {}
                time.sleep(1)
                
        return {}

    def extract_affiliations_from_crossref(self, crossref_data: Dict) -> tuple[List[str], List[str]]:
        """Извлекает аффилиации и страны из данных Crossref"""
        affiliations = set()
        countries = set()

        try:
            if 'author' in crossref_data:
                for author in crossref_data['author']:
                    if 'affiliation' in author:
                        for affil in author['affiliation']:
                            if 'name' in affil:
                                affiliation_name = affil['name'].strip()
                                if affiliation_name and affiliation_name not in ['', 'None']:
                                    main_org = self.fast_affiliation_processor.extract_main_organization_fast(affiliation_name)
                                    if main_org and main_org != "Unknown":
                                        affiliations.add(main_org)

                            country = self.extract_country_from_affiliation(affil)
                            if country:
                                countries.add(country)

        except Exception as e:
            pass

        return list(affiliations), list(countries)

    def extract_country_from_affiliation(self, affiliation_data: Dict) -> str:
        """Извлекает страну из данных аффилиации Crossref"""
        try:
            if 'country' in affiliation_data and affiliation_data['country']:
                return affiliation_data['country'].upper().strip()

            if 'name' in affiliation_data and affiliation_data['name']:
                name = affiliation_data['name'].upper()
                country_keywords = {
                    'UNITED STATES': 'US', 'USA': 'US', 'U.S.A': 'US', 'U.S.': 'US',
                    'UNITED KINGDOM': 'UK', 'UK': 'UK', 'GREAT BRITAIN': 'UK',
                    'GERMANY': 'DE', 'FRANCE': 'FR', 'CHINA': 'CN', 'JAPAN': 'JP',
                    'RUSSIA': 'RU', 'INDIA': 'IN', 'BRAZIL': 'BR', 'CANADA': 'CA',
                    'AUSTRALIA': 'AU', 'KOREA': 'KR', 'SOUTH KOREA': 'KR'
                }
                for keyword, code in country_keywords.items():
                    if keyword in name:
                        return code

        except Exception as e:
            pass

        return ""

    def get_openalex_data_batch(self, dois: List[str], progress_bar=None, status_text=None) -> Dict[str, Dict]:
        """Получает данные из OpenAlex для нескольких DOI сразу"""
        results = {}
        total_dois = len(dois)
        
        if status_text:
            status_text.text("📡 Получение данных из OpenAlex...")
        
        # OpenAlex batch запросы
        batch_size = 50
        for i in range(0, len(dois), batch_size):
            batch_dois = dois[i:i + batch_size]
            doi_filter = "|".join([f"https://doi.org/{doi}" for doi in batch_dois])
            url = f"https://api.openalex.org/works?filter=doi:{doi_filter}&per-page={batch_size}"
            
            try:
                response = self.session.get(url, timeout=Config.REQUEST_TIMEOUT)
                if response.status_code == 200:
                    data = response.json()
                    works = data.get('results', [])
                    
                    for work in works:
                        if work.get('doi'):
                            clean_doi = self.normalize_doi(work['doi'])
                            self._openalex_cache[clean_doi] = work
                            results[clean_doi] = work
                
                if progress_bar and total_dois > 0:
                    progress_bar.progress(min((i + batch_size) / total_dois, 1.0))
                    
            except Exception as e:
                continue
                
            time.sleep(Config.DELAY_BETWEEN_REQUESTS)

        # Индивидуальные запросы для отсутствующих DOI
        missing_dois = set(dois) - set(results.keys())
        if missing_dois:
            if status_text:
                status_text.text("📡 Дозагрузка отсутствующих данных из OpenAlex...")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                future_to_doi = {
                    executor.submit(self._get_single_openalex_data, doi): doi 
                    for doi in missing_dois
                }
                
                for i, future in enumerate(concurrent.futures.as_completed(future_to_doi)):
                    doi = future_to_doi[future]
                    try:
                        data = future.result()
                        results[doi] = data
                    except Exception:
                        results[doi] = {}
                    
                    if progress_bar and len(missing_dois) > 0:
                        progress_bar.progress((len(dois) - len(missing_dois) + i + 1) / total_dois)
        
        return results

    def _get_single_openalex_data(self, doi: str) -> Dict:
        """Получает данные для одного DOI из OpenAlex"""
        if doi in self._openalex_cache:
            return self._openalex_cache[doi]
            
        for attempt in range(Config.MAX_RETRIES):
            try:
                url = f"https://api.openalex.org/works/https://doi.org/{doi}"
                response = self.session.get(url, timeout=Config.REQUEST_TIMEOUT)
                
                if response.status_code == 200:
                    data = response.json()
                    self._openalex_cache[doi] = data
                    return data
                    
            except Exception as e:
                if attempt == Config.MAX_RETRIES - 1:
                    return {}
                time.sleep(1)
                
        return {}

    def get_article_data_batch(self, dois: List[str], progress_container=None) -> Dict[str, Dict]:
        """Получает полные данные о статьях"""
        if progress_container:
            progress_bar = progress_container.progress(0)
            status_text = progress_container.empty()
        else:
            progress_bar = None
            status_text = None
        
        # Получаем данные из обоих источников параллельно
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            crossref_future = executor.submit(self.get_crossref_data_batch, dois, progress_bar, status_text)
            openalex_future = executor.submit(self.get_openalex_data_batch, dois, progress_bar, status_text)
            
            crossref_results = crossref_future.result()
            openalex_results = openalex_future.result()
        
        # Объединяем данные
        results = {}
        for doi in dois:
            crossref_data = crossref_results.get(doi, {})
            openalex_data = openalex_results.get(doi, {})
            altmetric_data = self.altmetric_processor.get_altmetric_metrics(doi)
            
            article_data = self._combine_article_data(doi, crossref_data, openalex_data, altmetric_data)
            results[doi] = article_data
        
        if progress_bar:
            progress_bar.progress(1.0)
        if status_text:
            status_text.text("✅ Данные статей получены!")
            
        return results

    def _combine_article_data(self, doi: str, crossref_data: Dict, openalex_data: Dict, altmetric_data: Dict) -> Dict[str, Any]:
        """Объединяет данные из разных источников"""
        # Заголовок
        title = 'Unknown'
        if openalex_data and openalex_data.get('title'):
            title = openalex_data['title']
        elif crossref_data.get('title'):
            title_list = crossref_data['title']
            if title_list:
                title = title_list[0]

        # Год публикации
        year = 'Unknown'
        publication_year = None
        if openalex_data and openalex_data.get('publication_year'):
            publication_year = openalex_data['publication_year']
            year = str(publication_year)
        elif crossref_data.get('issued', {}).get('date-parts', [[]])[0]:
            year = str(crossref_data['issued']['date-parts'][0][0])
            publication_year = int(year)

        # Авторы
        authors = []
        authors_with_initials = []
        if openalex_data:
            for author in openalex_data.get('authorships', []):
                name = author.get('author', {}).get('display_name', 'Unknown')
                if name != 'Unknown':
                    authors.append(name)
                    surname_with_initial = self._extract_surname_with_initial(name)
                    authors_with_initials.append(surname_with_initial)

        if not authors and crossref_data.get('author'):
            for author in crossref_data['author']:
                given = author.get('given', '')
                family = author.get('family', '')
                if given or family:
                    name = f"{given} {family}".strip()
                    authors.append(name)
                    surname_with_initial = self._extract_surname_with_initial(name)
                    authors_with_initials.append(surname_with_initial)

        authors_str = ', '.join(authors) if authors else 'Unknown'
        authors_with_initials_str = ', '.join(authors_with_initials) if authors_with_initials else 'Unknown'
        author_count = len(authors) if authors else 0

        # Информация о журнале
        journal_info = self._extract_journal_info(crossref_data)

        # Цитирования
        citation_count_crossref = crossref_data.get('is-referenced-by-count', 0)
        citation_count_openalex = openalex_data.get('cited_by_count', 0)

        # Аффилиации и страны
        affiliations, countries = self._get_enhanced_affiliations_and_countries(openalex_data, crossref_data)

        # Расчет лет с публикации
        current_year = datetime.now().year
        years_since_pub = self._calculate_years_since_publication(publication_year, current_year)

        return {
            'doi': doi,
            'title': title,
            'year': year,
            'publication_year': publication_year,
            'authors': authors_str,
            'authors_with_initials': authors_with_initials_str,
            'author_count': author_count,
            'journal_full_name': journal_info['full_name'],
            'journal_abbreviation': journal_info['abbreviation'],
            'publisher': journal_info['publisher'],
            'citation_count_crossref': citation_count_crossref,
            'citation_count_openalex': citation_count_openalex,
            'annual_citation_rate_crossref': self._safe_calculate_annual_citation_rate(citation_count_crossref, publication_year),
            'annual_citation_rate_openalex': self._safe_calculate_annual_citation_rate(citation_count_openalex, publication_year),
            'years_since_publication': years_since_pub,
            'affiliations': '; '.join(affiliations),
            'countries': countries,
            'altmetric_score': altmetric_data['altmetric_score'],
            'number_of_mentions': altmetric_data['cited_by_posts_count'],
            'x_mentions': altmetric_data['cited_by_tweeters_count'],
            'rss_blogs': altmetric_data['cited_by_feeds_count'],
            'unique_accounts': altmetric_data['cited_by_accounts_count']
        }

    def _extract_journal_info(self, crossref_data: Dict) -> Dict[str, Any]:
        """Извлекает информацию о журнале"""
        try:
            container_title = crossref_data.get('container-title', [])
            short_container_title = crossref_data.get('short-container-title', [])
            full_name = container_title[0] if container_title else (short_container_title[0] if short_container_title else 'Unknown')
            abbreviation = short_container_title[0] if short_container_title else (container_title[0] if container_title else 'Unknown')
            return {
                'full_name': full_name,
                'abbreviation': abbreviation,
                'publisher': crossref_data.get('publisher', 'Unknown'),
                'issn': crossref_data.get('ISSN', [None])[0]
            }
        except:
            return {
                'full_name': 'Unknown',
                'abbreviation': 'Unknown',
                'publisher': 'Unknown',
                'issn': None
            }

    def _get_enhanced_affiliations_and_countries(self, openalex_data: Dict, crossref_data: Dict) -> tuple[List[str], str]:
        """Улучшенная обработка аффилиаций с группировкой"""
        try:
            # Аффилиации из OpenAlex
            openalex_affiliations = set()
            openalex_countries = set()
            if openalex_data:
                for authorship in openalex_data.get('authorships', []):
                    for institution in authorship.get('institutions', []):
                        display_name = institution.get('display_name', '')
                        country_code = institution.get('country_code', '')
                        if display_name:
                            main_org = self.fast_affiliation_processor.extract_main_organization_fast(display_name)
                            if main_org and main_org != "Unknown":
                                openalex_affiliations.add(main_org)
                        if country_code:
                            openalex_countries.add(country_code.upper())

            # Аффилиации из Crossref
            crossref_affiliations = crossref_data.get('extracted_affiliations', [])
            crossref_countries = crossref_data.get('extracted_countries', [])

            # Объединяем аффилиации
            all_affiliations = list(openalex_affiliations) + crossref_affiliations
            if not all_affiliations:
                all_affiliations = ['Unknown']

            # Объединяем страны
            all_countries = openalex_countries.union(set(crossref_countries))
            final_countries = ';'.join(sorted(all_countries)) if all_countries else 'Unknown'

            return all_affiliations, final_countries

        except Exception as e:
            return ['Unknown'], 'Unknown'

    def _calculate_years_since_publication(self, publication_year: Any, current_year: int = None) -> int:
        """Безопасный расчет лет с момента публикации"""
        try:
            if current_year is None:
                current_year = datetime.now().year

            if publication_year is None or publication_year == 'Unknown':
                return 1

            year_str = str(publication_year).strip()
            if not year_str or year_str == 'Unknown':
                return 1

            year_match = re.search(r'\b(19|20)\d{2}\b', year_str)
            if year_match:
                year = int(year_match.group())
            else:
                year = int(year_str)

            if 1900 < year <= current_year:
                return max(1, current_year - year)
            else:
                return 1
        except (ValueError, TypeError):
            return 1

    def _safe_calculate_annual_citation_rate(self, citation_count, publication_year, current_year=None):
        """Безопасный расчет ежегодной цитируемости"""
        try:
            if not isinstance(citation_count, (int, float)) or citation_count == 0:
                return 0.0

            years = self._calculate_years_since_publication(publication_year, current_year)
            if years is None or not isinstance(years, (int, float)) or years <= 0:
                return 0.0

            return round(citation_count / years, 2)
        except (TypeError, ZeroDivisionError, ValueError):
            return 0.0

    def _extract_surname_with_initial(self, author_name: str) -> str:
        """Извлекает фамилию с инициалами"""
        if not author_name or author_name in ['Unknown', 'Error']:
            return author_name
        clean_name = re.sub(r'[^\w\s\-\.]', ' ', author_name).strip()
        parts = clean_name.split()
        if not parts:
            return author_name
        surname = parts[-1]
        initial = parts[0][0].upper() if parts[0] else ''
        return f"{surname} {initial}." if initial else surname

    def get_references_batch(self, doi_list: List[str], progress_container=None) -> Dict[str, List[Dict]]:
        """Получает ссылки для списка DOI"""
        all_references = {}
        total_articles = len(doi_list)
        
        if progress_container:
            progress_bar = progress_container.progress(0)
            status_text = progress_container.empty()
        else:
            progress_bar = None
            status_text = None
        
        if status_text:
            status_text.text("🔍 Сбор ссылок на статьи...")
        
        # Получаем данные Crossref для всех DOI
        crossref_data = self.get_crossref_data_batch(doi_list, progress_bar, status_text)
        
        for i, doi in enumerate(doi_list):
            data = crossref_data.get(doi, {})
            references = data.get('reference', [])
            all_references[doi] = references
            
            if progress_bar and total_articles > 0:
                progress_bar.progress((i + 1) / total_articles)
        
        if status_text:
            status_text.text("✅ Ссылки собраны!")
            
        return all_references

    def get_citing_articles_batch(self, doi_list: List[str], progress_container=None) -> Dict[str, List[str]]:
        """Получает цитирующие статьи для списка DOI"""
        all_citing_articles = {}
        total_articles = len(doi_list)
        
        if progress_container:
            progress_bar = progress_container.progress(0)
            status_text = progress_container.empty()
        else:
            progress_bar = None
            status_text = None
        
        if status_text:
            status_text.text("🔍 Поиск цитирующих статей...")
        
        # Получаем данные OpenAlex для поиска цитирований
        openalex_data = self.get_openalex_data_batch(doi_list, progress_bar, status_text)
        
        for i, doi in enumerate(doi_list):
            citing_dois = []
            data = openalex_data.get(doi, {})
            
            # Используем OpenAlex для поиска цитирований
            if data and 'cited_by_count' in data and data['cited_by_count'] > 0:
                work_id = data.get('id', '').split('/')[-1]
                if work_id:
                    try:
                        # Получаем все страницы цитирований
                        page = 1
                        per_page = 200
                        while len(citing_dois) < data['cited_by_count']:
                            citing_url = f"https://api.openalex.org/works?filter=cites:{work_id}&per-page={per_page}&page={page}"
                            response = self.session.get(citing_url, timeout=Config.REQUEST_TIMEOUT)
                            if response.status_code == 200:
                                citing_data = response.json()
                                results = citing_data.get('results', [])
                                
                                for work in results:
                                    if work.get('doi'):
                                        citing_dois.append(work['doi'])
                                
                                if len(results) < per_page:
                                    break
                                    
                                page += 1
                                time.sleep(Config.DELAY_BETWEEN_REQUESTS)
                            else:
                                break
                                
                    except Exception as e:
                        pass
            
            all_citing_articles[doi] = citing_dois
            
            if progress_bar and total_articles > 0:
                progress_bar.progress((i + 1) / total_articles)
        
        if status_text:
            status_text.text("✅ Цитирующие статьи найдены!")
            
        return all_citing_articles

    def analyze_references_comprehensive(self, doi_list: List[str]) -> pd.DataFrame:
        """Полный анализ ссылок"""
        st.info(f"🔍 Начинаем анализ ссылок для {len(doi_list)} статей...")
        
        # Контейнеры для прогресса
        main_progress = st.progress(0)
        main_status = st.empty()
        
        # Шаг 1: Получаем данные исходных статей
        main_status.text("📊 Шаг 1/4: Получение данных исходных статей...")
        source_articles_data = self.get_article_data_batch(doi_list, st.empty())
        main_progress.progress(0.25)
        
        # Шаг 2: Собираем все ссылки
        main_status.text("📊 Шаг 2/4: Сбор ссылок на статьи...")
        all_references = self.get_references_batch(doi_list, st.empty())
        main_progress.progress(0.5)
        
        # Шаг 3: Собираем все DOI ссылок
        main_status.text("📊 Шаг 3/4: Подготовка данных ссылок...")
        all_reference_dois = set()
        for references in all_references.values():
            for ref in references:
                ref_doi = ref.get('DOI')
                if ref_doi and self.validate_doi(ref_doi):
                    all_reference_dois.add(ref_doi)
        
        # Шаг 4: Получаем данные всех ссылок
        main_status.text("📊 Шаг 4/4: Получение данных ссылок...")
        if all_reference_dois:
            reference_articles_data = self.get_article_data_batch(list(all_reference_dois), st.empty())
        else:
            reference_articles_data = {}
        
        main_progress.progress(0.75)
        
        # Собираем все данные
        all_data = []
        
        # Добавляем исходные статьи
        for doi, data in source_articles_data.items():
            data['type'] = 'source'
            data['source_doi'] = doi
            all_data.append(data)
        
        # Добавляем ссылки
        for source_doi, references in all_references.items():
            for i, ref in enumerate(references):
                ref_doi = ref.get('DOI')
                if ref_doi and self.validate_doi(ref_doi):
                    ref_data = reference_articles_data.get(ref_doi, {})
                    if ref_data:
                        ref_data['type'] = 'reference'
                        ref_data['source_doi'] = source_doi
                        ref_data['position'] = i + 1
                        all_data.append(ref_data)
        
        main_progress.progress(1.0)
        main_status.text("✅ Анализ ссылок завершен!")
        
        return pd.DataFrame(all_data)

    def analyze_citations_comprehensive(self, doi_list: List[str]) -> pd.DataFrame:
        """Полный анализ цитирований"""
        st.info(f"🔍 Начинаем анализ цитирований для {len(doi_list)} статей...")
        
        # Контейнеры для прогресса
        main_progress = st.progress(0)
        main_status = st.empty()
        
        # Шаг 1: Получаем данные исходных статей
        main_status.text("📊 Шаг 1/4: Получение данных исходных статей...")
        source_articles_data = self.get_article_data_batch(doi_list, st.empty())
        main_progress.progress(0.25)
        
        # Шаг 2: Собираем все цитирующие статьи
        main_status.text("📊 Шаг 2/4: Поиск цитирующих статей...")
        all_citing_articles = self.get_citing_articles_batch(doi_list, st.empty())
        main_progress.progress(0.5)
        
        # Шаг 3: Собираем все DOI цитирующих статей
        main_status.text("📊 Шаг 3/4: Подготовка данных цитирований...")
        all_citing_dois = set()
        for citing_dois in all_citing_articles.values():
            all_citing_dois.update(citing_dois)
        
        # Шаг 4: Получаем данные всех цитирующих статей
        main_status.text("📊 Шаг 4/4: Получение данных цитирующих статей...")
        if all_citing_dois:
            citing_articles_data = self.get_article_data_batch(list(all_citing_dois), st.empty())
        else:
            citing_articles_data = {}
        
        main_progress.progress(0.75)
        
        # Собираем все данные
        all_data = []
        
        # Добавляем исходные статьи
        for doi, data in source_articles_data.items():
            data['type'] = 'source'
            data['source_doi'] = doi
            all_data.append(data)
        
        # Добавляем цитирующие статьи
        for source_doi, citing_dois in all_citing_articles.items():
            for citing_doi in citing_dois:
                if self.validate_doi(citing_doi):
                    citing_data = citing_articles_data.get(citing_doi, {})
                    if citing_data:
                        citing_data['type'] = 'citation'
                        citing_data['source_doi'] = source_doi
                        all_data.append(citing_data)
        
        main_progress.progress(1.0)
        main_status.text("✅ Анализ цитирований завершен!")
        
        return pd.DataFrame(all_data)

    # Методы анализа для отображения статистики
    def analyze_authors_frequency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Анализ частоты авторов"""
        if df.empty:
            return pd.DataFrame()
            
        try:
            authors_series = df['authors_with_initials'].str.split(',', expand=True).stack()
            authors_series = authors_series[authors_series.str.strip().isin(['Unknown', 'Error']) == False]
            author_freq = authors_series.value_counts().reset_index()
            author_freq.columns = ['author', 'frequency']
            author_freq['percentage'] = round(author_freq['frequency'] / len(df) * 100, 2)
            return author_freq
        except:
            return pd.DataFrame()

    def analyze_journals_frequency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Анализ частоты журналов"""
        if df.empty:
            return pd.DataFrame()
            
        try:
            journals = df['journal_abbreviation']
            journals = journals[journals.isin(['Unknown', 'Error']) == False]
            journal_freq = journals.value_counts().reset_index()
            journal_freq.columns = ['journal', 'frequency']
            journal_freq['percentage'] = round(journal_freq['frequency'] / len(df) * 100, 2)
            return journal_freq
        except:
            return pd.DataFrame()

    def analyze_countries_frequency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Анализ частоты стран"""
        if df.empty:
            return pd.DataFrame()
            
        try:
            countries_series = df['countries'].str.split(';', expand=True).stack()
            countries_series = countries_series[countries_series.str.strip().isin(['Unknown', 'Error']) == False]
            country_freq = countries_series.value_counts().reset_index()
            country_freq.columns = ['country', 'frequency']
            country_freq['percentage'] = round(country_freq['frequency'] / len(df) * 100, 2)
            return country_freq
        except:
            return pd.DataFrame()

    def analyze_year_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """Анализ распределения по годам"""
        if df.empty:
            return pd.DataFrame()
            
        try:
            years = pd.to_numeric(df['year'], errors='coerce')
            years = years[years.notna() & years.between(1900, datetime.now().year)]
            year_counts = years.value_counts().reset_index()
            year_counts.columns = ['year', 'frequency']
            year_counts['percentage'] = round(year_counts['frequency'] / len(df) * 100, 2)
            return year_counts.sort_values('year', ascending=False)
        except:
            return pd.DataFrame()

def main():
    st.title("📚 Refs/Cits Analysis - Full Version")
    st.markdown("🔬 Полнофункциональный анализ ссылок и цитирований научных статей")
    
    if not NLTK_AVAILABLE:
        st.warning("⚠️ NLTK не доступен. Используется упрощенная обработка текста. Для полного функционала установите nltk в requirements.txt")
    
    analyzer = FullCitationAnalyzer()
    
    # Боковая панель с информацией
    with st.sidebar:
        st.header("ℹ️ Информация")
        st.info("""
        **Полный функционал:**
        - Анализ всех ссылок и цитирований
        - Данные из Crossref, OpenAlex, Altmetric
        - Анализ аффилиаций и стран
        - Статистика по авторам и журналам
        - Альтметрические показатели
        - Без ограничений на количество данных
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Поддерживаемые форматы DOI")
        st.markdown("""
        - `10.1234/abcd.1234`
        - `https://doi.org/10.1234/abcd.1234`
        - `doi:10.1234/abcd.1234`
        """)
    
    # Основной контент
    tab1, tab2, tab3 = st.tabs(["📥 Ввод данных", "🔗 Анализ ссылок", "📈 Анализ цитирований"])
    
    with tab1:
        st.header("Ввод DOI статей")
        
        input_method = st.radio("Способ ввода:", ["📝 Текст", "📁 Файл"])
        
        if input_method == "📝 Текст":
            doi_input = st.text_area(
                "Введите DOI статей:",
                height=200,
                placeholder="Введите DOI через запятую, точку с запятой или с новой строки:\n10.1038/s41586-023-06924-6\n10.1126/science.abl8921\n10.1016/j.cell.2023.08.012\n10.1038/s41557-023-01282-2"
            )
            
            if st.button("🔍 Проверить DOI", key="validate"):
                if doi_input:
                    dois = analyzer.parse_doi_input(doi_input)
                    if dois:
                        st.success(f"✅ Найдено {len(dois)} валидных DOI:")
                        
                        # Показываем DOI в виде таблицы
                        doi_df = pd.DataFrame(dois, columns=['DOI'])
                        st.dataframe(doi_df, use_container_width=True)
                        
                        # Сохраняем в session state
                        st.session_state.dois = dois
                        st.session_state.dois_input = doi_input
                    else:
                        st.error("❌ Не найдено валидных DOI. Проверьте формат.")
        
        else:
            uploaded_file = st.file_uploader("Загрузите файл с DOI", type=['txt', 'csv'])
            if uploaded_file:
                content = uploaded_file.getvalue().decode()
                dois = analyzer.parse_doi_input(content)
                if dois:
                    st.success(f"✅ Найдено {len(dois)} валидных DOI в файле")
                    
                    # Показываем DOI в виде таблицы
                    doi_df = pd.DataFrame(dois, columns=['DOI'])
                    st.dataframe(doi_df, use_container_width=True)
                    
                    # Сохраняем в session state
                    st.session_state.dois = dois
                    st.session_state.dois_input = content
                else:
                    st.error("❌ Не найдено валидных DOI в файле.")
    
    with tab2:
        st.header("🔗 Полный анализ ссылок (References)")
        
        if 'dois' in st.session_state and st.session_state.dois:
            dois = st.session_state.dois
            
            st.info(f"🔍 Будет проанализировано {len(dois)} статей и ВСЕ их ссылки")
            
            if st.button("🚀 Запустить полный анализ ссылок", type="primary", key="refs_full"):
                with st.spinner("⚡ Выполняется полный анализ ссылок... Это может занять некоторое время"):
                    references_df = analyzer.analyze_references_comprehensive(dois)
                
                if not references_df.empty:
                    st.success("✅ Анализ завершен!")
                    
                    # Основная статистика
                    st.subheader("📊 Общая статистика")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        total_articles = len(references_df)
                        st.metric("Всего записей", total_articles)
                    with col2:
                        source_articles = len(references_df[references_df['type'] == 'source'])
                        st.metric("Исходных статей", source_articles)
                    with col3:
                        references_count = len(references_df[references_df['type'] == 'reference'])
                        st.metric("Ссылок", references_count)
                    with col4:
                        unique_dois = references_df['doi'].nunique()
                        st.metric("Уникальных DOI", unique_dois)
                    
                    # Дополнительная статистика
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        unique_journals = references_df['journal_abbreviation'].nunique()
                        st.metric("Уникальных журналов", unique_journals)
                    with col2:
                        unique_authors = len(analyzer.analyze_authors_frequency(references_df))
                        st.metric("Уникальных авторов", unique_authors)
                    with col3:
                        unique_countries = len(analyzer.analyze_countries_frequency(references_df))
                        st.metric("Уникальных стран", unique_countries)
                    with col4:
                        avg_citations = references_df['citation_count_openalex'].mean()
                        st.metric("Средние цитирования", f"{avg_citations:.1f}")
                    
                    # Визуализации
                    st.subheader("📈 Визуализации")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Распределение по годам
                        year_df = analyzer.analyze_year_distribution(references_df)
                        if not year_df.empty:
                            st.bar_chart(year_df.set_index('year')['frequency'].head(15))
                    
                    with col2:
                        # Топ журналов
                        journal_df = analyzer.analyze_journals_frequency(references_df)
                        if not journal_df.empty:
                            st.bar_chart(journal_df.set_index('journal')['frequency'].head(10))
                    
                    # Детальные данные
                    st.subheader("📋 Детальные данные")
                    
                    # Фильтры
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        record_type = st.selectbox("Тип записей:", ["Все", "Только исходные", "Только ссылки"])
                    with col2:
                        sort_by = st.selectbox("Сортировка:", ["doi", "year", "citation_count_openalex", "altmetric_score"])
                    with col3:
                        sort_order = st.selectbox("Порядок:", ["По убыванию", "По возрастанию"])
                    
                    # Применяем фильтры
                    filtered_df = references_df
                    if record_type == "Только исходные":
                        filtered_df = references_df[references_df['type'] == 'source']
                    elif record_type == "Только ссылки":
                        filtered_df = references_df[references_df['type'] == 'reference']
                    
                    # Сортировка
                    ascending = sort_order == "По возрастанию"
                    filtered_df = filtered_df.sort_values(sort_by, ascending=ascending)
                    
                    # Показываем данные
                    st.dataframe(filtered_df, use_container_width=True, height=400)
                    
                    # Статистические таблицы
                    st.subheader("📊 Статистические таблицы")
                    
                    tab1, tab2, tab3, tab4 = st.tabs(["Авторы", "Журналы", "Страны", "Годы"])
                    
                    with tab1:
                        authors_df = analyzer.analyze_authors_frequency(references_df)
                        if not authors_df.empty:
                            st.dataframe(authors_df.head(20), use_container_width=True)
                    
                    with tab2:
                        journals_df = analyzer.analyze_journals_frequency(references_df)
                        if not journals_df.empty:
                            st.dataframe(journals_df.head(20), use_container_width=True)
                    
                    with tab3:
                        countries_df = analyzer.analyze_countries_frequency(references_df)
                        if not countries_df.empty:
                            st.dataframe(countries_df.head(20), use_container_width=True)
                    
                    with tab4:
                        years_df = analyzer.analyze_year_distribution(references_df)
                        if not years_df.empty:
                            st.dataframe(years_df.head(20), use_container_width=True)
                    
                    # Экспорт
                    st.subheader("💾 Экспорт данных")
                    csv = references_df.to_csv(index=False)
                    st.download_button(
                        "📥 Скачать полные данные (CSV)",
                        csv,
                        "full_references_analysis.csv",
                        "text/csv",
                        key='download_full_refs'
                    )
                else:
                    st.error("❌ Не удалось получить данные")
        else:
            st.warning("⚠️ Сначала введите DOI на вкладке 'Ввод данных'")
    
    with tab3:
        st.header("📈 Полный анализ цитирований (Citations)")
        
        if 'dois' in st.session_state and st.session_state.dois:
            dois = st.session_state.dois
            
            st.info(f"🔍 Будет проанализировано {len(dois)} статей и ВСЕ их цитирования")
            
            if st.button("🚀 Запустить полный анализ цитирований", type="primary", key="cits_full"):
                with st.spinner("⚡ Выполняется полный анализ цитирований... Это может занять некоторое время"):
                    citations_df = analyzer.analyze_citations_comprehensive(dois)
                
                if not citations_df.empty:
                    st.success("✅ Анализ завершен!")
                    
                    # Основная статистика
                    st.subheader("📊 Общая статистика")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        total_articles = len(citations_df)
                        st.metric("Всего записей", total_articles)
                    with col2:
                        source_articles = len(citations_df[citations_df['type'] == 'source'])
                        st.metric("Исходных статей", source_articles)
                    with col3:
                        citations_count = len(citations_df[citations_df['type'] == 'citation'])
                        st.metric("Цитирований", citations_count)
                    with col4:
                        unique_dois = citations_df['doi'].nunique()
                        st.metric("Уникальных DOI", unique_dois)
                    
                    # Дополнительная статистика
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        citing_journals = citations_df[citations_df['type'] == 'citation']['journal_abbreviation'].nunique()
                        st.metric("Цитирующих журналов", citing_journals)
                    with col2:
                        citing_authors = len(analyzer.analyze_authors_frequency(citations_df[citations_df['type'] == 'citation']))
                        st.metric("Цитирующих авторов", citing_authors)
                    with col3:
                        citing_countries = len(analyzer.analyze_countries_frequency(citations_df[citations_df['type'] == 'citation']))
                        st.metric("Цитирующих стран", citing_countries)
                    with col4:
                        recent_year = str(datetime.now().year)
                        recent_citations = len(citations_df[
                            (citations_df['type'] == 'citation') & 
                            (citations_df['year'] == recent_year)
                        ])
                        st.metric(f"Цитирований {recent_year}", recent_citations)
                    
                    # Визуализации
                    st.subheader("📈 Визуализации")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Распределение по годам
                        year_df = analyzer.analyze_year_distribution(citations_df)
                        if not year_df.empty:
                            st.bar_chart(year_df.set_index('year')['frequency'].head(15))
                    
                    with col2:
                        # Топ цитирующих журналов
                        citing_journals_df = analyzer.analyze_journals_frequency(citations_df[citations_df['type'] == 'citation'])
                        if not citing_journals_df.empty:
                            st.bar_chart(citing_journals_df.set_index('journal')['frequency'].head(10))
                    
                    # Детальные данные
                    st.subheader("📋 Детальные данные")
                    
                    # Фильтры
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        record_type = st.selectbox("Тип записей:", ["Все", "Только исходные", "Только цитирования"], key='cit_type')
                    with col2:
                        sort_by = st.selectbox("Сортировка:", ["doi", "year", "citation_count_openalex", "altmetric_score"], key='cit_sort')
                    with col3:
                        sort_order = st.selectbox("Порядок:", ["По убыванию", "По возрастанию"], key='cit_order')
                    
                    # Применяем фильтры
                    filtered_df = citations_df
                    if record_type == "Только исходные":
                        filtered_df = citations_df[citations_df['type'] == 'source']
                    elif record_type == "Только цитирования":
                        filtered_df = citations_df[citations_df['type'] == 'citation']
                    
                    # Сортировка
                    ascending = sort_order == "По возрастанию"
                    filtered_df = filtered_df.sort_values(sort_by, ascending=ascending)
                    
                    # Показываем данные
                    st.dataframe(filtered_df, use_container_width=True, height=400)
                    
                    # Статистические таблицы
                    st.subheader("📊 Статистические таблицы")
                    
                    tab1, tab2, tab3, tab4 = st.tabs(["Цитирующие авторы", "Цитирующие журналы", "Цитирующие страны", "Годы цитирований"])
                    
                    with tab1:
                        citing_authors_df = analyzer.analyze_authors_frequency(citations_df[citations_df['type'] == 'citation'])
                        if not citing_authors_df.empty:
                            st.dataframe(citing_authors_df.head(20), use_container_width=True)
                    
                    with tab2:
                        citing_journals_df = analyzer.analyze_journals_frequency(citations_df[citations_df['type'] == 'citation'])
                        if not citing_journals_df.empty:
                            st.dataframe(citing_journals_df.head(20), use_container_width=True)
                    
                    with tab3:
                        citing_countries_df = analyzer.analyze_countries_frequency(citations_df[citations_df['type'] == 'citation'])
                        if not citing_countries_df.empty:
                            st.dataframe(citing_countries_df.head(20), use_container_width=True)
                    
                    with tab4:
                        citing_years_df = analyzer.analyze_year_distribution(citations_df[citations_df['type'] == 'citation'])
                        if not citing_years_df.empty:
                            st.dataframe(citing_years_df.head(20), use_container_width=True)
                    
                    # Экспорт
                    st.subheader("💾 Экспорт данных")
                    csv = citations_df.to_csv(index=False)
                    st.download_button(
                        "📥 Скачать полные данные (CSV)",
                        csv,
                        "full_citations_analysis.csv",
                        "text/csv",
                        key='download_full_cits'
                    )
                else:
                    st.error("❌ Не удалось получить данные")
        else:
            st.warning("⚠️ Сначала введите DOI на вкладке 'Ввод данных'")

if __name__ == "__main__":
    main()
