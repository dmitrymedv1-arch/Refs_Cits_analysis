import streamlit as st
import pandas as pd
import requests
import time
from typing import List, Dict, Any, Tuple, Set
import re
from collections import Counter
import os
import concurrent.futures
from functools import lru_cache
import json
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from ratelimit import limits, sleep_and_retry
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging
from io import BytesIO
import tempfile
import base64
from pathlib import Path

# Настройки
st.set_page_config(
    page_title="Refs/Cits Analysis - Full Professional Version",
    page_icon="📚",
    layout="wide"
)

# Создаем временную директорию для файлов
TEMP_DIR = tempfile.mkdtemp()
os.makedirs(TEMP_DIR, exist_ok=True)

def get_temp_file_path(filename: str) -> str:
    """Возвращает полный путь к временному файлу"""
    return os.path.join(TEMP_DIR, filename)

def create_download_link(file_path: str, filename: str, link_text: str) -> str:
    """Создает HTML ссылку для скачивания файла"""
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}" style="background-color: #4CAF50; color: white; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; border-radius: 5px; font-weight: bold;">{link_text}</a>'
        
        # ОТЛАДОЧНЫЙ ВЫВОД В КОНСОЛЬ
        print(f"🔗 Создана HTML ссылка для скачивания:")
        print(f"   Файл: {filename}")
        print(f"   Путь: {file_path}")
        print(f"   Размер файла: {len(data)} байт")
        print(f"   HTML ссылка (первые 200 символов): {href[:200]}...")
        
        return href
    except Exception as e:
        error_msg = f"Ошибка создания ссылки: {str(e)}"
        print(f"❌ {error_msg}")
        return f"<p>{error_msg}</p>"

def cleanup_temp_files():
    """Очищает временные файлы старше 1 часа"""
    try:
        current_time = time.time()
        for filename in os.listdir(TEMP_DIR):
            file_path = os.path.join(TEMP_DIR, filename)
            if os.path.isfile(file_path):
                file_time = os.path.getmtime(file_path)
                if current_time - file_time > 3600:  # 1 час
                    os.remove(file_path)
    except Exception as e:
        print(f"Ошибка при очистке временных файлов: {e}")

# Инициализация NLTK
def initialize_nltk():
    """Инициализация NLTK с обработкой ошибок"""
    try:
        nltk_data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        
        resources = {
            'corpora/stopwords': 'stopwords',
            'tokenizers/punkt': 'punkt'
        }
        
        for resource_path, resource_name in resources.items():
            try:
                nltk.data.find(resource_path)
            except LookupError:
                try:
                    nltk.download(resource_name, quiet=True)
                except Exception as e:
                    print(f"⚠️ Не удалось загрузить {resource_name}: {e}")
        
        return True
    except Exception as e:
        print(f"❌ Ошибка инициализации NLTK: {e}")
        return False

# Инициализируем NLTK
NLTK_AVAILABLE = initialize_nltk()

class Config:
    REQUEST_TIMEOUT = 30
    MAX_RETRIES = 5
    DELAY_BETWEEN_REQUESTS = 0.1
    RETRY_DELAY = 1

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
            main_org = clean_affiliation
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

    def normalize_organization_name(self, org_name: str) -> str:
        """Нормализует название организации для группировки"""
        if not org_name or org_name == "Unknown":
            return org_name

        normalized = org_name.lower()

        remove_patterns = [
            r'^the\s+', r'\s+the$',
            r'\bdept\.?\s+of\b', r'\bdepartment\s+of\b',
            r'\bfaculty\s+of\b', r'\bschool\s+of\b',
            r'\binstitute\s+of\b', r'\binstitution\s+of\b',
            r'\bcollege\s+of\b', r'\bacademy\s+of\b',
            r'\blaboratory\b', r'\blab\b',
            r'\bcenter\b', r'\bcentre\b',
            r'\bdivision\b', r'\bgroup\b',
            r'\binc\.?$', r'\bltd\.?$', r'\bcorp\.?$', r'\bco\.?$',
            r'\bllc\.?$', r'\bgmbh\.?$'
        ]

        for pattern in remove_patterns:
            normalized = re.sub(pattern, '', normalized)

        normalized = re.sub(r'\s+', ' ', normalized).strip()
        normalized = re.sub(r'[^\w\s&]', '', normalized)

        return normalized.strip()

    def group_similar_organizations(self, organizations: List[str]) -> Dict[str, List[str]]:
        """Группирует похожие организации"""
        if not organizations:
            return {}

        normalized_map = {}
        for org in organizations:
            if org != "Unknown":
                normalized = self.normalize_organization_name(org)
                if normalized:
                    if normalized not in normalized_map:
                        normalized_map[normalized] = []
                    normalized_map[normalized].append(org)

        final_groups = {}
        normalized_keys = list(normalized_map.keys())

        for i, key1 in enumerate(normalized_keys):
            if key1 not in final_groups:
                final_groups[key1] = []

            final_groups[key1].extend(normalized_map[key1])

            for j, key2 in enumerate(normalized_keys[i+1:], i+1):
                if self.are_organizations_similar(key1, key2):
                    if key2 in normalized_map:
                        final_groups[key1].extend(normalized_map[key2])
                    if key2 in final_groups:
                        del final_groups[key2]

        return final_groups

    def are_organizations_similar(self, org1: str, org2: str) -> bool:
        """Проверяет, являются ли две организации похожими"""
        if not org1 or not org2:
            return False

        org1_lower = org1.lower()
        org2_lower = org2.lower()

        if org1_lower in org2_lower or org2_lower in org1_lower:
            return True

        words1 = set(org1_lower.split())
        words2 = set(org2_lower.split())

        if not words1 or not words2:
            return False

        stop_words = {'the', 'and', 'of', 'for', 'in', 'on', 'at', 'to', 'by'}
        words1 = words1 - stop_words
        words2 = words2 - stop_words

        if not words1 or not words2:
            return False

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        similarity = len(intersection) / len(union) if union else 0

        return similarity > 0.6

    def process_affiliations_list_fast(self, affiliations: List[str]) -> Tuple[Dict[str, int], Dict[str, List[str]]]:
        """Быстрая обработка списка аффилиаций с группировкой"""
        if not affiliations:
            return {}, {}

        main_organizations = []
        for aff in affiliations:
            if aff and aff not in ['Unknown', 'Error']:
                main_org = self.extract_main_organization_fast(aff)
                if main_org and main_org != "Unknown":
                    main_organizations.append(main_org)

        if not main_organizations:
            return {}, {}

        grouped_organizations = self.group_similar_organizations(main_organizations)

        group_representatives = {}
        for normalized_name, org_list in grouped_organizations.items():
            if org_list:
                representative = max(org_list, key=len)
                group_representatives[representative] = org_list

        frequency_count = {}
        for representative, org_list in group_representatives.items():
            frequency_count[representative] = len(org_list)

        return frequency_count, group_representatives

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

    @retry(stop=stop_after_attempt(Config.MAX_RETRIES),
           wait=wait_exponential(multiplier=1, min=Config.RETRY_DELAY, max=10),
           retry=retry_if_exception_type((requests.exceptions.RequestException,
                                        requests.exceptions.Timeout,
                                        requests.exceptions.ConnectionError)))
    @sleep_and_retry
    @limits(calls=15, period=1)
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
        
        # Многоуровневые кеши
        self._crossref_cache = {}
        self._openalex_cache = {}
        self._article_data_cache = {}
        self._references_cache = {}
        self._citations_cache = {}
        self._unique_references_cache = {}
        self._unique_citations_cache = {}
        self.unique_ref_data_cache = {}
        self.unique_citation_data_cache = {}
        
        # Процессоры
        self.fast_affiliation_processor = FastAffiliationProcessor()
        self.altmetric_processor = AltmetricProcessor()
        
        # NLP компоненты
        if NLTK_AVAILABLE:
            try:
                self.stop_words = set(stopwords.words('english'))
                self.stemmer = PorterStemmer()
            except:
                self.stop_words = set()
                self.stemmer = None
        else:
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
        
        if self.stemmer:
            self.scientific_stopwords_stemmed = {self.stemmer.stem(word) for word in self.scientific_stopwords}
        else:
            self.scientific_stopwords_stemmed = self.scientific_stopwords

        # Настройка логирования
        self.setup_logging()

    def setup_logging(self):
        """Настройка логирования"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def invalidate_cache(self, cache_type: str = None):
        """Инвалидация кеша"""
        if cache_type == 'crossref' or cache_type is None:
            self._crossref_cache.clear()
        if cache_type == 'openalex' or cache_type is None:
            self._openalex_cache.clear()
        if cache_type == 'article_data' or cache_type is None:
            self._article_data_cache.clear()
        if cache_type == 'references' or cache_type is None:
            self._references_cache.clear()
        if cache_type == 'citations' or cache_type is None:
            self._citations_cache.clear()
        if cache_type == 'unique_refs' or cache_type is None:
            self._unique_references_cache.clear()
        if cache_type == 'unique_cits' or cache_type is None:
            self._unique_citations_cache.clear()

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
        """Расширенный парсинг ввода DOI"""
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

    @retry(stop=stop_after_attempt(Config.MAX_RETRIES),
           wait=wait_exponential(multiplier=1, min=Config.RETRY_DELAY, max=10),
           retry=retry_if_exception_type((requests.exceptions.RequestException,
                                        requests.exceptions.Timeout,
                                        requests.exceptions.ConnectionError)))
    @sleep_and_retry
    @limits(calls=15, period=1)
    def get_crossref_data_batch(self, dois: List[str]) -> Dict[str, Dict]:
        """Получает данные из Crossref для нескольких DOI сразу"""
        results = {}
        total_dois = len(dois)
        
        print(f"📡 Получение данных из Crossref для {total_dois} DOI...")
        
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
                    self.logger.warning(f"Crossref error for {doi}: {e}")
                    results[doi] = {}
                
                time.sleep(Config.DELAY_BETWEEN_REQUESTS)
        
        print(f"✅ Данные Crossref получены для {len(results)} DOI")
        return results

    def _get_single_crossref_data(self, doi: str) -> Dict:
        """Получает данные для одного DOI из Crossref"""
        if doi in self._crossref_cache:
            return self._crossref_cache[doi]
            
        try:
            url = f"https://api.crossref.org/works/{doi}"
            response = self.session.get(url, timeout=Config.REQUEST_TIMEOUT)
            
            if response.status_code == 200:
                data = response.json().get('message', {})
                
                affiliations, countries = self.extract_affiliations_from_crossref(data)
                data['extracted_affiliations'] = affiliations
                data['extracted_countries'] = countries
                
                self._crossref_cache[doi] = data
                return data
            elif response.status_code == 404:
                self.logger.warning(f"Crossref 404 for {doi}")
                return {}
            else:
                self.logger.warning(f"Crossref {response.status_code} for {doi}")
                return {}
                
        except requests.exceptions.Timeout:
            self.logger.warning(f"Crossref timeout for {doi}")
            return {}
        except requests.exceptions.ConnectionError:
            self.logger.warning(f"Crossref connection error for {doi}")
            return {}
        except Exception as e:
            self.logger.warning(f"Crossref unexpected error for {doi}: {e}")
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
            self.logger.debug(f"Error extracting affiliations from Crossref: {e}")

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
            self.logger.debug(f"Error extracting country from affiliation: {e}")

        return ""

    @retry(stop=stop_after_attempt(Config.MAX_RETRIES),
           wait=wait_exponential(multiplier=1, min=Config.RETRY_DELAY, max=10),
           retry=retry_if_exception_type((requests.exceptions.RequestException,
                                        requests.exceptions.Timeout,
                                        requests.exceptions.ConnectionError)))
    @sleep_and_retry
    @limits(calls=15, period=1)
    def get_openalex_data_batch(self, dois: List[str]) -> Dict[str, Dict]:
        """Получает данные из OpenAlex для нескольких DOI сразу"""
        results = {}
        total_dois = len(dois)
        
        print(f"📡 Получение данных из OpenAlex для {total_dois} DOI...")
        
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
                elif response.status_code == 429:
                    self.logger.warning("OpenAlex rate limit hit, waiting...")
                    time.sleep(5)
                    continue
                    
            except requests.exceptions.Timeout:
                self.logger.warning("OpenAlex timeout in batch request")
                continue
            except requests.exceptions.ConnectionError:
                self.logger.warning("OpenAlex connection error in batch request")
                continue
            except Exception as e:
                self.logger.warning(f"OpenAlex batch error: {e}")
                continue
                
            time.sleep(Config.DELAY_BETWEEN_REQUESTS)

        # Индивидуальные запросы для отсутствующих DOI
        missing_dois = set(dois) - set(results.keys())
        if missing_dois:
            print(f"📡 Дозагрузка {len(missing_dois)} отсутствующих данных из OpenAlex...")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
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
        
        print(f"✅ Данные OpenAlex получены для {len(results)} DOI")
        return results

    def _get_single_openalex_data(self, doi: str) -> Dict:
        """Получает данные для одного DOI из OpenAlex"""
        if doi in self._openalex_cache:
            return self._openalex_cache[doi]
            
        try:
            url = f"https://api.openalex.org/works/https://doi.org/{doi}"
            response = self.session.get(url, timeout=Config.REQUEST_TIMEOUT)
            
            if response.status_code == 200:
                data = response.json()
                self._openalex_cache[doi] = data
                return data
            elif response.status_code == 404:
                self.logger.warning(f"OpenAlex 404 for {doi}")
                return {}
            elif response.status_code == 429:
                self.logger.warning(f"OpenAlex rate limit for {doi}")
                time.sleep(2)
                return {}
            else:
                self.logger.warning(f"OpenAlex {response.status_code} for {doi}")
                return {}
                
        except requests.exceptions.Timeout:
            self.logger.warning(f"OpenAlex timeout for {doi}")
            return {}
        except requests.exceptions.ConnectionError:
            self.logger.warning(f"OpenAlex connection error for {doi}")
            return {}
        except Exception as e:
            self.logger.warning(f"OpenAlex unexpected error for {doi}: {e}")
            return {}

    def quick_doi_search(self, title: str) -> str:
        """Быстрый поиск DOI по заголовку (fallback метод)"""
        if not title or title == 'Unknown':
            return None

        url = "https://api.crossref.org/works"
        params = {
            'query.title': title,
            'rows': 1,
            'select': 'DOI,title'
        }

        try:
            response = requests.get(url, params=params, timeout=Config.REQUEST_TIMEOUT)
            if response.status_code == 200:
                data = response.json()
                if data['message']['items']:
                    doi = data['message']['items'][0]['DOI']
                    if self.validate_doi(doi):
                        return doi
            return None
        except:
            return None

    def get_article_data_batch(self, dois: List[str]) -> Dict[str, Dict]:
        """Получает полные данные о статьях"""
        print(f"📊 Получение данных для {len(dois)} статей...")
        
        # Получаем данные из обоих источников параллельно
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            crossref_future = executor.submit(self.get_crossref_data_batch, dois)
            openalex_future = executor.submit(self.get_openalex_data_batch, dois)
            
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
        
        print(f"✅ Данные статей получены для {len(results)} DOI")
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
        
        try:
            if openalex_data and openalex_data.get('publication_year'):
                publication_year = openalex_data['publication_year']
                year = str(publication_year)
            elif crossref_data.get('issued', {}).get('date-parts', [[]])[0]:
                year_str = str(crossref_data['issued']['date-parts'][0][0])
                if year_str.isdigit() and len(year_str) == 4:
                    publication_year = int(year_str)
                    year = year_str
                else:
                    year = year_str
                    publication_year = None
        except (ValueError, TypeError, IndexError, KeyError) as e:
            self.logger.debug(f"Error parsing year for {doi}: {e}")
            year = 'Unknown'
            publication_year = None

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
                try:
                    year = int(year_str)
                except ValueError:
                    return 1

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

            if publication_year is None:
                return float(citation_count)

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

    def get_references_batch(self, doi_list: List[str]) -> Dict[str, List[Dict]]:
        """Получает ссылки для списка DOI"""
        print(f"🔍 Сбор ссылок для {len(doi_list)} статей...")
        
        # Получаем данные Crossref для всех DOI
        crossref_data = self.get_crossref_data_batch(doi_list)
        
        all_references = {}
        for doi in doi_list:
            data = crossref_data.get(doi, {})
            references = data.get('reference', [])
            all_references[doi] = references
        
        print(f"✅ Ссылки собраны для {len(doi_list)} статей")
        return all_references

    def get_citing_articles_batch(self, doi_list: List[str]) -> Dict[str, List[str]]:
        """Получает цитирующие статьи для списка DOI"""
        print(f"🔍 Поиск цитирующих статей для {len(doi_list)} статей...")
        
        # Получаем данные OpenAlex для поиска цитирований
        openalex_data = self.get_openalex_data_batch(doi_list)
        
        all_citing_articles = {}
        for doi in doi_list:
            citing_dois = []
            data = openalex_data.get(doi, {})
            
            # Используем OpenAlex для поиска цитирований
            if data and 'cited_by_count' in data and data['cited_by_count'] > 0:
                work_id = data.get('id', '').split('/')[-1]
                if work_id:
                    try:
                        # Получаем все страницы цитирований с пагинацией
                        page = 1
                        per_page = 200
                        total_retrieved = 0
                        
                        while total_retrieved < data['cited_by_count']:
                            citing_url = f"https://api.openalex.org/works?filter=cites:{work_id}&per-page={per_page}&page={page}"
                            response = self.session.get(citing_url, timeout=Config.REQUEST_TIMEOUT)
                            
                            if response.status_code == 200:
                                citing_data = response.json()
                                results = citing_data.get('results', [])
                                total_retrieved += len(results)
                                
                                for work in results:
                                    if work.get('doi'):
                                        citing_dois.append(work['doi'])
                                
                                if len(results) < per_page:
                                    break
                                    
                                page += 1
                                time.sleep(Config.DELAY_BETWEEN_REQUESTS)
                            elif response.status_code == 429:
                                self.logger.warning("OpenAlex rate limit in citations, waiting...")
                                time.sleep(5)
                                continue
                            else:
                                break
                                
                    except Exception as e:
                        self.logger.warning(f"Error getting citations for {doi}: {e}")
            
            all_citing_articles[doi] = citing_dois
        
        print(f"✅ Цитирующие статьи найдены для {len(doi_list)} статей")
        return all_citing_articles

    # Анализ заголовков через NLTK
    def preprocess_content_words(self, text: str) -> List[str]:
        """Предобработка контент-слов из текста"""
        if not text or text in ['Unknown', 'Error']:
            return []
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s-]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        words = text.split()
        content_words = []
        for word in words:
            if '-' in word:
                continue
            if len(word) > 2 and word not in self.stop_words:
                if self.stemmer:
                    stemmed_word = self.stemmer.stem(word)
                    if stemmed_word not in self.scientific_stopwords_stemmed:
                        content_words.append(stemmed_word)
                else:
                    if word not in self.scientific_stopwords:
                        content_words.append(word)
        return content_words

    def extract_compound_words(self, text: str) -> List[str]:
        """Извлекает составные слова из текста"""
        if not text or text in ['Unknown', 'Error']:
            return []
        text = text.lower()
        compound_words = re.findall(r'\b[a-z]{2,}-[a-z]{2,}(?:-[a-z]{2,})*\b', text)
        return [word for word in compound_words if not any(part in self.stop_words for part in word.split('-'))]

    def extract_scientific_stopwords(self, text: str) -> List[str]:
        """Извлекает научные стоп-слова из текста"""
        if not text or text in ['Unknown', 'Error']:
            return []
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        words = text.split()
        scientific_words = []
        for word in words:
            if len(word) > 2:
                if self.stemmer:
                    stemmed_word = self.stemmer.stem(word)
                    if stemmed_word in self.scientific_stopwords_stemmed:
                        for original_word in self.scientific_stopwords:
                            if self.stemmer.stem(original_word) == stemmed_word:
                                scientific_words.append(original_word)
                                break
                else:
                    if word in self.scientific_stopwords:
                        scientific_words.append(word)
        return scientific_words

    def analyze_titles(self, titles: List[str]) -> tuple[Counter, Counter, Counter]:
        """Анализ заголовков через NLTK"""
        content_words = []
        compound_words = []
        scientific_words = []
        valid_titles = [t for t in titles if t not in ['Unknown', 'Error']]
        for title in valid_titles:
            content_words.extend(self.preprocess_content_words(title))
            compound_words.extend(self.extract_compound_words(title))
            scientific_words.extend(self.extract_scientific_stopwords(title))
        return Counter(content_words), Counter(compound_words), Counter(scientific_words)

    # Многоуровневые кеши для уникальных записей
    def get_unique_references(self, references_df: pd.DataFrame) -> pd.DataFrame:
        """Получает уникальные ссылки"""
        if references_df.empty:
            return pd.DataFrame()

        cache_key = str(references_df.shape) + str(hash(str(references_df.columns.tolist())))
        if cache_key not in self._unique_references_cache:
            references_df['ref_id'] = references_df['doi'].fillna('') + '|' + references_df['title'].fillna('')
            unique_df = references_df.drop_duplicates(subset=['ref_id'], keep='first').drop(columns=['ref_id'])
            self._unique_references_cache[cache_key] = unique_df
        return self._unique_references_cache[cache_key]

    def get_unique_citations(self, citations_df: pd.DataFrame) -> pd.DataFrame:
        """Получает уникальные цитирования"""
        if citations_df.empty:
            return pd.DataFrame()

        cache_key = str(citations_df.shape) + str(hash(str(citations_df.columns.tolist())))
        if cache_key not in self._unique_citations_cache:
            citations_df['citation_id'] = citations_df['doi'].fillna('') + '|' + citations_df['title'].fillna('')
            unique_df = citations_df.drop_duplicates(subset=['citation_id'], keep='first').drop(columns=['citation_id'])
            self._unique_citations_cache[cache_key] = unique_df
        return self._unique_citations_cache[cache_key]

    # Расширенная статистика
    def analyze_authors_frequency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Анализ частоты авторов"""
        if df.empty:
            return pd.DataFrame()
            
        try:
            total_refs = len(df)
            unique_df = self.get_unique_references(df) if 'source_doi' in df.columns else df
            total_unique = len(unique_df)
            
            authors_total = df['authors_with_initials'].str.split(',', expand=True).stack()
            authors_total = authors_total[authors_total.str.strip().isin(['Unknown', 'Error']) == False]
            author_freq_total = authors_total.value_counts().reset_index()
            author_freq_total.columns = ['author_with_initial', 'frequency_total']
            author_freq_total['percentage_total'] = round(author_freq_total['frequency_total'] / total_refs * 100, 2)

            authors_unique = unique_df['authors_with_initials'].str.split(',', expand=True).stack()
            authors_unique = authors_unique[authors_unique.str.strip().isin(['Unknown', 'Error']) == False]
            author_freq_unique = authors_unique.value_counts().reset_index()
            author_freq_unique.columns = ['author_with_initial', 'frequency_unique']
            author_freq_unique['percentage_unique'] = round(author_freq_unique['frequency_unique'] / total_unique * 100, 2)

            author_freq = author_freq_total.merge(author_freq_unique, on='author_with_initial', how='outer').fillna(0)
            return author_freq[['author_with_initial', 'frequency_total', 'percentage_total', 'frequency_unique', 'percentage_unique']].sort_values('frequency_total', ascending=False)
        except:
            return pd.DataFrame()

    def analyze_journals_frequency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Анализ частоты журналов"""
        if df.empty:
            return pd.DataFrame()
            
        try:
            total_refs = len(df)
            unique_df = self.get_unique_references(df) if 'source_doi' in df.columns else df
            total_unique = len(unique_df)
            
            journals_total = df['journal_abbreviation']
            journals_total = journals_total[journals_total.isin(['Unknown', 'Error']) == False]
            journal_freq_total = journals_total.value_counts().reset_index()
            journal_freq_total.columns = ['journal_abbreviation', 'frequency_total']
            journal_freq_total['percentage_total'] = round(journal_freq_total['frequency_total'] / total_refs * 100, 2)

            journals_unique = unique_df['journal_abbreviation']
            journals_unique = journals_unique[journals_unique.isin(['Unknown', 'Error']) == False]
            journal_freq_unique = journals_unique.value_counts().reset_index()
            journal_freq_unique.columns = ['journal_abbreviation', 'frequency_unique']
            journal_freq_unique['percentage_unique'] = round(journal_freq_unique['frequency_unique'] / total_unique * 100, 2)

            journal_freq = journal_freq_total.merge(journal_freq_unique, on='journal_abbreviation', how='outer').fillna(0)

            # Дополнительные метрики для цитирований
            journal_citation_metrics = []
            for journal in journal_freq['journal_abbreviation']:
                journal_articles = unique_df[unique_df['journal_abbreviation'] == journal]

                total_crossref_citations = journal_articles['citation_count_crossref'].sum()
                total_openalex_citations = journal_articles['citation_count_openalex'].sum()

                avg_crossref_citations = journal_articles['citation_count_crossref'].mean() if len(journal_articles) > 0 else 0
                avg_openalex_citations = journal_articles['citation_count_openalex'].mean() if len(journal_articles) > 0 else 0

                journal_citation_metrics.append({
                    'journal_abbreviation': journal,
                    'total_crossref_citations': total_crossref_citations,
                    'total_openalex_citations': total_openalex_citations,
                    'avg_crossref_citations': round(avg_crossref_citations, 2),
                    'avg_openalex_citations': round(avg_openalex_citations, 2)
                })

            journal_metrics_df = pd.DataFrame(journal_citation_metrics)
            journal_freq = journal_freq.merge(journal_metrics_df, on='journal_abbreviation', how='left').fillna(0)

            return journal_freq[['journal_abbreviation', 'frequency_total', 'percentage_total',
                               'frequency_unique', 'percentage_unique', 'total_crossref_citations',
                               'total_openalex_citations', 'avg_crossref_citations', 'avg_openalex_citations']].sort_values('frequency_total', ascending=False)
        except:
            return pd.DataFrame()

    def analyze_affiliations_frequency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Анализ частоты аффилиаций"""
        if df.empty:
            return pd.DataFrame()

        try:
            total_refs = len(df)
            unique_df = self.get_unique_references(df) if 'source_doi' in df.columns else df
            total_unique = len(unique_df)

            all_affiliations = []
            for affil_string in df['affiliations']:
                if pd.isna(affil_string) or affil_string in ['Unknown', 'Error', '']:
                    continue
                try:
                    affil_list = affil_string.split(';')
                    for affil in affil_list:
                        clean_affil = affil.strip()
                        if clean_affil and clean_affil not in ['Unknown', 'Error']:
                            all_affiliations.append(clean_affil)
                except:
                    pass

            if not all_affiliations:
                return pd.DataFrame(columns=['affiliation', 'frequency_total', 'percentage_total', 'frequency_unique', 'percentage_unique'])

            affiliation_frequencies, grouped_organizations = self.fast_affiliation_processor.process_affiliations_list_fast(all_affiliations)

            affil_data = []
            for affil, freq in affiliation_frequencies.items():
                percentage_total = round(freq / total_refs * 100, 2) if total_refs > 0 else 0
                affil_data.append({
                    'affiliation': affil,
                    'frequency_total': freq,
                    'percentage_total': percentage_total
                })

            unique_affiliations = []
            for affil_string in unique_df['affiliations']:
                if affil_string and affil_string not in ['Unknown', 'Error']:
                    affil_list = affil_string.split(';')
                    unique_affiliations.extend([affil.strip() for affil in affil_list if affil.strip()])

            if unique_affiliations:
                unique_frequencies, _ = self.fast_affiliation_processor.process_affiliations_list_fast(unique_affiliations)
                affil_df = pd.DataFrame(affil_data)
                for affil, freq in unique_frequencies.items():
                    percentage_unique = round(freq / total_unique * 100, 2) if total_unique > 0 else 0
                    if affil in affil_df['affiliation'].values:
                        affil_df.loc[affil_df['affiliation'] == affil, 'frequency_unique'] = freq
                        affil_df.loc[affil_df['affiliation'] == affil, 'percentage_unique'] = percentage_unique
                    else:
                        affil_df = pd.concat([affil_df, pd.DataFrame([{
                            'affiliation': affil,
                            'frequency_total': 0,
                            'percentage_total': 0,
                            'frequency_unique': freq,
                            'percentage_unique': percentage_unique
                        }])], ignore_index=True)
            else:
                affil_df = pd.DataFrame(affil_data)
                affil_df['frequency_unique'] = affil_df['frequency_total']
                affil_df['percentage_unique'] = affil_df['percentage_total']

            affil_df = affil_df.fillna(0)
            return affil_df[['affiliation', 'frequency_total', 'percentage_total', 'frequency_unique', 'percentage_unique']].sort_values('frequency_total', ascending=False)

        except:
            return pd.DataFrame()

    def analyze_countries_frequency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Анализ частоты стран"""
        if df.empty:
            return pd.DataFrame()

        try:
            total_refs = len(df)
            unique_df = self.get_unique_references(df) if 'source_doi' in df.columns else df
            total_unique = len(unique_df)

            # Анализ отдельных стран
            single_countries = []
            collaborations = []

            for countries in df['countries']:
                if countries not in ['Unknown', 'Error']:
                    country_list = [c.strip() for c in countries.split(';')]
                    if len(country_list) == 1:
                        single_countries.extend(country_list)
                    else:
                        collaborations.append(countries)

            # Частотность отдельных стран
            single_country_counter = Counter(single_countries)
            single_country_freq = pd.DataFrame({
                'countries': list(single_country_counter.keys()),
                'type': ['single'] * len(single_country_counter),
                'frequency_total': list(single_country_counter.values())
            })
            single_country_freq['percentage_total'] = round(single_country_freq['frequency_total'] / total_refs * 100, 2)

            # Частотность коллабораций
            collaboration_counter = Counter(collaborations)
            collaboration_freq = pd.DataFrame({
                'countries': list(collaboration_counter.keys()),
                'type': ['collaboration'] * len(collaboration_counter),
                'frequency_total': list(collaboration_counter.values())
            })
            collaboration_freq['percentage_total'] = round(collaboration_freq['frequency_total'] / total_refs * 100, 2)

            # Объединяем
            country_freq_total = pd.concat([single_country_freq, collaboration_freq], ignore_index=True)

            # Аналогично для уникальных статей
            single_countries_unique = []
            collaborations_unique = []

            for countries in unique_df['countries']:
                if countries not in ['Unknown', 'Error']:
                    country_list = [c.strip() for c in countries.split(';')]
                    if len(country_list) == 1:
                        single_countries_unique.extend(country_list)
                    else:
                        collaborations_unique.append(countries)

            single_country_counter_unique = Counter(single_countries_unique)
            collaboration_counter_unique = Counter(collaborations_unique)

            single_country_freq_unique = pd.DataFrame({
                'countries': list(single_country_counter_unique.keys()),
                'frequency_unique': list(single_country_counter_unique.values())
            })
            single_country_freq_unique['percentage_unique'] = round(single_country_freq_unique['frequency_unique'] / total_unique * 100, 2)

            collaboration_freq_unique = pd.DataFrame({
                'countries': list(collaboration_counter_unique.keys()),
                'frequency_unique': list(collaboration_counter_unique.values())
            })
            collaboration_freq_unique['percentage_unique'] = round(collaboration_freq_unique['frequency_unique'] / total_unique * 100, 2)

            country_freq_unique = pd.concat([
                single_country_freq_unique.assign(type='single'),
                collaboration_freq_unique.assign(type='collaboration')
            ], ignore_index=True)

            country_freq = country_freq_total.merge(country_freq_unique, on=['countries', 'type'], how='outer').fillna(0)

            return country_freq[['countries', 'type', 'frequency_total', 'percentage_total', 'frequency_unique', 'percentage_unique']].sort_values('frequency_total', ascending=False)
        except:
            return pd.DataFrame()

    def analyze_year_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """Анализ распределения по годам"""
        if df.empty:
            return pd.DataFrame()

        try:
            total_refs = len(df)
            unique_df = self.get_unique_references(df) if 'source_doi' in df.columns else df
            total_unique = len(unique_df)
            
            years_total = pd.to_numeric(df['year'], errors='coerce')
            years_total = years_total[years_total.notna() & years_total.between(1900, 2026)].astype(int)
            year_counts_total = years_total.value_counts().reset_index()
            year_counts_total.columns = ['year', 'frequency_total']
            year_counts_total['percentage_total'] = round(year_counts_total['frequency_total'] / total_refs * 100, 2)

            years_unique = pd.to_numeric(unique_df['year'], errors='coerce')
            years_unique = years_unique[years_unique.notna() & years_unique.between(1900, 2026)].astype(int)
            year_counts_unique = years_unique.value_counts().reset_index()
            year_counts_unique.columns = ['year', 'frequency_unique']
            year_counts_unique['percentage_unique'] = round(year_counts_unique['frequency_unique'] / total_unique * 100, 2)

            year_counts = year_counts_total.merge(year_counts_unique, on='year', how='outer').fillna(0)
            return year_counts[['year', 'frequency_total', 'percentage_total', 'frequency_unique', 'percentage_unique']].sort_values('year', ascending=False)
        except:
            return pd.DataFrame()

    def analyze_five_year_periods(self, df: pd.DataFrame) -> pd.DataFrame:
        """Анализ пятилетних периодов"""
        if df.empty:
            return pd.DataFrame()

        try:
            total_refs = len(df)
            unique_df = self.get_unique_references(df) if 'source_doi' in df.columns else df
            total_unique = len(unique_df)

            start_year = 1900
            current_year = datetime.now().year + 4
            period_starts = list(range(start_year, current_year + 1, 5))
            bins = period_starts + [period_starts[-1] + 5]
            labels = [f"{s}-{s+4}" for s in period_starts]

            years_total = pd.to_numeric(df['year'], errors='coerce')
            years_total = years_total[years_total.notna() & years_total.between(1900, current_year)].astype(int)
            period_counts_total = pd.cut(years_total, bins=bins, labels=labels, right=False).astype(str)
            period_df_total = period_counts_total.value_counts().reset_index()
            period_df_total.columns = ['period', 'frequency_total']
            period_df_total['percentage_total'] = round(period_df_total['frequency_total'] / total_refs * 100, 2)
            period_df_total['period'] = period_df_total['period'].astype(str)

            years_unique = pd.to_numeric(unique_df['year'], errors='coerce')
            years_unique = years_unique[years_unique.notna() & years_unique.between(1900, current_year)].astype(int)
            period_counts_unique = pd.cut(years_unique, bins=bins, labels=labels, right=False).astype(str)
            period_df_unique = period_counts_unique.value_counts().reset_index()
            period_df_unique.columns = ['period', 'frequency_unique']
            period_df_unique['percentage_unique'] = round(period_df_unique['frequency_unique'] / total_unique * 100, 2)
            period_df_unique['period'] = period_df_unique['period'].astype(str)

            period_df = period_df_total.merge(period_df_unique, on='period', how='outer').fillna(0)
            return period_df[['period', 'frequency_total', 'percentage_total', 'frequency_unique', 'percentage_unique']].sort_values('period', ascending=False)
        except:
            return pd.DataFrame()

    def find_duplicate_references(self, references_df: pd.DataFrame) -> pd.DataFrame:
        """Находит дублирующиеся ссылки"""
        try:
            if references_df.empty:
                return pd.DataFrame()

            references_df['ref_id'] = references_df['doi'].fillna('') + '|' + references_df['title'].fillna('')
            ref_counts = references_df.groupby('ref_id')['source_doi'].nunique().reset_index()
            duplicate_ref_ids = ref_counts[ref_counts['source_doi'] > 1]['ref_id']
            if duplicate_ref_ids.empty:
                columns = list(references_df.columns) + ['frequency']
                columns.remove('ref_id')
                return pd.DataFrame(columns=columns)
            frequency_map = references_df['ref_id'].value_counts().to_dict()
            duplicates = references_df[references_df['ref_id'].isin(duplicate_ref_ids)].copy()
            duplicates = duplicates.drop_duplicates(subset=['ref_id'], keep='first')
            duplicates = duplicates[~((duplicates['doi'].isna()) & (duplicates['title'] == 'Unknown'))]
            duplicates['frequency'] = duplicates['ref_id'].map(frequency_map)
            duplicates = duplicates.drop(columns=['ref_id'])
            return duplicates.sort_values(['frequency', 'doi'], ascending=[False, True])
        except:
            return pd.DataFrame()

    def find_duplicate_citations(self, citations_df: pd.DataFrame) -> pd.DataFrame:
        """Находит дублирующиеся цитирования"""
        try:
            if citations_df.empty:
                return pd.DataFrame()

            citations_df['citation_id'] = citations_df['doi'].fillna('') + '|' + citations_df['title'].fillna('')

            citation_counts = citations_df.groupby('citation_id')['source_doi'].nunique().reset_index()
            duplicate_citation_ids = citation_counts[citation_counts['source_doi'] > 1]['citation_id']

            if duplicate_citation_ids.empty:
                columns = list(citations_df.columns) + ['frequency']
                columns.remove('citation_id')
                return pd.DataFrame(columns=columns)

            frequency_map = citation_counts.set_index('citation_id')['source_doi'].to_dict()

            duplicates = citations_df[citations_df['citation_id'].isin(duplicate_citation_ids)].copy()
            duplicates = duplicates.drop_duplicates(subset=['citation_id'], keep='first')

            duplicates = duplicates[~((duplicates['doi'].isna()) & (duplicates['title'] == 'Unknown'))]

            duplicates['frequency'] = duplicates['citation_id'].map(frequency_map)
            duplicates = duplicates.drop(columns=['citation_id'])

            return duplicates.sort_values(['frequency', 'doi'], ascending=[False, True])
        except:
            return pd.DataFrame()

    # Основные методы анализа с улучшенным прогрессом
    def analyze_references_comprehensive(self, doi_list: List[str]) -> pd.DataFrame:
        """Полный анализ ссылок с детальным прогрессом"""
        st.info(f"🔍 Начинаем анализ ссылок для {len(doi_list)} статей...")
        
        # Контейнеры для прогресса
        main_progress = st.progress(0)
        main_status = st.empty()
        step_progress = st.empty()
        step_status = st.empty()
        
        # Шаг 1: Получаем данные исходных статей
        main_status.text("📊 Шаг 1/5: Получение данных исходных статей...")
        step_status.text("Получение метаданных статей...")
        source_articles_data = self.get_article_data_batch(doi_list)
        main_progress.progress(0.2)
        
        # Шаг 2: Собираем все ссылки
        main_status.text("📊 Шаг 2/5: Сбор ссылок на статьи...")
        step_status.text("Сбор библиографии...")
        all_references = self.get_references_batch(doi_list)
        main_progress.progress(0.4)
        
        # Шаг 3: Собираем все DOI ссылок
        main_status.text("📊 Шаг 3/5: Подготовка данных ссылок...")
        step_status.text("Извлечение DOI из ссылок...")
        all_reference_dois = set()
        reference_titles = []
        
        total_refs = sum(len(refs) for refs in all_references.values())
        processed_refs = 0
        
        # ИСПРАВЛЕНИЕ: используем .items() вместо .values()
        for source_doi, references in all_references.items():
            for ref in references:
                ref_doi = ref.get('DOI')
                title = ref.get('article-title', 'Unknown')
                reference_titles.append(title)
                
                if ref_doi and self.validate_doi(ref_doi):
                    all_reference_dois.add(ref_doi)
                processed_refs += 1
                
                # Обновляем прогресс каждые 10 ссылок
                if processed_refs % 10 == 0:
                    progress_pct = processed_refs / total_refs * 100
                    step_progress.progress(progress_pct / 100)
                    step_status.text(f"Обработано {processed_refs}/{total_refs} ссылок ({progress_pct:.1f}%)")
        
        main_progress.progress(0.6)
        
        # Шаг 4: Поиск DOI по заголовкам для ссылок без DOI
        main_status.text("📊 Шаг 4/5: Поиск DOI по заголовкам...")
        step_status.text("Поиск недостающих DOI...")
        
        titles_to_search = [title for title in reference_titles if title != 'Unknown']
        total_titles = len(titles_to_search)
        found_dois = 0
        
        for i, title in enumerate(titles_to_search):
            found_doi = self.quick_doi_search(title)
            if found_doi:
                all_reference_dois.add(found_doi)
                found_dois += 1
            
            # Обновляем прогресс
            if i % 5 == 0:
                progress_pct = (i + 1) / total_titles * 100
                step_progress.progress(progress_pct / 100)
                step_status.text(f"Найдено DOI: {found_dois}/{i+1} ({progress_pct:.1f}%)")
            
            time.sleep(0.1)  # Чтобы не перегружать API
        
        main_progress.progress(0.8)
        
        # Шаг 5: Получаем данные всех ссылок
        main_status.text("📊 Шаг 5/5: Получение данных ссылок...")
        step_status.text("Загрузка метаданных ссылок...")
        
        if all_reference_dois:
            reference_dois_list = list(all_reference_dois)
            total_ref_dois = len(reference_dois_list)
            
            # Разбиваем на батчи для лучшего отображения прогресса
            batch_size = 50
            reference_articles_data = {}
            
            for i in range(0, total_ref_dois, batch_size):
                batch_dois = reference_dois_list[i:i + batch_size]
                batch_data = self.get_article_data_batch(batch_dois)
                reference_articles_data.update(batch_data)
                
                # Обновляем прогресс
                progress_pct = min((i + batch_size) / total_ref_dois * 100, 100)
                step_progress.progress(progress_pct / 100)
                step_status.text(f"Загружено {min(i + batch_size, total_ref_dois)}/{total_ref_dois} ссылок ({progress_pct:.1f}%)")
        else:
            reference_articles_data = {}
        
        main_progress.progress(1.0)
        
        # Собираем все данные
        main_status.text("📊 Сборка финального датасета...")
        step_status.text("Объединение данных...")
        
        all_data = []
        
        # Добавляем исходные статьи
        for doi, data in source_articles_data.items():
            data['type'] = 'source'
            data['source_doi'] = doi
            all_data.append(data)
        
        # Добавляем ссылки
        processed_connections = 0
        total_connections = sum(len(refs) for refs in all_references.values())
        
        # ИСПРАВЛЕНИЕ: снова используем .items() для итерации
        for source_doi, references in all_references.items():
            for i, ref in enumerate(references):
                ref_doi = ref.get('DOI')
                title = ref.get('article-title', 'Unknown')
                
                if ref_doi and self.validate_doi(ref_doi):
                    ref_data = reference_articles_data.get(ref_doi, {})
                    if ref_data:
                        ref_data['type'] = 'reference'
                        ref_data['source_doi'] = source_doi
                        ref_data['position'] = i + 1
                        all_data.append(ref_data)
                else:
                    # Fallback обработка
                    found_doi = self.quick_doi_search(title)
                    if found_doi and found_doi in reference_articles_data:
                        ref_data = reference_articles_data[found_doi].copy()
                        ref_data['type'] = 'reference'
                        ref_data['source_doi'] = source_doi
                        ref_data['position'] = i + 1
                        all_data.append(ref_data)
                    else:
                        # Базовая запись для ссылки без DOI
                        all_data.append({
                            'source_doi': source_doi,
                            'position': i + 1,
                            'doi': ref_doi,
                            'title': title,
                            'authors': 'Unknown',
                            'authors_with_initials': 'Unknown',
                            'author_count': 0,
                            'year': ref.get('year', 'Unknown'),
                            'journal_full_name': 'Unknown',
                            'journal_abbreviation': 'Unknown',
                            'publisher': 'Unknown',
                            'citation_count_crossref': 0,
                            'citation_count_openalex': 0,
                            'annual_citation_rate_crossref': 0,
                            'annual_citation_rate_openalex': 0,
                            'years_since_publication': 1,
                            'affiliations': 'Unknown',
                            'countries': 'Unknown',
                            'altmetric_score': 0,
                            'number_of_mentions': 0,
                            'x_mentions': 0,
                            'rss_blogs': 0,
                            'unique_accounts': 0,
                            'type': 'reference',
                            'error': f"Invalid DOI: {ref_doi}" if ref_doi else "No DOI found"
                        })
                
                processed_connections += 1
                if processed_connections % 10 == 0:
                    progress_pct = processed_connections / total_connections * 100
                    step_progress.progress(progress_pct / 100)
                    step_status.text(f"Обработано связей: {processed_connections}/{total_connections} ({progress_pct:.1f}%)")
        
        main_status.text("✅ Анализ ссылок завершен!")
        step_status.text("")
        step_progress.empty()
        
        # Создаем DataFrame и добавляем недостающие колонки
        result_df = pd.DataFrame(all_data)
        
        # Добавляем недостающие колонки, если их нет
        expected_columns = [
            'doi', 'title', 'year', 'publication_year', 'authors', 'authors_with_initials', 
            'author_count', 'journal_full_name', 'journal_abbreviation', 'publisher',
            'citation_count_crossref', 'citation_count_openalex', 'annual_citation_rate_crossref',
            'annual_citation_rate_openalex', 'years_since_publication', 'affiliations', 
            'countries', 'altmetric_score', 'number_of_mentions', 'x_mentions', 'rss_blogs',
            'unique_accounts', 'type', 'source_doi', 'position', 'error'
        ]
        
        for col in expected_columns:
            if col not in result_df.columns:
                result_df[col] = None
        
        return result_df

    def analyze_citations_comprehensive(self, doi_list: List[str]) -> pd.DataFrame:
        """Полный анализ цитирований с детальным прогрессом"""
        st.info(f"🔍 Начинаем анализ цитирований для {len(doi_list)} статей...")
        
        # Контейнеры для прогресса
        main_progress = st.progress(0)
        main_status = st.empty()
        step_progress = st.empty()
        step_status = st.empty()
        
        # Шаг 1: Получаем данные исходных статей
        main_status.text("📊 Шаг 1/4: Получение данных исходных статей...")
        step_status.text("Получение метаданных статей...")
        source_articles_data = self.get_article_data_batch(doi_list)
        main_progress.progress(0.25)
        
        # Шаг 2: Собираем все цитирующие статьи
        main_status.text("📊 Шаг 2/4: Поиск цитирующих статей...")
        step_status.text("Поиск статей, цитирующих исходные...")
        all_citing_articles = self.get_citing_articles_batch(doi_list)
        main_progress.progress(0.5)
        
        # Шаг 3: Собираем все DOI цитирующих статей
        main_status.text("📊 Шаг 3/4: Подготовка данных цитирований...")
        step_status.text("Сбор DOI цитирующих статей...")
        all_citing_dois = set()
        for citing_dois in all_citing_articles.values():
            all_citing_dois.update(citing_dois)
        
        main_progress.progress(0.75)
        
        # Шаг 4: Получаем данные всех цитирующих статей
        main_status.text("📊 Шаг 4/4: Получение данных цитирующих статей...")
        step_status.text("Загрузка метаданных цитирующих статей...")
        
        if all_citing_dois:
            citing_dois_list = list(all_citing_dois)
            total_citing_dois = len(citing_dois_list)
            
            # Разбиваем на батчи для лучшего отображения прогресса
            batch_size = 50
            citing_articles_data = {}
            
            for i in range(0, total_citing_dois, batch_size):
                batch_dois = citing_dois_list[i:i + batch_size]
                batch_data = self.get_article_data_batch(batch_dois)
                citing_articles_data.update(batch_data)
                
                # Обновляем прогресс
                progress_pct = min((i + batch_size) / total_citing_dois * 100, 100)
                step_progress.progress(progress_pct / 100)
                step_status.text(f"Загружено {min(i + batch_size, total_citing_dois)}/{total_citing_dois} цитирований ({progress_pct:.1f}%)")
        else:
            citing_articles_data = {}
        
        main_progress.progress(1.0)
        
        # Собираем все данные
        main_status.text("📊 Сборка финального датасета...")
        step_status.text("Объединение данных...")
        
        all_data = []
        
        # Добавляем исходные статьи
        for doi, data in source_articles_data.items():
            data['type'] = 'source'
            data['source_doi'] = doi
            all_data.append(data)
        
        # Добавляем цитирующие статьи
        total_citing = sum(len(citing_dois) for citing_dois in all_citing_articles.values())
        processed_citing = 0
        
        for source_doi, citing_dois in all_citing_articles.items():
            for citing_doi in citing_dois:
                if self.validate_doi(citing_doi):
                    citing_data = citing_articles_data.get(citing_doi, {})
                    if citing_data:
                        citing_data['type'] = 'citation'
                        citing_data['source_doi'] = source_doi
                        all_data.append(citing_data)
                
                processed_citing += 1
                if processed_citing % 10 == 0:
                    progress_pct = processed_citing / total_citing * 100
                    step_progress.progress(progress_pct / 100)
                    step_status.text(f"Обработано цитирований: {processed_citing}/{total_citing} ({progress_pct:.1f}%)")
        
        main_status.text("✅ Анализ цитирований завершен!")
        step_status.text("")
        step_progress.empty()
        
        # Создаем DataFrame и добавляем недостающие колонки
        result_df = pd.DataFrame(all_data)
        
        # Добавляем недостающие колонки, если их нет
        expected_columns = [
            'doi', 'title', 'year', 'publication_year', 'authors', 'authors_with_initials', 
            'author_count', 'journal_full_name', 'journal_abbreviation', 'publisher',
            'citation_count_crossref', 'citation_count_openalex', 'annual_citation_rate_crossref',
            'annual_citation_rate_openalex', 'years_since_publication', 'affiliations', 
            'countries', 'altmetric_score', 'number_of_mentions', 'x_mentions', 'rss_blogs',
            'unique_accounts', 'type', 'source_doi'
        ]
        
        for col in expected_columns:
            if col not in result_df.columns:
                result_df[col] = None
        
        return result_df

    # ИСПРАВЛЕННЫЕ МЕТОДЫ ЭКСПОРТА В EXCEL С СОХРАНЕНИЕМ ВО ВРЕМЕННЫЕ ФАЙЛЫ
    def save_references_analysis_to_excel(self, references_df: pd.DataFrame, source_articles_df: pd.DataFrame,
                                        doi_list: List[str], total_references: int, unique_dois: int,
                                        all_titles: List[str]) -> str:
        """Сохраняет полный анализ ссылок в Excel и возвращает путь к файлу"""
        try:
            # Создаем workbook в памяти
            wb = Workbook()
            wb.remove(wb.active)  # Удаляем дефолтный лист

            # Получаем все данные для анализа
            unique_df = self.get_unique_references(references_df)
            duplicate_df = self.find_duplicate_references(references_df)
            
            # Анализ заголовков
            content_freq, compound_freq, scientific_freq = self.analyze_titles(all_titles)

            # Создаем вкладку Report_Summary
            ws_summary = wb.create_sheet('Report_Summary')
            
            summary_content = f"""@MedvDmitry production

REFERENCES ANALYSIS REPORT

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

ANALYSIS OVERVIEW
=================
Total source articles: {len(doi_list)}
Total references collected: {total_references}
Unique DOIs identified: {unique_dois}
Total references processed: {len(references_df) if not references_df.empty else 0}
Unique references: {len(unique_df) if not unique_df.empty else 0}
Successful references: {len(references_df[references_df['error'].isna()]) if not references_df.empty else 0}
Failed references: {len(references_df[references_df['error'].notna()]) if not references_df.empty else 0}
Unique authors: {len(self.analyze_authors_frequency(references_df)) if not references_df.empty else 0}
Unique journals: {len(self.analyze_journals_frequency(references_df)) if not references_df.empty else 0}
Unique affiliations: {len(self.analyze_affiliations_frequency(references_df)) if not references_df.empty else 0}
Unique countries: {len(self.analyze_countries_frequency(references_df)) if not references_df.empty else 0}
Duplicate references: {len(duplicate_df) if not duplicate_df.empty else 0}

DATA COMPLETENESS
=================
References with country data: {(len(references_df[references_df['countries'].isin(['Unknown', 'Error']) == False]) / len(references_df) * 100) if not references_df.empty else 0:.1f}%
References with affiliation data: {(len(references_df[references_df['affiliations'].isin(['Unknown', 'Error']) == False]) / len(references_df) * 100) if not references_df.empty else 0:.1f}%
References with altmetric data: {(len(references_df[references_df['altmetric_score'] > 0]) / len(references_df) * 100) if not references_df.empty else 0:.1f}%

AFFILIATION PROCESSING
======================
Affiliations normalized and grouped by organization
Similar affiliations merged together
Frequency counts reflect grouped organizations

ALTMETRIC METRICS INCLUDED
==========================
Altmetric Score: Overall attention score
Number of Mentions: Posts mentioning the article
X Mentions: Twitter/X accounts mentioning
RSS/Blogs: Blog and RSS feed mentions
Unique Accounts: Unique accounts across platforms

TITLE ANALYSIS
==============
Content words analyzed: {len(content_freq)}
Compound words identified: {len(compound_freq)}
Scientific stopwords found: {len(scientific_freq)}

DATA QUALITY NOTES
==================
Analysis focuses on references cited by the source articles
Combined data from Crossref and OpenAlex improves completeness
All standard statistical analyses performed (authors, journals, countries, etc.)
Error handling ensures report generation even with partial data
Affiliations normalized and grouped for consistent organization names
Altmetric metrics provide social media and online attention analysis
Title word analysis helps identify key research topics and trends
"""

            for line in summary_content.split('\n'):
                ws_summary.append([line])

            # Основные таблицы
            sheets_data = [
                ('Source_Articles', source_articles_df),
                ('All_References', references_df),
                ('All_Unique_References', unique_df),
                ('Duplicate_References', duplicate_df)
            ]

            # Статистические таблицы
            analysis_methods = [
                ('Author_Frequency', self.analyze_authors_frequency),
                ('Journal_Frequency', self.analyze_journals_frequency),
                ('Affiliation_Frequency', self.analyze_affiliations_frequency),
                ('Country_Frequency', self.analyze_countries_frequency),
                ('Year_Distribution', self.analyze_year_distribution),
                ('5_Years_Period', self.analyze_five_year_periods)
            ]

            for sheet_name, method in analysis_methods:
                try:
                    result_df = method(references_df)
                    sheets_data.append((sheet_name, result_df))
                except Exception as e:
                    sheets_data.append((sheet_name, pd.DataFrame()))

            # Анализ заголовков
            try:
                title_word_data = []
                for i, (word, count) in enumerate(content_freq.most_common(50), 1):
                    title_word_data.append({'Category': 'Content_Words', 'Rank': i, 'Word': word, 'Frequency': count})
                for i, (word, count) in enumerate(compound_freq.most_common(50), 1):
                    title_word_data.append({'Category': 'Compound_Words', 'Rank': i, 'Word': word, 'Frequency': count})
                for i, (word, count) in enumerate(scientific_freq.most_common(50), 1):
                    title_word_data.append({'Category': 'Scientific_Stopwords', 'Rank': i, 'Word': word, 'Frequency': count})

                title_word_df = pd.DataFrame(title_word_data)
                sheets_data.append(('Title_Word_Frequency', title_word_df))
            except Exception as e:
                sheets_data.append(('Title_Word_Frequency', pd.DataFrame()))

            # Создаем все вкладки
            for sheet_name, df in sheets_data:
                try:
                    if not df.empty:
                        ws = wb.create_sheet(sheet_name)
                        for r in dataframe_to_rows(df, index=False, header=True):
                            ws.append(r)
                    else:
                        ws = wb.create_sheet(sheet_name)
                        ws.append([f"No data available for {sheet_name}"])
                except Exception as e:
                    pass

            # Сохраняем во временный файл
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"references_analysis_{timestamp}.xlsx"
            file_path = get_temp_file_path(filename)
            
            wb.save(file_path)
            
            return file_path

        except Exception as e:
            self.logger.error(f"Error saving Excel: {e}")
            # Создаем файл с ошибкой
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"error_references_analysis_{timestamp}.xlsx"
            file_path = get_temp_file_path(filename)
            
            wb = Workbook()
            ws = wb.active
            ws.title = "Error"
            ws.append(["Error creating Excel file"])
            ws.append([str(e)])
            wb.save(file_path)
            
            return file_path

    def save_citations_analysis_to_excel(self, citations_df: pd.DataFrame, citing_details_df: pd.DataFrame,
                                       doi_list: List[str], citing_results: Dict, all_citing_titles: List[str]) -> str:
        """Сохраняет полный анализ цитирований в Excel и возвращает путь к файлу"""
        try:
            wb = Workbook()
            wb.remove(wb.active)

            # Получаем все данные для анализа
            unique_citations_df = self.get_unique_citations(citations_df)
            duplicate_citations_df = self.find_duplicate_citations(citations_df)
            
            # Анализ заголовков
            content_freq, compound_freq, scientific_freq = self.analyze_titles(all_citing_titles)

            # Создаем вкладку Report_Summary
            ws_summary = wb.create_sheet('Report_Summary_Citations')
            
            citing_info = ""
            if citing_results:
                citing_info = f"\nCitations per source article:"
                for doi, data in citing_results.items():
                    citing_info += f"\n  - {doi}: {data['count']} citations"

            summary_content = f"""@MedvDmitry production

CITATION ANALYSIS REPORT (CITING ARTICLES)

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

ANALYSIS OVERVIEW
=================
Total source articles: {len(doi_list)}
Total citation relationships: {len(citations_df) if not citations_df.empty else 0}
Total unique citing articles: {len(unique_citations_df) if not unique_citations_df.empty else 0}
Successful citations: {len(citations_df[citations_df['error'].isna()]) if not citations_df.empty else 0}
Failed citations: {len(citations_df[citations_df['error'].notna()]) if not citations_df.empty else 0}
Unique authors: {len(self.analyze_authors_frequency(citations_df)) if not citations_df.empty else 0}
Unique journals: {len(self.analyze_journals_frequency(citations_df)) if not citations_df.empty else 0}
Unique affiliations: {len(self.analyze_affiliations_frequency(citations_df)) if not citations_df.empty else 0}
Unique countries: {len(self.analyze_countries_frequency(citations_df)) if not citations_df.empty else 0}
Duplicate citations: {len(duplicate_citations_df) if not duplicate_citations_df.empty else 0}

DATA COMPLETENESS
=================
Articles with country data: {(len(citations_df[citations_df['countries'].isin(['Unknown', 'Error']) == False]) / len(citations_df) * 100) if not citations_df.empty else 0:.1f}%
Articles with affiliation data: {(len(citations_df[citations_df['affiliations'].isin(['Unknown', 'Error']) == False]) / len(citations_df) * 100) if not citations_df.empty else 0:.1f}%

AFFILIATION PROCESSING
======================
Affiliations normalized and grouped by organization
Similar affiliations merged together
Frequency counts reflect grouped organizations

TITLE ANALYSIS
==============
Content words analyzed: {len(content_freq)}
Compound words identified: {len(compound_freq)}
Scientific stopwords found: {len(scientific_freq)}
{citing_info}

DATA QUALITY NOTES
==================
Analysis focuses on articles that cite the source articles
Combined data from Crossref and OpenAlex improves completeness
All standard statistical analyses performed (authors, journals, countries, etc.)
Error handling ensures report generation even with partial data
Duplicate citations show articles that cite multiple source articles
Affiliations normalized and grouped for consistent organization names
Altmetric metrics included for social media and online attention analysis
Title word analysis helps identify key research topics and trends in citing literature
"""

            for line in summary_content.split('\n'):
                ws_summary.append([line])

            # Основные таблицы
            sheets_data = [
                ('Source_Articles_Citations', citing_details_df),
                ('All_Citations', citations_df),
                ('All_Unique_Citations', unique_citations_df),
                ('Duplicate_Citations', duplicate_citations_df)
            ]

            # Статистические таблицы
            analysis_methods = [
                ('Author_Frequency_Citations', self.analyze_authors_frequency),
                ('Journal_Frequency_Citations', self.analyze_journals_frequency),
                ('Affiliation_Frequency_Citations', self.analyze_affiliations_frequency),
                ('Country_Frequency_Citations', self.analyze_countries_frequency),
                ('Year_Distribution_Citations', self.analyze_year_distribution),
                ('5_Years_Period_Citations', self.analyze_five_year_periods)
            ]

            for sheet_name, method in analysis_methods:
                try:
                    result_df = method(citations_df)
                    sheets_data.append((sheet_name, result_df))
                except Exception as e:
                    sheets_data.append((sheet_name, pd.DataFrame()))

            # Анализ заголовков
            try:
                title_word_data = []
                for i, (word, count) in enumerate(content_freq.most_common(50), 1):
                    title_word_data.append({'Category': 'Content_Words', 'Rank': i, 'Word': word, 'Frequency': count})
                for i, (word, count) in enumerate(compound_freq.most_common(50), 1):
                    title_word_data.append({'Category': 'Compound_Words', 'Rank': i, 'Word': word, 'Frequency': count})
                for i, (word, count) in enumerate(scientific_freq.most_common(50), 1):
                    title_word_data.append({'Category': 'Scientific_Stopwords', 'Rank': i, 'Word': word, 'Frequency': count})

                title_word_df = pd.DataFrame(title_word_data)
                sheets_data.append(('Title_Word_Frequency_Citations', title_word_df))
            except Exception as e:
                sheets_data.append(('Title_Word_Frequency_Citations', pd.DataFrame()))

            # Создаем все вкладки
            for sheet_name, df in sheets_data:
                try:
                    if not df.empty:
                        ws = wb.create_sheet(sheet_name)
                        for r in dataframe_to_rows(df, index=False, header=True):
                            ws.append(r)
                    else:
                        ws = wb.create_sheet(sheet_name)
                        ws.append([f"No data available for {sheet_name}"])
                except Exception as e:
                    pass

            # Сохраняем во временный файл
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"citations_analysis_{timestamp}.xlsx"
            file_path = get_temp_file_path(filename)
            
            wb.save(file_path)
            
            return file_path

        except Exception as e:
            self.logger.error(f"Error saving citations Excel: {e}")
            # Создаем файл с ошибкой
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"error_citations_analysis_{timestamp}.xlsx"
            file_path = get_temp_file_path(filename)
            
            wb = Workbook()
            ws = wb.active
            ws.title = "Error"
            ws.append(["Error creating Excel file"])
            ws.append([str(e)])
            wb.save(file_path)
            
            return file_path

def main():
    st.title("📚 Refs/Cits Analysis - Full Professional Version")
    st.markdown("🔬 Полнофункциональный анализ ссылок и цитирований научных статей")
    
    # Очистка временных файлов при старте
    cleanup_temp_files()
    
    if not NLTK_AVAILABLE:
        st.warning("⚠️ NLTK не доступен. Используется упрощенная обработка текста.")
    
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
        - Анализ заголовков через NLTK
        - Многоуровневое кеширование
        - Расширенная обработка ошибок
        - Экспорт в Excel с множеством вкладок
        - Без ограничений на количество данных
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Поддерживаемые форматы DOI")
        st.markdown("""
        - `10.1234/abcd.1234`
        - `https://doi.org/10.1234/abcd.1234`
        - `doi:10.1234/abcd.1234`
        - И многие другие форматы
        """)
        
        if st.button("🔄 Очистить кеш"):
            analyzer.invalidate_cache()
            st.success("✅ Кеш очищен!")
    
    # Основной контент
    tab1, tab2, tab3 = st.tabs(["📥 Ввод данных", "🔗 Анализ ссылок", "📈 Анализ цитирований"])
    
    with tab1:
        st.header("Ввод DOI статей")
        
        input_method = st.radio("Способ ввода:", ["📝 Текст", "📁 Файл"])
        
        if input_method == "📝 Текст":
            doi_input = st.text_area(
                "Введите DOI статей:",
                height=200,
                placeholder="Введите DOI через запятую, точку с запятой или с новой строки:\n10.1038/s41586-023-06924-6\n10.1126/science.abl8921\n10.1016/j.cell.2023.08.012\n10.1038/s41557-023-01282-2\nhttps://doi.org/10.1038/s41586-023-06924-6\ndoi:10.1126/science.abl8921"
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
                    
                    # Сохраняем результаты в session state
                    st.session_state.references_df = references_df
                    st.session_state.references_analysis_complete = True
                    
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
                        if not year_df.empty and 'frequency_total' in year_df.columns:
                            viz_df = year_df[['year', 'frequency_total']].set_index('year')
                            st.bar_chart(viz_df.head(15))
                        else:
                            st.info("Нет данных для визуализации по годам")
                    
                    with col2:
                        # Топ журналов
                        journal_df = analyzer.analyze_journals_frequency(references_df)
                        if not journal_df.empty and 'frequency_total' in journal_df.columns:
                            viz_df = journal_df[['journal_abbreviation', 'frequency_total']].set_index('journal_abbreviation')
                            st.bar_chart(viz_df.head(10))
                        else:
                            st.info("Нет данных для визуализации по журналам")
                    
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
                    
                    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                        "Авторы", "Журналы", "Аффилиации", "Страны", "Годы", "5-летние периоды", "Дубликаты"
                    ])
                    
                    with tab1:
                        authors_df = analyzer.analyze_authors_frequency(references_df)
                        if not authors_df.empty:
                            st.dataframe(authors_df.head(20), use_container_width=True)
                    
                    with tab2:
                        journals_df = analyzer.analyze_journals_frequency(references_df)
                        if not journals_df.empty:
                            st.dataframe(journals_df.head(20), use_container_width=True)
                    
                    with tab3:
                        affiliations_df = analyzer.analyze_affiliations_frequency(references_df)
                        if not affiliations_df.empty:
                            st.dataframe(affiliations_df.head(20), use_container_width=True)
                    
                    with tab4:
                        countries_df = analyzer.analyze_countries_frequency(references_df)
                        if not countries_df.empty:
                            st.dataframe(countries_df.head(20), use_container_width=True)
                    
                    with tab5:
                        years_df = analyzer.analyze_year_distribution(references_df)
                        if not years_df.empty:
                            st.dataframe(years_df.head(20), use_container_width=True)
                    
                    with tab6:
                        periods_df = analyzer.analyze_five_year_periods(references_df)
                        if not periods_df.empty:
                            st.dataframe(periods_df.head(20), use_container_width=True)
                    
                    with tab7:
                        duplicates_df = analyzer.find_duplicate_references(references_df)
                        if not duplicates_df.empty:
                            st.dataframe(duplicates_df.head(20), use_container_width=True)
                    
                    # Экспорт в Excel
                    st.subheader("💾 Экспорт данных")
                    
                    if st.button("📊 Создать полный отчет Excel", key="excel_refs"):
                        with st.spinner("Создание Excel отчета..."):
                            source_articles_df = references_df[references_df['type'] == 'source']
                            all_titles = references_df['title'].tolist()
                            excel_file_path = analyzer.save_references_analysis_to_excel(
                                references_df, source_articles_df, dois, 
                                len(references_df[references_df['type'] == 'reference']),
                                references_df['doi'].nunique(), all_titles
                            )
                            
                            # Сохраняем путь к файлу в session state
                            st.session_state.excel_refs_path = excel_file_path
                            st.session_state.excel_refs_filename = os.path.basename(excel_file_path)
                            
                            st.success("✅ Excel отчет создан!")
                    
                    # Кнопка скачивания - отображается только после создания файла
                    if 'excel_refs_path' in st.session_state:
                        # Создаем ссылку для скачивания
                        download_link = create_download_link(
                            st.session_state.excel_refs_path,
                            st.session_state.excel_refs_filename,
                            "📥 Скачать полный отчет (Excel)"
                        )
                        st.markdown(download_link, unsafe_allow_html=True)
                        
                        # Также показываем стандартную кнопку для надежности
                        with open(st.session_state.excel_refs_path, 'rb') as f:
                            st.download_button(
                                "📥 Скачать отчет (альтернативный способ)",
                                f,
                                st.session_state.excel_refs_filename,
                                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key='download_excel_refs'
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
                    # Создаем временные структуры для совместимости
                    citations_df = analyzer.analyze_citations_comprehensive(dois)
                    citing_details_df = citations_df[citations_df['type'] == 'citation'].copy()
                    citing_results = {}
                    for doi in dois:
                        citing_count = len(citations_df[
                            (citations_df['type'] == 'citation') & 
                            (citations_df['source_doi'] == doi)
                        ])
                        citing_results[doi] = {'count': citing_count, 'citing_dois': []}
                
                if not citations_df.empty:
                    st.success("✅ Анализ завершен!")
                    
                    # Сохраняем результаты в session state
                    st.session_state.citations_df = citations_df
                    st.session_state.citations_analysis_complete = True
                    
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
                        if not year_df.empty and 'frequency_total' in year_df.columns:
                            viz_df = year_df[['year', 'frequency_total']].set_index('year')
                            st.bar_chart(viz_df.head(15))
                        else:
                            st.info("Нет данных для визуализации по годам")
                    
                    with col2:
                        # Топ цитирующих журналов
                        citing_journals_df = analyzer.analyze_journals_frequency(citations_df[citations_df['type'] == 'citation'])
                        if not citing_journals_df.empty and 'frequency_total' in citing_journals_df.columns:
                            viz_df = citing_journals_df[['journal_abbreviation', 'frequency_total']].set_index('journal_abbreviation')
                            st.bar_chart(viz_df.head(10))
                        else:
                            st.info("Нет данных для визуализации по журналам")
                    
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
                    
                    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                        "Цитирующие авторы", "Цитирующие журналы", "Цитирующие аффилиации", 
                        "Цитирующие страны", "Годы цитирований", "5-летние периоды", "Дубликаты"
                    ])
                    
                    with tab1:
                        citing_authors_df = analyzer.analyze_authors_frequency(citations_df[citations_df['type'] == 'citation'])
                        if not citing_authors_df.empty:
                            st.dataframe(citing_authors_df.head(20), use_container_width=True)
                    
                    with tab2:
                        citing_journals_df = analyzer.analyze_journals_frequency(citations_df[citations_df['type'] == 'citation'])
                        if not citing_journals_df.empty:
                            st.dataframe(citing_journals_df.head(20), use_container_width=True)
                    
                    with tab3:
                        citing_affiliations_df = analyzer.analyze_affiliations_frequency(citations_df[citations_df['type'] == 'citation'])
                        if not citing_affiliations_df.empty:
                            st.dataframe(citing_affiliations_df.head(20), use_container_width=True)
                    
                    with tab4:
                        citing_countries_df = analyzer.analyze_countries_frequency(citations_df[citations_df['type'] == 'citation'])
                        if not citing_countries_df.empty:
                            st.dataframe(citing_countries_df.head(20), use_container_width=True)
                    
                    with tab5:
                        citing_years_df = analyzer.analyze_year_distribution(citations_df[citations_df['type'] == 'citation'])
                        if not citing_years_df.empty:
                            st.dataframe(citing_years_df.head(20), use_container_width=True)
                    
                    with tab6:
                        citing_periods_df = analyzer.analyze_five_year_periods(citations_df[citations_df['type'] == 'citation'])
                        if not citing_periods_df.empty:
                            st.dataframe(citing_periods_df.head(20), use_container_width=True)
                    
                    with tab7:
                        citing_duplicates_df = analyzer.find_duplicate_citations(citations_df)
                        if not citing_duplicates_df.empty:
                            st.dataframe(citing_duplicates_df.head(20), use_container_width=True)
                    
                    # Экспорт в Excel
                    st.subheader("💾 Экспорт данных")
                    
                    if st.button("📊 Создать полный отчет Excel", key="excel_cits"):
                        with st.spinner("Создание Excel отчета..."):
                            all_citing_titles = citations_df[citations_df['type'] == 'citation']['title'].tolist()
                            excel_file_path = analyzer.save_citations_analysis_to_excel(
                                citations_df, citing_details_df, dois, citing_results, all_citing_titles
                            )
                            
                            # Сохраняем путь к файлу в session state
                            st.session_state.excel_cits_path = excel_file_path
                            st.session_state.excel_cits_filename = os.path.basename(excel_file_path)
                            
                            st.success("✅ Excel отчет создан!")
                    
                    # Кнопка скачивания - отображается только после создания файла
                    if 'excel_cits_path' in st.session_state:
                        # Создаем ссылку для скачивания
                        download_link = create_download_link(
                            st.session_state.excel_cits_path,
                            st.session_state.excel_cits_filename,
                            "📥 Скачать полный отчет (Excel)"
                        )
                        st.markdown(download_link, unsafe_allow_html=True)
                        
                        # Также показываем стандартную кнопку для надежности
                        with open(st.session_state.excel_cits_path, 'rb') as f:
                            st.download_button(
                                "📥 Скачать отчет (альтернативный способ)",
                                f,
                                st.session_state.excel_cits_filename,
                                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key='download_excel_cits'
                            )
                else:
                    st.error("❌ Не удалось получить данные")
        else:
            st.warning("⚠️ Сначала введите DOI на вкладке 'Ввод данных'")

if __name__ == "__main__":
    main()

