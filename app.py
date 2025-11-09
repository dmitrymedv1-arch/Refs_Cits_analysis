import streamlit as st
import requests
import json
import pandas as pd
from habanero import Crossref
from crossref_commons.retrieval import get_publication_as_json
import time
from typing import List, Dict, Any, Tuple, Set
from datetime import datetime
from collections import Counter
import re
import numpy as np
from functools import lru_cache
import os
import tempfile
import zipfile
import shutil
from tqdm import tqdm
from contextlib import redirect_stdout
from io import StringIO, BytesIO
from ratelimit import limits, sleep_and_retry
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging
from bs4 import BeautifulSoup
import io
import csv
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import base64
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import pickle
import hashlib
import concurrent.futures
from threading import Lock
import asyncio
import aiohttp

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# =====================================================================
# CONFIGURATION AND UTILITIES
# =====================================================================

class Config:
    REQUEST_TIMEOUT = 30
    MAX_RETRIES = 5
    DELAY_BETWEEN_REQUESTS = 0.3
    RETRY_DELAY = 1
    MAX_WORKERS = 5
    BATCH_SIZE = 10
    CACHE_TTL_HOURS = 24
    MAX_CACHE_SIZE = 10000

class PerformanceMonitor:
    def __init__(self):
        self.start_time = None
        self.request_count = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def start(self):
        self.start_time = datetime.now()

    def increment_request(self):
        self.request_count += 1

    def increment_cache_hit(self):
        self.cache_hits += 1

    def increment_cache_miss(self):
        self.cache_misses += 1

    def get_stats(self):
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            cache_efficiency = (self.cache_hits / (self.cache_hits + self.cache_misses) * 100) if (self.cache_hits + self.cache_misses) > 0 else 0
            return {
                'total_requests': self.request_count,
                'elapsed_seconds': elapsed,
                'elapsed_minutes': elapsed / 60,
                'requests_per_second': self.request_count / elapsed if elapsed > 0 else 0,
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'cache_efficiency': cache_efficiency
            }
        return {}

# =====================================================================
# CACHING SYSTEM
# =====================================================================

class OptimizedPersistentCache:
    """Persistent cache with TTL and LRU eviction"""
    
    def __init__(self, cache_dir=".citation_cache", ttl_hours=Config.CACHE_TTL_HOURS, max_size=Config.MAX_CACHE_SIZE):
        self.cache_dir = cache_dir
        self.ttl = ttl_hours * 3600
        self.max_size = max_size
        self.access_times = {}
        self.lock = Lock()
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_filename(self, key):
        """Get filename for cache key"""
        key_hash = hashlib.md5(str(key).encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{key_hash}.pkl")
    
    def get(self, key):
        """Get value from cache with LRU tracking"""
        filename = self._get_filename(key)
        with self.lock:
            if os.path.exists(filename):
                try:
                    with open(filename, 'rb') as f:
                        data, timestamp = pickle.load(f)
                    if time.time() - timestamp < self.ttl:
                        self.access_times[key] = time.time()
                        return data
                    os.remove(filename)
                    if key in self.access_times:
                        del self.access_times[key]
                except Exception:
                    pass
            return None
    
    def set(self, key, value):
        """Set value in cache with LRU eviction"""
        with self.lock:
            # LRU eviction
            if len(self.access_times) >= self.max_size:
                oldest_key = min(self.access_times.items(), key=lambda x: x[1])[0]
                oldest_file = self._get_filename(oldest_key)
                if os.path.exists(oldest_file):
                    os.remove(oldest_file)
                del self.access_times[oldest_key]
            
            filename = self._get_filename(key)
            try:
                with open(filename, 'wb') as f:
                    pickle.dump((value, time.time()), f)
                self.access_times[key] = time.time()
            except Exception:
                pass
    
    def clear_expired(self):
        """Clear all expired cache entries"""
        if not os.path.exists(self.cache_dir):
            return
        
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pkl'):
                filepath = os.path.join(self.cache_dir, filename)
                try:
                    with open(filepath, 'rb') as f:
                        _, timestamp = pickle.load(f)
                        if time.time() - timestamp >= self.ttl:
                            os.remove(filepath)
                            # Remove from access times if exists
                            for key in list(self.access_times.keys()):
                                if self._get_filename(key) == filepath:
                                    del self.access_times[key]
                except:
                    try:
                        os.remove(filepath)
                    except:
                        pass

    def get_stats(self):
        """Get cache statistics"""
        with self.lock:
            return {
                'total_cached': len(self.access_times),
                'cache_dir': self.cache_dir,
                'max_size': self.max_size
            }

class ProgressTracker:
    """Track progress for resuming analysis after interruptions"""
    
    def __init__(self, progress_file="progress_state.pkl"):
        self.progress_file = progress_file
        self.lock = Lock()
    
    def save_progress(self, analysis_type, doi_list, processed_dois, current_step, 
                     additional_data=None, progress_percentage=0):
        """Save progress state"""
        progress = {
            'analysis_type': analysis_type,
            'doi_list': doi_list,
            'processed_dois': processed_dois,
            'current_step': current_step,
            'additional_data': additional_data or {},
            'progress_percentage': progress_percentage,
            'timestamp': datetime.now(),
            'timestamp_epoch': time.time()
        }
        with self.lock:
            try:
                with open(self.progress_file, 'wb') as f:
                    pickle.dump(progress, f)
            except Exception:
                pass
    
    def load_progress(self):
        """Load progress state"""
        with self.lock:
            if os.path.exists(self.progress_file):
                try:
                    with open(self.progress_file, 'rb') as f:
                        progress = pickle.load(f)
                        if time.time() - progress.get('timestamp_epoch', 0) < 3600:
                            return progress
                        else:
                            self.clear_progress()
                except Exception:
                    pass
            return None
    
    def clear_progress(self):
        """Clear progress state"""
        with self.lock:
            if os.path.exists(self.progress_file):
                try:
                    os.remove(self.progress_file)
                except:
                    pass
    
    def update_progress(self, progress_percentage, current_step=None, additional_data=None):
        """Update progress percentage and optionally current step"""
        progress = self.load_progress()
        if progress:
            progress['progress_percentage'] = progress_percentage
            if current_step:
                progress['current_step'] = current_step
            if additional_data:
                progress['additional_data'].update(additional_data)
            self.save_progress(
                progress['analysis_type'],
                progress['doi_list'],
                progress['processed_dois'],
                progress['current_step'],
                progress['additional_data'],
                progress_percentage
            )

# =====================================================================
# AFFILIATION PROCESSING
# =====================================================================

class OptimizedAffiliationProcessor:
    """Fast affiliation processor with organization grouping and optimized regex"""

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
        
        # Precompiled regex patterns for performance
        self.email_pattern = re.compile(r'\S+@\S+')
        self.zipcode_pattern = re.compile(r'\d{5,}(?:-\d{4})?')
        self.address_pattern = re.compile(r'p\.?o\.? box \d+', re.IGNORECASE)
        self.street_pattern = re.compile(r'\b\d+\s+[a-zA-Z]+\s+[a-zA-Z]+\b')
        self.split_pattern = re.compile(r'[,;]')
        self.whitespace_pattern = re.compile(r'\s+')
        self.non_alphanumeric_pattern = re.compile(r'[^\w\s&]')
        
        # Common organization pattern for quick matching
        self.common_org_pattern = re.compile(
            r'\b(university|college|institute|school|department|faculty|'
            r'laboratory|center|centre|academy|hospital|clinic)\b', 
            re.IGNORECASE
        )

    def extract_main_organization_fast(self, affiliation: str) -> str:
        """Fast extraction of main organization from full affiliation"""
        if not affiliation or affiliation in ['Unknown', 'Error', '']:
            return "Unknown"

        # Cache lookup
        if affiliation in self.organization_cache:
            return self.organization_cache[affiliation]

        clean_affiliation = affiliation.strip()

        # Quick pattern matching for common organizations
        match = self.common_org_pattern.search(clean_affiliation)
        if match:
            # Extract context around the keyword
            start = max(0, match.start() - 50)
            end = min(len(clean_affiliation), match.end() + 50)
            context = clean_affiliation[start:end]
            
            # Clean up the result
            result = self.whitespace_pattern.sub(' ', context).strip()
            result = re.sub(r'^[^a-zA-Z0-9]*|[^a-zA-Z0-9]*$', '', result)
            
            if result:
                self.organization_cache[affiliation] = result
                return result

        # Fallback to detailed processing
        return self._extract_main_organization_detailed(clean_affiliation)

    def _extract_main_organization_detailed(self, affiliation: str) -> str:
        """Detailed organization extraction when quick method fails"""
        # Clean text
        clean_affiliation = self.email_pattern.sub('', affiliation)
        clean_affiliation = self.zipcode_pattern.sub('', clean_affiliation)
        clean_affiliation = self.address_pattern.sub('', clean_affiliation)
        clean_affiliation = self.street_pattern.sub('', clean_affiliation)

        # Split by separators
        parts = self.split_pattern.split(clean_affiliation)
        main_org_candidates = []

        for part in parts:
            part = part.strip()
            if not part or len(part) < 5:
                continue

            part_lower = part.lower()

            # Check for organization keywords
            has_org_keyword = any(keyword in part_lower for keyword in self.common_keywords)
            has_country = any(country in part_lower for keyword in self.country_keywords)

            if has_org_keyword and not has_country:
                main_org_candidates.append(part)

        # Select best candidate
        if main_org_candidates:
            main_org_candidates.sort(key=len, reverse=True)
            main_org = main_org_candidates[0]
        else:
            # Fallback strategies
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

        # Final cleaning
        main_org = self.whitespace_pattern.sub(' ', main_org).strip()
        main_org = re.sub(r'^[^a-zA-Z0-9]*|[^a-zA-Z0-9]*$', '', main_org)

        result = main_org if main_org else "Unknown"
        self.organization_cache[affiliation] = result
        return result

    def normalize_organization_name(self, org_name: str) -> str:
        """Normalizes organization name for grouping"""
        if not org_name or org_name == "Unknown":
            return org_name

        normalized = org_name.lower().strip()

        # Remove common prefixes and suffixes
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

        # Clean up
        normalized = self.whitespace_pattern.sub(' ', normalized).strip()
        normalized = self.non_alphanumeric_pattern.sub('', normalized)

        return normalized.strip()

    def group_similar_organizations(self, organizations: List[str]) -> Dict[str, List[str]]:
        """Groups similar organizations"""
        if not organizations:
            return {}

        # Create normalized versions
        normalized_map = {}
        for org in organizations:
            if org != "Unknown":
                normalized = self.normalize_organization_name(org)
                if normalized:
                    if normalized not in normalized_map:
                        normalized_map[normalized] = []
                    normalized_map[normalized].append(org)

        # Merge very similar groups
        final_groups = {}
        normalized_keys = list(normalized_map.keys())

        for i, key1 in enumerate(normalized_keys):
            if key1 not in final_groups:
                final_groups[key1] = []

            # Add all organizations for this group
            final_groups[key1].extend(normalized_map[key1])

            # Find similar groups to merge
            for j, key2 in enumerate(normalized_keys[i+1:], i+1):
                if self.are_organizations_similar(key1, key2):
                    if key2 in normalized_map:
                        final_groups[key1].extend(normalized_map[key2])
                    # Mark for deletion
                    if key2 in final_groups:
                        del final_groups[key2]

        return final_groups

    def are_organizations_similar(self, org1: str, org2: str) -> bool:
        """Checks if two organizations are similar"""
        if not org1 or not org2:
            return False

        org1_lower = org1.lower()
        org2_lower = org2.lower()

        # Simple containment check
        if org1_lower in org2_lower or org2_lower in org1_lower:
            return True

        # Word-based similarity
        words1 = set(org1_lower.split())
        words2 = set(org2_lower.split())

        if not words1 or not words2:
            return False

        # Remove stop words
        stop_words = {'the', 'and', 'of', 'for', 'in', 'on', 'at', 'to', 'by'}
        words1 = words1 - stop_words
        words2 = words2 - stop_words

        if not words1 or not words2:
            return False

        # Jaccard similarity
        intersection = words1.intersection(words2)
        union = words1.union(words2)

        similarity = len(intersection) / len(union) if union else 0

        return similarity > 0.6

    def process_affiliations_list_fast(self, affiliations: List[str]) -> Tuple[Dict[str, int], Dict[str, List[str]]]:
        """Fast processing of affiliation list with grouping"""
        if not affiliations:
            return {}, {}

        # Extract main organizations
        main_organizations = []
        for aff in affiliations:
            if aff and aff not in ['Unknown', 'Error']:
                main_org = self.extract_main_organization_fast(aff)
                if main_org and main_org != "Unknown":
                    main_organizations.append(main_org)

        if not main_organizations:
            return {}, {}

        # Group similar organizations
        grouped_organizations = self.group_similar_organizations(main_organizations)

        # Choose representative for each group
        group_representatives = {}
        for normalized_name, org_list in grouped_organizations.items():
            if org_list:
                representative = max(org_list, key=len)
                group_representatives[representative] = org_list

        # Count frequencies for representatives
        frequency_count = {}
        for representative, org_list in group_representatives.items():
            frequency_count[representative] = len(org_list)

        return frequency_count, group_representatives

# =====================================================================
# ALT METRIC PROCESSING
# =====================================================================

class AltmetricProcessor:
    """Processor for collecting altmetric data"""

    def __init__(self):
        self.altmetric_cache = {}
        self.doi_clean_pattern = re.compile(r'^(doi:)?\s*', re.IGNORECASE)
        self.whitespace_pattern = re.compile(r'\s+')
        self.valid_doi_pattern = re.compile(r'^10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+$')

    def clean_doi(self, doi: str) -> str:
        """Cleans DOI from extra characters, prefixes and spaces."""
        if not doi or doi in ['Unknown', 'Error', '']:
            return None

        doi = doi.strip().lower()
        doi = self.doi_clean_pattern.sub('', doi)
        doi = self.whitespace_pattern.sub('', doi)
        
        if self.valid_doi_pattern.match(doi):
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
        """Gets data from free Altmetric API by DOI."""
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
        """Extracts key altmetric indicators for DOI"""
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

# =====================================================================
# BATCH API PROCESSING
# =====================================================================

class BatchAPIProcessor:
    """Batch processing for Crossref and OpenAlex APIs"""
    
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.lock = Lock()
    
    def get_crossref_data_batch(self, dois: List[str]) -> Dict[str, Dict]:
        """Get Crossref data in batches"""
        results = {}
        
        for i in range(0, len(dois), Config.BATCH_SIZE):
            batch = dois[i:i + Config.BATCH_SIZE]
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(Config.MAX_WORKERS, len(batch))) as executor:
                batch_results = list(executor.map(self.analyzer.get_crossref_data, batch))
            
            for doi, result in zip(batch, batch_results):
                results[doi] = result
            
            # Rate limiting
            if i + Config.BATCH_SIZE < len(dois):
                time.sleep(Config.DELAY_BETWEEN_REQUESTS)
        
        return results
    
    def get_openalex_data_batch(self, dois: List[str]) -> Dict[str, Dict]:
        """Get OpenAlex data using batch queries where possible"""
        results = {}
        
        # Process in batches for OpenAlex
        for i in range(0, len(dois), Config.BATCH_SIZE):
            batch = dois[i:i + Config.BATCH_SIZE]
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(Config.MAX_WORKERS, len(batch))) as executor:
                batch_results = list(executor.map(self.analyzer.get_openalex_data, batch))
            
            for doi, result in zip(batch, batch_results):
                results[doi] = result
            
            # Rate limiting
            if i + Config.BATCH_SIZE < len(dois):
                time.sleep(Config.DELAY_BETWEEN_REQUESTS)
        
        return results

# =====================================================================
# MAIN CITATION ANALYZER
# =====================================================================

class OptimizedCitationAnalyzer:
    def __init__(self, rate_limit_calls=10, rate_limit_period=1):
        # Persistent caching with optimization
        self.persistent_cache = OptimizedPersistentCache(ttl_hours=Config.CACHE_TTL_HOURS)
        self.progress_tracker = ProgressTracker()
        self.batch_processor = BatchAPIProcessor(self)
        
        # Initialize caches with restoration from persistent storage
        self._restore_caches()
        
        self.rate_limit_calls = rate_limit_calls
        self.rate_limit_period = rate_limit_period
        self.performance_monitor = PerformanceMonitor()
        self._unique_references_cache = {}
        self._unique_citations_cache = {}
        self.unique_ref_data_cache = {}
        self.unique_citation_data_cache = {}
        self.ltwa_map = None
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.fast_affiliation_processor = OptimizedAffiliationProcessor()
        self.altmetric_processor = AltmetricProcessor()
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
        self.scientific_stopwords_stemmed = {self.stemmer.stem(word) for word in self.scientific_stopwords}
        
        # Precompiled patterns
        self.doi_pattern = re.compile(r'10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+', re.IGNORECASE)
        self.year_pattern = re.compile(r'\b(19|20)\d{2}\b')
        
        self.setup_logging()

    def _restore_caches(self):
        """Restore caches from persistent storage"""
        cache_keys = [
            'crossref_cache', 'openalex_cache', 'altmetric_cache',
            'unique_ref_data_cache', 'unique_citation_data_cache'
        ]
        
        for key in cache_keys:
            cached_data = self.persistent_cache.get(key)
            if cached_data is not None:
                setattr(self, key, cached_data)
            else:
                setattr(self, key, {})

    def _save_caches(self):
        """Save caches to persistent storage"""
        cache_keys = [
            'crossref_cache', 'openalex_cache', 'altmetric_cache',
            'unique_ref_data_cache', 'unique_citation_data_cache'
        ]
        
        for key in cache_keys:
            self.persistent_cache.set(key, getattr(self, key, {}))

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'doi_analyzer_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    # =====================================================================
    # DOI PARSING AND VALIDATION
    # =====================================================================

    def validate_doi(self, doi: str) -> bool:
        """Validates DOI with improved processing"""
        if not doi or not isinstance(doi, str):
            return False

        doi = self.normalize_doi(doi)

        if not bool(self.doi_pattern.match(doi)):
            return False

        if len(doi) < 10:
            return False

        if re.search(r'[^\w\.\-_;()/:]', doi):
            return False

        return True

    def normalize_doi(self, doi: str) -> str:
        """Normalizes DOI by removing prefixes and extra characters"""
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
        doi = doi.strip()

        return doi.lower()

    def parse_doi_input(self, input_text: str, max_dois: int = 200) -> List[str]:
        """Parses DOI input with improved processing of various formats"""
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
            found_dois = self.doi_pattern.findall(line)

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
            st.info("Please check the format. Valid examples:")
            st.code("  - 10.1234/abcd.1234")
            st.code("  - https://doi.org/10.1234/abcd.1234")
            st.code("  - doi:10.1234/abcd.1234")
        else:
            st.success(f"Found {len(unique_dois)} valid DOI(s)")
            if len(cleaned_dois) > len(unique_dois):
                st.info(f"Removed {len(cleaned_dois) - len(unique_dois)} duplicate DOI(s)")
            if len(cleaned_dois) > max_dois:
                st.info(f"Limited to first {max_dois} unique DOI(s) from {len(cleaned_dois)} found")

        return unique_dois

    def preprocess_doi_list(self, doi_list: List[str]) -> List[str]:
        """Preprocesses DOI list by checking cache first"""
        cached_dois = []
        uncached_dois = []
        
        for doi in doi_list:
            if (doi in self.crossref_cache and doi in self.openalex_cache and 
                self.crossref_cache[doi] and self.openalex_cache[doi]):
                cached_dois.append(doi)
            else:
                uncached_dois.append(doi)
        
        # Return cached DOIs first for faster display
        return cached_dois + uncached_dois

    # =====================================================================
    # CROSSREF ANALYSIS
    # =====================================================================

    @retry(stop=stop_after_attempt(Config.MAX_RETRIES),
           wait=wait_exponential(multiplier=1, min=Config.RETRY_DELAY, max=10),
           retry=retry_if_exception_type((requests.exceptions.RequestException,
                                        Exception)))
    @sleep_and_retry
    @limits(calls=15, period=1)
    def get_crossref_data(self, doi: str) -> Dict:
        """Gets data from Crossref with retries and improved affiliation processing"""
        if doi in self.crossref_cache:
            self.performance_monitor.increment_cache_hit()
            return self.crossref_cache[doi]
        
        self.performance_monitor.increment_cache_miss()
        try:
            cr = Crossref()
            result = cr.works(ids=doi)
            self.performance_monitor.increment_request()
            data = result['message']

            year = None
            for key in ['published-print', 'published-online', 'issued']:
                if key in data and 'date-parts' in data[key]:
                    date_parts = data[key]['date-parts'][0]
                    year = date_parts[0] if date_parts else None
                    break
            data['publication_year'] = year if year else 'Unknown'

            affiliations, countries = self.extract_affiliations_from_crossref(data)
            data['extracted_affiliations'] = affiliations
            data['extracted_countries'] = countries

            self.crossref_cache[doi] = data
            self._save_caches()
            return data
        except Exception as e:
            self.logger.warning(f"Crossref request failed for {doi}: {e}")
            self.crossref_cache[doi] = {'publication_year': 'Unknown', 'extracted_affiliations': [], 'extracted_countries': []}
            self._save_caches()
            return {'publication_year': 'Unknown', 'extracted_affiliations': [], 'extracted_countries': []}

    def extract_affiliations_from_crossref(self, crossref_data: Dict) -> tuple[List[str], List[str]]:
        """Extracts affiliations and countries from Crossref data"""
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

            for field in ['institution', 'organization', 'department']:
                if field in crossref_data:
                    if isinstance(crossref_data[field], list):
                        for item in crossref_data[field]:
                            if isinstance(item, dict) and 'name' in item:
                                main_org = self.fast_affiliation_processor.extract_main_organization_fast(item['name'].strip())
                                if main_org and main_org != "Unknown":
                                    affiliations.add(main_org)
                            elif isinstance(item, str):
                                main_org = self.fast_affiliation_processor.extract_main_organization_fast(item.strip())
                                if main_org and main_org != "Unknown":
                                    affiliations.add(main_org)
                    elif isinstance(crossref_data[field], str):
                        main_org = self.fast_affiliation_processor.extract_main_organization_fast(crossref_data[field].strip())
                        if main_org and main_org != "Unknown":
                            affiliations.add(main_org)

        except Exception as e:
            self.logger.debug(f"Error extracting affiliations from Crossref: {e}")

        return list(affiliations), list(countries)

    def extract_country_from_affiliation(self, affiliation_data: Dict) -> str:
        """Extracts country from Crossref affiliation data"""
        try:
            if 'country' in affiliation_data and affiliation_data['country']:
                return affiliation_data['country'].upper().strip()

            if 'address' in affiliation_data and affiliation_data['address']:
                address = affiliation_data['address'].upper()
                country_codes = ['USA', 'US', 'UK', 'GB', 'DE', 'FR', 'CN', 'JP', 'RU', 'IN', 'BR', 'CA', 'AU', 'KR']
                for code in country_codes:
                    if code in address:
                        return code

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

    # =====================================================================
    # OPENALEX ANALYSIS
    # =====================================================================

    @retry(stop=stop_after_attempt(Config.MAX_RETRIES),
           wait=wait_exponential(multiplier=1, min=Config.RETRY_DELAY, max=10),
           retry=retry_if_exception_type((requests.exceptions.RequestException,
                                        requests.exceptions.Timeout,
                                        requests.exceptions.ConnectionError)))
    @sleep_and_retry
    @limits(calls=15, period=1)
    def get_openalex_data(self, doi: str) -> Dict:
        """Gets data from OpenAlex with retries"""
        if doi in self.openalex_cache:
            self.performance_monitor.increment_cache_hit()
            return self.openalex_cache[doi]
        
        self.performance_monitor.increment_cache_miss()
        try:
            openalex_url = f"https://api.openalex.org/works/https://doi.org/{doi}"
            response = requests.get(openalex_url, timeout=Config.REQUEST_TIMEOUT)
            self.performance_monitor.increment_request()
            if response.status_code == 404:
                self.openalex_cache[doi] = {}
                self._save_caches()
                return {}
            response.raise_for_status()
            result = response.json()
            self.openalex_cache[doi] = result
            self._save_caches()
            return result
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"OpenAlex request failed for {doi}: {e}")
            if doi not in self.openalex_cache:
                self.openalex_cache[doi] = {}
                self._save_caches()
            return {}
        except Exception as e:
            self.logger.error(f"Unexpected error in get_openalex_data for {doi}: {e}")
            if doi not in self.openalex_cache:
                self.openalex_cache[doi] = {}
                self._save_caches()
            return {}

    # =====================================================================
    # MAIN ANALYSIS METHODS
    # =====================================================================

    def get_combined_article_data(self, doi: str) -> Dict[str, Any]:
        """Get combined data from both Crossref and OpenAlex with improved processing"""
        try:
            # Use batch processing for better performance
            crossref_data = self.get_crossref_data(doi)
            openalex_data = self.get_openalex_data(doi)
            altmetric_data = self.altmetric_processor.get_altmetric_metrics(doi)

            title = 'Unknown'
            if openalex_data and openalex_data.get('title'):
                title = openalex_data['title']
            elif crossref_data.get('title'):
                title_list = crossref_data['title']
                if title_list:
                    title = title_list[0]

            year = 'Unknown'
            publication_year = None

            if openalex_data and openalex_data.get('publication_year'):
                publication_year = openalex_data['publication_year']
                year = str(publication_year)
            elif crossref_data.get('publication_year') != 'Unknown':
                publication_year = crossref_data['publication_year']
                year = str(publication_year)

            authors = []
            authors_surnames = []
            authors_with_initials = []

            if openalex_data:
                for author in openalex_data.get('authorships', []):
                    name = author.get('author', {}).get('display_name', 'Unknown')
                    if name != 'Unknown':
                        authors.append(name)
                        surname_with_initial = self.extract_surname_with_initial(name)
                        authors_surnames.append(surname_with_initial)
                        authors_with_initials.append(surname_with_initial)

            if not authors and crossref_data.get('author'):
                for author in crossref_data['author']:
                    given = author.get('given', '')
                    family = author.get('family', '')
                    if given or family:
                        name = f"{given} {family}".strip()
                        authors.append(name)
                        surname_with_initial = self.extract_surname_with_initial(name)
                        authors_surnames.append(surname_with_initial)
                        authors_with_initials.append(surname_with_initial)

            authors_str = ', '.join(authors) if authors else 'Unknown'
            authors_surnames_str = ', '.join(authors_surnames) if authors_surnames else 'Unknown'
            authors_with_initials_str = ', '.join(authors_with_initials) if authors_with_initials else 'Unknown'
            author_count = len(authors) if authors else 0

            journal_info = self.get_journal_info_from_crossref(doi)

            _, crossref_citations, openalex_citations = self.get_citation_data(doi)

            affiliations, countries = self.get_enhanced_affiliations_and_countries(openalex_data, crossref_data)

            current_year = datetime.now().year
            years_since_pub = self.calculate_years_since_publication(publication_year, current_year)

            return {
                'doi': doi,
                'title': title,
                'year': year,
                'publication_year': publication_year,
                'authors': authors_str,
                'authors_surnames': authors_surnames_str,
                'authors_with_initials': authors_with_initials_str,
                'author_count': author_count,
                'journal_full_name': journal_info['full_name'],
                'journal_abbreviation': journal_info['abbreviation'],
                'publisher': journal_info['publisher'],
                'citation_count_crossref': crossref_citations,
                'citation_count_openalex': openalex_citations,
                'affiliations': '; '.join(affiliations),
                'countries': countries,
                'years_since_publication': years_since_pub,
                'altmetric_score': altmetric_data['altmetric_score'],
                'number_of_mentions': altmetric_data['cited_by_posts_count'],
                'x_mentions': altmetric_data['cited_by_tweeters_count'],
                'rss_blogs': altmetric_data['cited_by_feeds_count'],
                'unique_accounts': altmetric_data['cited_by_accounts_count']
            }
        except Exception as e:
            return {
                'doi': doi,
                'title': 'Error',
                'year': 'Unknown',
                'publication_year': None,
                'authors': 'Error',
                'authors_surnames': 'Error',
                'authors_with_initials': 'Error',
                'author_count': 0,
                'journal_full_name': 'Error',
                'journal_abbreviation': 'Error',
                'publisher': 'Error',
                'citation_count_crossref': 0,
                'citation_count_openalex': 0,
                'affiliations': 'Error',
                'countries': 'Error',
                'years_since_publication': 1,
                'altmetric_score': 0,
                'number_of_mentions': 0,
                'x_mentions': 0,
                'rss_blogs': 0,
                'unique_accounts': 0,
                'error': str(e)
            }

    def get_enhanced_affiliations_and_countries(self, openalex_data: Dict, crossref_data: Dict) -> tuple[List[str], str]:
        """Enhanced affiliation processing with grouping"""
        try:
            openalex_affiliations, openalex_countries = self.get_affiliations_and_countries_from_openalex_data(openalex_data)
            crossref_affiliations, crossref_countries = self.get_affiliations_and_countries_from_crossref_data(crossref_data)

            all_affiliations = []
            if openalex_affiliations and openalex_affiliations != ['Unknown']:
                all_affiliations.extend(openalex_affiliations)
            if crossref_affiliations:
                all_affiliations.extend(crossref_affiliations)

            if all_affiliations:
                affiliation_frequencies, grouped_organizations = self.fast_affiliation_processor.process_affiliations_list_fast(all_affiliations)
                final_affiliations = list(affiliation_frequencies.keys())
            else:
                final_affiliations = ['Unknown']

            all_countries = set()
            if openalex_countries and openalex_countries != 'Unknown':
                countries_list = openalex_countries.split(';')
                all_countries.update([c.strip() for c in countries_list if c.strip()])
            if crossref_countries:
                all_countries.update(crossref_countries)

            final_countries = ';'.join(sorted(all_countries)) if all_countries else 'Unknown'

            return final_affiliations, final_countries

        except Exception as e:
            self.logger.error(f"Error in enhanced affiliations processing: {e}")
            return ['Unknown'], 'Unknown'

    def get_affiliations_and_countries_from_openalex_data(self, openalex_data: Dict) -> tuple[List[str], str]:
        """Extracts affiliations and countries from OpenAlex data"""
        try:
            if not openalex_data:
                return ['Unknown'], 'Unknown'

            affiliations = set()
            countries = set()

            for authorship in openalex_data.get('authorships', []):
                for institution in authorship.get('institutions', []):
                    display_name = institution.get('display_name', '')
                    country_code = institution.get('country_code', '')

                    if display_name:
                        main_org = self.fast_affiliation_processor.extract_main_organization_fast(display_name)
                        if main_org and main_org != "Unknown":
                            affiliations.add(main_org)

                    if country_code:
                        countries.add(country_code.upper())

            return list(affiliations) or ['Unknown'], ';'.join(sorted(countries)) if countries else 'Unknown'
        except Exception as e:
            self.logger.debug(f"Error extracting from OpenAlex: {e}")
            return ['Unknown'], 'Unknown'

    def get_affiliations_and_countries_from_crossref_data(self, crossref_data: Dict) -> tuple[List[str], List[str]]:
        """Extracts affiliations and countries from Crossref data"""
        try:
            if not crossref_data:
                return [], []

            affiliations = crossref_data.get('extracted_affiliations', [])
            countries = crossref_data.get('extracted_countries', [])

            return affiliations, countries
        except Exception as e:
            self.logger.debug(f"Error extracting from Crossref: {e}")
            return [], []

    def get_citation_data(self, doi: str) -> tuple:
        try:
            crossref_data = self.get_crossref_data(doi)
            crossref_citations = crossref_data.get('is-referenced-by-count', 0)

            openalex_data = self.get_openalex_data(doi)
            openalex_citations = openalex_data.get('cited_by_count', 0)

            return doi, crossref_citations, openalex_citations
        except:
            return doi, 0, 0

    def get_journal_info_from_crossref(self, doi: str) -> Dict[str, Any]:
        try:
            data = self.get_crossref_data(doi)
            container_title = data.get('container-title', [])
            short_container_title = data.get('short-container-title', [])
            full_name = container_title[0] if container_title else (short_container_title[0] if short_container_title else 'Unknown')
            abbreviation = short_container_title[0] if short_container_title else (container_title[0] if container_title else 'Unknown')
            return {
                'full_name': full_name,
                'abbreviation': abbreviation,
                'publisher': data.get('publisher', 'Unknown'),
                'issn': data.get('ISSN', [None])[0]
            }
        except:
            return {
                'full_name': 'Unknown',
                'abbreviation': 'Unknown',
                'publisher': 'Unknown',
                'issn': None
            }

    @lru_cache(maxsize=1000)
    def extract_surname_with_initial(self, author_name: str) -> str:
        if not author_name or author_name in ['Unknown', 'Error']:
            return author_name
        clean_name = re.sub(r'[^\w\s\-\.]', ' ', author_name).strip()
        parts = clean_name.split()
        if not parts:
            return author_name
        surname = parts[-1]
        initial = parts[0][0].upper() if parts[0] else ''
        return f"{surname} {initial}." if initial else surname

    def safe_calculate_annual_citation_rate(self, citation_count, publication_year, current_year=None):
        """Safe calculation of annual citation rate with error handling"""
        try:
            if not isinstance(citation_count, (int, float)) or citation_count == 0:
                return 0.0

            years = self.calculate_years_since_publication(publication_year, current_year)

            if years is None or not isinstance(years, (int, float)) or years <= 0:
                return 0.0

            return round(citation_count / years, 2)
        except (TypeError, ZeroDivisionError, ValueError):
            return 0.0

    def calculate_years_since_publication(self, publication_year: Any, current_year: int = None) -> int:
        """Safe calculation of years since publication"""
        try:
            if current_year is None:
                current_year = datetime.now().year

            if publication_year is None or publication_year == 'Unknown':
                return 1

            year_str = str(publication_year).strip()
            if not year_str or year_str == 'Unknown':
                return 1

            year_match = self.year_pattern.search(year_str)
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

    # =====================================================================
    # REFERENCES PROCESSING METHODS
    # =====================================================================

    def get_references_from_crossref(self, doi: str) -> List[Dict[str, Any]]:
        try:
            article_data = get_publication_as_json(doi)
            return article_data.get('reference', [])
        except:
            return []

    @sleep_and_retry
    @limits(calls=10, period=1)
    def quick_doi_search(self, title: str) -> str:
        """Quick DOI search by title"""
        if not title or title == 'Unknown':
            return None

        url = "https://api.crossref.org/works"
        params = {
            'query.title': title,
            'rows': 1,
            'select': 'DOI,title'
        }
        headers = {'User-Agent': 'CitationAnalyzer/1.0 (mailto:your@email.com)'}

        try:
            response = requests.get(url, params=params, headers=headers, timeout=Config.REQUEST_TIMEOUT)
            self.performance_monitor.increment_request()
            response.raise_for_status()
            data = response.json()

            if data['message']['items']:
                doi = data['message']['items'][0]['DOI']
                if self.validate_doi(doi):
                    return doi
            return None
        except:
            return None

    # =====================================================================
    # CITING ARTICLES PROCESSING METHODS
    # =====================================================================

    @retry(stop=stop_after_attempt(Config.MAX_RETRIES),
           wait=wait_exponential(multiplier=1, min=Config.RETRY_DELAY, max=10))
    def get_citing_articles_from_openalex(self, doi: str) -> List[str]:
        """Gets citing works through OpenAlex API"""
        citing_dois = []
        try:
            work_id = doi.replace('/', '%2F')
            url = f"https://api.openalex.org/works/https://doi.org/{work_id}"

            response = requests.get(url, timeout=Config.REQUEST_TIMEOUT)
            self.performance_monitor.increment_request()

            if response.status_code == 200:
                data = response.json()
                cited_by_count = data.get('cited_by_count', 0)
                work_openalex_id = data.get('id', '')

                if cited_by_count > 0:
                    per_page = 200
                    total_pages = (cited_by_count + per_page - 1) // per_page

                    for page in range(1, total_pages + 1):
                        citing_url = f"https://api.openalex.org/works?filter=cites:{work_openalex_id}&per-page={per_page}&page={page}"

                        response = requests.get(citing_url, timeout=Config.REQUEST_TIMEOUT)
                        self.performance_monitor.increment_request()

                        if response.status_code == 200:
                            citing_data = response.json()
                            results = citing_data.get('results', [])
                            results_count = len(results)

                            for work in results:
                                if work.get('doi'):
                                    citing_dois.append(work['doi'])

                            if results_count < per_page:
                                break

                            time.sleep(Config.DELAY_BETWEEN_REQUESTS)

                        else:
                            if response.status_code == 429:
                                time.sleep(3)
                                page -= 1
                            else:
                                break

                        if len(citing_dois) >= cited_by_count:
                            break

            return citing_dois

        except Exception as e:
            return []

    def get_citing_articles_from_crossref(self, doi: str) -> List[str]:
        """Gets citing works through Crossref API"""
        citing_dois = []
        try:
            url = f"https://api.crossref.org/works/{doi}"
            response = requests.get(url, timeout=Config.REQUEST_TIMEOUT)
            self.performance_monitor.increment_request()

            if response.status_code == 200:
                data = response.json()
                if 'message' in data and 'is-referenced-by' in data['message']:
                    references = data['message']['is-referenced-by']
                    for ref in references:
                        if isinstance(ref, dict) and 'DOI' in ref:
                            citing_dois.append(ref['DOI'])

        except Exception as e:
            pass

        return citing_dois

    def find_citing_articles(self, doi_list: List[str]) -> Dict[str, Dict]:
        """Main function for finding citing articles"""
        results = {}

        for i, doi in enumerate(doi_list, 1):
            doi = doi.strip()
            if not doi:
                continue

            openalex_citations = self.get_citing_articles_from_openalex(doi)
            crossref_citations = self.get_citing_articles_from_crossref(doi)

            all_citations = list(set(openalex_citations + crossref_citations))

            results[doi] = {
                'count': len(all_citations),
                'citing_dois': all_citations
            }

            time.sleep(Config.DELAY_BETWEEN_REQUESTS)

        return results

    # =====================================================================
    # PARALLEL PROCESSING METHODS
    # =====================================================================

    def process_dois_parallel(self, doi_list: List[str], max_workers: int = None) -> List[Dict]:
        """Parallel processing of DOIs"""
        if max_workers is None:
            max_workers = min(Config.MAX_WORKERS, len(doi_list))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(self.get_combined_article_data, doi_list))
        return results

    def process_references_parallel(self, doi_list: List[str]) -> tuple[pd.DataFrame, pd.DataFrame, int, int, List[str]]:
        """Optimized parallel processing for references analysis"""
        self.performance_monitor.start()
        
        # Preprocess DOI list
        optimized_dois = self.preprocess_doi_list(doi_list)
        
        # Check for existing progress
        progress = self.progress_tracker.load_progress()
        if progress and progress.get('analysis_type') == 'references':
            st.info("🔄 Resuming previous references analysis...")
            return self._resume_references_analysis(progress)
        
        # Start new analysis with parallel processing
        return self._process_references_parallel_new(optimized_dois)

    def _resume_references_analysis(self, progress: Dict) -> tuple[pd.DataFrame, pd.DataFrame, int, int, List[str]]:
        """Resume references analysis from progress state"""
        doi_list = progress['doi_list']
        processed_dois = progress['processed_dois']
        additional_data = progress['additional_data']
        
        # Continue processing from where we left off
        remaining_dois = [doi for doi in doi_list if doi not in processed_dois]
        
        if remaining_dois:
            # Process remaining DOIs in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
                results = list(executor.map(self.get_combined_article_data, remaining_dois))
            
            # Update progress
            processed_dois.extend(remaining_dois)
            self.progress_tracker.update_progress(
                100, 'completed', {'processed_dois': processed_dois}
            )
        
        # Reconstruct results from progress data
        combined_references_df = pd.DataFrame(additional_data.get('results', []))
        source_articles_df = pd.DataFrame(additional_data.get('source_articles', []))
        total_references = additional_data.get('total_references', 0)
        unique_dois = additional_data.get('unique_dois', 0)
        all_titles = additional_data.get('all_titles', [])
        
        return combined_references_df, source_articles_df, total_references, unique_dois, all_titles

    def _process_references_parallel_new(self, doi_list: List[str]) -> tuple[pd.DataFrame, pd.DataFrame, int, int, List[str]]:
        """New parallel processing for references analysis"""
        self.progress_tracker.save_progress('references', doi_list, [], 'starting', {}, 0)
        
        # Step 1: Get source articles data in parallel
        with st.spinner("Processing source articles in parallel..."):
            source_articles_data = self.process_dois_parallel(doi_list)
            source_articles_df = pd.DataFrame(source_articles_data)
        
        self.progress_tracker.update_progress(25, 'processing_source', {
            'source_articles': source_articles_data
        })
        
        # Step 2: Get references in parallel
        with st.spinner("Collecting references in parallel..."):
            all_references = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
                reference_results = list(executor.map(self.get_references_from_crossref, doi_list))
            
            for doi, references in zip(doi_list, reference_results):
                for j, ref in enumerate(references):
                    all_references.append({
                        'source_doi': doi,
                        'position': j + 1,
                        'ref': ref
                    })
        
        self.progress_tracker.update_progress(50, 'processing_references', {
            'all_references': all_references,
            'total_references': len(all_references)
        })
        
        # Step 3: Process unique references in parallel
        with st.spinner("Processing unique references in parallel..."):
            unique_dois = set()
            titles_to_search = set()
            all_titles = []
            
            for ref_data in all_references:
                ref = ref_data['ref']
                ref_doi = ref.get('DOI')
                title = ref.get('article-title', 'Unknown')
                all_titles.append(title)

                if ref_doi and self.validate_doi(ref_doi):
                    unique_dois.add(ref_doi)
                elif title != 'Unknown':
                    titles_to_search.add(title)
            
            # Process unique DOIs in parallel
            unique_dois_list = list(unique_dois)
            with concurrent.futures.ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
                unique_results = list(executor.map(self.get_combined_article_data, unique_dois_list))
            
            for doi, result in zip(unique_dois_list, unique_results):
                self.unique_ref_data_cache[doi] = result
        
        self.progress_tracker.update_progress(75, 'processing_unique', {
            'unique_dois': len(unique_dois),
            'all_titles': all_titles
        })
        
        # Step 4: Build final results
        with st.spinner("Building final results..."):
            results = []
            for ref_data in all_references:
                ref = ref_data['ref']
                position = ref_data['position']
                ref_doi = ref.get('DOI')
                title = ref.get('article-title', 'Unknown')
                
                if ref_doi and ref_doi in self.unique_ref_data_cache:
                    ref_info = self.unique_ref_data_cache[ref_doi].copy()
                    results.append(self._build_reference_row(ref_data, ref_info))
                else:
                    results.append(self._build_missing_reference_row(ref_data, ref_doi, title))
            
            combined_references_df = pd.DataFrame(results)
        
        self.progress_tracker.clear_progress()
        self._save_caches()
        
        return combined_references_df, source_articles_df, len(all_references), len(unique_dois), all_titles

    def _build_reference_row(self, ref_data: Dict, ref_info: Dict) -> Dict:
        """Build a reference row from cached data"""
        return {
            'source_doi': ref_data['source_doi'],
            'position': ref_data['position'],
            'doi': ref_info['doi'],
            'title': ref_info['title'],
            'authors': ref_info['authors'],
            'authors_with_initials': ref_info['authors_with_initials'],
            'author_count': ref_info['author_count'],
            'year': ref_info['year'],
            'journal_full_name': ref_info['journal_full_name'],
            'journal_abbreviation': ref_info['journal_abbreviation'],
            'publisher': ref_info['publisher'],
            'citation_count_crossref': ref_info['citation_count_crossref'],
            'citation_count_openalex': ref_info['citation_count_openalex'],
            'annual_citation_rate_crossref': self.safe_calculate_annual_citation_rate(
                ref_info['citation_count_crossref'], ref_info.get('publication_year')
            ),
            'annual_citation_rate_openalex': self.safe_calculate_annual_citation_rate(
                ref_info['citation_count_openalex'], ref_info.get('publication_year')
            ),
            'years_since_publication': ref_info['years_since_publication'],
            'affiliations': ref_info['affiliations'],
            'countries': ref_info['countries'],
            'altmetric_score': ref_info['altmetric_score'],
            'number_of_mentions': ref_info['number_of_mentions'],
            'x_mentions': ref_info['x_mentions'],
            'rss_blogs': ref_info['rss_blogs'],
            'unique_accounts': ref_info['unique_accounts'],
            'error': None
        }

    def _build_missing_reference_row(self, ref_data: Dict, ref_doi: str, title: str) -> Dict:
        """Build a reference row for missing DOI data"""
        return {
            'source_doi': ref_data['source_doi'],
            'position': ref_data['position'],
            'doi': ref_doi,
            'title': title,
            'authors': 'Unknown',
            'authors_with_initials': 'Unknown',
            'author_count': 0,
            'year': ref_data['ref'].get('year', 'Unknown'),
            'journal_full_name': 'Unknown',
            'journal_abbreviation': 'Unknown',
            'publisher': 'Unknown',
            'citation_count_crossref': 'N/A',
            'citation_count_openalex': 'N/A',
            'annual_citation_rate_crossref': 'N/A',
            'annual_citation_rate_openalex': 'N/A',
            'years_since_publication': 'N/A',
            'affiliations': 'Unknown',
            'countries': 'Unknown',
            'altmetric_score': 0,
            'number_of_mentions': 0,
            'x_mentions': 0,
            'rss_blogs': 0,
            'unique_accounts': 0,
            'error': f"Invalid or missing DOI: {ref_doi}"
        }

    # =====================================================================
    # CITING ARTICLES ANALYSIS
    # =====================================================================

    def process_citing_articles_parallel(self, doi_list: List[str]) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict], List[str]]:
        """Parallel processing for citing articles analysis"""
        self.performance_monitor.start()
        
        # Check for existing progress
        progress = self.progress_tracker.load_progress()
        if progress and progress.get('analysis_type') == 'citing_articles':
            st.info("🔄 Resuming previous citing articles analysis...")
            return self._resume_citing_analysis(progress)
        
        # Start new analysis with parallel processing
        return self._process_citing_articles_parallel_new(doi_list)

    def _resume_citing_analysis(self, progress: Dict) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict], List[str]]:
        """Resume citing articles analysis from progress state"""
        doi_list = progress['doi_list']
        additional_data = progress['additional_data']
        
        citing_results = additional_data.get('citing_results', {})
        all_citing_connections = additional_data.get('all_citing_connections', [])
        citing_data_cache = additional_data.get('citing_data_cache', {})
        all_citing_titles = additional_data.get('all_citing_titles', [])
        
        # Reconstruct DataFrames
        all_citing_articles_data = []
        for connection in all_citing_connections:
            citing_doi = connection['citing_doi']
            source_doi = connection['source_doi']
            
            article_data = citing_data_cache.get(citing_doi, {})
            citing_row = self._build_citing_article_row(source_doi, citing_doi, article_data)
            all_citing_articles_data.append(citing_row)
        
        citing_articles_df = pd.DataFrame(all_citing_articles_data) if all_citing_articles_data else pd.DataFrame()
        
        # Build details DataFrame
        citing_articles_details = []
        for source_doi, data in citing_results.items():
            for citing_doi in data['citing_dois']:
                article_data = citing_data_cache.get(citing_doi, {})
                citing_articles_details.append({
                    'source_doi': source_doi,
                    'citing_doi': citing_doi,
                    'citing_title': article_data.get('title', 'Unknown'),
                    'citing_authors': article_data.get('authors_with_initials', 'Unknown'),
                    'citing_year': article_data.get('year', 'Unknown'),
                    'citing_journal': article_data.get('journal_abbreviation', 'Unknown'),
                    'citation_count': article_data.get('citation_count_openalex', 0),
                    'altmetric_score': article_data.get('altmetric_score', 0),
                    'number_of_mentions': article_data.get('number_of_mentions', 0)
                })
        
        citing_details_df = pd.DataFrame(citing_articles_details) if citing_articles_details else pd.DataFrame()
        
        self.progress_tracker.clear_progress()
        
        return citing_articles_df, citing_details_df, citing_results, all_citing_titles

    def _process_citing_articles_parallel_new(self, doi_list: List[str]) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict], List[str]]:
        """New parallel processing for citing articles analysis"""
        self.progress_tracker.save_progress('citing_articles', doi_list, [], 'finding_citing', {}, 0)
        
        # Step 1: Find citing articles in parallel
        with st.spinner("Finding citing articles in parallel..."):
            citing_results = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
                future_to_doi = {executor.submit(self.get_citing_articles_from_openalex, doi): doi for doi in doi_list}
                
                for future in concurrent.futures.as_completed(future_to_doi):
                    doi = future_to_doi[future]
                    try:
                        openalex_citations = future.result()
                        crossref_citations = self.get_citing_articles_from_crossref(doi)
                        all_citations = list(set(openalex_citations + crossref_citations))
                        
                        citing_results[doi] = {
                            'count': len(all_citations),
                            'citing_dois': all_citations
                        }
                    except Exception as e:
                        citing_results[doi] = {'count': 0, 'citing_dois': []}
        
        self.progress_tracker.update_progress(33, 'finding_citing', {
            'citing_results': citing_results
        })
        
        # Step 2: Collect all citing connections
        all_citing_connections = []
        all_citing_dois = set()
        
        for source_doi, source_data in citing_results.items():
            for citing_doi in source_data['citing_dois']:
                all_citing_connections.append({
                    'source_doi': source_doi,
                    'citing_doi': citing_doi
                })
                all_citing_dois.add(citing_doi)
        
        self.progress_tracker.update_progress(66, 'collecting_connections', {
            'citing_results': citing_results,
            'all_citing_connections': all_citing_connections
        })
        
        # Step 3: Process citing articles in parallel
        with st.spinner("Processing citing articles in parallel..."):
            citing_data_cache = {}
            all_citing_titles = []
            
            citing_dois_list = list(all_citing_dois)
            with concurrent.futures.ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
                citing_results_data = list(executor.map(self.get_combined_article_data, citing_dois_list))
            
            for doi, article_data in zip(citing_dois_list, citing_results_data):
                citing_data_cache[doi] = article_data
                all_citing_titles.append(article_data['title'])
        
        self.progress_tracker.update_progress(90, 'processing_citing', {
            'citing_results': citing_results,
            'all_citing_connections': all_citing_connections,
            'citing_data_cache': citing_data_cache,
            'all_citing_titles': all_citing_titles
        })
        
        # Step 4: Build final DataFrames
        citing_articles_df, citing_details_df = self._build_citing_dataframes(
            all_citing_connections, citing_data_cache
        )
        
        self.progress_tracker.clear_progress()
        self._save_caches()
        
        return citing_articles_df, citing_details_df, citing_results, all_citing_titles

    def _build_citing_article_row(self, source_doi: str, citing_doi: str, article_data: Dict) -> Dict:
        """Build a row for citing articles DataFrame"""
        return {
            'source_doi': source_doi,
            'position': 'N/A',
            'doi': citing_doi,
            'title': article_data.get('title', 'Unknown'),
            'authors': article_data.get('authors', 'Unknown'),
            'authors_with_initials': article_data.get('authors_with_initials', 'Unknown'),
            'author_count': article_data.get('author_count', 0),
            'year': article_data.get('year', 'Unknown'),
            'journal_full_name': article_data.get('journal_full_name', 'Unknown'),
            'journal_abbreviation': article_data.get('journal_abbreviation', 'Unknown'),
            'publisher': article_data.get('publisher', 'Unknown'),
            'citation_count_crossref': article_data.get('citation_count_crossref', 0),
            'citation_count_openalex': article_data.get('citation_count_openalex', 0),
            'annual_citation_rate_crossref': self.safe_calculate_annual_citation_rate(
                article_data.get('citation_count_crossref', 0), article_data.get('publication_year')
            ),
            'annual_citation_rate_openalex': self.safe_calculate_annual_citation_rate(
                article_data.get('citation_count_openalex', 0), article_data.get('publication_year')
            ),
            'years_since_publication': article_data.get('years_since_publication', 1),
            'affiliations': article_data.get('affiliations', 'Unknown'),
            'countries': article_data.get('countries', 'Unknown'),
            'altmetric_score': article_data.get('altmetric_score', 0),
            'number_of_mentions': article_data.get('number_of_mentions', 0),
            'x_mentions': article_data.get('x_mentions', 0),
            'rss_blogs': article_data.get('rss_blogs', 0),
            'unique_accounts': article_data.get('unique_accounts', 0),
            'error': None
        }

    def _build_citing_dataframes(self, all_citing_connections: List[Dict], citing_data_cache: Dict) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Build citing articles and details DataFrames"""
        all_citing_articles_data = []
        citing_articles_details = []

        for connection in all_citing_connections:
            citing_doi = connection['citing_doi']
            source_doi = connection['source_doi']

            article_data = citing_data_cache.get(citing_doi, {})
            
            # Main citing articles DataFrame
            citing_row = self._build_citing_article_row(source_doi, citing_doi, article_data)
            all_citing_articles_data.append(citing_row)
            
            # Details DataFrame
            citing_articles_details.append({
                'source_doi': source_doi,
                'citing_doi': citing_doi,
                'citing_title': article_data.get('title', 'Unknown'),
                'citing_authors': article_data.get('authors_with_initials', 'Unknown'),
                'citing_year': article_data.get('year', 'Unknown'),
                'citing_journal': article_data.get('journal_abbreviation', 'Unknown'),
                'citation_count': article_data.get('citation_count_openalex', 0),
                'altmetric_score': article_data.get('altmetric_score', 0),
                'number_of_mentions': article_data.get('number_of_mentions', 0)
            })

        citing_articles_df = pd.DataFrame(all_citing_articles_data) if all_citing_articles_data else pd.DataFrame()
        citing_details_df = pd.DataFrame(citing_articles_details) if citing_articles_details else pd.DataFrame()

        return citing_articles_df, citing_details_df

    # =====================================================================
    # ANALYSIS METHODS FOR REFERENCES
    # =====================================================================

    def get_unique_references(self, references_df: pd.DataFrame) -> pd.DataFrame:
        """Gets unique references"""
        if references_df.empty:
            return pd.DataFrame()

        cache_key = id(references_df)
        if cache_key not in self._unique_references_cache:
            references_df['ref_id'] = references_df['doi'].fillna('') + '|' + references_df['title'].fillna('')
            unique_df = references_df.drop_duplicates(subset=['ref_id'], keep='first').drop(columns=['ref_id'])
            self._unique_references_cache[cache_key] = unique_df
        return self._unique_references_cache[cache_key]

    def analyze_authors_frequency(self, references_df: pd.DataFrame) -> pd.DataFrame:
        """Analyzes author frequency in references"""
        try:
            if references_df.empty:
                return pd.DataFrame()

            total_refs = len(references_df)
            unique_df = self.get_unique_references(references_df)
            total_unique = len(unique_df)
            
            authors_total = references_df['authors_with_initials'].str.split(',', expand=True).stack()
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
        except Exception as e:
            return pd.DataFrame()

    def analyze_journals_frequency(self, references_df: pd.DataFrame) -> pd.DataFrame:
        """Analyzes journal frequency in references with additional metrics"""
        try:
            if references_df.empty:
                return pd.DataFrame()

            total_refs = len(references_df)
            unique_df = self.get_unique_references(references_df)
            total_unique = len(unique_df)
            
            journals_total = references_df['journal_abbreviation']
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

            # Add citation metrics
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
        except Exception as e:
            return pd.DataFrame()

    def analyze_affiliations_frequency(self, references_df: pd.DataFrame) -> pd.DataFrame:
        """Analyzes affiliation frequency in references with grouping"""
        try:
            if references_df.empty:
                return pd.DataFrame()

            total_refs = len(references_df)
            unique_df = self.get_unique_references(references_df)
            total_unique = len(unique_df)

            all_affiliations = []
            for affil_string in references_df['affiliations']:
                if pd.isna(affil_string) or affil_string in ['Unknown', 'Error', '']:
                    continue

                try:
                    affil_list = affil_string.split(';')
                    for affil in affil_list:
                        clean_affil = affil.strip()
                        if clean_affil and clean_affil not in ['Unknown', 'Error']:
                            all_affiliations.append(clean_affil)
                except Exception:
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

        except Exception as e:
            return pd.DataFrame()

    def analyze_countries_frequency(self, references_df: pd.DataFrame) -> pd.DataFrame:
        """Analyzes country frequency in references with separation into single countries and collaborations"""
        try:
            if references_df.empty:
                return pd.DataFrame()

            total_refs = len(references_df)
            unique_df = self.get_unique_references(references_df)
            total_unique = len(unique_df)

            # Analysis of individual countries
            single_countries = []
            collaborations = []

            for countries in references_df['countries']:
                if countries not in ['Unknown', 'Error']:
                    country_list = [c.strip() for c in countries.split(';')]
                    if len(country_list) == 1:
                        single_countries.extend(country_list)
                    else:
                        collaborations.append(countries)

            # Frequency of individual countries
            single_country_counter = Counter(single_countries)
            single_country_freq = pd.DataFrame({
                'countries': list(single_country_counter.keys()),
                'type': ['single'] * len(single_country_counter),
                'frequency_total': list(single_country_counter.values())
            })
            single_country_freq['percentage_total'] = round(single_country_freq['frequency_total'] / total_refs * 100, 2)

            # Frequency of collaborations
            collaboration_counter = Counter(collaborations)
            collaboration_freq = pd.DataFrame({
                'countries': list(collaboration_counter.keys()),
                'type': ['collaboration'] * len(collaboration_counter),
                'frequency_total': list(collaboration_counter.values())
            })
            collaboration_freq['percentage_total'] = round(collaboration_freq['frequency_total'] / total_refs * 100, 2)

            # Combine
            country_freq_total = pd.concat([single_country_freq, collaboration_freq], ignore_index=True)

            # Similarly for unique articles
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
        except Exception as e:
            return pd.DataFrame()

    def analyze_year_distribution(self, references_df: pd.DataFrame) -> pd.DataFrame:
        """Analyzes year distribution for references (from new to old)"""
        try:
            if references_df.empty:
                return pd.DataFrame()

            total_refs = len(references_df)
            unique_df = self.get_unique_references(references_df)
            total_unique = len(unique_df)
            
            years_total = pd.to_numeric(references_df['year'], errors='coerce')
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
        except Exception as e:
            return pd.DataFrame()

    def analyze_five_year_periods(self, references_df: pd.DataFrame) -> pd.DataFrame:
        """Analyzes five-year periods for references (from new to old)"""
        try:
            if references_df.empty:
                return pd.DataFrame()

            total_refs = len(references_df)
            unique_df = self.get_unique_references(references_df)
            total_unique = len(unique_df)
            
            start_year = 1900
            current_year = datetime.now().year + 4
            period_starts = list(range(start_year, current_year + 1, 5))
            bins = period_starts + [period_starts[-1] + 5]
            labels = [f"{s}-{s+4}" for s in period_starts]
            
            years_total = pd.to_numeric(references_df['year'], errors='coerce')
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
        except Exception as e:
            return pd.DataFrame()

    def find_duplicate_references(self, references_df: pd.DataFrame) -> pd.DataFrame:
        """Finds duplicate references (articles that are cited by multiple analyzed works)"""
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
        except Exception as e:
            return pd.DataFrame()

    # =====================================================================
    # ANALYSIS METHODS FOR CITING ARTICLES
    # =====================================================================

    def get_unique_citations(self, citations_df: pd.DataFrame) -> pd.DataFrame:
        """Gets unique citing articles"""
        if citations_df.empty:
            return pd.DataFrame()

        cache_key = id(citations_df)
        if cache_key not in self._unique_citations_cache:
            citations_df['citation_id'] = citations_df['doi'].fillna('') + '|' + citations_df['title'].fillna('')
            unique_df = citations_df.drop_duplicates(subset=['citation_id'], keep='first').drop(columns=['citation_id'])
            self._unique_citations_cache[cache_key] = unique_df
        return self._unique_citations_cache[cache_key]

    def find_duplicate_citations(self, citations_df: pd.DataFrame) -> pd.DataFrame:
        """Finds duplicate citations (articles that cite multiple analyzed works)"""
        try:
            if citations_df.empty:
                return pd.DataFrame()

            citations_df['citation_id'] = citations_df['doi'].fillna('') + '|' + citations_df['title'].fillna('')

            # Count how many different source articles each citing article cites
            citation_counts = citations_df.groupby('citation_id')['source_doi'].nunique().reset_index()
            duplicate_citation_ids = citation_counts[citation_counts['source_doi'] > 1]['citation_id']

            if duplicate_citation_ids.empty:
                columns = list(citations_df.columns) + ['frequency']
                columns.remove('citation_id')
                return pd.DataFrame(columns=columns)

            # Create mapping citation_id -> frequency (number of cited source articles)
            frequency_map = citation_counts.set_index('citation_id')['source_doi'].to_dict()

            # Select duplicate citations and keep one record per citing article
            duplicates = citations_df[citations_df['citation_id'].isin(duplicate_citation_ids)].copy()
            duplicates = duplicates.drop_duplicates(subset=['citation_id'], keep='first')

            # Filter invalid records
            duplicates = duplicates[~((duplicates['doi'].isna()) & (duplicates['title'] == 'Unknown'))]

            # Add frequency column
            duplicates['frequency'] = duplicates['citation_id'].map(frequency_map)
            duplicates = duplicates.drop(columns=['citation_id'])

            return duplicates.sort_values(['frequency', 'doi'], ascending=[False, True])
        except Exception as e:
            return pd.DataFrame()

    def analyze_citation_authors_frequency(self, citations_df: pd.DataFrame) -> pd.DataFrame:
        """Analyzes author frequency in citing articles"""
        try:
            if citations_df.empty:
                return pd.DataFrame()

            total_citations = len(citations_df)
            unique_df = self.get_unique_citations(citations_df)
            total_unique = len(unique_df)

            authors_total = citations_df['authors_with_initials'].str.split(',', expand=True).stack()
            authors_total = authors_total[authors_total.str.strip().isin(['Unknown', 'Error']) == False]
            author_freq_total = authors_total.value_counts().reset_index()
            author_freq_total.columns = ['author_with_initial', 'frequency_total']
            author_freq_total['percentage_total'] = round(author_freq_total['frequency_total'] / total_citations * 100, 2)

            authors_unique = unique_df['authors_with_initials'].str.split(',', expand=True).stack()
            authors_unique = authors_unique[authors_unique.str.strip().isin(['Unknown', 'Error']) == False]
            author_freq_unique = authors_unique.value_counts().reset_index()
            author_freq_unique.columns = ['author_with_initial', 'frequency_unique']
            author_freq_unique['percentage_unique'] = round(author_freq_unique['frequency_unique'] / total_unique * 100, 2)

            author_freq = author_freq_total.merge(author_freq_unique, on='author_with_initial', how='outer').fillna(0)
            return author_freq[['author_with_initial', 'frequency_total', 'percentage_total', 'frequency_unique', 'percentage_unique']].sort_values('frequency_total', ascending=False)
        except Exception as e:
            return pd.DataFrame()

    def analyze_citation_journals_frequency(self, citations_df: pd.DataFrame) -> pd.DataFrame:
        """Analyzes journal frequency in citing articles with additional metrics"""
        try:
            if citations_df.empty:
                return pd.DataFrame()

            total_citations = len(citations_df)
            unique_df = self.get_unique_citations(citations_df)
            total_unique = len(unique_df)

            journals_total = citations_df['journal_abbreviation']
            journals_total = journals_total[journals_total.isin(['Unknown', 'Error']) == False]
            journal_freq_total = journals_total.value_counts().reset_index()
            journal_freq_total.columns = ['journal_abbreviation', 'frequency_total']
            journal_freq_total['percentage_total'] = round(journal_freq_total['frequency_total'] / total_citations * 100, 2)

            journals_unique = unique_df['journal_abbreviation']
            journals_unique = journals_unique[journals_unique.isin(['Unknown', 'Error']) == False]
            journal_freq_unique = journals_unique.value_counts().reset_index()
            journal_freq_unique.columns = ['journal_abbreviation', 'frequency_unique']
            journal_freq_unique['percentage_unique'] = round(journal_freq_unique['frequency_unique'] / total_unique * 100, 2)

            journal_freq = journal_freq_total.merge(journal_freq_unique, on='journal_abbreviation', how='outer').fillna(0)

            # Add citation metrics for citations
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
        except Exception as e:
            return pd.DataFrame()

    def analyze_citation_affiliations_frequency(self, citations_df: pd.DataFrame) -> pd.DataFrame:
        """Analyzes affiliation frequency in citing articles with grouping"""
        try:
            if citations_df.empty:
                return pd.DataFrame()

            total_citations = len(citations_df)
            unique_df = self.get_unique_citations(citations_df)
            total_unique = len(unique_df)

            all_affiliations = []
            for affil_string in citations_df['affiliations']:
                if pd.isna(affil_string) or affil_string in ['Unknown', 'Error', '']:
                    continue

                try:
                    affil_list = affil_string.split(';')
                    for affil in affil_list:
                        clean_affil = affil.strip()
                        if clean_affil and clean_affil not in ['Unknown', 'Error']:
                            all_affiliations.append(clean_affil)
                except Exception:
                    pass

            if not all_affiliations:
                return pd.DataFrame(columns=['affiliation', 'frequency_total', 'percentage_total', 'frequency_unique', 'percentage_unique'])

            affiliation_frequencies, grouped_organizations = self.fast_affiliation_processor.process_affiliations_list_fast(all_affiliations)

            affil_data = []
            for affil, freq in affiliation_frequencies.items():
                percentage_total = round(freq / total_citations * 100, 2) if total_citations > 0 else 0
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

        except Exception as e:
            return pd.DataFrame()

    def analyze_citation_countries_frequency(self, citations_df: pd.DataFrame) -> pd.DataFrame:
        """Analyzes country frequency in citing articles with separation into single countries and collaborations"""
        try:
            if citations_df.empty:
                return pd.DataFrame()

            total_citations = len(citations_df)
            unique_df = self.get_unique_citations(citations_df)
            total_unique = len(unique_df)

            # Analysis of individual countries
            single_countries = []
            collaborations = []

            for countries in citations_df['countries']:
                if countries not in ['Unknown', 'Error']:
                    country_list = [c.strip() for c in countries.split(';')]
                    if len(country_list) == 1:
                        single_countries.extend(country_list)
                    else:
                        collaborations.append(countries)

            # Frequency of individual countries
            single_country_counter = Counter(single_countries)
            single_country_freq = pd.DataFrame({
                'countries': list(single_country_counter.keys()),
                'type': ['single'] * len(single_country_counter),
                'frequency_total': list(single_country_counter.values())
            })
            single_country_freq['percentage_total'] = round(single_country_freq['frequency_total'] / total_citations * 100, 2)

            # Frequency of collaborations
            collaboration_counter = Counter(collaborations)
            collaboration_freq = pd.DataFrame({
                'countries': list(collaboration_counter.keys()),
                'type': ['collaboration'] * len(collaboration_counter),
                'frequency_total': list(collaboration_counter.values())
            })
            collaboration_freq['percentage_total'] = round(collaboration_freq['frequency_total'] / total_citations * 100, 2)

            # Combine
            country_freq_total = pd.concat([single_country_freq, collaboration_freq], ignore_index=True)

            # Similarly for unique articles
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
        except Exception as e:
            return pd.DataFrame()

    def analyze_citation_year_distribution(self, citations_df: pd.DataFrame) -> pd.DataFrame:
        """Analyzes year distribution for citing articles (from new to old)"""
        try:
            if citations_df.empty:
                return pd.DataFrame()

            total_citations = len(citations_df)
            unique_df = self.get_unique_citations(citations_df)
            total_unique = len(unique_df)

            years_total = pd.to_numeric(citations_df['year'], errors='coerce')
            years_total = years_total[years_total.notna() & years_total.between(1900, 2026)].astype(int)
            year_counts_total = years_total.value_counts().reset_index()
            year_counts_total.columns = ['year', 'frequency_total']
            year_counts_total['percentage_total'] = round(year_counts_total['frequency_total'] / total_citations * 100, 2)

            years_unique = pd.to_numeric(unique_df['year'], errors='coerce')
            years_unique = years_unique[years_unique.notna() & years_unique.between(1900, 2026)].astype(int)
            year_counts_unique = years_unique.value_counts().reset_index()
            year_counts_unique.columns = ['year', 'frequency_unique']
            year_counts_unique['percentage_unique'] = round(year_counts_unique['frequency_unique'] / total_unique * 100, 2)

            year_counts = year_counts_total.merge(year_counts_unique, on='year', how='outer').fillna(0)
            return year_counts[['year', 'frequency_total', 'percentage_total', 'frequency_unique', 'percentage_unique']].sort_values('year', ascending=False)
        except Exception as e:
            return pd.DataFrame()

    def analyze_citation_five_year_periods(self, citations_df: pd.DataFrame) -> pd.DataFrame:
        """Analyzes five-year periods for citing articles (from new to old)"""
        try:
            if citations_df.empty:
                return pd.DataFrame()

            total_citations = len(citations_df)
            unique_df = self.get_unique_citations(citations_df)
            total_unique = len(unique_df)

            start_year = 1900
            current_year = datetime.now().year + 4
            period_starts = list(range(start_year, current_year + 1, 5))
            bins = period_starts + [period_starts[-1] + 5]
            labels = [f"{s}-{s+4}" for s in period_starts]

            years_total = pd.to_numeric(citations_df['year'], errors='coerce')
            years_total = years_total[years_total.notna() & years_total.between(1900, current_year)].astype(int)
            period_counts_total = pd.cut(years_total, bins=bins, labels=labels, right=False).astype(str)
            period_df_total = period_counts_total.value_counts().reset_index()
            period_df_total.columns = ['period', 'frequency_total']
            period_df_total['percentage_total'] = round(period_df_total['frequency_total'] / total_citations * 100, 2)
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
        except Exception as e:
            return pd.DataFrame()

    # =====================================================================
    # TITLE ANALYSIS METHODS
    # =====================================================================

    def preprocess_content_words(self, text: str) -> List[str]:
        """Preprocesses content words from text"""
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
                stemmed_word = self.stemmer.stem(word)
                if stemmed_word not in self.scientific_stopwords_stemmed:
                    content_words.append(stemmed_word)
        return content_words

    def extract_compound_words(self, text: str) -> List[str]:
        """Extracts compound words from text"""
        if not text or text in ['Unknown', 'Error']:
            return []
        text = text.lower()
        compound_words = re.findall(r'\b[a-z]{2,}-[a-z]{2,}(?:-[a-z]{2,})*\b', text)
        return [word for word in compound_words if not any(part in self.stop_words for part in word.split('-'))]

    def extract_scientific_stopwords(self, text: str) -> List[str]:
        """Extracts scientific stopwords from text"""
        if not text or text in ['Unknown', 'Error']:
            return []
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        words = text.split()
        scientific_words = []
        for word in words:
            if len(word) > 2:
                stemmed_word = self.stemmer.stem(word)
                if stemmed_word in self.scientific_stopwords_stemmed:
                    for original_word in self.scientific_stopwords:
                        if self.stemmer.stem(original_word) == stemmed_word:
                            scientific_words.append(original_word)
                            break
        return scientific_words

    def analyze_titles(self, titles: List[str]) -> tuple[Counter, Counter, Counter]:
        """Analyzes titles for content words, compound words, and scientific stopwords"""
        content_words = []
        compound_words = []
        scientific_words = []
        valid_titles = [t for t in titles if t not in ['Unknown', 'Error']]
        for title in valid_titles:
            content_words.extend(self.preprocess_content_words(title))
            compound_words.extend(self.extract_compound_words(title))
            scientific_words.extend(self.extract_scientific_stopwords(title))
        return Counter(content_words), Counter(compound_words), Counter(scientific_words)

    # =====================================================================
    # DATA ENHANCEMENT METHODS
    # =====================================================================

    def enhance_incomplete_data(self, references_df: pd.DataFrame) -> pd.DataFrame:
        """Enhances references with incomplete data"""
        if references_df.empty:
            return references_df

        enhanced_rows = []
        incomplete_count = 0

        for index, row in references_df.iterrows():
            doi = row['doi']

            needs_enhancement = (
                pd.isna(doi) or
                row['title'] == 'Unknown' or
                row['authors'] == 'Unknown' or
                row['affiliations'] == 'Unknown' or
                row['countries'] == 'Unknown' or
                pd.notna(row.get('error'))
            )

            if needs_enhancement and doi and self.validate_doi(doi):
                incomplete_count += 1
                try:
                    enhanced_data = self.get_combined_article_data(doi)
                    enhanced_row = {
                        'source_doi': row['source_doi'],
                        'position': row['position'],
                        'doi': doi,
                        'title': enhanced_data['title'],
                        'authors': enhanced_data['authors'],
                        'authors_with_initials': enhanced_data['authors_with_initials'],
                        'author_count': enhanced_data['author_count'],
                        'year': enhanced_data['year'],
                        'journal_full_name': enhanced_data['journal_full_name'],
                        'journal_abbreviation': enhanced_data['journal_abbreviation'],
                        'publisher': enhanced_data['publisher'],
                        'citation_count_crossref': enhanced_data['citation_count_crossref'],
                        'citation_count_openalex': enhanced_data['citation_count_openalex'],
                        'annual_citation_rate_crossref': self.safe_calculate_annual_citation_rate(
                            enhanced_data['citation_count_crossref'], enhanced_data.get('publication_year')
                        ),
                        'annual_citation_rate_openalex': self.safe_calculate_annual_citation_rate(
                            enhanced_data['citation_count_openalex'], enhanced_data.get('publication_year')
                        ),
                        'years_since_publication': enhanced_data['years_since_publication'],
                        'affiliations': enhanced_data['affiliations'],
                        'countries': enhanced_data['countries'],
                        'altmetric_score': enhanced_data['altmetric_score'],
                        'number_of_mentions': enhanced_data['number_of_mentions'],
                        'x_mentions': enhanced_data['x_mentions'],
                        'rss_blogs': enhanced_data['rss_blogs'],
                        'unique_accounts': enhanced_data['unique_accounts'],
                        'error': None
                    }
                    enhanced_rows.append(enhanced_row)
                    time.sleep(Config.DELAY_BETWEEN_REQUESTS)
                    continue
                except Exception as e:
                    pass

            enhanced_rows.append(row.to_dict())

        return pd.DataFrame(enhanced_rows)

    def reprocess_failed_references(self, failed_references_df: pd.DataFrame) -> pd.DataFrame:
        """Reprocesses failed references to find missing DOIs and data"""
        if failed_references_df.empty:
            return failed_references_df

        updated_rows = []

        for index, row in failed_references_df.iterrows():
            original_doi = row.get('reference_doi') if 'reference_doi' in row else row.get('doi')
            error_description = row.get('error_description', '') if 'error_description' in row else row.get('error', '')

            title_match = re.search(r"title '([^']+)'|title ([^,]+)", str(error_description))
            title = None
            if title_match:
                title = next((g for g in title_match.groups() if g), None)

            if not title and 'title' in row:
                title = row['title']

            found_doi = None
            if title and title != 'Unknown':
                found_doi = self.quick_doi_search(title)
                time.sleep(Config.DELAY_BETWEEN_REQUESTS)

            if found_doi and self.validate_doi(found_doi):
                try:
                    enhanced_data = self.get_combined_article_data(found_doi)
                    enhanced_data.update({
                        'source_doi': row.get('source_doi'),
                        'position': row.get('position'),
                        'error': None,
                        'annual_citation_rate_crossref': self.safe_calculate_annual_citation_rate(
                            enhanced_data['citation_count_crossref'], enhanced_data.get('publication_year')
                        ),
                        'annual_citation_rate_openalex': self.safe_calculate_annual_citation_rate(
                            enhanced_data['citation_count_openalex'], enhanced_data.get('publication_year')
                        )
                    })
                    updated_rows.append(enhanced_data)
                    continue
                except Exception as e:
                    pass

            updated_row = row.copy()
            if 'updated_doi' not in updated_row:
                updated_row['updated_doi'] = found_doi if found_doi else original_doi
            if 'updated_error' not in updated_row:
                updated_row['updated_error'] = f"DOI found: {found_doi}" if found_doi else f"No DOI found for title '{title}'" if title else "No title available"

            updated_rows.append(updated_row)

        return pd.DataFrame(updated_rows)

    # =====================================================================
    # EXPORT METHODS
    # =====================================================================

    def save_all_data_to_excel(self, combined_df: pd.DataFrame, source_articles_df: pd.DataFrame,
                         doi_list: List[str], total_references: int, unique_dois: int,
                         all_titles: List[str]) -> BytesIO:
        """Saves references analysis to Excel with altmetrics"""
        try:
            timestamp = int(time.time())
            excel_buffer = BytesIO()

            wb = Workbook()

            # First create Report_Summary tab
            ws_summary = wb.active
            ws_summary.title = 'Report_Summary'

            wb.remove(wb.active)

            try:
                unique_df = self.get_unique_references(combined_df)
            except Exception as e:
                unique_df = pd.DataFrame()

            try:
                duplicate_df = self.find_duplicate_references(combined_df)
            except Exception as e:
                duplicate_df = pd.DataFrame()

            try:
                failed_df = combined_df[combined_df['error'].notna()][['source_doi', 'position', 'doi', 'error']].copy()
                failed_df.columns = ['source_doi', 'ref_number', 'reference_doi', 'error_description']
            except Exception as e:
                failed_df = pd.DataFrame()

            stats = self.performance_monitor.get_stats()

            try:
                content_freq, compound_freq, scientific_freq = self.analyze_titles(all_titles)
            except Exception as e:
                content_freq, compound_freq, scientific_freq = Counter(), Counter(), Counter()

            try:
                total_processed = len(combined_df) if not combined_df.empty else 0

                countries_with_data = combined_df[combined_df['countries'].isin(['Unknown', 'Error']) == False]
                countries_percentage = (len(countries_with_data) / total_processed * 100) if total_processed > 0 else 0

                affiliations_with_data = combined_df[combined_df['affiliations'].isin(['Unknown', 'Error']) == False]
                affiliations_percentage = (len(affiliations_with_data) / total_processed * 100) if total_processed > 0 else 0

                altmetric_with_data = combined_df[combined_df['altmetric_score'] > 0]
                altmetric_percentage = (len(altmetric_with_data) / total_processed * 100) if total_processed > 0 else 0

            except Exception as e:
                countries_percentage = 0
                affiliations_percentage = 0
                altmetric_percentage = 0

            summary_content = f"""@MedvDmitry production

REFERENCES ANALYSIS REPORT

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

ANALYSIS OVERVIEW
=================
Total source articles: {len(doi_list)}
Total references collected: {total_references}
Unique DOIs identified: {unique_dois}
Total references processed: {len(combined_df) if not combined_df.empty else 0}
Unique references: {len(unique_df) if not unique_df.empty else 0}
Successful references: {len(combined_df[combined_df['error'].isna()]) if not combined_df.empty else 0}
Failed references: {len(combined_df[combined_df['error'].notna()]) if not combined_df.empty else 0}
Unique authors: {len(self.analyze_authors_frequency(combined_df)) if not combined_df.empty else 0}
Unique journals: {len(self.analyze_journals_frequency(combined_df)) if not combined_df.empty else 0}
Unique affiliations: {len(self.analyze_affiliations_frequency(combined_df)) if not combined_df.empty else 0}
Unique countries: {len(self.analyze_countries_frequency(combined_df)) if not combined_df.empty else 0}
Duplicate references: {len(duplicate_df) if not duplicate_df.empty else 0}

DATA COMPLETENESS
=================
References with country data: {countries_percentage:.1f}%
References with affiliation data: {affiliations_percentage:.1f}%
References with altmetric data: {altmetric_percentage:.1f}%

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

PERFORMANCE STATISTICS
======================
Total processing time: {stats.get('elapsed_seconds', 0):.2f} seconds ({stats.get('elapsed_minutes', 0):.2f} minutes)
Total API requests: {stats.get('total_requests', 0)}
Requests per second: {stats.get('requests_per_second', 0):.2f}

DATA QUALITY NOTES
==================
Analysis focuses on references cited by the source articles
Combined data from Crossref and OpenAlex improves completeness
All standard statistical analyses performed (authors, journals, countries, etc.)
Error handling ensures report generation even with partial data
Affiliations normalized and grouped for consistent organization names
Altmetric metrics provide social media and online attention analysis
"""

            # Create Report_Summary tab first
            ws_summary = wb.create_sheet('Report_Summary')
            for line in summary_content.split('\n'):
                ws_summary.append([line])

            # Remove authors_surnames column and add author_count for main tables
            if not source_articles_df.empty and 'authors_surnames' in source_articles_df.columns:
                source_articles_df = source_articles_df.drop(columns=['authors_surnames'])
            if not combined_df.empty and 'authors_surnames' in combined_df.columns:
                combined_df = combined_df.drop(columns=['authors_surnames'])
            if not unique_df.empty and 'authors_surnames' in unique_df.columns:
                unique_df = unique_df.drop(columns=['authors_surnames'])
            if not duplicate_df.empty and 'authors_surnames' in duplicate_df.columns:
                duplicate_df = duplicate_df.drop(columns=['authors_surnames'])

            sheets_data = [
                ('Source_Articles', source_articles_df),
                ('All_References', combined_df),
                ('All_Unique_References', unique_df),
                ('Duplicate_References', duplicate_df),
                ('Failed_References', failed_df)
            ]

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
                    result_df = method(combined_df)
                    sheets_data.append((sheet_name, result_df))
                except Exception as e:
                    sheets_data.append((sheet_name, pd.DataFrame()))

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

            wb.save(excel_buffer)
            excel_buffer.seek(0)
            return excel_buffer

        except Exception as e:
            try:
                timestamp = int(time.time())
                excel_buffer = BytesIO()
                wb = Workbook()
                ws = wb.active
                ws.title = "Error_Report_References"
                ws.append(["ERROR REPORT - REFERENCES ANALYSIS"])
                ws.append([f"Critical error during references analysis: {str(e)}"])
                ws.append([f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])
                ws.append(["DOIs processed:", ', '.join(doi_list)])
                wb.save(excel_buffer)
                excel_buffer.seek(0)
                return excel_buffer
            except:
                return BytesIO()

    def save_citation_analysis_to_excel(self, citing_articles_df: pd.DataFrame, citing_details_df: pd.DataFrame,
                                      doi_list: List[str], citing_results: Dict, all_citing_titles: List[str]) -> BytesIO:
        """Saves full citing articles analysis to Excel"""
        try:
            timestamp = int(time.time())
            excel_buffer = BytesIO()

            wb = Workbook()

            # First create Report_Summary tab
            ws_summary = wb.active
            ws_summary.title = 'Report_Summary_Citations'

            wb.remove(wb.active)

            try:
                unique_citations_df = self.get_unique_citations(citing_articles_df)
            except Exception as e:
                unique_citations_df = pd.DataFrame()

            try:
                duplicate_citations_df = self.find_duplicate_citations(citing_articles_df)
            except Exception as e:
                duplicate_citations_df = pd.DataFrame()

            stats = self.performance_monitor.get_stats()

            try:
                content_freq, compound_freq, scientific_freq = self.analyze_titles(all_citing_titles)
            except Exception as e:
                content_freq, compound_freq, scientific_freq = Counter(), Counter(), Counter()

            try:
                total_citation_relationships = len(citing_articles_df) if not citing_articles_df.empty else 0
                total_unique_citations = len(unique_citations_df) if not unique_citations_df.empty else 0

                countries_with_data = citing_articles_df[citing_articles_df['countries'].isin(['Unknown', 'Error']) == False]
                countries_percentage = (len(countries_with_data) / total_citation_relationships * 100) if total_citation_relationships > 0 else 0

                affiliations_with_data = citing_articles_df[citing_articles_df['affiliations'].isin(['Unknown', 'Error']) == False]
                affiliations_percentage = (len(affiliations_with_data) / total_citation_relationships * 100) if total_citation_relationships > 0 else 0

            except Exception as e:
                total_citation_relationships = 0
                total_unique_citations = 0
                countries_percentage = 0
                affiliations_percentage = 0

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
Total citation relationships: {total_citation_relationships}
Total unique citing articles: {total_unique_citations}
Successful citations: {len(citing_articles_df[citing_articles_df['error'].isna()]) if not citing_articles_df.empty else 0}
Failed citations: {len(citing_articles_df[citing_articles_df['error'].notna()]) if not citing_articles_df.empty else 0}
Unique authors: {len(self.analyze_citation_authors_frequency(citing_articles_df)) if not citing_articles_df.empty else 0}
Unique journals: {len(self.analyze_citation_journals_frequency(citing_articles_df)) if not citing_articles_df.empty else 0}
Unique affiliations: {len(self.analyze_citation_affiliations_frequency(citing_articles_df)) if not citing_articles_df.empty else 0}
Unique countries: {len(self.analyze_citation_countries_frequency(citing_articles_df)) if not citing_articles_df.empty else 0}
Duplicate citations: {len(duplicate_citations_df) if not duplicate_citations_df.empty else 0}

DATA COMPLETENESS
=================
Articles with country data: {countries_percentage:.1f}%
Articles with affiliation data: {affiliations_percentage:.1f}%

AFFILIATION PROCESSING
======================
Affiliations normalized and grouped by organization
Similar affiliations merged together
Frequency counts reflect grouped organizations

PERFORMANCE STATISTICS
======================
Total processing time: {stats.get('elapsed_seconds', 0):.2f} seconds ({stats.get('elapsed_minutes', 0):.2f} minutes)
Total API requests: {stats.get('total_requests', 0)}
Requests per second: {stats.get('requests_per_second', 0):.2f}
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
"""

            # Create Report_Summary tab first
            ws_summary = wb.create_sheet('Report_Summary_Citations')
            for line in summary_content.split('\n'):
                ws_summary.append([line])

            # Remove authors_surnames column and add author_count for main tables
            if not citing_articles_df.empty and 'authors_surnames' in citing_articles_df.columns:
                citing_articles_df = citing_articles_df.drop(columns=['authors_surnames'])
            if not unique_citations_df.empty and 'authors_surnames' in unique_citations_df.columns:
                unique_citations_df = unique_citations_df.drop(columns=['authors_surnames'])
            if not duplicate_citations_df.empty and 'authors_surnames' in duplicate_citations_df.columns:
                duplicate_citations_df = duplicate_citations_df.drop(columns=['authors_surnames'])

            sheets_data = [
                ('Source_Articles_Citations', citing_details_df),
                ('All_Citations', citing_articles_df),
                ('All_Unique_Citations', unique_citations_df),
                ('Duplicate_Citations', duplicate_citations_df)
            ]

            analysis_methods = [
                ('Author_Frequency_Citations', self.analyze_citation_authors_frequency),
                ('Journal_Frequency_Citations', self.analyze_citation_journals_frequency),
                ('Affiliation_Frequency_Citations', self.analyze_citation_affiliations_frequency),
                ('Country_Frequency_Citations', self.analyze_citation_countries_frequency),
                ('Year_Distribution_Citations', self.analyze_citation_year_distribution),
                ('5_Years_Period_Citations', self.analyze_citation_five_year_periods)
            ]

            for sheet_name, method in analysis_methods:
                try:
                    result_df = method(citing_articles_df)
                    sheets_data.append((sheet_name, result_df))
                except Exception as e:
                    sheets_data.append((sheet_name, pd.DataFrame()))

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

            wb.save(excel_buffer)
            excel_buffer.seek(0)
            return excel_buffer

        except Exception as e:
            try:
                timestamp = int(time.time())
                excel_buffer = BytesIO()
                wb = Workbook()
                ws = wb.active
                ws.title = "Error_Report_Citations"
                ws.append(["ERROR REPORT - CITATION ANALYSIS"])
                ws.append([f"Critical error during citation analysis: {str(e)}"])
                ws.append([f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])
                ws.append(["DOIs processed:", ', '.join(doi_list)])
                wb.save(excel_buffer)
                excel_buffer.seek(0)
                return excel_buffer
            except:
                return BytesIO()

    def display_analysis_results(self, combined_df: pd.DataFrame, source_articles_df: pd.DataFrame,
                         doi_list: List[str], total_references: int, unique_dois: int,
                         all_titles: List[str]) -> None:
        """Displays references analysis results with altmetrics"""
        try:
            st.markdown(f"{'='*80}\n**REFERENCES ANALYSIS RESULTS FOR {len(doi_list)} ARTICLES**\n{'='*80}")

            if combined_df.empty and source_articles_df.empty:
                st.error("No data available - generating error report")
                excel_buffer = self.save_all_data_to_excel(combined_df, source_articles_df, doi_list, total_references, unique_dois, all_titles)
                st.download_button(
                    label="Download Error Report",
                    data=excel_buffer.getvalue(),
                    file_name="error_references_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                return

            unique_df = self.get_unique_references(combined_df)
            successful_refs = len(combined_df[combined_df['error'].isna()])
            stats = self.performance_monitor.get_stats()

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total references found", total_references)
            col2.metric("Unique DOIs", unique_dois)
            col3.metric("Total references processed", len(combined_df))
            col4.metric("Unique references", len(unique_df))

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Successful references", successful_refs)
            col2.metric("Failed references", len(combined_df[combined_df['error'].notna()]))
            col3.metric("Total processing time", f"{stats.get('elapsed_seconds', 0):.2f} seconds")
            col4.metric("Requests per second", f"{stats.get('requests_per_second', 0):.2f}")

            st.subheader("References per article:")
            for doi in doi_list:
                ref_count = len(combined_df[combined_df['source_doi'] == doi]) if not combined_df.empty else 0
                st.write(f"  {doi}: {ref_count} references")

            display_cols = ['source_doi', 'position', 'doi', 'title', 'authors_with_initials', 'author_count', 'year',
                          'journal_abbreviation', 'publisher', 'countries', 'citation_count_crossref',
                          'citation_count_openalex', 'altmetric_score', 'number_of_mentions']

            pd.set_option('display.max_colwidth', 25)
            pd.set_option('display.max_rows', 50)

            if not source_articles_df.empty:
                st.subheader("SOURCE ARTICLES:")
                try:
                    st.dataframe(source_articles_df[display_cols].head(10))
                except Exception as e:
                    pass

            if not unique_df.empty:
                st.subheader("UNIQUE REFERENCES:")
                try:
                    st.dataframe(unique_df[display_cols].head(10))
                except Exception as e:
                    pass

            analyses = [
                ('DUPLICATE REFERENCES', self.find_duplicate_references),
                ('COUNTRIES FREQUENCY', self.analyze_countries_frequency),
                ('YEAR DISTRIBUTION', self.analyze_year_distribution),
                ('FIVE-YEAR PERIODS', self.analyze_five_year_periods),
                ('TOP 10 AUTHORS', self.analyze_authors_frequency),
                ('TOP 10 JOURNALS', self.analyze_journals_frequency),
                ('TOP 10 AFFILIATIONS', self.analyze_affiliations_frequency)
            ]

            for title, method in analyses:
                st.markdown(f"{'='*60}\n**{title}**\n{'='*60}")
                try:
                    result = method(combined_df)
                    if not result.empty:
                        if title == 'TOP 10 JOURNALS':
                            st.dataframe(result[['journal_abbreviation', 'frequency_total', 'percentage_total', 'frequency_unique', 'percentage_unique']].head(10))
                        else:
                            st.dataframe(result.head(10))
                    else:
                        st.info("No data available")
                except Exception as e:
                    st.error("Error in analysis")

            st.markdown(f"{'='*60}\n**TOP 15 TITLE WORDS**\n{'='*60}")
            try:
                content_freq, compound_freq, scientific_freq = self.analyze_titles(all_titles)
                content_df = pd.DataFrame(content_freq.most_common(15), columns=['Word', 'Frequency'])
                content_df['Category'] = 'Content_Words'
                content_df['Rank'] = range(1, len(content_df) + 1)
                compound_df = pd.DataFrame(compound_freq.most_common(15), columns=['Word', 'Frequency'])
                compound_df['Category'] = 'Compound_Words'
                compound_df['Rank'] = range(1, len(compound_df) + 1)
                scientific_df = pd.DataFrame(scientific_freq.most_common(15), columns=['Word', 'Frequency'])
                scientific_df['Category'] = 'Scientific_Stopwords'
                scientific_df['Rank'] = range(1, len(scientific_df) + 1)
                title_word_freq_df = pd.concat([content_df, compound_df, scientific_df], ignore_index=True)[['Category', 'Rank', 'Word', 'Frequency']]
                st.dataframe(title_word_freq_df)
            except Exception as e:
                st.error("Error in title analysis")

            excel_buffer = self.save_all_data_to_excel(combined_df, source_articles_df, doi_list, total_references, unique_dois, all_titles)
            st.download_button(
                label="Download Full Report (Excel)",
                data=excel_buffer.getvalue(),
                file_name="references_analysis_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        except Exception as e:
            st.error(f"Critical error in display_analysis_results: {e}")
            excel_buffer = self.save_all_data_to_excel(combined_df, source_articles_df, doi_list, total_references, unique_dois, all_titles)
            st.download_button(
                label="Download Error Report",
                data=excel_buffer.getvalue(),
                file_name="error_references_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    # =====================================================================
    # LEGACY METHODS (maintained for compatibility)
    # =====================================================================

    def process_doi_sequential(self, doi_list: List[str]) -> tuple[pd.DataFrame, pd.DataFrame, int, int, List[str]]:
        """Legacy method - now uses parallel processing internally"""
        return self.process_references_parallel(doi_list)

    def process_citing_articles_sequential(self, doi_list: List[str]) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict], List[str]]:
        """Legacy method - now uses parallel processing internally"""
        return self.process_citing_articles_parallel(doi_list)

# =====================================================================
# STREAMLIT APP
# =====================================================================

def main():
    st.title("Optimized Citation Analyzer")
    st.markdown("Analyze references and citing articles for given DOIs with parallel processing")

    # Initialize optimized analyzer
    analyzer = OptimizedCitationAnalyzer()

    # Clear expired cache on startup
    analyzer.persistent_cache.clear_expired()

    # Display cache statistics
    cache_stats = analyzer.persistent_cache.get_stats()
    st.sidebar.markdown(f"**Cache Stats:** {cache_stats['total_cached']} items")

    tab1, tab2 = st.tabs(["References Analysis", "Citing Articles Analysis"])

    with tab1:
        st.markdown("### Analyze References (Parallel)")
        st.markdown("Analyze the references cited by the input articles using parallel processing")
        
        # Check for existing progress
        progress = analyzer.progress_tracker.load_progress()
        if progress and progress.get('analysis_type') == 'references':
            st.warning("🔄 Previous references analysis detected! You can resume it.")
            if st.button("Resume Previous Analysis", type="primary"):
                with st.spinner("Resuming analysis..."):
                    doi_list = progress['doi_list']
                    try:
                        combined_references_df, source_articles_df, total_references, unique_dois, all_titles = analyzer.process_references_parallel(doi_list)
                        analyzer.display_analysis_results(combined_references_df, source_articles_df, doi_list, total_references, unique_dois, all_titles)
                    except Exception as e:
                        st.error(f"Error resuming analysis: {e}")
            st.markdown("---")
        
        doi_input_references = st.text_area(
            "DOIs:",
            value='10.1038/s41586-023-06924-6',
            placeholder='Enter DOIs for references analysis (e.g., 10.1010/XYZ, doi:10.1010/XYZ, https://doi.org/10.1010/XYZ, etc.) separated by any punctuation or newlines',
            height=200,
            key="references_doi_input"
        )
        
        if st.button("Analyze References (Parallel)", type="primary", key="analyze_references_btn"):
            with st.spinner("Processing in parallel..."):
                input_text = doi_input_references
                st.info(f"Input text: '{input_text}'")

                doi_list = analyzer.parse_doi_input(input_text)
                st.info(f"Parsed DOI list: {doi_list}")

                if not doi_list:
                    st.error("No valid DOIs provided. Please enter at least one valid DOI.")
                    st.info("Example formats:")
                    st.code("  10.1038/s41586-023-06924-6")
                    st.code("  https://doi.org/10.1038/s41586-023-06924-6")
                    st.code("  doi:10.1038/s41586-023-06924-6")
                    st.stop()

                st.success("Starting parallel processing for references analysis...")
                try:
                    combined_references_df, source_articles_df, total_references, unique_dois, all_titles = analyzer.process_references_parallel(doi_list)
                    analyzer.display_analysis_results(combined_references_df, source_articles_df, doi_list, total_references, unique_dois, all_titles)
                except Exception as e:
                    st.error(f"Critical error during processing: {e}")
                    empty_df = pd.DataFrame()
                    excel_buffer = analyzer.save_all_data_to_excel(empty_df, empty_df, doi_list, 0, 0, [])
                    st.download_button(
                        label="Download Error Report",
                        data=excel_buffer.getvalue(),
                        file_name="error_report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    st.info("Error report generated despite processing failure.")

    with tab2:
        st.markdown("### Analyze Citing Articles (Parallel)")
        st.markdown("Find articles that cite the input articles using parallel processing")
        
        # Check for existing progress
        progress = analyzer.progress_tracker.load_progress()
        if progress and progress.get('analysis_type') == 'citing_articles':
            st.warning("🔄 Previous citing articles analysis detected! You can resume it.")
            if st.button("Resume Previous Analysis", type="secondary"):
                with st.spinner("Resuming analysis..."):
                    doi_list = progress['doi_list']
                    try:
                        citing_articles_df, citing_details_df, citing_results, all_citing_titles = analyzer.process_citing_articles_parallel(doi_list)

                        if citing_results:
                            st.markdown(f"{'='*80}\n**CITING ARTICLES ANALYSIS RESULTS**\n{'='*80}")

                            total_citation_relationships = len(citing_articles_df) if citing_articles_df is not None else 0
                            total_unique_citations = len(analyzer.get_unique_citations(citing_articles_df)) if citing_articles_df is not None and not citing_articles_df.empty else 0

                            col1, col2 = st.columns(2)
                            col1.metric("Total source articles", len(doi_list))
                            col2.metric("Total citation relationships", total_citation_relationships)
                            st.metric("Total unique citing articles", total_unique_citations)

                            st.subheader("Citations per source article:")
                            for doi, data in citing_results.items():
                                st.write(f"  {doi}: {data['count']} citations")

                            if citing_articles_df is not None and not citing_articles_df.empty:
                                st.subheader("First 10 citing articles:")
                                display_cols = ['source_doi', 'doi', 'title', 'authors_with_initials', 'author_count', 'year', 'journal_abbreviation',
                                              'citation_count_openalex', 'altmetric_score', 'number_of_mentions']
                                st.dataframe(citing_articles_df[display_cols].head(10))

                            excel_buffer = analyzer.save_citation_analysis_to_excel(citing_articles_df, citing_details_df, doi_list, citing_results, all_citing_titles)
                            st.download_button(
                                label="Download Complete Citation Analysis (Excel)",
                                data=excel_buffer.getvalue(),
                                file_name="citing_analysis_report.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                            st.success("Complete citation analysis archived and ready for download")
                        else:
                            st.warning("No citing articles found.")
                    except Exception as e:
                        st.error(f"Error resuming analysis: {e}")
            st.markdown("---")
        
        doi_input_citing = st.text_area(
            "DOIs:",
            value='10.1038/s41586-023-06924-6',
            placeholder='Enter DOIs for citing articles analysis separated by any punctuation or newlines',
            height=200,
            key="citing_doi_input"
        )
        
        if st.button("Analyze Citing Articles (Parallel)", type="secondary", key="analyze_citing_btn"):
            with st.spinner("Processing in parallel..."):
                input_text = doi_input_citing
                doi_list = analyzer.parse_doi_input(input_text)

                if not doi_list:
                    st.error("No valid DOIs provided.")
                    st.stop()

                st.success("Starting parallel processing for citing articles analysis...")
                try:
                    citing_articles_df, citing_details_df, citing_results, all_citing_titles = analyzer.process_citing_articles_parallel(doi_list)

                    if citing_results:
                        st.markdown(f"{'='*80}\n**CITING ARTICLES ANALYSIS RESULTS**\n{'='*80}")

                        total_citation_relationships = len(citing_articles_df) if citing_articles_df is not None else 0
                        total_unique_citations = len(analyzer.get_unique_citations(citing_articles_df)) if citing_articles_df is not None and not citing_articles_df.empty else 0

                        col1, col2 = st.columns(2)
                        col1.metric("Total source articles", len(doi_list))
                        col2.metric("Total citation relationships", total_citation_relationships)
                        st.metric("Total unique citing articles", total_unique_citations)

                        st.subheader("Citations per source article:")
                        for doi, data in citing_results.items():
                            st.write(f"  {doi}: {data['count']} citations")

                        if citing_articles_df is not None and not citing_articles_df.empty:
                            st.subheader("First 10 citing articles:")
                            display_cols = ['source_doi', 'doi', 'title', 'authors_with_initials', 'author_count', 'year', 'journal_abbreviation',
                                          'citation_count_openalex', 'altmetric_score', 'number_of_mentions']
                            st.dataframe(citing_articles_df[display_cols].head(10))

                        excel_buffer = analyzer.save_citation_analysis_to_excel(citing_articles_df, citing_details_df, doi_list, citing_results, all_citing_titles)
                        st.download_button(
                            label="Download Complete Citation Analysis (Excel)",
                            data=excel_buffer.getvalue(),
                            file_name="citing_analysis_report.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        st.success("Complete citation analysis archived and ready for download")
                    else:
                        st.warning("No citing articles found.")

                except Exception as e:
                    st.error(f"Critical error during processing: {e}")
                    empty_df = pd.DataFrame()
                    excel_buffer = analyzer.save_citation_analysis_to_excel(empty_df, empty_df, doi_list, {}, [])
                    st.download_button(
                        label="Download Error Report",
                        data=excel_buffer.getvalue(),
                        file_name="error_citing_report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    st.info("Error report generated despite processing failure.")

    # Cache management
    with st.sidebar:
        st.markdown("---")
        st.markdown("### Cache Management")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Clear All Cache"):
                try:
                    analyzer.persistent_cache = OptimizedPersistentCache()
                    analyzer.progress_tracker.clear_progress()
                    st.success("Cache cleared successfully!")
                except Exception as e:
                    st.error(f"Error clearing cache: {e}")
        
        with col2:
            if st.button("Clear Progress Only"):
                try:
                    analyzer.progress_tracker.clear_progress()
                    st.success("Progress cleared successfully!")
                except Exception as e:
                    st.error(f"Error clearing progress: {e}")

        # Performance stats
        stats = analyzer.performance_monitor.get_stats()
        if stats:
            st.markdown("### Performance Stats")
            st.write(f"Requests: {stats['total_requests']}")
            st.write(f"Cache Efficiency: {stats['cache_efficiency']:.1f}%")
            st.write(f"Time: {stats['elapsed_seconds']:.1f}s")

if __name__ == "__main__":
    main()
