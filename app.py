# ============================================
# СЕКЦИЯ ПАРАМЕТРОВ (настройка запросов)
# ============================================

# Параметры API запросов
BATCH_SIZE = 50 
MAX_RETRIES = 3 
TIMEOUT = 30 
DELAY_BETWEEN_BATCHES = 0.5 
MAX_CONCURRENT_REQUESTS = 3
RETRY_DELAY = 2 
ORCID_REQUEST_DELAY = 0.2 

# Параметры вывода
SHOW_DEBUG_LOGS = True  # Показывать детальные логи
GENERATE_HTML_REPORT = True  # Генерировать HTML отчет
USE_CACHE = True  # Кэширование результатов
LOGO_PATH = None  # Путь к логотипу журнала (устанавливается через виджет)

# Лимиты для анализа
MAX_PUBLICATIONS_TO_ANALYZE = 1000  # Максимум статей для анализа
MIN_YEAR_FOR_TREND = 5  # Сколько лет для тренда

# Режим анализа источников данных
ANALYSIS_MODE = "orcid_openalex"  # "orcid_only" | "orcid_openalex"
# orcid_only: только публикации из ORCID профиля
# orcid_openalex: ORCID + OpenAlex (максимальная полнота)

# Параметры для обнаружения временных разрывов
MIN_GAP_YEARS_FOR_WARNING = 10  # Минимальный разрыв в годах для предупреждения

# ============================================
# ИМПОРТЫ
# ============================================

import asyncio
import aiohttp
import pandas as pd
import streamlit as st
from streamlit import session_state as ss
import re
import time
from datetime import datetime
import json
from typing import List, Set, Dict, Tuple, Optional, Any
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from wordcloud import WordCloud
from io import BytesIO
import base64
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
from bs4 import BeautifulSoup
import os
import hashlib
from matplotlib.ticker import MaxNLocator
import html
import html as html_module
import colorsys
from tenacity import retry, stop_after_attempt, wait_exponential, wait_random
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
from itertools import combinations
import difflib
from threading import Lock
from tqdm import tqdm
import random

# ============================================
# СЛОВАРЬ ПЕРЕВОДОВ (только английский)
# ============================================

LANG = {
    'en': {
        'app_title': 'Ref-Cit-Analysis',
        'app_icon': '📚',
        'settings': '⚙️ Settings',
        'language': '🌐 Language',
        'language_en': 'English',
        'language_ru': 'Russian',
        'color_theme': '🎨 Color Theme',
        'preset_themes': '🎨 Theme Presets',
        'use_preset': 'Use preset',
        'select_primary': '🎨 Select primary color',
        'select_secondary': '🎨 Select secondary color',
        'primary': 'Primary',
        'secondary': 'Secondary',
        'analysis_params': '📊 Analysis Parameters',
        'use_cache': '💾 Use cache',
        'clear_cache': '🗑️ Clear cache',
        'cache_cleared': '✅ Cache cleared!',
        'load_data': '📥 Load Data',
        'profile': 'Scholar Profile',
        'reports': '📄 Reports',
        'doi_input': 'DOI(s) to analyze',
        'doi_placeholder': '10.1000/xxx.yyy\nhttps://doi.org/10.1000/xxx.yyy',
        'doi_help': 'Enter one or more DOIs. Supports: plain DOI, URL format, separated by newlines, commas, semicolons, or spaces. Maximum 100 DOIs.',
        'workers': 'Parallel Workers',
        'workers_help': 'Number of parallel threads for API requests',
        'analyze_button': '🔍 Analyze DOI Network',
        'no_doi': '⚠️ Enter at least one valid DOI',
        'too_many_dois': '⚠️ Maximum 100 DOIs allowed. You entered {count}.',
        'duplicate_dois': '⚠️ Found {count} duplicate DOIs. Using unique values.',
        'analysis_complete': '✅ Analysis complete! Found {level1} references, {level2} analyzed, {level3} citing works in {time:.1f} sec.',
        'no_data': '👈 Load data and click "Analyze DOI Network"',
        'no_data_reports': '👈 First run analysis in "Load Data" tab',
        'html_report': '📄 HTML Report Generation',
        'download_report': '💾 Download HTML Report',
        'report_preview': '📋 HTML Report Preview',
        'download_hint': 'Click "Download HTML Report" for full report',
        'generating_report': 'Generating HTML report...',
        'publications': 'Publications',
        'citations': 'Citations',
        'h_index': 'h-index',
        'g_index': 'g-index',
        'i10_index': 'i10-index',
        'i100_index': 'i100-index',
        'total_citations': 'Total citations',
        'avg_citations': 'Average citations',
        'median_citations': 'Median citations',
        'max_citations': 'Max citations',
        'open_access': 'Open Access',
        'active_years': 'Active years',
        'risk_flags': 'Risk flags',
        'collaborations': 'Collaboration Analysis',
        'domestic': '🇷🇺 Domestic collaborations',
        'international': '🌐 International collaborations',
        'papers': 'Papers',
        'no_data_collab': 'No data',
        'collab_index': 'Collaboration index: {index:.2f}',
        'country_diversity': 'Country diversity: {count} countries',
        'most_collaborative': 'Most collaborative country: {country}',
        'top_coauthors': 'Top co-authors',
        'joint_works': 'joint works',
        'publications_list': '📚 Publication list',
        'showing_limited': 'Showing {shown} of {total} publications',
        'title': 'Title',
        'year': 'Year',
        'journal': 'Journal',
        'doi': 'DOI',
        'no_publications': 'No publications',
        'orcid': 'ORCID',
        'affiliations': 'Affiliations',
        'countries': 'Countries',
        'total_analyzed': 'Total analyzed publications',
        'retractions': 'Retractions',
        'corrections': 'Corrections',
        'first_publication': 'First publication',
        'last_publication': 'Last publication',
        'papers_per_year': 'Papers per year',
        'trend': 'Trend',
        'unique_coauthors': 'Unique co-authors',
        'avg_authors_per_paper': 'Average authors per paper',
        'thematic_diversity': 'Thematic diversity (Shannon)',
        'domestic_ratio': 'Domestic collaboration ratio',
        'international_ratio': 'International collaboration ratio',
        'years_chart_title': 'Publication activity dynamics',
        'journals_chart_title': 'Top journals by publications',
        'oa_chart_title': 'Open access status',
        'publishers_chart_title': 'Distribution by publishers',
        'affiliations_chart_title': 'Top affiliations',
        'citations_chart_title': 'Most cited articles',
        'citation_distribution_title': 'Citation distribution',
        'thematic_structure_title': 'Thematic structure of research',
        'wordcloud_title': 'Key research concepts',
        'radar_title': 'Thematic profile (Radar Chart)',
        'concepts': 'Concepts',
        'fields': 'Fields',
        'domains': 'Domains',
        'topics': 'Topics',
        'subtopics': 'Subtopics',
        'publication_year': 'Publication year',
        'number': 'Number',
        'citation_range': 'Citation range',
        'articles': 'Articles',
        'x_label_pubs': 'Number of publications',
        'y_label_pubs': 'Number of publications',
        'trend_label': 'Trend',
        'confidence_interval': 'Confidence interval',
        'footer': '© Ref-Cit-Analysis / Created by daM / Chimica Techno Acta',
        'journal_url': 'https://chimicatechnoacta.ru',
        'no_profile_data': 'No profile data available',
        'enter_orcid': 'Enter ORCID to analyze',
        'analyze_multiple': 'Analyze multiple authors',
        'profile_analysis': 'Comprehensive scholar profile analysis by ORCID',
        'select_language': 'Select language',
        'theme_presets_label': 'Theme presets',
        'primary_color_label': 'Primary color',
        'secondary_color_label': 'Secondary color',
        'analysis_progress': 'Analysis progress',
        'loading_data': 'Loading data',
        'analyzing_data': 'Analyzing data',
        'generating_viz': 'Generating visualizations',
        'orcid_format_error': 'Invalid ORCID format',
        'data_not_found': 'Data not found. Check ORCID correctness.',
        'error_occurred': 'Error occurred',
        'retracted_publications': 'retracted publications',
        'possible_unethical': 'Possible unethical practices detected!',
        'analyzing_authors': 'Analyzing {count} author(s)...',
        'starting_analysis': 'Starting analysis...',
        'fetching_data': 'Fetching data',
        'analysis_complete_text': 'Analysis complete',
        'creating_charts': 'Creating charts',
        'retractions_in_profile': 'retractions in profile',
        'source_types': 'Sources by Type',
        'source_journal_articles': 'Journal articles',
        'source_repositories': 'Preprints/Repositories',
        'source_ebooks': 'Electronic books',
        'source_proceedings': 'Proceedings',
        'source_other': 'Other items (non-DOI)',
        'source_count': 'Count',
        'source_examples': 'Examples',
        'source_no_doi': 'No DOI available',
        'source_view_link': 'View',
        'source_doi_available': 'DOI available',
        'source_no_link': 'No link available',
        'coauthor_orcid': 'ORCID',
        'coauthor_scopus': 'Scopus',
        'coauthor_researcherid': 'ResearcherID',
        'coauthor_website': 'Personal website',
        'coauthor_other': 'Other profiles',
        'no_orcid_found': 'No ORCID found',
        'coauthor_info': 'Co-author information',
        'coauthor_profiles': 'External profiles',
        'main_metrics': 'Main Metrics',
        'citations_per_year': 'Citations/year',
        'fetching_orcid_profiles': '🆔 Fetching ORCID profiles...',
        'orcid_profiles_fetched': '✅ ORCID profiles fetched: {count}',
        'no_orcid_profiles_found': 'No ORCID profiles found',
        'analysis_source': '📊 Data source for analysis:',
        'analysis_source_orcid_only': '🔒 ORCID only (safe)',
        'analysis_source_orcid_openalex': '🔓 ORCID + OpenAlex (max. completeness)',
        'analysis_source_help': 'Select data source for publication analysis',
        'temporal_gap_warning': '⚠️ Significant temporal gap detected in publications!',
        'temporal_gap_detected': 'Gap of {gap_years} years between {gap_start} and {gap_end}',
        'temporal_gap_suggestion': 'This may indicate: - Wrongly attributed publications from another scientist with the same name - Long break in scientific activity',
        'temporal_gap_recommendation': 'Recommended to cut off publications before {cut_off_year}',
        'temporal_gap_apply_filter': '📅 Apply year filter for report',
        'temporal_gap_select_period': 'Select analysis period:',
        'temporal_gap_publications_total': 'Total publications',
        'temporal_gap_after_filter': 'After filtering',
        'temporal_gap_filter_info': '📅 Period: {start_year} - {end_year}',
        'temporal_gap_original_count': 'Original: {count} publications',
        'temporal_gap_filtered_count': 'Filtered: {count} publications',
        'show_filtered_report': 'Show report with filtering',
        'show_original_report': 'Show original report (no filtering)',
        'temporal_gap_use_filter': 'Use year filter for report',
        # ====== КЛЮЧИ ДЛЯ DOI ANALYSIS ======
        'stage_fetch_level_ii': 'Stage 1: Fetching Level II (analyzed DOIs)',
        'stage_fetch_level_i': 'Stage 2: Fetching Level I (references)',
        'stage_fetch_level_iii': 'Stage 3: Fetching Level III (citing works)',
        'stage_fetch_metadata': 'Stage 4: Fetching metadata for all levels',
        'stage_analyze_report': 'Stage 5: Analyzing data and generating report',
        'stage_doi_found': 'Found {count} Level II DOIs',
        'stage_ref_found': 'Found {count} Level I references (unique: {unique})',
        'stage_citing_found': 'Found {count} Level III citing works (unique: {unique})',
        'stage_metadata_fetched': 'Fetched metadata for {count} works',
        'stage_processing': 'Processing {current}/{total}...',
        'overview': 'Overview',
        'level_i': 'Level I (References)',
        'level_ii': 'Level II (Analyzed)',
        'level_iii': 'Level III (Citing)',
        'total_items': 'Total Items',
        'total_weighted': 'Total Weighted Count',
        'unique_items': 'Unique Items',
        'total_citations': 'Total Citations',
        'avg_citations': 'Avg Citations',
        'active_years': 'Active Years',
        'unique_authors': 'Unique Authors',
        'unique_affiliations': 'Unique Affiliations',
        'unique_countries': 'Unique Countries',
        'avg_authors_per_paper': 'Avg Authors/Paper',
        'avg_affiliations_per_paper': 'Avg Affiliations/Paper',
        'avg_countries_per_paper': 'Avg Countries/Paper',
        'international_collaboration_rate': 'International Collaboration Rate',
        'unique_citing_authors': 'Unique Citing Authors',
        'unique_citing_affiliations': 'Unique Citing Affiliations',
        'unique_citing_countries': 'Unique Citing Countries',
        'unique_citing_journals': 'Unique Citing Journals',
        'unique_citing_publishers': 'Unique Citing Publishers',
        'open_access_breakdown': 'Open Access Breakdown',
        'gold': 'Gold',
        'hybrid': 'Hybrid',
        'green': 'Green',
        'bronze': 'Bronze',
        'closed': 'Closed',
        'unknown': 'Unknown',
        'analyzed_articles': 'Analyzed Articles',
        'author_analysis': 'Author Analysis',
        'rank': 'Rank',
        'authors': 'Authors',
        'publications_count': 'Publications',
        'citations_count': 'Citations',
        'top_affiliations': 'Top Affiliations',
        'geographic_analysis': 'Geographic Analysis',
        'unique_countries_per_publication': 'Unique Countries per Publication',
        'authors_per_country': 'Authors per Country',
        'collaboration_patterns': 'Collaboration Patterns',
        'single_country': 'Single-Country',
        'multi_country': 'Multi-Country',
        'collaboration_couples': 'Collaboration Couples',
        'country_pair': 'Country Pair',
        'frequency': 'Frequency',
        'citation_analysis': 'Citation Analysis',
        'citation_dynamics_by_year': 'Citation Dynamics by Year',
        'publication_year': 'Publication Year',
        'citation_year': 'Citation Year',
        'citations_count': 'Citations Count',
        'first_citation_analysis': 'First Citation Analysis',
        'min_lag': 'Min lag (days)',
        'max_lag': 'Max lag (days)',
        'avg_lag': 'Avg lag (days)',
        'median_lag': 'Median lag (days)',
        'cumulative_citations': 'Cumulative Citations',
        'citation_network_heatmap': 'Citation Network Heatmap',
        'most_cited_publications': 'Most Cited Publications',
        'citing_works_analysis': 'Citing Works Analysis',
        'total_citing_works': 'Total Citing Works',
        'top_citing_authors': 'Top Citing Authors',
        'top_citing_affiliations': 'Top Citing Affiliations',
        'top_citing_countries': 'Top Citing Countries',
        'top_citing_journals': 'Top Citing Journals',
        'top_citing_publishers': 'Top Citing Publishers',
        'topics_analysis': 'Topics Analysis',
        'analyzed_count': 'Count in Level I',
        'citing_count': 'Count in Level II',
        'citing_count_iii': 'Count in Level III',
        'analyzed_norm_count': 'Norm. Level I',
        'citing_norm_count': 'Norm. Level II',
        'citing_norm_count_iii': 'Norm. Level III',
        'total_norm_count': 'Total Norm.',
        'first_year': 'First Year',
        'peak_year': 'Peak Year',
        'top_cited_topics': 'Top Cited Topics',
        'top_cited_subtopics': 'Top Cited Subtopics',
        'top_cited_fields': 'Top Cited Fields',
        'top_cited_domains': 'Top Cited Domains',
        'top_cited_concepts': 'Top Cited Concepts',
        'detailed_citations': 'Detailed Citations',
        'show_citations': 'Show Citations',
        'hide_citations': 'Hide Citations',
        'citing_journal': 'Citing Journal',
        'citing_year': 'Citing Year',
        'citing_date': 'Citing Date',
        'citation_lag': 'Citation Lag (days)',
        'all_publications': 'All Publications',
        'filter_by_year': 'Filter by Year',
        'filter_by_author': 'Filter by Author',
        'filter_by_affiliation': 'Filter by Affiliation',
        'filter_by_citations': 'Filter by Citations (min)',
        'filter_by_title': 'Filter by Title Word(s)',
        'search_publications': 'Search Publications',
        'all_years': 'All Years',
        'visible_count': 'Showing {shown} of {total} publications',
        'data_source': 'Data source: OpenAlex',
        'generated_on': 'Generated',
        'click_to_toggle': 'Click to toggle citations',
        'no_citations_found': 'No citations found for this publication',
        'citations_per_year_label': 'Citations/Year',
        'reset_analysis': 'Reset Analysis',
        'days': 'days',
        'analysis_data_from_cache': 'Using cached data from previous analysis',
        'regenerate_report': 'Regenerate Report',
        # ====== КЛЮЧИ ДЛЯ AUTHOR DISTRIBUTION ======
        'author_distribution': 'Author Distribution',
        'author_distribution_analyzed': 'Distribution of Level II Publications by Author Count',
        'author_distribution_citing': 'Distribution of Level III Publications by Author Count',
        'authors_per_paper': 'Authors per paper',
        'num_papers': 'Number of papers',
        'num_citing_papers': 'Number of citing papers',
        'one_author': '1 author',
        'two_authors': '2 authors',
        'three_authors': '3 authors',
        'four_plus_authors': '4+ authors',
        # ====== КЛЮЧИ ДЛЯ ORCID IN CITING AUTHORS ======
        'citing_author_orcid': 'ORCID',
        # ====== КЛЮЧИ ДЛЯ MULTILEVEL RELATIONSHIPS ======
        'multilevel_relationships': 'Multilevel Relationships',
        'author_matrix': 'Author Matrix (All Levels)',
        'affiliation_matrix': 'Affiliation Matrix (All Levels)',
        'journal_matrix': 'Journal Matrix (All Levels)',
        'publisher_matrix': 'Publisher Matrix (All Levels)',
        'count_level_i': 'Count I',
        'count_level_ii': 'Count II',
        'count_level_iii': 'Count III',
        'norm_level_i': 'Norm I',
        'norm_level_ii': 'Norm II',
        'norm_level_iii': 'Norm III',
        'total_norm': 'Total Norm',
        'references_list': 'References (Level I)',
        'weighted_count': 'Weighted Count',
        'duplicate_warning': '⚠️ Duplicate DOIs detected and removed. Using {unique} unique DOIs from {total} entered.',
        'cross_level_citation': '⚠️ Cross-level citation detected: Level II DOI "{doi}" appears in Level I or Level III with count {count}',
        'references': 'References',
        'citing_works': 'Citing Works',
        'level_i_description': 'Articles that are cited by Level II DOIs (references)',
        'level_ii_description': 'Analyzed DOIs entered by user',
        'level_iii_description': 'Articles that cite Level II DOIs (citing works)',
        # ====== НОВЫЕ КЛЮЧИ ДЛЯ WEIGHTED CITING ======
        'citing_weighted_count': 'Weighted Count (Cited Analyzed)',
        'citing_weighted_count_desc': 'Number of Level II articles cited by this citing work',
        'citing_works_weighted': 'Citing Works with Weighted Counts',
        # ====== НОВЫЕ КЛЮЧИ ДЛЯ TITLE KEYWORDS ======
        'title_keywords_analysis': 'Title Keywords Analysis',
        'title_keywords_desc': 'Key terms extracted from titles across all levels with lemmatization',
        'title_term': 'Title Term (Lemma)',
        'variants': 'Variants',
        'term_type': 'Type',
        'level_i_count': 'Count I (Ref)',
        'level_ii_count': 'Count II (Analyzed)',
        'level_iii_count': 'Count III (Citing)',
        'norm_i': 'Norm I',
        'norm_ii': 'Norm II',
        'norm_iii': 'Norm III',
        'total_norm_keywords': 'Total Norm',
        'term_content': 'Content',
        'term_scientific': 'Scientific',
        'term_compound': 'Compound',
        # ====== НОВЫЕ КЛЮЧИ ДЛЯ TEMPORAL RELATIONSHIPS ======
        'temporal_relationships': 'Temporal Relationships',
        'temporal_desc': 'Time lag between publications across different levels',
        'reference_to_analyzed': 'Reference → Analyzed Connections',
        'analyzed_to_citing': 'Analyzed → Citing Connections',
        'ref_doi': 'Reference DOI',
        'analyzed_doi': 'Analyzed DOI',
        'citing_doi': 'Citing DOI',
        'ref_date': 'Ref Date',
        'analyzed_date': 'Analyzed Date',
        'citing_date': 'Citing Date',
        'time_lag_days': 'Time Lag (days)',
        'time_lag_stats': 'Temporal Lag Statistics',
        'min_lag_days': 'Min Lag (days)',
        'max_lag_days': 'Max Lag (days)',
        'avg_lag_days': 'Avg Lag (days)',
        'median_lag_days': 'Median Lag (days)',
        'total_connections': 'Total Connections',
        'ref_analyzed_connections': 'Ref→Analyzed Connections',
        'analyzed_citing_connections': 'Analyzed→Citing Connections',
        'analyzed_articles_list': 'Analyzed Articles List',
        'analyzed_articles_count': 'Analyzed Articles',
        'unique_journals': 'Unique Journals',
    }
}

def translate(key: str, lang: str = 'en', **kwargs) -> str:
    """Get translated string by key and language"""
    if lang not in LANG:
        lang = 'en'
    text = LANG[lang].get(key, key)
    if kwargs:
        try:
            text = text.format(**kwargs)
        except:
            pass
    return text

# ============================================
# COLOR UTILITIES FOR DYNAMIC THEMES
# ============================================

def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb: tuple) -> str:
    """Convert RGB tuple to hex color"""
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

def get_complementary_color(hex_color: str) -> str:
    """
    Generate complementary color by rotating hue by 180 degrees
    Returns a color that pairs well with the base color
    """
    rgb = hex_to_rgb(hex_color)
    h, s, v = colorsys.rgb_to_hsv(rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0)
    complementary_hue = (h + 0.5) % 1.0
    complementary_rgb = colorsys.hsv_to_rgb(complementary_hue, s, v)
    return rgb_to_hex(tuple(int(c * 255) for c in complementary_rgb))

def get_analogous_colors(hex_color: str, count: int = 2) -> List[str]:
    """Generate analogous colors (colors adjacent on color wheel)"""
    rgb = hex_to_rgb(hex_color)
    h, s, v = colorsys.rgb_to_hsv(rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0)
    
    colors_list = []
    step = 30 / 360.0
    
    for i in range(count):
        offset = (i + 1) * step
        new_hue = (h + offset) % 1.0
        new_rgb = colorsys.hsv_to_rgb(new_hue, s, v)
        colors_list.append(rgb_to_hex(tuple(int(c * 255) for c in new_rgb)))
    
    return colors_list

def get_gradient_colors(hex_color: str, steps: int = 5) -> List[str]:
    """Generate gradient colors from base color to lighter shades"""
    rgb = hex_to_rgb(hex_color)
    colors_list = []
    
    for i in range(steps):
        factor = 0.3 + (i * 0.14)
        new_rgb = tuple(min(255, int(c * (1 + factor * 0.5))) for c in rgb)
        colors_list.append(rgb_to_hex(new_rgb))
    
    return colors_list

def get_contrast_color(hex_color: str) -> str:
    """Get contrasting color (black or white) for text on a colored background"""
    rgb = hex_to_rgb(hex_color)
    luminance = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255
    return '#FFFFFF' if luminance < 0.5 else '#000000'

def generate_css_variables(base_color: str, accent_color: str) -> Dict[str, str]:
    """Generate complete CSS variable set for the theme"""
    gradient_start = base_color
    gradient_end = accent_color
    
    lighter_base = get_gradient_colors(base_color, 1)[0]
    lighter_accent = get_gradient_colors(accent_color, 1)[0]
    
    base_contrast = get_contrast_color(base_color)
    accent_contrast = get_contrast_color(accent_color)
    
    analogous = get_analogous_colors(base_color, 2)
    
    return {
        '--primary-color': base_color,
        '--secondary-color': accent_color,
        '--primary-light': lighter_base,
        '--secondary-light': lighter_accent,
        '--primary-contrast': base_contrast,
        '--secondary-contrast': accent_contrast,
        '--gradient-start': gradient_start,
        '--gradient-end': gradient_end,
        '--accent-1': analogous[0] if len(analogous) > 0 else accent_color,
        '--accent-2': analogous[1] if len(analogous) > 1 else accent_color,
        '--hover-light': f"{base_color}20",
    }

def apply_theme_css(base_color: str, accent_color: str):
    """Apply dynamic CSS theme based on selected colors"""
    css_vars = generate_css_variables(base_color, accent_color)
    
    theme_css = f"""
    <style>
        :root {{
            --primary: {css_vars['--primary-color']};
            --secondary: {css_vars['--secondary-color']};
            --primary-light: {css_vars['--primary-light']};
            --secondary-light: {css_vars['--secondary-light']};
            --primary-contrast: {css_vars['--primary-contrast']};
            --secondary-contrast: {css_vars['--secondary-contrast']};
            --gradient-start: {css_vars['--gradient-start']};
            --gradient-end: {css_vars['--gradient-end']};
            --accent-1: {css_vars['--accent-1']};
            --accent-2: {css_vars['--accent-2']};
            --hover-light: {css_vars['--hover-light']};
        }}
        
        .stApp {{
            background: linear-gradient(135deg, 
                rgba({int(hex_to_rgb(css_vars['--gradient-start'])[0])}, {int(hex_to_rgb(css_vars['--gradient-start'])[1])}, {int(hex_to_rgb(css_vars['--gradient-start'])[2])}, 0.05) 0%,
                rgba({int(hex_to_rgb(css_vars['--gradient-end'])[0])}, {int(hex_to_rgb(css_vars['--gradient-end'])[1])}, {int(hex_to_rgb(css_vars['--gradient-end'])[2])}, 0.08) 100%);
        }}
        
        .metric-number {{
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .section-header {{
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        }}
        
        .rank-item {{
            border-left: 3px solid var(--primary);
        }}
        
        .rank-number {{
            color: var(--primary);
        }}
        
        .progress-fill {{
            background: linear-gradient(90deg, var(--primary), var(--secondary));
        }}
        
        .custom-tab-button.active {{
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        }}
        
        .custom-tab-button:hover {{
            background: linear-gradient(135deg, var(--primary-light) 0%, var(--secondary-light) 100%);
        }}
        
        .colored-progress-bar {{
            background: linear-gradient(90deg, 
                var(--primary) 0%, 
                var(--secondary) 50%,
                var(--primary) 100%);
        }}
        
        .section-title {{
            border-bottom: 3px solid var(--primary);
        }}
        
        .concept-card {{
            background: linear-gradient(135deg, var(--hover-light) 0%, var(--secondary-light) 100%);
            border: 1px solid var(--primary-light);
        }}
        
        .concept-name {{
            color: var(--primary);
        }}
        
        .clickable-link {{
            color: var(--primary);
        }}
        
        .clickable-link:hover {{
            color: var(--secondary);
        }}
        
        .badge-success {{
            background: var(--primary-light);
            color: var(--primary-contrast);
        }}
        
        .custom-tab-button .custom-tab-title {{
            color: inherit;
        }}
        
        .metric-card:hover {{
            box-shadow: 0 6px 12px rgba({int(hex_to_rgb(css_vars['--primary-color'])[0])}, {int(hex_to_rgb(css_vars['--primary-color'])[1])}, {int(hex_to_rgb(css_vars['--primary-color'])[2])}, 0.15);
        }}
        
        * {{
            transition: background-color 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
        }}
        
        .color-preview {{
            display: inline-block;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            margin-left: 10px;
            vertical-align: middle;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }}
        
        .color-preview:hover {{
            transform: scale(1.1);
        }}
        
        .complementary-preview {{
            display: inline-block;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            margin-left: 10px;
            vertical-align: middle;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        .coauthor-card {{
            background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 12px;
            border: 1px solid #e0e0e0;
            border-left: 4px solid var(--primary);
            transition: transform 0.2s, box-shadow 0.2s;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        
        .coauthor-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.1);
            border-color: var(--primary);
        }}
        
        .coauthor-name {{
            font-size: 16px;
            font-weight: 600;
            color: var(--primary);
            margin-bottom: 6px;
        }}
        
        .coauthor-joint {{
            font-size: 13px;
            color: #666;
            margin-bottom: 8px;
        }}
        
        .coauthor-profile-link {{
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 3px 10px;
            border-radius: 15px;
            font-size: 11px;
            font-weight: 500;
            text-decoration: none;
            transition: all 0.2s;
            margin: 2px;
        }}
        
        .coauthor-profile-link:hover {{
            transform: translateY(-1px);
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }}
        
        .coauthor-profile-link.orcid {{
            background: #a6ce39;
            color: #1a1a1a;
        }}
        
        .coauthor-profile-link.orcid:hover {{
            background: #8cb82e;
        }}
        
        .coauthor-profile-link.website {{
            background: #6c757d;
            color: white;
        }}
        
        .coauthor-profile-link.website:hover {{
            background: #5a6268;
        }}
        
        .coauthor-profile-link.other {{
            background: #17a2b8;
            color: white;
        }}
        
        .coauthor-profile-link.other:hover {{
            background: #138496;
        }}
        
        .coauthor-no-orcid {{
            font-size: 12px;
            color: #999;
            font-style: italic;
        }}
        
        .no-links {{
            color: #999;
            font-style: italic;
            margin: 5px 0;
            font-size: 12px;
        }}

        .theme-info {{
            background: var(--hover-light);
            border-radius: 10px;
            padding: 12px;
            margin-top: 15px;
            font-size: 12px;
            text-align: center;
        }}
        
        .reviewer-card {{
            background: white;
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 16px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: transform 0.2s, box-shadow 0.2s;
            border-left: 4px solid var(--primary);
        }}
        
        .reviewer-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        
        .reviewer-name {{
            font-size: 18px;
            font-weight: 600;
            color: var(--primary);
            margin-bottom: 8px;
        }}
        
        .reviewer-orcid {{
            font-family: monospace;
            font-size: 12px;
            margin-bottom: 8px;
        }}
        
        .reviewer-section {{
            margin-top: 12px;
            padding-top: 8px;
            border-top: 1px solid #e0e0e0;
        }}
        
        .reviewer-section-title {{
            font-weight: 600;
            font-size: 13px;
            margin-bottom: 8px;
            color: #555;
        }}
        
        .external-id-link {{
            display: inline-block;
            background: #f0f0f0;
            padding: 4px 10px;
            border-radius: 15px;
            font-size: 11px;
            margin: 3px;
            text-decoration: none;
            color: #333;
            transition: background 0.2s;
        }}
        
        .external-id-link:hover {{
            background: var(--primary);
            color: white;
        }}
        
        .reviewer-website {{
            display: inline-block;
            margin: 3px 6px 3px 0;
            font-size: 12px;
        }}
        
        .confidential-banner {{
            background: linear-gradient(135deg, #fff3cd 0%, #ffe69e 100%);
            border-left: 4px solid #dc3545;
            padding: 12px 20px;
            margin-bottom: 20px;
            border-radius: 8px;
            font-weight: 500;
            text-align: center;
        }}
        
        .author-card {{
            background: white;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 25px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-left: 5px solid var(--primary);
            transition: transform 0.2s;
        }}
        
        .author-card:hover {{
            transform: translateX(5px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }}
        
        .author-card.best {{
            border-left-color: #FFD700;
            background: linear-gradient(135deg, #fff9e6 0%, #ffffff 100%);
        }}
        
        .author-rank {{
            font-size: 20px;
            font-weight: bold;
            color: var(--primary);
            display: inline-block;
            margin-right: 10px;
        }}
        
        .author-name-main {{
            font-size: 22px;
            font-weight: 600;
            color: var(--primary);
            display: inline-block;
        }}
        
        .author-hindex {{
            font-size: 18px;
            color: #666;
            margin-left: 10px;
        }}
        
        .best-badge {{
            background: #FFD700;
            color: #333;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: bold;
            display: inline-block;
            margin-left: 15px;
        }}
        
        .author-section {{
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 2px solid #e0e0e0;
        }}
        
        .author-section:last-child {{
            border-bottom: none;
        }}
        
        .source-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-family: 'Times New Roman', serif;
        }}
        .source-table th {{
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: white;
            padding: 12px;
            text-align: left;
        }}
        .source-table td {{
            padding: 10px;
            border-bottom: 1px solid #BDC3C7;
            vertical-align: top;
        }}
        .source-table tr:hover {{
            background-color: #f5f5f5;
        }}
        .source-example-item {{
            margin: 3px 0;
            font-size: 13px;
        }}
        .source-example-link {{
            color: #2980B9;
            text-decoration: none;
            font-size: 12px;
        }}
        .source-example-link:hover {{
            text-decoration: underline;
        }}
        .source-badge {{
            display: inline-block;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            margin-left: 5px;
        }}
        .source-badge-doi {{
            background: #d4edda;
            color: #155724;
        }}
        .source-badge-nodoi {{
            background: #f8d7da;
            color: #721c24;
        }}
        
        /* ===== COLOR SCALE FOR NUMERIC VALUES ===== */
        .color-scale-value {{
            display: inline-block;
            padding: 2px 10px;
            border-radius: 8px;
            font-weight: 600;
            text-align: center;
            min-width: 30px;
            transition: all 0.2s;
        }}
        .color-scale-value:hover {{
            transform: scale(1.05);
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        /* ===== HEATMAP CELL COLORS ===== */
        .heatmap-cell {{
            text-align: center;
            padding: 6px 10px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
            transition: all 0.3s;
            min-width: 40px;
        }}
        .heatmap-cell:hover {{
            transform: scale(1.05);
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            z-index: 5;
        }}
        
        /* ===== SORTABLE HEADERS ===== */
        th.sortable {{
            cursor: pointer;
            user-select: none;
            position: relative;
        }}
        th.sortable:hover {{
            opacity: 0.85;
        }}
        th.sortable::after {{
            content: ' ↕';
            opacity: 0.4;
            font-size: 10px;
        }}
        th.sortable.asc::after {{
            content: ' ↑';
            opacity: 0.8;
        }}
        th.sortable.desc::after {{
            content: ' ↓';
            opacity: 0.8;
        }}
    </style>
    """
    st.markdown(theme_css, unsafe_allow_html=True)

def update_colored_progress(progress_percent: float, status_text: str = "", color: str = None, badge_text: str = None):
    """Update progress bar with theme colors"""
    if color is None:
        primary_color = st.session_state.get('primary_color', '#667eea')
        secondary_color = st.session_state.get('secondary_color', '#f39c12')
        color = primary_color
    
    if badge_text is None:
        if progress_percent >= 80:
            badge_text = "✅ Excellent"
        elif progress_percent >= 60:
            badge_text = "📊 Good"
        elif progress_percent >= 40:
            badge_text = "⚠️ Average"
        elif progress_percent >= 20:
            badge_text = "⚠️ Low"
        else:
            badge_text = "❌ Critical"
    
    progress_html = f"""
    <style>
    @keyframes shimmer{{
        0% {{ background-position: -1000px 0; }}
        100% {{ background-position: 1000px 0; }}
    }}
    
    .colored-progress-container {{
        width: 100%;
        background-color: #f0f0f0;
        border-radius: 20px;
        overflow: hidden;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.2);
        margin: 10px 0;
    }}
    
    .colored-progress-bar {{
        width: {progress_percent:.1f}%;
        height: 32px;
        background: linear-gradient(90deg, 
            {color} 0%, 
            {color}DD 25%,
            {color} 50%,
            {color}DD 75%,
            {color} 100%);
        background-size: 200% 100%;
        animation: shimmer 2s infinite linear;
        border-radius: 20px;
        transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        font-size: 13px;
        text-shadow: 0 0 2px rgba(0,0,0,0.5);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    
    .colored-progress-bar::after {{
        content: "{progress_percent:.1f}%";
        position: absolute;
        left: 50%;
        transform: translateX(-50%);
        white-space: nowrap;
    }}
    
    .progress-stats {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 8px;
        font-size: 12px;
    }}
    
    .progress-badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        background: {color}20;
        color: {color};
        border: 1px solid {color}40;
    }}
    
    .progress-status {{
        font-size: 14px;
        font-weight: 500;
        color: #333;
    }}
    </style>
    
    <div class="colored-progress-container">
        <div class="colored-progress-bar"></div>
    </div>
    <div class="progress-stats">
        <span class="progress-status">{status_text}</span>
        <span class="progress-badge">{badge_text}</span>
    </div>
    """
    
    return progress_html

# ============================================
# ФУНКЦИИ ДЛЯ РАБОТЫ С DOI
# ============================================

def normalize_doi(doi_str: str) -> str:
    """Normalize DOI string to standard format"""
    if not doi_str:
        return ""
    
    doi_str = str(doi_str).strip()
    
    # Remove URL prefixes
    doi_str = re.sub(r'https?://(?:dx\.)?doi\.org/', '', doi_str, flags=re.IGNORECASE)
    doi_str = re.sub(r'^doi[:\s=]+', '', doi_str, flags=re.IGNORECASE)
    
    # Remove trailing punctuation
    doi_str = re.sub(r'[\s\.\,>]+$', '', doi_str)
    
    return doi_str.strip()

def parse_doi_input(text: str) -> List[str]:
    """Parse DOI input from text. Supports multiple formats."""
    if not text or not text.strip():
        return []
    
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = text.replace(',', ' ').replace(';', ' ')
    
    # Find all DOI patterns
    doi_pattern = r'10\.\d{4,9}/[^\s]+'
    matches = re.findall(doi_pattern, text)
    
    # Also find URLs with DOI
    url_pattern = r'doi\.org/(10\.\d{4,9}/[^\s]+)'
    url_matches = re.findall(url_pattern, text, re.IGNORECASE)
    
    all_dois = matches + url_matches
    
    # Normalize and deduplicate
    cleaned = [normalize_doi(d) for d in all_dois if normalize_doi(d)]
    return list(dict.fromkeys(cleaned))

def get_color_for_value(value: float, max_value: float, min_value: float = 0) -> str:
    """
    Get color from green-yellow-red scale based on value relative to max
    Green = highest value, Yellow = middle, Red = lowest
    """
    if max_value == min_value:
        return "rgba(46, 204, 113, 0.15)"
    
    # Normalize value to 0-1 range
    normalized = (value - min_value) / (max_value - min_value)
    
    # Clamp to 0-1
    normalized = max(0, min(1, normalized))
    
    # Define colors: Red (0) -> Yellow (0.5) -> Green (1)
    if normalized < 0.5:
        # Red to Yellow: (255,0,0) to (255,255,0)
        ratio = normalized / 0.5
        r = 255
        g = int(255 * ratio)
        b = 0
    else:
        # Yellow to Green: (255,255,0) to (0,255,0)
        ratio = (normalized - 0.5) / 0.5
        r = int(255 * (1 - ratio))
        g = 255
        b = 0
    
    # Return with semi-transparent alpha
    return f"rgba({r}, {g}, {b}, 0.25)"

def get_color_for_value_text(value: float, max_value: float, min_value: float = 0) -> str:
    """
    Get color from green-yellow-red scale for text (more opaque for readability)
    """
    if max_value == min_value:
        return "rgba(46, 204, 113, 0.3)"
    
    normalized = (value - min_value) / (max_value - min_value)
    normalized = max(0, min(1, normalized))
    
    if normalized < 0.5:
        ratio = normalized / 0.5
        r = 200
        g = int(200 * ratio)
        b = 50
    else:
        ratio = (normalized - 0.5) / 0.5
        r = int(200 * (1 - ratio))
        g = 200
        b = 50
    
    return f"rgba({r}, {g}, {b}, 0.35)"

def get_heatmap_cell_color(value: float, max_value: float) -> str:
    """
    Get color for heatmap cells using green-yellow-red scale
    Returns transparent for None or 0 values
    """
    if value is None or value == 0 or max_value == 0:
        return "transparent"
    
    normalized = value / max_value
    normalized = max(0, min(1, normalized))
    
    if normalized < 0.5:
        ratio = normalized / 0.5
        r = 200
        g = int(200 * ratio)
        b = 50
    else:
        ratio = (normalized - 0.5) / 0.5
        r = int(200 * (1 - ratio))
        g = 200
        b = 50
    
    return f"rgba({r}, {g}, {b}, 0.45)"

def get_color_scale_html_with_format(value: float, max_value: float, min_value: float = 0, decimals: int = 3) -> str:
    """
    Get color scale HTML with formatted number (for Topics table with 3 decimal places)
    """
    if max_value == min_value:
        return f'<span class="color-scale-value" style="background: rgba(200,200,200,0.15); color: #1a1a1a;">{value:.{decimals}f}</span>'
    
    # Normalize value to 0-1 range
    normalized = (value - min_value) / (max_value - min_value)
    normalized = max(0, min(1, normalized))
    
    # Define colors: Red (0) -> Yellow (0.5) -> Green (1)
    if normalized < 0.5:
        ratio = normalized / 0.5
        r = 200
        g = int(200 * ratio)
        b = 50
    else:
        ratio = (normalized - 0.5) / 0.5
        r = int(200 * (1 - ratio))
        g = 200
        b = 50
    
    bg_color = f"rgba({r}, {g}, {b}, 0.35)"
    
    # Format number with specified decimal places
    formatted_value = f"{value:.{decimals}f}"
    
    return f'<span class="color-scale-value" style="background: {bg_color}; color: #1a1a1a;">{formatted_value}</span>'

def format_ror_link(ror_short: str) -> str:
    """
    Format ROR ID for display in HTML
    
    Args:
        ror_short: ROR ID without https://ror.org/ prefix
        
    Returns:
        str: HTML link to colab.ws
    """
    if not ror_short:
        return '-'
    # Show only first 8 characters for compactness
    display_id = ror_short[:8] + '...' if len(ror_short) > 8 else ror_short
    return f'<a href="https://colab.ws/organizations/{ror_short}" target="_blank" class="doi-link" style="font-family: monospace; font-size: 11px;">{display_id}</a>'

# ============================================
# НАСТРОЙКА НАУЧНОГО СТИЛЯ ДЛЯ ГРАФИКОВ
# ============================================

def apply_scientific_style():
    """Improved scientific style for matplotlib for materials science publications"""
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        try:
            plt.style.use('seaborn-whitegrid')
        except:
            pass
    
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
        'mathtext.fontset': 'stix',
        
        'axes.labelsize': 12,
        'axes.labelweight': 'bold',
        'axes.titlesize': 13,
        'axes.titleweight': 'bold',
        'axes.facecolor': '#FFFFFF',
        'axes.edgecolor': '#000000',
        'axes.linewidth': 1.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        
        'xtick.color': '#000000',
        'ytick.color': '#000000',
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 7,
        'xtick.major.width': 1.5,
        'ytick.major.size': 7,
        'ytick.major.width': 1.5,
        'xtick.minor.size': 3,
        'xtick.minor.width': 1.0,
        'ytick.minor.size': 3,
        'ytick.minor.width': 1.0,
        
        'legend.fontsize': 10,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '#000000',
        'legend.fancybox': False,
        'legend.borderaxespad': 0.5,
        'legend.handlelength': 1.5,
        'legend.handletextpad': 0.8,
        
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'figure.facecolor': 'white',
        'figure.constrained_layout.use': True,
        'figure.figsize': (8, 6),
        
        'lines.linewidth': 2,
        'lines.markersize': 7,
        'lines.markeredgewidth': 1.0,
        'errorbar.capsize': 3,
        
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

apply_scientific_style()

# ============================================
# ДОПОЛНИТЕЛЬНЫЕ СЛОВАРИ И УТИЛИТЫ
# ============================================

# Dictionary for converting country codes to full names
COUNTRY_CODE_TO_NAME = {
    'GR': 'Greece',
    'CN': 'China',
    'PT': 'Portugal',
    'BY': 'Belarus',
    'PL': 'Poland',
    'SK': 'Slovakia',
    'SA': 'Saudi Arabia',
    'US': 'United States',
    'AU': 'Australia',
    'PK': 'Pakistan',
    'GB': 'United Kingdom',
    'HK': 'Hong Kong',
    'DE': 'Germany',
    'NO': 'Norway',
    'FR': 'France',
    'IN': 'India',
    'KR': 'South Korea',
    'RU': 'Russia',
    'UA': 'Ukraine',
    'IT': 'Italy',
    'ES': 'Spain',
    'NL': 'Netherlands',
    'CH': 'Switzerland',
    'SE': 'Sweden',
    'BE': 'Belgium',
    'AT': 'Austria',
    'DK': 'Denmark',
    'FI': 'Finland',
    'IE': 'Ireland',
    'NZ': 'New Zealand',
    'ZA': 'South Africa',
    'AR': 'Argentina',
    'MX': 'Mexico',
    'CL': 'Chile',
    'CO': 'Colombia',
    'BR': 'Brazil',
    'JP': 'Japan',
    'SG': 'Singapore',
    'TW': 'Taiwan',
    'IL': 'Israel',
    'TR': 'Turkey',
    'EG': 'Egypt',
    'NG': 'Nigeria',
    'KE': 'Kenya',
}

def get_full_country_name(country_code: str) -> str:
    """Convert country code to full name"""
    if not country_code:
        return 'Unknown'
    
    if len(country_code) > 3:
        return country_code
    
    return COUNTRY_CODE_TO_NAME.get(country_code.upper(), country_code)

def is_author_affiliation(affiliation_name: str, author_affiliations: List[str]) -> bool:
    """Check if affiliation belongs to the author"""
    if not affiliation_name or not author_affiliations:
        return False
    
    aff_normalized = affiliation_name.strip().lower()
    
    for author_aff in author_affiliations:
        if not author_aff:
            continue
        author_aff_normalized = author_aff.strip().lower()
        if aff_normalized == author_aff_normalized:
            return True
        if aff_normalized in author_aff_normalized or author_aff_normalized in aff_normalized:
            if len(aff_normalized) > 10 and len(author_aff_normalized) > 10:
                return True
    
    return False

# ============================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================

def clean_orcid(orcid_input: str) -> str:
    """Clean ORCID from extra characters and convert to standard format"""
    orcid = orcid_input.strip().upper()
    
    if 'orcid.org/' in orcid:
        orcid = orcid.split('orcid.org/')[-1]
    
    orcid = re.sub(r'[^0-9X-]', '', orcid)
    
    if re.match(r'^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$', orcid):
        return orcid
    
    if len(orcid) == 16 and orcid.isdigit():
        return f"{orcid[:4]}-{orcid[4:8]}-{orcid[8:12]}-{orcid[12:]}"
    
    return orcid

def format_boolean(value: bool) -> str:
    return "✅" if value else "❌"

def extract_country_from_affiliation(affiliation: str) -> str:
    """Extract country from affiliation name"""
    countries = [
        'USA', 'UK', 'China', 'Germany', 'France', 'Japan', 'Russia', 'Italy', 
        'Canada', 'Australia', 'Spain', 'Brazil', 'India', 'Netherlands', 'Switzerland',
        'South Korea', 'Sweden', 'Belgium', 'Poland', 'Austria', 'Norway', 'Denmark',
        'Finland', 'Ireland', 'Portugal', 'Greece', 'Czech Republic', 'Hungary',
        'New Zealand', 'South Africa', 'Argentina', 'Mexico', 'Chile', 'Colombia',
        'United States', 'United Kingdom', 'England', 'Scotland', 'Wales'
    ]
    
    for country in countries:
        if country.lower() in affiliation.lower():
            return country
    return "Unknown"

def normalize_author_name(name: str) -> str:
    """Normalize author name for comparison (initial + last name)"""
    if not name:
        return name
    
    name = name.strip()
    parts = name.split()
    
    if len(parts) >= 2:
        first_initial = parts[0][0].upper()
        last_name = parts[-1]
        return f"{first_initial} {last_name}"
    elif len(parts) == 1:
        return parts[0]
    else:
        return name

def normalize_author_name_for_grouping(name: str) -> str:
    """
    Normalize author name for grouping (last name + first initial)
    
    Examples:
    - "D.A. Osinkin" → "Osinkin D"
    - "Denis Osinkin" → "Osinkin D"
    - "Dmitry A. Medvedev" → "Medvedev D"
    - "D. Medvedev" → "Medvedev D"
    - "Osinkin D.A." → "Osinkin D"
    """
    if not name:
        return name
    
    name = name.strip()
    
    # Remove dots and extra spaces
    name = name.replace('.', ' ')
    name = ' '.join(name.split())
    
    parts = name.split()
    
    if len(parts) == 0:
        return name
    
    # Determine last name and initials
    if len(parts) == 2:
        first_part = parts[0]
        second_part = parts[1]
        
        if len(first_part) <= 2:
            last_name = second_part
            first_initial = first_part[0]
        else:
            last_name = first_part
            first_initial = second_part[0]
        
        return f"{last_name} {first_initial}"
    
    elif len(parts) >= 3:
        first_part = parts[0]
        last_part = parts[-1]
        
        if len(first_part) <= 2:
            last_name = last_part
            first_initial = first_part[0]
        else:
            last_name = last_part
            first_initial = parts[0][0]
        
        return f"{last_name} {first_initial}"
    
    elif len(parts) == 1:
        if '.' in name:
            subparts = name.split('.')
            if len(subparts) >= 2:
                last_name = subparts[0].strip()
                first_initial = subparts[1].strip() if subparts[1] else ''
                if first_initial:
                    return f"{last_name} {first_initial[0]}"
        return name
    
    elif ',' in name:
        parts_comma = name.split(',')
        if len(parts_comma) == 2:
            last_name = parts_comma[0].strip()
            first_part = parts_comma[1].strip()
            first_initial = first_part[0] if first_part else ''
            return f"{last_name} {first_initial}"
    
    elif len(parts) >= 2:
        last_name = parts[-1]
        first_initial = parts[0][0]
        return f"{last_name} {first_initial}"
    
    return name

def format_orcid_id(orcid: str) -> str:
    """Format ORCID ID to full URL"""
    if not orcid or not isinstance(orcid, str):
        return ""
    
    if orcid.startswith('https://orcid.org/'):
        return orcid
    
    clean_id = re.sub(r'[^\dXx-]', '', orcid.strip())
    
    if '-' in clean_id:
        if re.match(r'^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$', clean_id, re.IGNORECASE):
            return f"https://orcid.org/{clean_id}"
    
    if len(clean_id) == 16:
        formatted = f"{clean_id[:4]}-{clean_id[4:8]}-{clean_id[8:12]}-{clean_id[12:]}"
        return f"https://orcid.org/{formatted}"
    elif len(clean_id) == 15 and clean_id[15] in ['X', 'x']:
        formatted = f"{clean_id[:4]}-{clean_id[4:8]}-{clean_id[8:12]}-{clean_id[12:15]}X"
        return f"https://orcid.org/{formatted}"
    else:
        return f"https://orcid.org/{clean_id}"

def parse_orcids(text: str) -> List[str]:
    """Parse ORCID from text. Supports multiple input."""
    if not text or not text.strip():
        return []
    
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = text.replace(',', ' ').replace(';', ' ')
    
    orcid_pattern = r'\d{4}-\d{4}-\d{4}-\d{3}[\dX]'
    matches = re.findall(orcid_pattern, text, re.IGNORECASE)
    
    url_pattern = r'orcid\.org/(\d{4}-\d{4}-\d{4}-\d{3}[\dX])'
    url_matches = re.findall(url_pattern, text, re.IGNORECASE)
    
    all_orcids = matches + url_matches
    
    cleaned = [clean_orcid(o) for o in all_orcids]
    return list(dict.fromkeys(cleaned))

async def fetch_with_retry(session, url, params=None, headers=None, method='GET'):
    """Execute request with retry attempts on error"""
    for attempt in range(MAX_RETRIES):
        try:
            async with session.request(method, url, params=params, headers=headers, timeout=TIMEOUT) as response:
                if response.status == 429:
                    retry_after = int(response.headers.get('Retry-After', RETRY_DELAY * (attempt + 1)))
                    if SHOW_DEBUG_LOGS:
                        print(f"⚠️ Rate limit, waiting {retry_after} sec...")
                    await asyncio.sleep(retry_after)
                    continue
                
                if response.status == 200:
                    return await response.json()
                else:
                    if SHOW_DEBUG_LOGS:
                        print(f"⚠️ Error {response.status} for {url}")
                    return None
        except Exception as e:
            if SHOW_DEBUG_LOGS:
                print(f"⚠️ Attempt {attempt+1}/{MAX_RETRIES} error: {str(e)[:100]}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
            else:
                return None
    return None

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def safe_get(data, *keys, default=None):
    """Safe get value from nested dictionary"""
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key, default)
        else:
            return default
    return data

def get_cache_path(identifier: str, cache_type: str = "doi") -> str:
    """Return path to cache file"""
    if not os.path.exists('cache_doi'):
        os.makedirs('cache_doi')
    return f"cache_doi/{identifier}_{cache_type}.json"

def load_from_cache(identifier: str, cache_type: str = "doi") -> Optional[Dict]:
    """Load data from cache"""
    if not USE_CACHE:
        return None
    
    cache_path = get_cache_path(identifier, cache_type)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if SHOW_DEBUG_LOGS:
                print(f"✅ Loaded from cache: {cache_path}")
            return data
        except Exception as e:
            if SHOW_DEBUG_LOGS:
                print(f"⚠️ Cache load error: {e}")
            return None
    return None

def save_to_cache(identifier: str, data: Dict, cache_type: str = "doi"):
    """Save data to cache"""
    if not USE_CACHE:
        return
    
    cache_path = get_cache_path(identifier, cache_type)
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        if SHOW_DEBUG_LOGS:
            print(f"✅ Data saved to cache: {cache_path}")
    except Exception as e:
        if SHOW_DEBUG_LOGS:
            print(f"⚠️ Cache save error: {e}")

def normalize_issn(issn_str):
    """Normalize ISSN string to standard format"""
    cleaned = re.sub(r'[^0-9Xx]', '', str(issn_str).strip())
    if len(cleaned) == 8:
        return f"{cleaned[:4]}-{cleaned[4:]}".upper()
    return cleaned.upper()

def smart_request(params, retries=5):
    """Smart request to OpenAlex API with rate limiting and retries"""
    base_url = "https://api.openalex.org/works"
    lock = Lock()
    
    for attempt in range(retries):
        try:
            with lock:
                time.sleep(random.uniform(0.2, 0.45))
            
            resp = requests.get(base_url, params=params, timeout=30)
            
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 3))
                time.sleep(wait + random.uniform(1, 2))
                continue
                
            if resp.status_code == 200:
                return resp.json()
            
            time.sleep(1.2 ** attempt)
        except:
            time.sleep(1.5 ** attempt)
    return None

def get_work_metadata_batch(work_ids: List[str]) -> List[Dict]:
    """Get metadata for a batch of works by OpenAlex IDs"""
    if not work_ids:
        return []
    
    results = []
    
    for batch in chunks(work_ids, 50):
        id_query = '|'.join(batch)
        params = {
            'filter': f'openalex:{id_query}',
            'per_page': len(batch)
        }
        
        data = smart_request(params)
        if data and data.get('results'):
            results.extend(data['results'])
        
        time.sleep(random.uniform(0.1, 0.3))
    
    return results

def get_work_by_doi(doi: str) -> Optional[Dict]:
    """Fetch work metadata by DOI from OpenAlex"""
    if not doi:
        return None
    
    doi = normalize_doi(doi)
    url = f"https://api.openalex.org/works/doi:{doi}"
    
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            if SHOW_DEBUG_LOGS:
                print(f"⚠️ DOI not found: {doi} (status {response.status_code})")
            return None
    except Exception as e:
        if SHOW_DEBUG_LOGS:
            print(f"⚠️ Error fetching DOI {doi}: {e}")
        return None

def get_citing_works(oa_id: str, cursor: str = "*", per_page: int = 50) -> Optional[Dict]:
    """Get citing works for a given OpenAlex ID with pagination"""
    url = f"https://api.openalex.org/works?filter=cites:{oa_id}&per_page={per_page}&cursor={cursor}"
    
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        if SHOW_DEBUG_LOGS:
            print(f"⚠️ Error fetching citing works: {e}")
        return None

def get_referenced_works_batch(openalex_ids: List[str]) -> List[Dict]:
    if not openalex_ids:
        return []
    results = []
    for batch in chunks(openalex_ids, 50):
        id_query = '|'.join(batch)
        params = {
            'filter': f'openalex:{id_query}',
            'per_page': len(batch)
        }
        data = smart_request(params)
        if data and data.get('results'):
            for item in data['results']:
                if item is not None:
                    results.append(item)
        time.sleep(random.uniform(0.1, 0.3))
    return results

def extract_doi_from_openalex_id(oa_id: str) -> Optional[str]:
    """Extract DOI from OpenAlex ID URL"""
    if not oa_id:
        return None
    
    # Try to get DOI from OpenAlex ID
    # This is a fallback - usually we get DOI directly from the work
    return None

def parse_work_metadata(work: Dict) -> Dict:
    if work is None:
        return None
    try:
        parsed = {}
        
        parsed['id'] = work.get('id', '').replace('https://openalex.org/', '')
        if work is None:
            return None
        parsed['doi'] = work.get('doi', '').replace('https://doi.org/', '') if work.get('doi') else ''
        parsed['title'] = work.get('title', 'No title')
        parsed['publication_year'] = work.get('publication_year')
        parsed['publication_date'] = work.get('publication_date')
        parsed['cited_by_count'] = work.get('cited_by_count', 0)
        parsed['type'] = work.get('type', 'unknown')
        parsed['raw_type'] = work.get('raw_type', '')
        
        # Open Access
        oa = work.get('open_access', {})
        parsed['is_oa'] = oa.get('is_oa', False)
        parsed['oa_status'] = oa.get('oa_status', 'unknown')
        
        # Primary location (source/journal)
        if work.get('primary_location'):
            source = work['primary_location'].get('source', {})
            parsed['journal_name'] = source.get('display_name', 'Unknown')
            parsed['publisher'] = source.get('host_organization_name') or source.get('publisher', 'Unknown')
            parsed['source_type'] = source.get('type', 'unknown')
            parsed['issn'] = source.get('issn', [])
        else:
            parsed['journal_name'] = 'Unknown'
            parsed['publisher'] = 'Unknown'
            parsed['source_type'] = 'unknown'
            parsed['issn'] = []
        
        # Authors
        authors = []
        author_orcids = []
        authors_with_orcids = []
        authorships_raw = []  # Store raw author data with their affiliations
        
        for auth in work.get('authorships', []):
            raw_author_name = auth.get('raw_author_name', '')
            if not raw_author_name:
                author_data = auth.get('author', {})
                raw_author_name = author_data.get('display_name', '')
            
            author_orcid = ''
            author_data = auth.get('author', {})
            if author_data:
                author_orcid = author_data.get('orcid', '')
            
            if raw_author_name:
                authors.append(raw_author_name)
                if author_orcid:
                    author_orcids.append(author_orcid)
                authors_with_orcids.append({
                    'name': raw_author_name,
                    'orcid': author_orcid.replace('https://orcid.org/', '') if author_orcid else None
                })
                
                # Store raw author data with affiliations
                authorships_raw.append({
                    'author': raw_author_name,
                    'orcid': author_orcid,
                    'institutions': auth.get('institutions', []),
                    'countries': auth.get('countries', []),
                    'raw_affiliation_strings': auth.get('raw_affiliation_strings', [])
                })
        
        # Store authorships_raw for detailed country analysis
        parsed['authorships_raw'] = authorships_raw
        
        # ===== COLLECT AFFILIATIONS FROM institutions (WITH ROR) =====
        affiliations = []
        affiliation_countries = []
        institutions = []
        
        for auth in work.get('authorships', []):
            if auth.get('institutions'):
                for inst in auth['institutions']:
                    inst_name = inst.get('display_name', '')
                    country_code = inst.get('country_code', '')
                    ror = inst.get('ror', '')
                    inst_type = inst.get('type', '')
                    
                    if inst_name and inst_name not in affiliations:
                        affiliations.append(inst_name)
                    
                    if country_code:
                        country_name = get_full_country_name(country_code)
                        if country_name and country_name not in affiliation_countries:
                            affiliation_countries.append(country_name)
                    
                    institutions.append({
                        'id': inst.get('id', ''),
                        'display_name': inst.get('display_name', ''),
                        'country_code': inst.get('country_code', ''),
                        'ror': ror,
                        'type': inst_type
                    })
        
        parsed['authors'] = authors
        parsed['author_orcids'] = author_orcids
        parsed['authors_with_orcids'] = authors_with_orcids
        parsed['author_count'] = len(authors)
        parsed['affiliations'] = affiliations
        parsed['affiliation_countries'] = affiliation_countries
        parsed['institutions'] = institutions
        
        # Determine primary country from first affiliation
        if affiliation_countries:
            parsed['country'] = affiliation_countries[0]
        elif affiliations:
            parsed['country'] = extract_country_from_affiliation(affiliations[0])
        else:
            parsed['country'] = 'Unknown'
        
        # Topics
        topics_from_field = []
        for topic in work.get('topics', []):
            topic_name = topic.get('display_name', '')
            if topic_name:
                topics_from_field.append(topic_name)
        parsed['topics'] = topics_from_field[:15]
        
        # Concepts
        concepts = []
        concept_levels = {}
        fields = []
        domains = []
        subtopics = []
        subfields = []
        
        for concept in work.get('concepts', []):
            concept_name = concept.get('display_name', '')
            concept_level = concept.get('level', 0)
            concept_score = concept.get('score', 0)
            
            if concept_name:
                concepts.append(concept_name)
                concept_levels[concept_name] = {
                    'level': concept_level,
                    'score': concept_score
                }
            
            if concept_level >= 3:
                domains.append(concept_name)
            elif concept_level == 2:
                fields.append(concept_name)
            elif concept_level == 1:
                subfields.append(concept_name)
            elif concept_level == 0:
                subtopics.append(concept_name)
        
        parsed['concepts'] = concepts[:15]
        parsed['concept_levels'] = concept_levels
        parsed['fields'] = fields[:10]
        parsed['domains'] = domains[:5]
        parsed['subtopics'] = subtopics[:20]
        parsed['subfields'] = subfields[:15]
        parsed['topics_old'] = subfields[:15]
        
        return parsed
        
    except Exception as e:
        if SHOW_DEBUG_LOGS:
            print(f"⚠️ Error parsing work metadata: {e}")
        return None

# ============================================
# ============================================
# TITLE KEYWORDS ANALYZER (FROM OLD CODE)
# ============================================
# ============================================

class TitleKeywordsAnalyzer:
    """
    Analyzer for extracting and normalizing keywords from article titles
    with lemmatization, compound word detection, and scientific stopword filtering
    """
    
    def __init__(self):
        # Initialize stopwords and lemmatizer
        try:
            import nltk
            from nltk.corpus import stopwords
            from nltk.stem import WordNetLemmatizer
            
            # Load necessary NLTK resources
            try:
                nltk.download('wordnet', quiet=True)
                nltk.download('omw-eng', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('punkt', quiet=True)
            except:
                pass
            
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
            
            # Rules for special cases (irregular plurals)
            self.irregular_plurals = {
                'analyses': 'analysis',
                'axes': 'axis',
                'bases': 'basis',
                'crises': 'crisis',
                'criteria': 'criterion',
                'data': 'datum',
                'diagnoses': 'diagnosis',
                'ellipses': 'ellipsis',
                'emphases': 'emphasis',
                'genera': 'genus',
                'hypotheses': 'hypothesis',
                'indices': 'index',
                'media': 'medium',
                'memoranda': 'memorandum',
                'parentheses': 'parenthesis',
                'phenomena': 'phenomenon',
                'prognoses': 'prognosis',
                'radii': 'radius',
                'stimuli': 'stimulus',
                'syntheses': 'synthesis',
                'theses': 'thesis',
                'vertebrae': 'vertebra',
                # Add scientific terms
                'oxides': 'oxide',
                'composites': 'composite',
                'applications': 'application',
                'materials': 'material',
                'methods': 'method',
                'systems': 'system',
                'techniques': 'technique',
                'properties': 'property',
                'structures': 'structure',
                'devices': 'device',
                'processes': 'process',
                'mechanisms': 'mechanism',
                'models': 'model',
                'approaches': 'approach',
                'frameworks': 'framework',
                'strategies': 'strategy',
                'solutions': 'solution',
                'technologies': 'technology',
                'nanoparticles': 'nanoparticle',
                'nanostructures': 'nanostructure',
                'polymers': 'polymer',
                'ceramics': 'ceramic',
                'alloys': 'alloy',
                'coatings': 'coating',
                'films': 'film',
                'layers': 'layer',
                'interfaces': 'interface',
                'surfaces': 'surface',
                'catalysts': 'catalyst',
                'sensors': 'sensor',
                'actuators': 'actuator',
                'transistors': 'transistor',
                'diodes': 'diode',
                'circuits': 'circuit',
                'networks': 'network',
                'algorithms': 'algorithm',
                'protocols': 'protocol',
                'databases': 'database',
                'architectures': 'architecture',
                'platforms': 'platform',
                'environments': 'environment',
                'simulations': 'simulation',
                'experiments': 'experiment',
                'measurements': 'measurement',
                'observations': 'observation',
                'evaluations': 'evaluation',
                'assessments': 'assessment',
                'comparisons': 'comparison',
                'classifications': 'classification',
                'predictions': 'prediction',
                'optimizations': 'optimization',
                'characterizations': 'characterization',
                'syntheses': 'synthesis',
                'fabrications': 'fabrication',
                'preparations': 'preparation',
                'treatments': 'treatment',
                'modifications': 'modification',
                'enhancements': 'enhancement',
                'improvements': 'improvement',
                'developments': 'development',
                'innovations': 'innovation',
                'discoveries': 'discovery',
                'inventions': 'invention',
                'implementations': 'implementation',
                'utilizations': 'utilization',
                'integrations': 'integration',
                'combinations': 'combination',
                'interactions': 'interaction',
                'relationships': 'relationship',
                'dependencies': 'dependency',
                'correlations': 'correlation',
                'associations': 'association',
                'connections': 'connection',
                'communications': 'communication',
                'collaborations': 'collaboration',
                'cooperations': 'cooperation',
                'competitions': 'competition',
                'challenges': 'challenge',
                'problems': 'problem',
            }
            
            # Suffixes that need conversion
            self.suffix_replacements = {
                'ies': 'y',
                'es': '',
                's': '',
                'ed': '',
                'ing': '',
                'ly': '',
                'ally': 'al',
                'ically': 'ic',
                'ization': 'ize',
                'isation': 'ise',
                'ment': '',
                'ness': '',
                'ity': '',
                'ty': '',
                'ic': '',
                'ical': '',
                'ive': '',
                'ous': '',
                'ful': '',
                'less': '',
                'est': '',
                'er': '',
                'ors': 'or',
                'ings': 'ing',
                'ments': 'ment',
            }
            
        except:
            # Fallback if nltk not available
            self.stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            self.lemmatizer = None
            self.irregular_plurals = {}
            self.suffix_replacements = {}
        
        # Scientific stopwords (already lemmatized)
        self.scientific_stopwords = {
            'activate', 'adapt', 'advance', 'analyze', 'apply',
            'approach', 'architect', 'artificial', 'assess',
            'base', 'behave', 'capacity', 'characterize',
            'coat', 'compare', 'compute', 'composite',
            'control', 'cycle', 'damage', 'data', 'density', 'design',
            'detect', 'develop', 'device', 'diagnose', 'discover',
            'dynamic', 'economic', 'effect', 'efficacy',
            'efficient', 'energy', 'engineer', 'enhance', 'environment',
            'evaluate', 'experiment', 'explore', 'factor', 'fail',
            'fabricate', 'field', 'film', 'flow', 'framework', 'frequency',
            'function', 'grow', 'high', 'impact', 'improve',
            'induce', 'influence', 'inform', 'innovate', 'intelligent',
            'interact', 'interface', 'investigate', 'know',
            'layer', 'learn', 'magnetic', 'manage', 'material',
            'measure', 'mechanism', 'medical',
            'method', 'model', 'modify', 'modulate',
            'molecule', 'monitor', 'motion', 'nanoparticle',
            'nanostructure', 'network', 'neural', 'new', 'nonlinear',
            'novel', 'numerical', 'optical', 'optimize', 'pattern', 'perform',
            'phenomenon', 'potential', 'power', 'predict', 'prepare', 'process',
            'produce', 'progress', 'property', 'quality', 'regulate', 'relate',
            'reliable', 'remote', 'repair', 'research', 'resist', 'respond',
            'review', 'risk', 'role', 'safe', 'sample', 'scale', 'screen',
            'separate', 'signal', 'simulate', 'specific', 'stable', 'state',
            'store', 'strain', 'strength', 'stress', 'structure', 'study',
            'sustain', 'synergy', 'synthesize', 'system', 'target',
            'technique', 'technology', 'test', 'theoretical', 'therapy',
            'thermal', 'tissue', 'tolerate', 'toxic', 'transform', 'transition',
            'transmit', 'transport', 'type', 'understand', 'use', 'validate',
            'value', 'vary', 'virtual', 'waste', 'wave',
            # Additional scientific stopwords
            'application', 'approach', 'assessment', 'behavior', 'capability',
            'characterization', 'comparison', 'concept', 'condition', 'configuration',
            'construction', 'contribution', 'demonstration', 'description', 'detection',
            'determination', 'development', 'effectiveness', 'efficiency', 'evaluation',
            'examination', 'experimentation', 'explanation', 'exploration', 'fabrication',
            'formation', 'implementation', 'improvement', 'indication', 'investigation',
            'management', 'manufacture', 'measurement', 'modification', 'observation',
            'operation', 'optimization', 'performance', 'preparation', 'presentation',
            'production', 'realization', 'recognition', 'regulation', 'representation',
            'simulation', 'solution', 'specification', 'synthesis', 'transformation',
            'treatment', 'utilization', 'validation', 'verification'
        }
    
    def _get_lemma(self, word: str) -> str:
        """Get word lemma considering special rules"""
        if not word or len(word) < 3:
            return word
        
        # Convert to lowercase for processing
        lower_word = word.lower()
        
        # Check irregular plurals FIRST
        if lower_word in self.irregular_plurals:
            return self.irregular_plurals[lower_word]
        
        # Check regular plurals
        # If word ends with 's' or 'es' but not 'ss' or 'us'
        if lower_word.endswith('s') and not (lower_word.endswith('ss') or lower_word.endswith('us')):
            # Try to remove 's' or 'es'
            if lower_word.endswith('es') and len(lower_word) > 2:
                base_word = lower_word[:-2]
                # Check that after removing 'es' word not too short
                if len(base_word) >= 3:
                    return base_word
            elif len(lower_word) > 1:
                base_word = lower_word[:-1]
                # Check that after removing 's' word not too short
                if len(base_word) >= 3:
                    return base_word
        
        # Use lemmatizer if available
        if self.lemmatizer:
            # Try different parts of speech
            for pos in ['n', 'v', 'a', 'r']:  # noun, verb, adjective, adverb
                lemma = self.lemmatizer.lemmatize(lower_word, pos=pos)
                if lemma != lower_word:
                    return lemma
        
        # Apply suffix rules in reverse order (long to short)
        sorted_suffixes = sorted(self.suffix_replacements.keys(), key=len, reverse=True)
        for suffix in sorted_suffixes:
            if lower_word.endswith(suffix) and len(lower_word) > len(suffix) + 2:
                replacement = self.suffix_replacements[suffix]
                base = lower_word[:-len(suffix)] + replacement
                # Check result not too short
                if len(base) >= 3:
                    # Also check base doesn't end with double consonant
                    if len(base) >= 4 and base[-1] == base[-2]:
                        base = base[:-1]
                    return base
        
        return lower_word
    
    def _get_base_form(self, word: str) -> str:
        """Get base word form with aggressive lemmatization"""
        lemma = self._get_lemma(word)
        
        # Additional rules for scientific terms
        if lemma.endswith('isation'):
            return lemma[:-7] + 'ize'
        elif lemma.endswith('ization'):
            return lemma[:-7] + 'ize'
        elif lemma.endswith('ication'):
            return lemma[:-7] + 'y'
        elif lemma.endswith('ation'):
            return lemma[:-5] + 'e'
        elif lemma.endswith('ition'):
            return lemma[:-5] + 'e'
        elif lemma.endswith('ution'):
            return lemma[:-5] + 'e'
        elif lemma.endswith('ment'):
            return lemma[:-4]
        elif lemma.endswith('ness'):
            return lemma[:-4]
        elif lemma.endswith('ity'):
            return lemma[:-3] + 'e'
        elif lemma.endswith('ty'):
            base = lemma[:-2]
            if base.endswith('i'):
                return base[:-1] + 'y'
            return base
        elif lemma.endswith('ic'):
            return lemma[:-2] + 'y'
        elif lemma.endswith('al'):
            return lemma[:-2]
        elif lemma.endswith('ive'):
            return lemma[:-3] + 'e'
        elif lemma.endswith('ous'):
            return lemma[:-3]
        
        return lemma
    
    def preprocess_content_words(self, text: str) -> List[Dict]:
        """Clean and normalize content words, return dictionaries with lemmas and forms"""
        if not text or text in ['Title not found', 'Request timeout', 'Network error', 'Retrieval error', 'No title']:
            return []

        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s-]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        words = text.split()
        content_words = []

        for word in words:
            # EXCLUDE word "sub"
            if word == 'sub':
                continue
            if '-' in word:
                continue
            if len(word) > 2 and word not in self.stop_words:
                lemma = self._get_base_form(word)
                if lemma not in self.scientific_stopwords:
                    content_words.append({
                        'original': word,
                        'lemma': lemma,
                        'type': 'content'
                    })

        return content_words

    def extract_compound_words(self, text: str) -> List[Dict]:
        """Extract hyphenated compound words"""
        if not text or text in ['Title not found', 'Request timeout', 'Network error', 'Retrieval error', 'No title']:
            return []

        text = text.lower()
        compound_words = re.findall(r'\b[a-z]{2,}-[a-z]{2,}(?:-[a-z]{2,})*\b', text)

        compounds = []
        for word in compound_words:
            parts = word.split('-')
            if not any(part in self.stop_words for part in parts):
                # For compound words lemmatize each part
                lemmatized_parts = []
                for part in parts:
                    lemma = self._get_base_form(part)
                    lemmatized_parts.append(lemma)
                
                compounds.append({
                    'original': word,
                    'lemma': '-'.join(lemmatized_parts),
                    'type': 'compound'
                })

        return compounds

    def extract_scientific_stopwords(self, text: str) -> List[Dict]:
        """Extract scientific stopwords"""
        if not text or text in ['Title not found', 'Request timeout', 'Network error', 'Retrieval error', 'No title']:
            return []

        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        words = text.split()
        scientific_words = []

        for word in words:
            if len(word) > 2:
                lemma = self._get_base_form(word)
                if lemma in self.scientific_stopwords:
                    scientific_words.append({
                        'original': word,
                        'lemma': lemma,
                        'type': 'scientific'
                    })

        return scientific_words

    def analyze_titles(self, analyzed_titles: List[str], reference_titles: List[str], citing_titles: List[str]) -> dict:
        """
        Analyze keywords in analyzed, reference and citing article titles
        
        Returns:
            dict with 'analyzed', 'reference', 'citing' keys
            Each contains 'words' (list of dicts with lemma, variants, count) and 'total_titles'
        """
        
        # Analyze analyzed articles
        analyzed_words = []
        valid_analyzed_titles = [t for t in analyzed_titles if t and t not in ['Title not found', 'Request timeout', 'Network error', 'Retrieval error', 'No title']]
        
        for title in valid_analyzed_titles:
            analyzed_words.extend(self.preprocess_content_words(title))
            analyzed_words.extend(self.extract_compound_words(title))
            analyzed_words.extend(self.extract_scientific_stopwords(title))
        
        # Analyze reference articles
        reference_words = []
        valid_reference_titles = [t for t in reference_titles if t and t not in ['Title not found', 'Request timeout', 'Network error', 'Retrieval error', 'No title']]
        
        for title in valid_reference_titles:
            reference_words.extend(self.preprocess_content_words(title))
            reference_words.extend(self.extract_compound_words(title))
            reference_words.extend(self.extract_scientific_stopwords(title))
        
        # Analyze citing articles
        citing_words = []
        valid_citing_titles = [t for t in citing_titles if t and t not in ['Title not found', 'Request timeout', 'Network error', 'Retrieval error', 'No title']]
        
        for title in valid_citing_titles:
            citing_words.extend(self.preprocess_content_words(title))
            citing_words.extend(self.extract_compound_words(title))
            citing_words.extend(self.extract_scientific_stopwords(title))
        
        # Create aggregated data by lemmas
        def aggregate_by_lemma(word_list):
            lemma_dict = {}
            for word_info in word_list:
                lemma = word_info['lemma']
                original = word_info['original']
                
                # Exclude too short lemmas
                if len(lemma) < 3:
                    continue
                    
                if lemma not in lemma_dict:
                    lemma_dict[lemma] = {
                        'lemma': lemma,
                        'type': word_info['type'],
                        'variants': Counter(),
                        'count': 0
                    }
                
                lemma_dict[lemma]['variants'][original] += 1
                lemma_dict[lemma]['count'] += 1
            
            return lemma_dict
        
        analyzed_aggregated = aggregate_by_lemma(analyzed_words)
        reference_aggregated = aggregate_by_lemma(reference_words)
        citing_aggregated = aggregate_by_lemma(citing_words)
        
        # Merge similar lemmas (e.g., "composite" and "composites")
        def merge_similar_lemmas(lemma_dict):
            # Create list for removal after merging
            to_remove = set()
            
            lemmas = list(lemma_dict.keys())
            for i in range(len(lemmas)):
                lemma1 = lemmas[i]
                if lemma1 in to_remove:
                    continue
                    
                for j in range(i+1, len(lemmas)):
                    lemma2 = lemmas[j]
                    if lemma2 in to_remove:
                        continue
                    
                    # Check if lemmas are similar
                    if self._are_similar_lemmas(lemma1, lemma2):
                        # Merge into lemma1
                        lemma_dict[lemma1]['count'] += lemma_dict[lemma2]['count']
                        for variant, count in lemma_dict[lemma2]['variants'].items():
                            lemma_dict[lemma1]['variants'][variant] += count
                        
                        to_remove.add(lemma2)
            
            # Remove merged lemmas
            for lemma in to_remove:
                if lemma in lemma_dict:
                    del lemma_dict[lemma]
            
            return lemma_dict
        
        analyzed_aggregated = merge_similar_lemmas(analyzed_aggregated)
        reference_aggregated = merge_similar_lemmas(reference_aggregated)
        citing_aggregated = merge_similar_lemmas(citing_aggregated)
        
        # Get top 100 for each type
        def get_top_100(aggregated_dict):
            items = list(aggregated_dict.values())
            items.sort(key=lambda x: x['count'], reverse=True)
            return items[:100]
        
        top_100_analyzed = get_top_100(analyzed_aggregated)
        top_100_reference = get_top_100(reference_aggregated)
        top_100_citing = get_top_100(citing_aggregated)
        
        return {
            'analyzed': {
                'words': top_100_analyzed,
                'total_titles': len(valid_analyzed_titles)
            },
            'reference': {
                'words': top_100_reference,
                'total_titles': len(valid_reference_titles)
            },
            'citing': {
                'words': top_100_citing,
                'total_titles': len(valid_citing_titles)
            }
        }
    
    def _are_similar_lemmas(self, lemma1: str, lemma2: str) -> bool:
        """Check if lemmas are similar (e.g., singular/plural)"""
        if lemma1 == lemma2:
            return True
        
        # Check if they are forms of the same word
        # Example: "composite" and "composites"
        if lemma1.endswith('s') and lemma1[:-1] == lemma2:
            return True
        if lemma2.endswith('s') and lemma2[:-1] == lemma1:
            return True
        
        # Check if they are forms with different suffixes
        # Example: "characterization" and "characterize"
        common_prefix = self._get_common_prefix(lemma1, lemma2)
        if len(common_prefix) >= 5:  # If common prefix long enough
            # Check length difference
            if abs(len(lemma1) - len(lemma2)) <= 3:
                return True
        
        return False
    
    def _get_common_prefix(self, str1: str, str2: str) -> str:
        """Return common prefix of two strings"""
        min_length = min(len(str1), len(str2))
        common_prefix = []
        
        for i in range(min_length):
            if str1[i] == str2[i]:
                common_prefix.append(str1[i])
            else:
                break
        
        return ''.join(common_prefix)

# ============================================
# ============================================
# КЛАСС АНАЛИЗАТОРА DOI (3 УРОВНЯ) - РАСШИРЕННЫЙ
# ============================================
# ============================================

class DOIAnalyzer:
    def __init__(self, doi_list: List[str], max_workers: int = 6):
        self.doi_list = list(dict.fromkeys([normalize_doi(d) for d in doi_list if normalize_doi(d)]))  # Level II (analyzed)
        self.max_workers = max_workers
        
        # Three levels with duplicate counting
        self.level_I = defaultdict(int)   # {doi: count} - references (cited by Level II)
        self.level_II = set(self.doi_list)  # Unique analyzed DOIs
        self.level_III = defaultdict(int) # {doi: count} - citing works (cite Level II)
        
        # Metadata for each level
        self.metadata_I = {}   # {doi: parsed_meta}
        self.metadata_II = {}  # {doi: parsed_meta}
        self.metadata_III = {} # {doi: parsed_meta}
        
        # Relationships (for Detailed Citations)
        self.citations_from_II_to_I = defaultdict(list)  # {II_doi: [I_doi, ...]}
        self.citations_from_III_to_II = defaultdict(list) # {III_doi: [II_doi, ...]}
        
        # Track cross-level citations (when a Level II DOI appears in Level I or III)
        self.cross_level_citations = []
        
        # Analysis results
        self.analysis_results = {}
        self.lock = Lock()
        
        # Track progress
        self.total_references = 0
        self.total_citing = 0
        
        # ===== NEW: Title Keywords Analyzer =====
        self.title_keywords_analyzer = TitleKeywordsAnalyzer()
        
        # ===== NEW: Temporal Relationships Data =====
        self.temporal_relationships = {
            'ref_to_analyzed': [],  # List of (ref_doi, analyzed_doi, lag_days)
            'analyzed_to_citing': [],  # List of (analyzed_doi, citing_doi, lag_days)
        }
        
    def get_cache_identifier(self) -> str:
        """Generate cache identifier from sorted DOIs"""
        doi_hash = hashlib.md5(','.join(sorted(self.doi_list)).encode()).hexdigest()[:16]
        return f"doi_{doi_hash}"
    
    def fetch_level_II(self, progress_callback=None) -> Dict:
        """Stage 1: Fetch Level II (analyzed DOIs) metadata and collect references and citations"""
        if SHOW_DEBUG_LOGS:
            print(f"🚀 Fetching Level II for {len(self.doi_list)} DOIs...")
        
        # Check cache first
        cache_id = self.get_cache_identifier()
        cached = load_from_cache(cache_id, "level_ii")
        if cached:
            self.level_II = set(cached.get('level_ii', []))
            self.level_I = defaultdict(int, cached.get('level_i', {}))
            self.level_III = defaultdict(int, cached.get('level_iii', {}))
            self.citations_from_II_to_I = defaultdict(list, cached.get('citations_from_ii_to_i', {}))
            self.citations_from_III_to_II = defaultdict(list, cached.get('citations_from_iii_to_ii', {}))
            self.metadata_II = cached.get('metadata_ii', {})
            if SHOW_DEBUG_LOGS:
                print(f"✅ Loaded Level II from cache")
            return self.metadata_II
        
        metadata = {}
        total = len(self.doi_list)
        
        for idx, doi in enumerate(self.doi_list):
            if SHOW_DEBUG_LOGS:
                print(f"  Processing {idx+1}/{total}: {doi}")
            
            # Fetch work
            work = get_work_by_doi(doi)
            if not work:
                if SHOW_DEBUG_LOGS:
                    print(f"    ⚠️ Work not found for {doi}")
                if progress_callback:
                    progress_callback(idx + 1, total)
                continue
            
            # Parse metadata
            parsed = parse_work_metadata(work)
            if not parsed:
                if progress_callback:
                    progress_callback(idx + 1, total)
                continue
            
            metadata[doi] = parsed
            oa_id = work.get('id', '').replace('https://openalex.org/', '')
            
            # Get referenced works (REFERENCES - Level I)
            ref_ids = work.get('referenced_works', [])
            if ref_ids:
                # Extract OpenAlex IDs
                oa_ref_ids = [rid.replace('https://openalex.org/', '') for rid in ref_ids]
                
                # Fetch referenced works metadata
                ref_works = get_referenced_works_batch(oa_ref_ids)
                for ref_work in ref_works:
                    if ref_work is None:
                        continue
                    ref_doi = ref_work.get('doi', '')
                    if ref_doi:
                        ref_doi = ref_doi.replace('https://doi.org/', '')
                        ref_doi = normalize_doi(ref_doi)
                        self.level_I[ref_doi] += 1
                        self.citations_from_II_to_I[doi].append(ref_doi)
                        if ref_doi not in self.metadata_I:
                            parsed_ref = parse_work_metadata(ref_work)
                            if parsed_ref:
                                self.metadata_I[ref_doi] = parsed_ref
            
            # Get citing works (CITATIONS - Level III) - with pagination
            cursor = "*"
            citing_count = 0
            max_citing = 1000  # Limit to prevent excessive requests, but can be increased
            
            while cursor and citing_count < max_citing:
                data = get_citing_works(oa_id, cursor, per_page=50)
                if not data or not data.get('results'):
                    break
                
                for citing_work in data.get('results', []):
                    if citing_work is None:
                        continue
                    citing_doi = citing_work.get('doi', '')
                    if citing_doi:
                        citing_doi = citing_doi.replace('https://doi.org/', '')
                        citing_doi = normalize_doi(citing_doi)
                        self.level_III[citing_doi] += 1
                        self.citations_from_III_to_II[citing_doi].append(doi)
                        citing_count += 1
                        if citing_doi not in self.metadata_III:
                            parsed_citing = parse_work_metadata(citing_work)
                            if parsed_citing:
                                self.metadata_III[citing_doi] = parsed_citing
                
                cursor = data.get('meta', {}).get('next_cursor')
                if cursor:
                    time.sleep(DELAY_BETWEEN_BATCHES)
            
            # Check for cross-level citations (Level II DOI in Level I or III)
            if doi in self.level_I:
                self.cross_level_citations.append({
                    'doi': doi,
                    'level': 'I',
                    'count': self.level_I[doi]
                })
            if doi in self.level_III:
                self.cross_level_citations.append({
                    'doi': doi,
                    'level': 'III',
                    'count': self.level_III[doi]
                })
            
            if progress_callback:
                progress_callback(idx + 1, total)
            
            time.sleep(DELAY_BETWEEN_BATCHES)
        
        # Store metadata
        self.metadata_II = metadata
        
        # Save to cache
        cache_data = {
            'level_ii': list(self.level_II),
            'level_i': dict(self.level_I),
            'level_iii': dict(self.level_III),
            'citations_from_ii_to_i': dict(self.citations_from_II_to_I),
            'citations_from_iii_to_ii': dict(self.citations_from_III_to_II),
            'metadata_ii': self.metadata_II
        }
        save_to_cache(cache_id, cache_data, "level_ii")
        
        self.total_references = len(self.level_I)
        self.total_citing = len(self.level_III)
        
        if SHOW_DEBUG_LOGS:
            print(f"✅ Level II fetched: {len(metadata)} works")
            print(f"   Level I references: {len(self.level_I)} unique (total weighted: {sum(self.level_I.values())})")
            print(f"   Level III citing: {len(self.level_III)} unique (total weighted: {sum(self.level_III.values())})")
        
        return metadata
    
    def fetch_level_I_metadata(self, progress_callback=None) -> Dict:
        """Stage 2: Fetch missing metadata for Level I (references)"""
        if SHOW_DEBUG_LOGS:
            print(f"📖 Fetching Level I metadata for {len(self.level_I)} references...")
        
        # Check which DOIs already have metadata
        missing_dois = [doi for doi in self.level_I.keys() if doi not in self.metadata_I]
        
        if not missing_dois:
            if SHOW_DEBUG_LOGS:
                print(f"✅ All Level I metadata already present")
            return self.metadata_I
        
        # Fetch metadata for missing DOIs
        for batch in chunks(missing_dois, 50):
            # Use DOI filter
            doi_query = '|'.join(batch)
            params = {
                'filter': f'doi:{doi_query}',
                'per_page': len(batch)
            }
            data = smart_request(params)
            if data and data.get('results'):
                for work in data['results']:
                    if work is None:
                        continue
                    parsed = parse_work_metadata(work)
                    if parsed and parsed.get('doi'):
                        self.metadata_I[parsed['doi']] = parsed
            
            if progress_callback:
                progress_callback(len([d for d in missing_dois if d in self.metadata_I]), len(missing_dois))
            
            time.sleep(DELAY_BETWEEN_BATCHES)
        
        if SHOW_DEBUG_LOGS:
            print(f"✅ Level I metadata: {len(self.metadata_I)} works")
        
        return self.metadata_I
    
    def fetch_level_III_metadata(self, progress_callback=None) -> Dict:
        """Stage 3: Fetch missing metadata for Level III (citing works)"""
        if SHOW_DEBUG_LOGS:
            print(f"📖 Fetching Level III metadata for {len(self.level_III)} citing works...")
        
        # Check which DOIs already have metadata
        missing_dois = [doi for doi in self.level_III.keys() if doi not in self.metadata_III]
        
        if not missing_dois:
            if SHOW_DEBUG_LOGS:
                print(f"✅ All Level III metadata already present")
            return self.metadata_III
        
        # Fetch metadata for missing DOIs
        for batch in chunks(missing_dois, 50):
            doi_query = '|'.join(batch)
            params = {
                'filter': f'doi:{doi_query}',
                'per_page': len(batch)
            }
            data = smart_request(params)
            if data and data.get('results'):
                for work in data['results']:
                    parsed = parse_work_metadata(work)
                    if parsed and parsed.get('doi'):
                        self.metadata_III[parsed['doi']] = parsed
            
            if progress_callback:
                progress_callback(len([d for d in missing_dois if d in self.metadata_III]), len(missing_dois))
            
            time.sleep(DELAY_BETWEEN_BATCHES)
        
        if SHOW_DEBUG_LOGS:
            print(f"✅ Level III metadata: {len(self.metadata_III)} works")
        
        return self.metadata_III
    
    def fetch_all_metadata(self, progress_callback=None) -> Dict:
        """Stage 4: Fetch all metadata for all levels"""
        if SHOW_DEBUG_LOGS:
            print(f"📖 Fetching all metadata...")
        
        # Fetch Level I metadata
        self.fetch_level_I_metadata(progress_callback)
        
        # Fetch Level III metadata
        self.fetch_level_III_metadata(progress_callback)
        
        total_metadata = len(self.metadata_I) + len(self.metadata_II) + len(self.metadata_III)
        if SHOW_DEBUG_LOGS:
            print(f"✅ Total metadata: {total_metadata} works")
        
        return {
            'level_I': self.metadata_I,
            'level_II': self.metadata_II,
            'level_III': self.metadata_III
        }
    
    def analyze_data(self, progress_callback=None) -> Dict:
        """Stage 5: Analyze all collected data"""
        if SHOW_DEBUG_LOGS:
            print("📊 Analyzing collected data...")
        
        results = {}
        
        # 1. Basic metrics for all levels
        results['basic_metrics'] = self._analyze_basic_metrics()
        
        # 2.1 Author analysis (Level II only)
        results['author_analysis'] = self._analyze_authors()
        
        # 2.2 Analyzed articles list (Level II) =====
        results['analyzed_articles_list'] = self._get_analyzed_articles_list()
        
        # 3. Affiliation analysis (Level II only)
        results['affiliation_analysis'] = self._analyze_affiliations()
        
        # 4. Geographic analysis (Level II only)
        results['geographic_analysis'] = self._analyze_geographic()
        
        # 5. Citation analysis (Level II only)
        results['citation_analysis'] = self._analyze_citations()
        
        # 6. Citing works analysis (Level III) - UPDATED with weighted count
        results['citing_analysis'] = self._analyze_citing_works()
        
        # 7. Topics analysis (All 3 levels)
        results['topics_analysis'] = self._analyze_topics()
        
        # 8. Detailed citations (Level II only)
        results['detailed_citations'] = self._get_detailed_citations()
        
        # 9. Author distribution (Level II and Level III)
        results['author_distribution'] = self._analyze_author_distribution()
        
        # 10. Multilevel Relationships (NEW!)
        results['multilevel_relationships'] = self._analyze_multilevel_relationships()
        
        # 11. References list (Level I)
        results['references_list'] = self._get_references_list()
        
        # 12. Title Keywords Analysis =====
        results['title_keywords'] = self._analyze_title_keywords()
        
        # 13. Temporal Relationships =====
        results['temporal_relationships'] = self._analyze_temporal_relationships()
        
        self.analysis_results = results
        
        if progress_callback:
            progress_callback(100, 100)
        
        return results
    
    def _calc_group_metrics(self, data_dict: dict, metadata_dict: dict, is_weighted: bool = True) -> Dict:
        """Calculate metrics for a group (I, II, or III)"""
        if not data_dict:
            return {
                'total_items': 0,
                'total_weighted': 0,
                'unique_items': 0,
                'total_citations': 0,
                'avg_citations': 0,
                'median_citations': 0,
                'max_citations': 0,
                'active_years': 0,
                'unique_authors': 0,
                'unique_affiliations': 0,
                'unique_countries': 0,
                'avg_authors_per_paper': 0,
                'avg_affiliations_per_paper': 0,
                'avg_countries_per_paper': 0,
                'oa_percentage': 0,
                'oa_breakdown': {},
                'h_index': 0,
                'g_index': 0,
                'i10_index': 0,
                'i100_index': 0,
                'international_collaboration_rate': 0  # ДОБАВИТЬ
            }
        
        # Determine if data_dict is a dict with counts (I, III) or set/list (II)
        if is_weighted:
            # Level I or III: data_dict is defaultdict(int) with counts
            items = list(data_dict.keys())
            total_weighted = sum(data_dict.values())
            total_items = len(items)
        else:
            # Level II: data_dict is set
            items = list(data_dict)
            total_weighted = len(items)
            total_items = len(items)
        
        # Collect citations
        citations = []
        years = []
        authors_set = set()
        affiliations_set = set()
        countries_set = set()
        total_authors = 0
        total_affiliations = 0
        total_countries = 0
        oa_statuses = []
        
        for doi in items:
            meta = metadata_dict.get(doi, {})
            
            # Citations
            cited_by = meta.get('cited_by_count', 0)
            citations.append(cited_by)
            
            # Year
            year = meta.get('publication_year')
            if year:
                years.append(year)
            
            # Authors
            authors = meta.get('authors', [])
            authors_set.update(authors)
            total_authors += len(authors)
            
            # Affiliations
            affiliations = meta.get('affiliations', [])
            affiliations_set.update(affiliations)
            total_affiliations += len(affiliations)
            
            # Countries
            countries = meta.get('affiliation_countries', [])
            countries_set.update(countries)
            total_countries += len(set(countries))
            
            # OA
            oa_statuses.append(meta.get('oa_status', 'unknown'))
        
        # Calculate metrics
        total_citations = sum(citations)
        avg_citations = total_citations / total_items if total_items > 0 else 0
        median_citations = np.median(citations) if citations else 0
        max_citations = max(citations) if citations else 0
        
        # h-index
        citations_sorted = sorted([c for c in citations if c > 0], reverse=True)
        h_index = 0
        for i, c in enumerate(citations_sorted, 1):
            if c >= i:
                h_index = i
            else:
                break
        
        # g-index
        total_citations_sorted = 0
        g_index = 0
        for i, c in enumerate(citations_sorted, 1):
            total_citations_sorted += c
            if total_citations_sorted >= i**2:
                g_index = i
        
        # i10-index, i100-index
        i10_index = sum(1 for c in citations if c >= 10)
        i100_index = sum(1 for c in citations if c >= 100)
        
        # Active years
        active_years = len(set(years)) if years else 0
        
        # OA breakdown
        oa_breakdown = dict(Counter(oa_statuses))
        oa_count = sum(1 for s in oa_statuses if s not in ['closed', 'unknown'])
        oa_percentage = (oa_count / total_items * 100) if total_items > 0 else 0
        
        # Averages
        avg_authors = total_authors / total_items if total_items > 0 else 0
        avg_affiliations = total_affiliations / total_items if total_items > 0 else 0
        avg_countries = total_countries / total_items if total_items > 0 else 0
        
        # ===== NEW: International Collaboration Rate (only for Level II) =====
        international_collaboration_rate = 0
        if not is_weighted and items:
            multi_country_count = 0
            for doi in items:
                meta = metadata_dict.get(doi, {})
                countries = meta.get('affiliation_countries', [])
                # Убираем дубликаты и пустые значения
                unique_countries = set([c for c in countries if c and c != 'Unknown'])
                if len(unique_countries) > 1:
                    multi_country_count += 1
            international_collaboration_rate = (multi_country_count / len(items) * 100) if items else 0
        
        return {
            'total_items': total_items,
            'total_weighted': total_weighted,
            'unique_items': total_items,
            'total_citations': total_citations,
            'avg_citations': avg_citations,
            'median_citations': median_citations,
            'max_citations': max_citations,
            'active_years': active_years,
            'unique_authors': len(authors_set),
            'unique_affiliations': len(affiliations_set),
            'unique_countries': len(countries_set),
            'avg_authors_per_paper': avg_authors,
            'avg_affiliations_per_paper': avg_affiliations,
            'avg_countries_per_paper': avg_countries,
            'oa_percentage': oa_percentage,
            'oa_breakdown': oa_breakdown,
            'h_index': h_index,
            'g_index': g_index,
            'i10_index': i10_index,
            'i100_index': i100_index,
            'international_collaboration_rate': international_collaboration_rate  # ДОБАВЛЕНО
        }

    def _analyze_basic_metrics(self) -> Dict:
        # Level I (weighted)
        metrics_I = self._calc_group_metrics(self.level_I, self.metadata_I, is_weighted=True)
        
        # Level II (not weighted) - теперь будет содержать international_collaboration_rate
        metrics_II = self._calc_group_metrics(self.level_II, self.metadata_II, is_weighted=False)
        
        # Level III (weighted)
        metrics_III = self._calc_group_metrics(self.level_III, self.metadata_III, is_weighted=True)
        
        return {
            'level_I': metrics_I,
            'level_II': metrics_II,
            'level_III': metrics_III
        }
    
    def _analyze_authors(self) -> Dict:
        """Analyze authors for Level II only with ORCID-based merging and personal affiliations only"""
        author_stats = defaultdict(lambda: {
            'publications': 0,
            'citations': 0,
            'orcid': None,
            'affiliations': set(),
            'countries': set()
        })
        
        for doi in self.level_II:
            meta = self.metadata_II.get(doi, {})
            citations = meta.get('cited_by_count', 0)
            authors_with_orcids = meta.get('authors_with_orcids', [])
            authorships_raw = meta.get('authorships_raw', [])
            
            # Create mapping from author name to their affiliations in this work
            author_affiliations_map = {}
            author_countries_map = {}
            
            for auth in authorships_raw:
                auth_name = auth.get('author', '')
                if not auth_name:
                    continue
                
                # Get institutions for this specific author
                inst_names = []
                inst_countries = []
                
                for inst in auth.get('institutions', []):
                    inst_name = inst.get('display_name', '')
                    if inst_name:
                        inst_names.append(inst_name)
                    
                    country_code = inst.get('country_code', '')
                    if country_code:
                        country_name = get_full_country_name(country_code)
                        if country_name and country_name != 'Unknown':
                            inst_countries.append(country_name)
                
                # Also check raw_affiliation_strings if institutions are empty
                if not inst_names and auth.get('raw_affiliation_strings'):
                    for aff_str in auth.get('raw_affiliation_strings', []):
                        if aff_str:
                            inst_names.append(aff_str)
                            # Try to extract country from raw string
                            country = extract_country_from_affiliation(aff_str)
                            if country and country != 'Unknown':
                                inst_countries.append(country)
                
                author_affiliations_map[auth_name] = inst_names
                author_countries_map[auth_name] = list(set(inst_countries))
            
            # Now process each author and add only their personal affiliations
            for auth in authors_with_orcids:
                name = auth.get('name', '')
                orcid = auth.get('orcid')
                
                if not name:
                    continue
                
                author_stats[name]['publications'] += 1
                author_stats[name]['citations'] += citations
                if orcid:
                    author_stats[name]['orcid'] = orcid
                
                # Add only this author's personal affiliations from this work
                personal_affs = author_affiliations_map.get(name, [])
                for aff in personal_affs:
                    if aff:
                        author_stats[name]['affiliations'].add(aff)
                
                personal_countries = author_countries_map.get(name, [])
                for country in personal_countries:
                    if country:
                        author_stats[name]['countries'].add(country)
        
        # MERGE AUTHORS BY ORCID AND NORMALIZED NAMES
        merged_stats = self._merge_authors_by_orcid(dict(author_stats))
        
        # Sort by publications
        sorted_authors = sorted(
            merged_stats.items(),
            key=lambda x: x[1]['publications'],
            reverse=True
        )
        
        return {
            'top_authors': [
                {
                    'name': name,
                    'publications': data['publications'],
                    'citations': data['citations'],
                    'orcid': data.get('orcid'),
                    'affiliations': list(data.get('affiliations', []))[:5],
                    'countries': list(data.get('countries', []))[:5]
                }
                for name, data in sorted_authors
            ]
        }
    
    def _analyze_affiliations(self) -> Dict:
        """Analyze affiliations for Level II only with ROR-based aggregation"""
        affiliations_by_ror = defaultdict(lambda: {
            'name': '',
            'count': 0,
            'ror': '',
            'ror_short': ''
        })
        
        for doi in self.level_II:
            meta = self.metadata_II.get(doi, {})
            
            work_ror_ids = set()
            
            for inst in meta.get('institutions', []):
                ror = inst.get('ror', '')
                inst_name = inst.get('display_name', '')
                
                if ror:
                    work_ror_ids.add(ror)
                elif inst_name:
                    key = f"no_ror_{inst_name}"
                    work_ror_ids.add(key)
            
            for ror_id in work_ror_ids:
                if ror_id.startswith('no_ror_'):
                    inst_name = ror_id.replace('no_ror_', '')
                    if not affiliations_by_ror.get(ror_id):
                        affiliations_by_ror[ror_id] = {
                            'name': inst_name,
                            'count': 0,
                            'ror': '',
                            'ror_short': ''
                        }
                    affiliations_by_ror[ror_id]['count'] += 1
                else:
                    inst_name = ''
                    for inst in meta.get('institutions', []):
                        if inst.get('ror', '') == ror_id:
                            inst_name = inst.get('display_name', '')
                            break
                    
                    if not affiliations_by_ror[ror_id]['name']:
                        ror_short = ror_id.replace('https://ror.org/', '') if ror_id else ''
                        affiliations_by_ror[ror_id] = {
                            'name': inst_name,
                            'count': 0,
                            'ror': ror_id,
                            'ror_short': ror_short
                        }
                    affiliations_by_ror[ror_id]['count'] += 1
        
        sorted_affs = sorted(
            [{
                'name': data['name'],
                'count': data['count'],
                'ror': data['ror'],
                'ror_short': data['ror_short']
            } for data in affiliations_by_ror.values() if data['count'] > 0],
            key=lambda x: x['count'],
            reverse=True
        )
        
        return {
            'top_affiliations': sorted_affs
        }
    
    def _analyze_geographic(self) -> Dict:
        """Analyze geographic data for Level II only"""
        countries_per_work = []
        authors_per_country = defaultdict(int)
        single_country_papers = 0
        multi_country_papers = 0
        country_pairs = defaultdict(int)
        
        for doi in self.level_II:
            meta = self.metadata_II.get(doi, {})
            
            work_countries = set()
            authorships = meta.get('authorships_raw', [])
            
            if authorships:
                for auth in authorships:
                    for inst in auth.get('institutions', []):
                        country_code = inst.get('country_code', '')
                        if country_code:
                            country_name = get_full_country_name(country_code)
                            if country_name and country_name != 'Unknown':
                                work_countries.add(country_name)
                                authors_per_country[country_name] += 1
            else:
                for inst in meta.get('institutions', []):
                    country_code = inst.get('country_code', '')
                    if country_code:
                        country_name = get_full_country_name(country_code)
                        if country_name and country_name != 'Unknown':
                            work_countries.add(country_name)
                            for _ in meta.get('authors', []):
                                authors_per_country[country_name] += 1
            
            if not work_countries:
                work_countries = set(meta.get('affiliation_countries', []))
                work_countries = {c for c in work_countries if c and c != 'Unknown'}
            
            if work_countries:
                countries_per_work.append({
                    'work_doi': doi,
                    'countries': list(work_countries),
                    'count': len(work_countries)
                })
                
                if len(work_countries) == 1:
                    single_country_papers += 1
                else:
                    multi_country_papers += 1
                
                country_list = list(work_countries)
                for i in range(len(country_list)):
                    for j in range(i+1, len(country_list)):
                        pair = tuple(sorted([country_list[i], country_list[j]]))
                        country_pairs[pair] += 1
        
        country_counts = [item['count'] for item in countries_per_work]
        
        unique_countries_stats = {
            'avg': np.mean(country_counts) if country_counts else 0,
            'min': min(country_counts) if country_counts else 0,
            'max': max(country_counts) if country_counts else 0,
            'total_works': len(country_counts)
        }
        
        country_stats = defaultdict(lambda: {
            'unique_works': 0,
            'authors_count': 0,
            'work_dois': set()
        })
        
        for work_data in countries_per_work:
            for country in work_data['countries']:
                country_stats[country]['unique_works'] += 1
                country_stats[country]['work_dois'].add(work_data['work_doi'])
        
        for country, author_count in authors_per_country.items():
            country_stats[country]['authors_count'] = author_count
        
        country_stats_list = []
        for country, stats in country_stats.items():
            country_stats_list.append({
                'country': country,
                'unique_works': stats['unique_works'],
                'authors_count': stats['authors_count'],
                'work_dois': list(stats['work_dois'])
            })
        
        country_stats_list.sort(key=lambda x: x['unique_works'], reverse=True)
        
        sorted_pairs = sorted(country_pairs.items(), key=lambda x: x[1], reverse=True)[:20]
        
        return {
            'country_stats': country_stats_list,
            'unique_countries_per_publication': unique_countries_stats,
            'collaboration_patterns': {
                'single_country': single_country_papers,
                'multi_country': multi_country_papers,
                'total': single_country_papers + multi_country_papers,
                'single_country_ratio': single_country_papers / (single_country_papers + multi_country_papers) if (single_country_papers + multi_country_papers) > 0 else 0
            },
            'collaboration_couples': [{'country1': pair[0], 'country2': pair[1], 'frequency': freq} for pair, freq in sorted_pairs]
        }
    
    def _analyze_citations(self) -> Dict:
        """Analyze citations for Level II only"""
        current_year = datetime.now().year
        
        # Get publication years for Level II
        pub_years = {}
        for doi in self.level_II:
            meta = self.metadata_II.get(doi, {})
            pub_years[doi] = {
                'year': meta.get('publication_year'),
                'date': meta.get('publication_date')
            }
        
        # Citation dynamics
        dynamics = defaultdict(lambda: defaultdict(int))
        first_citation_lags = []
        cumulative = defaultdict(int)
        heatmap = defaultdict(lambda: defaultdict(int))
        
        # Get all citation years from Level III
        for citing_doi, count in self.level_III.items():
            meta = self.metadata_III.get(citing_doi, {})
            cite_year = meta.get('publication_year')
            if not cite_year:
                continue
            
            # Find which Level II DOIs this citing work cites
            cited_dois = self.citations_from_III_to_II.get(citing_doi, [])
            for cited_doi in cited_dois:
                pub_year = pub_years.get(cited_doi, {}).get('year')
                if not pub_year or pub_year > cite_year:
                    continue
                
                # Dynamics
                dynamics[pub_year][cite_year] += 1
                heatmap[pub_year][cite_year] += 1
                cumulative[cite_year] += 1
                
                # First citation lag
                pub_date = pub_years.get(cited_doi, {}).get('date')
                if pub_date and meta.get('publication_date'):
                    try:
                        pub_dt = datetime.fromisoformat(pub_date[:10])
                        cite_dt = datetime.fromisoformat(meta['publication_date'][:10])
                        lag = (cite_dt - pub_dt).days
                        if lag >= 0:
                            first_citation_lags.append(lag)
                    except:
                        pass
        
        # Build dynamics matrix
        all_pub_years = sorted([y for y in dynamics.keys() if y <= current_year])
        all_cite_years = sorted([y for y in set([y for sub in dynamics.values() for y in sub.keys()]) if y <= current_year])
        
        if not all_cite_years:
            all_cite_years = all_pub_years
        
        complete_dynamics = []
        for pub_year in all_pub_years:
            for cite_year in all_cite_years:
                if cite_year < pub_year:
                    continue
                value = dynamics[pub_year].get(cite_year, 0)
                complete_dynamics.append({
                    'publication_year': pub_year,
                    'citation_year': cite_year,
                    'citations_count': value
                })
        
        sorted_dynamics = sorted(complete_dynamics, key=lambda x: (x['publication_year'], x['citation_year']))
        
        # Cumulative
        sorted_cumulative = sorted(cumulative.items())
        cumulative_list = []
        running_total = 0
        for year, count in sorted_cumulative:
            if year <= current_year:
                running_total += count
                cumulative_list.append({
                    'year': year,
                    'citations': running_total
                })
        
        # First citation stats
        first_citation_stats = {}
        if first_citation_lags:
            first_citation_stats = {
                'min': min([lag for lag in first_citation_lags if lag > 0]) if any(lag > 0 for lag in first_citation_lags) else 0,
                'max': max(first_citation_lags),
                'avg': np.mean(first_citation_lags),
                'median': np.median(first_citation_lags),
                'count': len(first_citation_lags)
            }
        
        # Heatmap
        all_years = list(range(min(all_pub_years) if all_pub_years else current_year - 5, current_year + 1))
        
        heatmap_data = []
        for pub_year in all_years:
            row = {'publication_year': pub_year}
            has_data = False
            
            for cite_year in all_years:
                if cite_year < pub_year:
                    row[cite_year] = None
                    continue
                
                value = heatmap[pub_year].get(cite_year, 0)
                if value > 0:
                    has_data = True
                    row[cite_year] = value
                else:
                    row[cite_year] = 0
            
            if has_data or pub_year in dynamics:
                heatmap_data.append(row)
        
        heatmap_data.sort(key=lambda x: x['publication_year'])
        
        # Most Cited Publications (Level II)
        most_cited = []
        for doi in self.level_II:
            meta = self.metadata_II.get(doi, {})
            citations = meta.get('cited_by_count', 0)
            year = meta.get('publication_year')
            
            if year and year > current_year:
                continue
            
            years_since = current_year - year + 1 if year else 1
            citations_per_year = citations / max(years_since, 1)
            
            authors = meta.get('authors', [])
            authors_str = ', '.join(authors[:3])
            if len(authors) > 3:
                authors_str += f' +{len(authors)-3} more'
            
            most_cited.append({
                'title': meta.get('title', 'No title'),
                'year': year,
                'citations': citations,
                'citations_per_year': citations_per_year,
                'authors': authors_str,
                'doi': doi,
                'journal': meta.get('journal_name', 'Unknown')
            })
        
        most_cited.sort(key=lambda x: x['citations'], reverse=True)
        
        return {
            'dynamics': sorted_dynamics,
            'first_citation_stats': first_citation_stats,
            'cumulative': cumulative_list,
            'heatmap': heatmap_data,
            'heatmap_years': all_years,
            'most_cited': most_cited[:10]
        }
    
    def _analyze_citing_works(self) -> Dict:
        """
        Analyze citing works (Level III) with weighted counts
        WEIGHTED COUNT = number of Level II articles cited by this citing work
        """
        total_citing = sum(self.level_III.values())
        
        # Authors aggregation with ORCID deduplication
        author_stats = defaultdict(lambda: {
            'count': 0,
            'orcid': None,
            'names_seen': set()
        })
        
        # Affiliations with ROR
        affiliations_by_ror = defaultdict(lambda: {
            'name': '',
            'count': 0,
            'ror': '',
            'ror_short': ''
        })
        
        countries = defaultdict(int)
        journals = defaultdict(int)
        publishers = defaultdict(int)
        
        # Track weighted count for each citing work
        citing_works_weighted = {}
        
        for citing_doi, weight in self.level_III.items():
            meta = self.metadata_III.get(citing_doi, {})
            
            # Calculate weighted count = number of Level II DOIs cited
            weighted_count = len(self.citations_from_III_to_II.get(citing_doi, []))
            citing_works_weighted[citing_doi] = weighted_count
            
            # Collect authors
            authors_with_orcids = meta.get('authors_with_orcids', [])
            for auth in authors_with_orcids:
                name = auth.get('name', '')
                orcid = auth.get('orcid')
                
                if not name:
                    continue
                
                author_stats[name]['count'] += weight
                author_stats[name]['names_seen'].add(name)
                if orcid and not author_stats[name]['orcid']:
                    author_stats[name]['orcid'] = orcid
            
            # Countries (with weight)
            work_countries = set(meta.get('affiliation_countries', []))
            for country in work_countries:
                countries[country] += weight
            
            # Journals
            journal = meta.get('journal_name', 'Unknown')
            journals[journal] += weight
            
            # Publishers
            publisher = meta.get('publisher', 'Unknown')
            publishers[publisher] += weight
            
            # Affiliations (with ROR, weighted)
            work_ror_ids = set()
            for inst in meta.get('institutions', []):
                ror = inst.get('ror', '')
                inst_name = inst.get('display_name', '')
                
                if ror:
                    work_ror_ids.add(ror)
                elif inst_name:
                    key = f"no_ror_{inst_name}"
                    work_ror_ids.add(key)
            
            for ror_id in work_ror_ids:
                if ror_id.startswith('no_ror_'):
                    inst_name = ror_id.replace('no_ror_', '')
                    if not affiliations_by_ror.get(ror_id):
                        affiliations_by_ror[ror_id] = {
                            'name': inst_name,
                            'count': 0,
                            'ror': '',
                            'ror_short': ''
                        }
                    affiliations_by_ror[ror_id]['count'] += weight
                else:
                    inst_name = ''
                    for inst in meta.get('institutions', []):
                        if inst.get('ror', '') == ror_id:
                            inst_name = inst.get('display_name', '')
                            break
                    
                    if not affiliations_by_ror[ror_id]['name']:
                        ror_short = ror_id.replace('https://ror.org/', '') if ror_id else ''
                        affiliations_by_ror[ror_id] = {
                            'name': inst_name,
                            'count': 0,
                            'ror': ror_id,
                            'ror_short': ror_short
                        }
                    affiliations_by_ror[ror_id]['count'] += weight
        
        # Convert author_stats to mergeable format
        mergeable_stats = {}
        for name, data in author_stats.items():
            mergeable_stats[name] = {
                'publications': data['count'],
                'citations': 0,
                'orcid': data['orcid'],
                'affiliations': [],
                'countries': []
            }
        
        # MERGE AUTHORS BY ORCID AND NORMALIZED NAMES
        merged_stats = self._merge_authors_by_orcid(mergeable_stats)
        
        # Convert back to citing format
        top_authors = []
        for name, data in merged_stats.items():
            top_authors.append({
                'name': name,
                'orcid': data.get('orcid'),
                'count': data['publications']
            })
        
        top_authors.sort(key=lambda x: x['count'], reverse=True)
        
        # Sort affiliations
        top_affiliations = sorted(
            [{
                'name': data['name'],
                'count': data['count'],
                'ror': data['ror'],
                'ror_short': data['ror_short']
            } for data in affiliations_by_ror.values() if data['count'] > 0],
            key=lambda x: x['count'],
            reverse=True
        )
        
        top_countries = sorted(countries.items(), key=lambda x: x[1], reverse=True)
        top_journals = sorted(journals.items(), key=lambda x: x[1], reverse=True)
        top_publishers = sorted(publishers.items(), key=lambda x: x[1], reverse=True)
        
        # Sort citing works by weighted count
        sorted_citing_by_weight = sorted(
            citing_works_weighted.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Build citing works with weighted counts for display
        citing_works_with_weight = []
        for citing_doi, weighted_count in sorted_citing_by_weight[:50]:
            meta = self.metadata_III.get(citing_doi, {})
            citing_works_with_weight.append({
                'doi': citing_doi,
                'weighted_count': weighted_count,
                'title': meta.get('title', 'No title'),
                'year': meta.get('publication_year'),
                'journal': meta.get('journal_name', 'Unknown'),
                'authors': meta.get('authors', [])[:3]
            })
        
        return {
            'total_citing_works': total_citing,
            'total_unique': len(self.level_III),
            'top_authors': [{'name': item['name'], 'orcid': item['orcid'], 'count': item['count']} for item in top_authors],
            'top_affiliations': top_affiliations,
            'top_countries': [{'name': name, 'count': count} for name, count in top_countries],
            'top_journals': [{'name': name, 'count': count} for name, count in top_journals],
            'top_publishers': [{'name': name, 'count': count} for name, count in top_publishers],
            'citing_works_weighted': citing_works_with_weight,
            'max_weighted_count': max(citing_works_weighted.values()) if citing_works_weighted else 0
        }
    
    def _analyze_topics(self) -> Dict:
        """Analyze topics for all three levels with weighted counts"""
        # Level I (weighted)
        level_I_topics = defaultdict(lambda: {'count': 0, 'years': []})
        for doi, weight in self.level_I.items():
            meta = self.metadata_I.get(doi, {})
            year = meta.get('publication_year')
            for topic in meta.get('topics', []):
                level_I_topics[topic]['count'] += weight
                if year:
                    level_I_topics[topic]['years'].append(year)
        
        # Level II (not weighted)
        level_II_topics = defaultdict(lambda: {'count': 0, 'years': []})
        for doi in self.level_II:
            meta = self.metadata_II.get(doi, {})
            year = meta.get('publication_year')
            for topic in meta.get('topics', []):
                level_II_topics[topic]['count'] += 1
                if year:
                    level_II_topics[topic]['years'].append(year)
        
        # Level III (weighted)
        level_III_topics = defaultdict(lambda: {'count': 0, 'years': []})
        for doi, weight in self.level_III.items():
            meta = self.metadata_III.get(doi, {})
            year = meta.get('publication_year')
            for topic in meta.get('topics', []):
                level_III_topics[topic]['count'] += weight
                if year:
                    level_III_topics[topic]['years'].append(year)
        
        # Combine all topics
        all_topics = set(level_I_topics.keys()) | set(level_II_topics.keys()) | set(level_III_topics.keys())
        
        total_I = sum(self.level_I.values()) if self.level_I else 1
        total_II = len(self.level_II) if self.level_II else 1
        total_III = sum(self.level_III.values()) if self.level_III else 1
        
        topic_results = []
        for topic in all_topics:
            count_I = level_I_topics[topic]['count']
            count_II = level_II_topics[topic]['count']
            count_III = level_III_topics[topic]['count']
            
            norm_I = count_I / total_I if total_I > 0 else 0
            norm_II = count_II / total_II if total_II > 0 else 0
            norm_III = count_III / total_III if total_III > 0 else 0
            total_norm = norm_I + norm_II + norm_III
            
            years = level_I_topics[topic]['years'] + level_II_topics[topic]['years'] + level_III_topics[topic]['years']
            first_year = min(years) if years else None
            peak_year = max(Counter(years).items(), key=lambda x: x[1])[0] if years else None
            
            topic_results.append({
                'topic': topic,
                'count_I': count_I,
                'count_II': count_II,
                'count_III': count_III,
                'norm_I': norm_I,
                'norm_II': norm_II,
                'norm_III': norm_III,
                'total_norm': total_norm,
                'first_year': first_year,
                'peak_year': peak_year
            })
        
        topic_results.sort(key=lambda x: x['total_norm'], reverse=True)
        
        # ===== FIXED: Top cited for each topic category (using UNIQUE counts, not weighted) =====
        def get_top_cited_count(items_key):
            """Get top cited items using UNIQUE counts per level, not weighted"""
            counter = defaultdict(int)
            
            # Level I - use unique DOIs (not weighted)
            for doi in self.level_I.keys():  # Changed: iterate over keys, not items
                meta = self.metadata_I.get(doi, {})
                for item in meta.get(items_key, []):
                    counter[item] += 1  # Changed: +1 instead of +weight
            
            # Level II - unique DOIs
            for doi in self.level_II:
                meta = self.metadata_II.get(doi, {})
                for item in meta.get(items_key, []):
                    counter[item] += 1
            
            # Level III - use unique DOIs (not weighted)
            for doi in self.level_III.keys():  # Changed: iterate over keys, not items
                meta = self.metadata_III.get(doi, {})
                for item in meta.get(items_key, []):
                    counter[item] += 1  # Changed: +1 instead of +weight
            
            return sorted(counter.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'topics': topic_results[:30],
            'top_cited_topics': get_top_cited_count('topics'),
            'top_cited_subtopics': get_top_cited_count('subtopics'),
            'top_cited_fields': get_top_cited_count('fields'),
            'top_cited_domains': get_top_cited_count('domains'),
            'top_cited_concepts': get_top_cited_count('concepts')
        }
    
    def _get_detailed_citations(self) -> Dict:
        """Get detailed citations for Level II only"""
        detailed = {}
        
        for doi in self.level_II:
            meta = self.metadata_II.get(doi, {})
            
            # Find citing works for this DOI
            citing_list = []
            for citing_doi, cited_list in self.citations_from_III_to_II.items():
                if doi in cited_list:
                    citing_meta = self.metadata_III.get(citing_doi, {})
                    
                    # Calculate citation lag
                    citation_lag = None
                    pub_date = meta.get('publication_date')
                    cite_date = citing_meta.get('publication_date')
                    if pub_date and cite_date:
                        try:
                            pub_dt = datetime.fromisoformat(pub_date[:10])
                            cite_dt = datetime.fromisoformat(cite_date[:10])
                            citation_lag = (cite_dt - pub_dt).days
                        except:
                            pass
                    
                    citing_list.append({
                        'citing_title': citing_meta.get('title', 'No title'),
                        'citing_year': citing_meta.get('publication_year'),
                        'citing_date': cite_date,
                        'citing_journal': citing_meta.get('journal_name', 'Unknown'),
                        'citing_publisher': citing_meta.get('publisher', 'Unknown'),
                        'citing_doi': citing_doi,
                        'citation_lag': citation_lag,
                        'citing_authors': citing_meta.get('authors', []),
                        'citing_countries': citing_meta.get('affiliation_countries', []),
                        'citing_topics': citing_meta.get('topics', [])
                    })
            
            detailed[doi] = {
                'title': meta.get('title', 'No title'),
                'year': meta.get('publication_year'),
                'doi': doi,
                'total_citations': len(citing_list),
                'citations': citing_list
            }
        
        return detailed
    
    def _analyze_author_distribution(self) -> Dict:
        """Analyze distribution of publications by number of authors for Level II and Level III"""
        # Level II distribution
        level_II_dist = defaultdict(int)
        level_II_total = 0
        
        for doi in self.level_II:
            meta = self.metadata_II.get(doi, {})
            author_count = meta.get('author_count', 0)
            if author_count > 0:
                if author_count == 1:
                    level_II_dist['1'] += 1
                elif author_count == 2:
                    level_II_dist['2'] += 1
                elif 3 <= author_count <= 5:
                    level_II_dist['3-5'] += 1
                elif 6 <= author_count <= 7:
                    level_II_dist['6-7'] += 1
                elif 8 <= author_count <= 10:
                    level_II_dist['8-10'] += 1
                elif 11 <= author_count <= 15:
                    level_II_dist['11-15'] += 1
                else:
                    level_II_dist['15+'] += 1
                level_II_total += 1
        
        # Level III distribution (with weights)
        level_III_dist = defaultdict(int)
        level_III_total = 0
        
        for doi, weight in self.level_III.items():
            meta = self.metadata_III.get(doi, {})
            author_count = meta.get('author_count', 0)
            if author_count > 0:
                if author_count == 1:
                    level_III_dist['1'] += weight
                elif author_count == 2:
                    level_III_dist['2'] += weight
                elif 3 <= author_count <= 5:
                    level_III_dist['3-5'] += weight
                elif 6 <= author_count <= 7:
                    level_III_dist['6-7'] += weight
                elif 8 <= author_count <= 10:
                    level_III_dist['8-10'] += weight
                elif 11 <= author_count <= 15:
                    level_III_dist['11-15'] += weight
                else:
                    level_III_dist['15+'] += weight
                level_III_total += weight
        
        category_order = ['1', '2', '3-5', '6-7', '8-10', '11-15', '15+']
        
        sorted_level_II = {}
        for cat in category_order:
            if cat in level_II_dist:
                sorted_level_II[cat] = level_II_dist[cat]
        
        sorted_level_III = {}
        for cat in category_order:
            if cat in level_III_dist:
                sorted_level_III[cat] = level_III_dist[cat]
        
        return {
            'level_II': {
                'distribution': sorted_level_II,
                'total': level_II_total
            },
            'level_III': {
                'distribution': sorted_level_III,
                'total': level_III_total
            }
        }
    
    def _analyze_multilevel_relationships(self) -> Dict:
        """
        Build multilevel matrices for authors, affiliations, journals, publishers
        with counts and normalized values for all three levels
        """
        total_I = sum(self.level_I.values()) if self.level_I else 1
        total_II = len(self.level_II) if self.level_II else 1
        total_III = sum(self.level_III.values()) if self.level_III else 1
        
        # Helper to build matrix for a given attribute
        def build_matrix(attribute: str, get_orcid: bool = False, is_string: bool = False) -> List[Dict]:
            stats = defaultdict(lambda: {
                'count_I': 0,
                'count_II': 0,
                'count_III': 0,
                'orcid': None,
                'name': ''
            })
            
            def get_items(meta: Dict, attr: str):
                """Get items from metadata, handling both lists and strings"""
                value = meta.get(attr, [])
                if is_string:
                    # For journal_name and publisher, return as single item list
                    return [value] if value and isinstance(value, str) else []
                else:
                    # For authors and affiliations, already a list
                    return value if isinstance(value, list) else []
            
            # Level I (weighted)
            for doi, weight in self.level_I.items():
                meta = self.metadata_I.get(doi, {})
                items = get_items(meta, attribute)
                for item in items:
                    if item:
                        stats[item]['count_I'] += weight
                        stats[item]['name'] = item
                        if get_orcid and attribute == 'authors':
                            for auth in meta.get('authors_with_orcids', []):
                                if auth.get('name') == item and auth.get('orcid'):
                                    stats[item]['orcid'] = auth.get('orcid')
                                    break
            
            # Level II (not weighted)
            for doi in self.level_II:
                meta = self.metadata_II.get(doi, {})
                items = get_items(meta, attribute)
                for item in items:
                    if item:
                        stats[item]['count_II'] += 1
                        stats[item]['name'] = item
                        if get_orcid and attribute == 'authors':
                            for auth in meta.get('authors_with_orcids', []):
                                if auth.get('name') == item and auth.get('orcid'):
                                    stats[item]['orcid'] = auth.get('orcid')
                                    break
            
            # Level III (weighted)
            for doi, weight in self.level_III.items():
                meta = self.metadata_III.get(doi, {})
                items = get_items(meta, attribute)
                for item in items:
                    if item:
                        stats[item]['count_III'] += weight
                        stats[item]['name'] = item
                        if get_orcid and attribute == 'authors':
                            for auth in meta.get('authors_with_orcids', []):
                                if auth.get('name') == item and auth.get('orcid'):
                                    stats[item]['orcid'] = auth.get('orcid')
                                    break
            
            # Calculate norms
            result = []
            for item, data in stats.items():
                norm_I = data['count_I'] / total_I if total_I > 0 else 0
                norm_II = data['count_II'] / total_II if total_II > 0 else 0
                norm_III = data['count_III'] / total_III if total_III > 0 else 0
                total_norm = norm_I + norm_II + norm_III
                
                result.append({
                    'name': data['name'],
                    'orcid': data.get('orcid'),
                    'count_I': data['count_I'],
                    'count_II': data['count_II'],
                    'count_III': data['count_III'],
                    'norm_I': norm_I,
                    'norm_II': norm_II,
                    'norm_III': norm_III,
                    'total_norm': total_norm
                })
            
            return sorted(result, key=lambda x: x['total_norm'], reverse=True)[:50]
        
        return {
            'author_matrix': build_matrix('authors', get_orcid=True, is_string=False),
            'affiliation_matrix': build_matrix('affiliations', get_orcid=False, is_string=False),
            'journal_matrix': build_matrix('journal_name', get_orcid=False, is_string=True),
            'publisher_matrix': build_matrix('publisher', get_orcid=False, is_string=True)
        }
    
    def _get_references_list(self) -> List[Dict]:
        """Get list of references (Level I) with counts"""
        result = []
        for doi, count in sorted(self.level_I.items(), key=lambda x: x[1], reverse=True):
            meta = self.metadata_I.get(doi, {})
            result.append({
                'doi': doi,
                'count': count,
                'title': meta.get('title', 'No title'),
                'year': meta.get('publication_year'),
                'journal': meta.get('journal_name', 'Unknown'),
                'authors': meta.get('authors', [])
            })
        return result

    def _get_analyzed_articles_list(self) -> List[Dict]:
        """Get list of analyzed articles (Level II) with full metadata"""
        result = []
        
        for doi in self.level_II:
            meta = self.metadata_II.get(doi, {})
            
            # Get authors with ORCID
            authors_with_orcids = meta.get('authors_with_orcids', [])
            authors_list = [a.get('name', '') for a in authors_with_orcids if a.get('name')]
            authors_str = ', '.join(authors_list[:5])
            if len(authors_list) > 5:
                authors_str += f' +{len(authors_list)-5} more'
            
            # Get affiliations
            affiliations = meta.get('affiliations', [])
            affiliations_str = ', '.join(affiliations[:3])
            if len(affiliations) > 3:
                affiliations_str += f' +{len(affiliations)-3} more'
            
            # Get countries
            countries = meta.get('affiliation_countries', [])
            countries_str = ', '.join(countries[:3])
            if len(countries) > 3:
                countries_str += f' +{len(countries)-3} more'
            
            # Get citation info
            citations = meta.get('cited_by_count', 0)
            year = meta.get('publication_year')
            
            # Get journal and publisher
            journal = meta.get('journal_name', 'Unknown')
            publisher = meta.get('publisher', 'Unknown')
            
            # Get Open Access status
            oa_status = meta.get('oa_status', 'unknown')
            is_oa = meta.get('is_oa', False)
            
            # Get topics
            topics = meta.get('topics', [])
            topics_str = ', '.join(topics[:3])
            if len(topics) > 3:
                topics_str += f' +{len(topics)-3} more'
            
            result.append({
                'doi': doi,
                'title': meta.get('title', 'No title'),
                'year': year,
                'authors': authors_str,
                'authors_full': authors_list,
                'authors_with_orcids': authors_with_orcids,
                'affiliations': affiliations_str,
                'affiliations_full': affiliations,
                'countries': countries_str,
                'countries_full': countries,
                'citations': citations,
                'journal': journal,
                'publisher': publisher,
                'oa_status': oa_status,
                'is_oa': is_oa,
                'topics': topics_str,
                'topics_full': topics,
                'publication_date': meta.get('publication_date'),
                'type': meta.get('type', 'unknown'),
                'raw_type': meta.get('raw_type', '')
            })
        
        # Sort by year (newest first), then by citations
        result.sort(key=lambda x: (x.get('year') or 0, x.get('citations', 0)), reverse=True)
        
        return result

    def _merge_authors_by_orcid(self, author_stats: dict) -> dict:
        """Merge authors by ORCID and normalized names"""
        
        def normalize_for_merge(name: str) -> str:
            """Extract last name + first initial for matching"""
            if not name:
                return ''
            name = name.strip()
            parts = name.split()
            if len(parts) >= 2:
                last = parts[-1]
                first_init = parts[0][0] if parts[0] else ''
                return f"{last} {first_init}".lower()
            return name.lower()
        
        # Group by ORCID (if exists)
        orcid_groups = {}
        no_orcid_groups = {}
        name_variants = defaultdict(set)
        
        for name, data in author_stats.items():
            orcid = data.get('orcid')
            norm_name = normalize_for_merge(name)
            name_variants[norm_name].add(name)
            
            if orcid:
                if orcid not in orcid_groups:
                    orcid_groups[orcid] = []
                orcid_groups[orcid].append((name, data))
            else:
                key = (norm_name, tuple(sorted(data.get('affiliations', []))[:2]))
                if key not in no_orcid_groups:
                    no_orcid_groups[key] = []
                no_orcid_groups[key].append((name, data))
        
        # Merge by ORCID
        merged = {}
        for orcid, entries in orcid_groups.items():
            if len(entries) == 1:
                name, data = entries[0]
                merged[name] = data.copy()
                continue
            
            # Merge multiple entries with same ORCID
            merged_data = {
                'publications': 0,
                'citations': 0,
                'orcid': orcid,
                'affiliations': set(),
                'countries': set(),
                'names': set()
            }
            
            for name, data in entries:
                merged_data['publications'] += data['publications']
                merged_data['citations'] += data['citations']
                merged_data['affiliations'].update(data.get('affiliations', []))
                merged_data['countries'].update(data.get('countries', []))
                merged_data['names'].add(name)
            
            # Choose best name (most complete or most common)
            best_name = max(merged_data['names'], key=lambda x: (len(x), x.count(' ')))
            merged[best_name] = {
                'publications': merged_data['publications'],
                'citations': merged_data['citations'],
                'orcid': merged_data['orcid'],
                'affiliations': list(merged_data['affiliations']),
                'countries': list(merged_data['countries'])
            }
        
        # Merge no-ORCID entries with similar names and affiliations
        for key, entries in no_orcid_groups.items():
            if len(entries) == 1:
                name, data = entries[0]
                if name not in merged:
                    merged[name] = data.copy()
                continue
            
            # Check if any entry matches with ORCID entries by name
            for name, data in entries:
                norm_name = normalize_for_merge(name)
                found_match = False
                
                # Try to match with ORCID entries
                for existing_name, existing_data in merged.items():
                    if normalize_for_merge(existing_name) == norm_name:
                        # Merge
                        existing_data['publications'] += data['publications']
                        existing_data['citations'] += data['citations']
                        existing_data['affiliations'] = list(set(existing_data.get('affiliations', set())) | set(data.get('affiliations', set())))
                        existing_data['countries'] = list(set(existing_data.get('countries', set())) | set(data.get('countries', set())))
                        found_match = True
                        break
                
                if not found_match:
                    if name not in merged:
                        merged[name] = data.copy()
        
        # Re-aggregate similar names within merged dict
        final_merged = {}
        used_names = set()
        
        for name, data in merged.items():
            norm_name = normalize_for_merge(name)
            
            # Check if similar name already exists
            matched = False
            for existing_name in list(final_merged.keys()):
                if normalize_for_merge(existing_name) == norm_name:
                    # Merge
                    final_merged[existing_name]['publications'] += data['publications']
                    final_merged[existing_name]['citations'] += data['citations']
                    final_merged[existing_name]['affiliations'] = list(
                        set(final_merged[existing_name].get('affiliations', set())) | set(data.get('affiliations', set()))
                    )
                    final_merged[existing_name]['countries'] = list(
                        set(final_merged[existing_name].get('countries', set())) | set(data.get('countries', set()))
                    )
                    if not final_merged[existing_name].get('orcid') and data.get('orcid'):
                        final_merged[existing_name]['orcid'] = data['orcid']
                    matched = True
                    break
            
            if not matched:
                final_merged[name] = data.copy()
        
        return final_merged

    # ============================================
    # ============================================
    # NEW: TITLE KEYWORDS ANALYSIS
    # ============================================
    # ============================================

    def _analyze_title_keywords(self) -> Dict:
        """
        Analyze keywords from titles across all three levels
        Uses TitleKeywordsAnalyzer for lemmatization and normalization
        """
        # Collect titles from all levels
        level_I_titles = []
        for doi in self.level_I.keys():
            meta = self.metadata_I.get(doi, {})
            title = meta.get('title', '')
            if title and title not in ['No title', 'Title not found']:
                level_I_titles.append(title)
        
        level_II_titles = []
        for doi in self.level_II:
            meta = self.metadata_II.get(doi, {})
            title = meta.get('title', '')
            if title and title not in ['No title', 'Title not found']:
                level_II_titles.append(title)
        
        level_III_titles = []
        for doi in self.level_III.keys():
            meta = self.metadata_III.get(doi, {})
            title = meta.get('title', '')
            if title and title not in ['No title', 'Title not found']:
                level_III_titles.append(title)
        
        # Run analysis
        analysis_result = self.title_keywords_analyzer.analyze_titles(
            level_II_titles,  # analyzed
            level_I_titles,   # reference
            level_III_titles  # citing
        )
        
        # Prepare aggregated data for HTML display
        # Combine all lemmas from all three levels
        all_lemmas = {}
        
        # Process analyzed (Level II)
        for word_info in analysis_result.get('analyzed', {}).get('words', []):
            lemma = word_info.get('lemma', '')
            if lemma:
                if lemma not in all_lemmas:
                    all_lemmas[lemma] = {
                        'type': word_info.get('type', 'content'),
                        'count_I': 0,
                        'count_II': 0,
                        'count_III': 0,
                        'variants': set()
                    }
                all_lemmas[lemma]['count_II'] = word_info.get('count', 0)
                for variant in word_info.get('variants', {}).keys():
                    all_lemmas[lemma]['variants'].add(variant)
        
        # Process reference (Level I)
        for word_info in analysis_result.get('reference', {}).get('words', []):
            lemma = word_info.get('lemma', '')
            if lemma:
                if lemma not in all_lemmas:
                    all_lemmas[lemma] = {
                        'type': word_info.get('type', 'content'),
                        'count_I': 0,
                        'count_II': 0,
                        'count_III': 0,
                        'variants': set()
                    }
                all_lemmas[lemma]['count_I'] = word_info.get('count', 0)
                for variant in word_info.get('variants', {}).keys():
                    all_lemmas[lemma]['variants'].add(variant)
        
        # Process citing (Level III)
        for word_info in analysis_result.get('citing', {}).get('words', []):
            lemma = word_info.get('lemma', '')
            if lemma:
                if lemma not in all_lemmas:
                    all_lemmas[lemma] = {
                        'type': word_info.get('type', 'content'),
                        'count_I': 0,
                        'count_II': 0,
                        'count_III': 0,
                        'variants': set()
                    }
                all_lemmas[lemma]['count_III'] = word_info.get('count', 0)
                for variant in word_info.get('variants', {}).keys():
                    all_lemmas[lemma]['variants'].add(variant)
        
        # Calculate total counts and normalized values
        total_I = sum(self.level_I.values()) if self.level_I else 1
        total_II = len(self.level_II) if self.level_II else 1
        total_III = sum(self.level_III.values()) if self.level_III else 1
        
        keywords_data = []
        for lemma, stats in all_lemmas.items():
            count_I = stats['count_I']
            count_II = stats['count_II']
            count_III = stats['count_III']
            
            norm_I = count_I / total_I if total_I > 0 else 0
            norm_II = count_II / total_II if total_II > 0 else 0
            norm_III = count_III / total_III if total_III > 0 else 0
            total_norm = norm_I + norm_II + norm_III
            
            keywords_data.append({
                'lemma': lemma,
                'variants': ', '.join(sorted(stats['variants'])) if stats['variants'] else lemma,
                'type': stats['type'],
                'count_I': count_I,
                'count_II': count_II,
                'count_III': count_III,
                'norm_I': norm_I,
                'norm_II': norm_II,
                'norm_III': norm_III,
                'total_norm': total_norm
            })
        
        # Sort by total_norm descending
        keywords_data.sort(key=lambda x: x['total_norm'], reverse=True)
        
        return {
            'keywords': keywords_data[:100],  # Top 100
            'total_titles_I': len(level_I_titles),
            'total_titles_II': len(level_II_titles),
            'total_titles_III': len(level_III_titles),
            'max_count_I': max([k['count_I'] for k in keywords_data]) if keywords_data else 0,
            'max_count_II': max([k['count_II'] for k in keywords_data]) if keywords_data else 0,
            'max_count_III': max([k['count_III'] for k in keywords_data]) if keywords_data else 0,
            'max_norm_I': max([k['norm_I'] for k in keywords_data]) if keywords_data else 0,
            'max_norm_II': max([k['norm_II'] for k in keywords_data]) if keywords_data else 0,
            'max_norm_III': max([k['norm_III'] for k in keywords_data]) if keywords_data else 0,
            'max_total_norm': max([k['total_norm'] for k in keywords_data]) if keywords_data else 0
        }

    # ============================================
    # ============================================
    # NEW: TEMPORAL RELATIONSHIPS ANALYSIS
    # ============================================
    # ============================================

    def _analyze_temporal_relationships(self) -> Dict:
        """
        Analyze temporal relationships between levels with heatmap visualization
        1. Reference → Analyzed (time lag between reference and analyzed article)
        2. Analyzed → Citing (time lag between analyzed and citing article)
        """
        ref_to_analyzed_connections = []
        analyzed_to_citing_connections = []
        
        # Heatmap data: years vs years
        ref_analyzed_heatmap = defaultdict(lambda: defaultdict(int))
        analyzed_citing_heatmap = defaultdict(lambda: defaultdict(int))
        
        # Distribution of lags
        ref_analyzed_lags = []
        analyzed_citing_lags = []
        
        # Helper to parse date
        def parse_date(date_str):
            if not date_str:
                return None
            try:
                return datetime.fromisoformat(date_str[:10])
            except:
                return None
        
        # Get all years from all levels for heatmap axes
        all_years = set()
        
        # For Reference → Analyzed: use years from Level I (references)
        ref_years = set()
        for doi in self.level_I.keys():
            meta = self.metadata_I.get(doi, {})
            year = meta.get('publication_year')
            if year:
                ref_years.add(year)
                all_years.add(year)
        
        # For Analyzed → Citing: use years from Level II (analyzed) and Level III (citing)
        analyzed_years = set()
        citing_years = set()
        
        for doi in self.level_II:
            meta = self.metadata_II.get(doi, {})
            year = meta.get('publication_year')
            if year:
                analyzed_years.add(year)
                all_years.add(year)
        
        for doi in self.level_III.keys():
            meta = self.metadata_III.get(doi, {})
            year = meta.get('publication_year')
            if year:
                citing_years.add(year)
                all_years.add(year)
        
        all_years = sorted([y for y in all_years if y and y > 1900])
        ref_years = sorted([y for y in ref_years if y and y > 1900])
        analyzed_years = sorted([y for y in analyzed_years if y and y > 1900])
        citing_years = sorted([y for y in citing_years if y and y > 1900])
        
        # 1. Reference → Analyzed connections with heatmap
        for ref_doi, weight in self.level_I.items():
            ref_meta = self.metadata_I.get(ref_doi, {})
            ref_date = parse_date(ref_meta.get('publication_date'))
            ref_year = ref_meta.get('publication_year')
            
            if not ref_year:
                continue
            
            # Find which analyzed articles cite this reference
            analyzed_dois = []
            for analyzed_doi, ref_list in self.citations_from_II_to_I.items():
                if ref_doi in ref_list:
                    analyzed_dois.append(analyzed_doi)
            
            for analyzed_doi in analyzed_dois:
                analyzed_meta = self.metadata_II.get(analyzed_doi, {})
                analyzed_date = parse_date(analyzed_meta.get('publication_date'))
                analyzed_year = analyzed_meta.get('publication_year')
                
                if not analyzed_year:
                    continue
                
                if ref_date and analyzed_date:
                    lag_days = (analyzed_date - ref_date).days
                    if lag_days >= 0:
                        ref_to_analyzed_connections.append({
                            'ref_doi': ref_doi,
                            'analyzed_doi': analyzed_doi,
                            'ref_date': ref_date.strftime('%Y-%m-%d'),
                            'analyzed_date': analyzed_date.strftime('%Y-%m-%d'),
                            'lag_days': lag_days,
                            'ref_year': ref_year,
                            'analyzed_year': analyzed_year,
                            'ref_title': ref_meta.get('title', 'No title')[:50],
                            'analyzed_title': analyzed_meta.get('title', 'No title')[:50]
                        })
                        ref_analyzed_lags.append(lag_days)
                        
                        # Heatmap: ref_year -> analyzed_year
                        if ref_year in ref_years and analyzed_year in all_years:
                            ref_analyzed_heatmap[ref_year][analyzed_year] += 1
        
        # 2. Analyzed → Citing connections with heatmap
        for citing_doi, weight in self.level_III.items():
            citing_meta = self.metadata_III.get(citing_doi, {})
            citing_date = parse_date(citing_meta.get('publication_date'))
            citing_year = citing_meta.get('publication_year')
            
            if not citing_year:
                continue
            
            # Find which analyzed articles this citing work cites
            analyzed_dois = self.citations_from_III_to_II.get(citing_doi, [])
            
            for analyzed_doi in analyzed_dois:
                analyzed_meta = self.metadata_II.get(analyzed_doi, {})
                analyzed_date = parse_date(analyzed_meta.get('publication_date'))
                analyzed_year = analyzed_meta.get('publication_year')
                
                if not analyzed_year:
                    continue
                
                if citing_date and analyzed_date:
                    lag_days = (citing_date - analyzed_date).days
                    if lag_days >= 0:
                        analyzed_to_citing_connections.append({
                            'analyzed_doi': analyzed_doi,
                            'citing_doi': citing_doi,
                            'analyzed_date': analyzed_date.strftime('%Y-%m-%d'),
                            'citing_date': citing_date.strftime('%Y-%m-%d'),
                            'lag_days': lag_days,
                            'analyzed_year': analyzed_year,
                            'citing_year': citing_year,
                            'analyzed_title': analyzed_meta.get('title', 'No title')[:50],
                            'citing_title': citing_meta.get('title', 'No title')[:50]
                        })
                        analyzed_citing_lags.append(lag_days)
                        
                        # Heatmap: analyzed_year -> citing_year
                        if analyzed_year in analyzed_years and citing_year in citing_years:
                            analyzed_citing_heatmap[analyzed_year][citing_year] += 1
        
        # Build heatmap data for HTML
        def build_heatmap_data(heatmap_dict, x_years, y_years):
            heatmap_rows = []
            for pub_year in y_years:
                row = {'publication_year': pub_year}
                has_data = False
                for cite_year in x_years:
                    if cite_year < pub_year:
                        row[cite_year] = None
                        continue
                    value = heatmap_dict.get(pub_year, {}).get(cite_year, 0)
                    if value > 0:
                        has_data = True
                        row[cite_year] = value
                    else:
                        row[cite_year] = 0
                if has_data or pub_year in heatmap_dict:
                    heatmap_rows.append(row)
            return heatmap_rows
        
        # For Reference → Analyzed: Y-axis = ref_years, X-axis = all_years (analyzed years)
        ref_heatmap_data = build_heatmap_data(ref_analyzed_heatmap, all_years, ref_years)
        
        # For Analyzed → Citing: Y-axis = analyzed_years, X-axis = citing_years
        citing_heatmap_data = build_heatmap_data(analyzed_citing_heatmap, citing_years, analyzed_years)
        
        # Calculate statistics
        ref_analyzed_stats = {}
        if ref_analyzed_lags:
            ref_analyzed_stats = {
                'min': min(ref_analyzed_lags),
                'max': max(ref_analyzed_lags),
                'avg': np.mean(ref_analyzed_lags),
                'median': np.median(ref_analyzed_lags),
                'count': len(ref_analyzed_lags),
                'std': np.std(ref_analyzed_lags)
            }
        
        analyzed_citing_stats = {}
        if analyzed_citing_lags:
            analyzed_citing_stats = {
                'min': min(analyzed_citing_lags),
                'max': max(analyzed_citing_lags),
                'avg': np.mean(analyzed_citing_lags),
                'median': np.median(analyzed_citing_lags),
                'count': len(analyzed_citing_lags),
                'std': np.std(analyzed_citing_lags)
            }
        
        # Calculate lag distribution bins
        def get_lag_distribution(lags, bins=10):
            if not lags:
                return []
            max_lag = max(lags)
            bin_size = max(1, max_lag // bins)
            hist = defaultdict(int)
            for lag in lags:
                bin_idx = lag // bin_size
                hist[bin_idx * bin_size] += 1
            return sorted(hist.items())
        
        ref_lag_dist = get_lag_distribution(ref_analyzed_lags)
        citing_lag_dist = get_lag_distribution(analyzed_citing_lags)
        
        return {
            'ref_to_analyzed': {
                'connections': ref_to_analyzed_connections[:100],
                'total_connections': len(ref_to_analyzed_connections),
                'stats': ref_analyzed_stats,
                'heatmap': ref_heatmap_data,
                'heatmap_years': all_years,
                'lag_distribution': ref_lag_dist
            },
            'analyzed_to_citing': {
                'connections': analyzed_to_citing_connections[:100],
                'total_connections': len(analyzed_to_citing_connections),
                'stats': analyzed_citing_stats,
                'heatmap': citing_heatmap_data,
                'heatmap_years': citing_years,
                'lag_distribution': citing_lag_dist
            },
            'all_years': all_years,
            'ref_years': ref_years,
            'analyzed_years': analyzed_years,
            'citing_years': citing_years
        }

# ============================================
# ФУНКЦИИ ДЛЯ ГЕНЕРАЦИИ ОТЧЕТОВ
# ============================================

def generate_multilevel_html_report(analyzer: DOIAnalyzer, 
                                    app_logo_base64: Optional[str] = None, 
                                    theme_colors: Optional[Dict] = None, 
                                    lang: str = 'en') -> str:
    """Generate HTML report for multi-level DOI analysis with rich visual elements"""
    
    results = analyzer.analysis_results
    
    if theme_colors is None:
        theme_colors = {
            'primary': '#667eea',
            'secondary': '#f39c12'
        }
    
    primary = theme_colors.get('primary', '#667eea')
    secondary = theme_colors.get('secondary', '#f39c12')

    # ============================================
    # ЗАГРУЗКА ИКОНОК В BASE64 (НОВЫЙ КОД)
    # ============================================
    
    icons = {}
    
    icon_files = [
        ("overview", "01.png"),
        ("references", "02.png"),
        ("analyzed", "03.png"),
        ("citation", "04.png"),
        ("citing", "05.png"),
        ("topics", "06.png"),
        ("detailed", "07.png"),
        ("multilevel", "08.png"),
        ("keywords", "09.png"),
        ("temporal", "10.png"),
    ]
    
    for key, filename in icon_files:
        try:
            icon_path = os.path.join("icons", filename)
            if os.path.exists(icon_path):
                with open(icon_path, "rb") as f:
                    icons[key] = f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"
            else:
                icons[key] = ""
                if SHOW_DEBUG_LOGS:
                    print(f"⚠️ Icon not found: {icon_path}")
        except Exception as e:
            icons[key] = ""
            if SHOW_DEBUG_LOGS:
                print(f"⚠️ Error loading icon {filename}: {e}")
    
    # Helper function to create icon HTML
    def icon_img(icon_key: str, alt: str = "", size: int = 20, cls: str = "") -> str:
        """Generate HTML for icon image"""
        icon_src = icons.get(icon_key, "")
        if icon_src:
            return f'<img src="{icon_src}" alt="{alt}" style="width:{size}px;height:{size}px;vertical-align:middle;" class="{cls}">'
        else:
            # Fallback to emoji if icon not loaded
            fallback_emojis = {
                "overview": "📋",
                "references": "📖",
                "analyzed": "📄",
                "citation": "📈",
                "citing": "📚",
                "topics": "🏷️",
                "detailed": "📋",
                "multilevel": "🔗",
                "keywords": "🔤",
                "temporal": "⏰",
            }
            return fallback_emojis.get(icon_key, "📄")
    
    # OA colors
    oa_colors = {
        'gold': '#FFD700',
        'hybrid': '#F1C40F',
        'green': '#2ECC71',
        'bronze': '#CD7F32',
        'closed': '#95A5A6',
        'unknown': '#BDC3C7'
    }
    
    def t(key: str, **kwargs) -> str:
        return translate(key, lang, **kwargs)
    
    # Get data
    basic = results.get('basic_metrics', {})
    level_I_metrics = basic.get('level_I', {})
    level_II_metrics = basic.get('level_II', {})
    level_III_metrics = basic.get('level_III', {})
    
    author_analysis = results.get('author_analysis', {})
    affiliation_analysis = results.get('affiliation_analysis', {})
    geographic = results.get('geographic_analysis', {})
    citation = results.get('citation_analysis', {})
    citing = results.get('citing_analysis', {})
    topics = results.get('topics_analysis', {})
    detailed_citations = results.get('detailed_citations', {})
    author_distribution = results.get('author_distribution', {})
    multilevel = results.get('multilevel_relationships', {})
    references_list = results.get('references_list', [])
    analyzed_articles_list = results.get('analyzed_articles_list', [])
    
    # ===== NEW: Get Title Keywords and Temporal Relationships =====
    title_keywords = results.get('title_keywords', {})
    temporal = results.get('temporal_relationships', {})
    
    # Max values for color scales
    def get_max_for_metric(data_list, metric_key):
        if not data_list:
            return 1
        return max([item.get(metric_key, 0) for item in data_list]) if data_list else 1
    
    # Get max values for matrices
    author_matrix = multilevel.get('author_matrix', [])
    aff_matrix = multilevel.get('affiliation_matrix', [])
    journal_matrix = multilevel.get('journal_matrix', [])
    pub_matrix = multilevel.get('publisher_matrix', [])
    
    max_author_count = max([a.get('count_I', 0) + a.get('count_II', 0) + a.get('count_III', 0) for a in author_matrix]) if author_matrix else 1
    max_aff_count = max([a.get('count_I', 0) + a.get('count_II', 0) + a.get('count_III', 0) for a in aff_matrix]) if aff_matrix else 1
    max_journal_count = max([a.get('count_I', 0) + a.get('count_II', 0) + a.get('count_III', 0) for a in journal_matrix]) if journal_matrix else 1
    max_pub_count = max([a.get('count_I', 0) + a.get('count_II', 0) + a.get('count_III', 0) for a in pub_matrix]) if pub_matrix else 1
    
    max_author_norm = max([a.get('total_norm', 0) for a in author_matrix]) if author_matrix else 1
    max_aff_norm = max([a.get('total_norm', 0) for a in aff_matrix]) if aff_matrix else 1
    max_journal_norm = max([a.get('total_norm', 0) for a in journal_matrix]) if journal_matrix else 1
    max_pub_norm = max([a.get('total_norm', 0) for a in pub_matrix]) if pub_matrix else 1
    
    # Max values for topics
    topics_data = topics.get('topics', [])
    max_count_I = max([t.get('count_I', 0) for t in topics_data]) if topics_data else 1
    max_count_II = max([t.get('count_II', 0) for t in topics_data]) if topics_data else 1
    max_count_III = max([t.get('count_III', 0) for t in topics_data]) if topics_data else 1
    max_norm_I = max([t.get('norm_I', 0) for t in topics_data]) if topics_data else 1
    max_norm_II = max([t.get('norm_II', 0) for t in topics_data]) if topics_data else 1
    max_norm_III = max([t.get('norm_III', 0) for t in topics_data]) if topics_data else 1
    max_total_norm = max([t.get('total_norm', 0) for t in topics_data]) if topics_data else 1
    
    # Max value for heatmap
    heatmap_max = 0
    for row in citation.get('heatmap', []):
        for year, val in row.items():
            if year != 'publication_year' and isinstance(val, (int, float)):
                heatmap_max = max(heatmap_max, val)
    
    # ===== NEW: Max values for Title Keywords =====
    keywords_data = title_keywords.get('keywords', [])
    max_keyword_count_I = title_keywords.get('max_count_I', 1)
    max_keyword_count_II = title_keywords.get('max_count_II', 1)
    max_keyword_count_III = title_keywords.get('max_count_III', 1)
    max_keyword_norm_I = title_keywords.get('max_norm_I', 1)
    max_keyword_norm_II = title_keywords.get('max_norm_II', 1)
    max_keyword_norm_III = title_keywords.get('max_norm_III', 1)
    max_keyword_total_norm = title_keywords.get('max_total_norm', 1)
    
    # ===== NEW: Max values for Temporal Heatmaps =====
    ref_heatmap = temporal.get('ref_to_analyzed', {}).get('heatmap', [])
    citing_heatmap = temporal.get('analyzed_to_citing', {}).get('heatmap', [])
    heatmap_max_ref = 0
    for row in ref_heatmap:
        for year, val in row.items():
            if year != 'publication_year' and isinstance(val, (int, float)):
                heatmap_max_ref = max(heatmap_max_ref, val)
    heatmap_max_citing = 0
    for row in citing_heatmap:
        for year, val in row.items():
            if year != 'publication_year' and isinstance(val, (int, float)):
                heatmap_max_citing = max(heatmap_max_citing, val)
    
    # Lag distributions
    ref_lag_dist = temporal.get('ref_to_analyzed', {}).get('lag_distribution', [])
    citing_lag_dist = temporal.get('analyzed_to_citing', {}).get('lag_distribution', [])
    
    # Shortcuts for connections
    ref_analyzed_connections = temporal.get('ref_to_analyzed', {}).get('connections', [])
    analyzed_citing_connections = temporal.get('analyzed_to_citing', {}).get('connections', [])
    
    # Statistics for display
    ref_analyzed_stats = temporal.get('ref_to_analyzed', {}).get('stats', {})
    analyzed_citing_stats = temporal.get('analyzed_to_citing', {}).get('stats', {})
    
    # Helper for color scale in matrices
    def get_color_scale_html(value, max_val, min_val=0):
        if max_val == min_val:
            return f'<span class="color-scale-value" style="background: rgba(200,200,200,0.15); color: #1a1a1a;">{value}</span>'
        
        normalized = (value - min_val) / (max_val - min_val)
        normalized = max(0, min(1, normalized))
        
        if normalized < 0.5:
            ratio = normalized / 0.5
            r = 200
            g = int(200 * ratio)
            b = 50
        else:
            ratio = (normalized - 0.5) / 0.5
            r = int(200 * (1 - ratio))
            g = 200
            b = 50
        
        bg_color = f"rgba({r}, {g}, {b}, 0.35)"
        return f'<span class="color-scale-value" style="background: {bg_color}; color: #1a1a1a;">{value}</span>'
    
    # Helper for norm values with 3 decimals
    def get_norm_scale_html(value, max_val, min_val=0, decimals=3):
        if max_val == min_val:
            return f'<span class="color-scale-value" style="background: rgba(200,200,200,0.15); color: #1a1a1a;">{value:.{decimals}f}</span>'
        
        normalized = (value - min_val) / (max_val - min_val)
        normalized = max(0, min(1, normalized))
        
        if normalized < 0.5:
            ratio = normalized / 0.5
            r = 200
            g = int(200 * ratio)
            b = 50
        else:
            ratio = (normalized - 0.5) / 0.5
            r = int(200 * (1 - ratio))
            g = 200
            b = 50
        
        bg_color = f"rgba({r}, {g}, {b}, 0.35)"
        formatted_value = f"{value:.{decimals}f}"
        return f'<span class="color-scale-value" style="background: {bg_color}; color: #1a1a1a;">{formatted_value}</span>'
    
    # Build HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Multi-Level DOI Analysis</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{
                font-family: 'Times New Roman', 'DejaVu Serif', serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                color: #333;
            }}
            .report-wrapper {{
                max-width: 1600px;
                margin: 0 auto;
                background: white;
                box-shadow: 0 10px 40px rgba(0,0,0,0.1);
                border-radius: 10px;
                overflow: hidden;
            }}
            
            /* ===== SIDEBAR NAVIGATION ===== */
            .sidebar {{
                position: fixed;
                left: 0;
                top: 0;
                width: 280px;
                height: 100vh;
                background: linear-gradient(135deg, {primary} 0%, {secondary} 100%);
                color: white;
                padding: 25px 18px;
                overflow-y: auto;
                z-index: 1000;
                box-shadow: 2px 0 20px rgba(0,0,0,0.15);
            }}
            .sidebar::-webkit-scrollbar {{ width: 4px; }}
            .sidebar::-webkit-scrollbar-thumb {{ background: rgba(255,255,255,0.3); border-radius: 4px; }}
            
            .sidebar h3 {{
                margin-bottom: 20px;
                font-size: 18px;
                font-weight: 700;
                color: white;
                border-bottom: 2px solid rgba(255,255,255,0.3);
                padding-bottom: 15px;
                letter-spacing: 0.5px;
                word-wrap: break-word;
            }}
            .sidebar .nav-section {{
                margin-top: 5px;
            }}
            .sidebar .nav-section-title {{
                font-size: 11px;
                text-transform: uppercase;
                letter-spacing: 1px;
                opacity: 0.7;
                padding: 8px 12px 4px 12px;
                font-weight: 600;
                display: none;
            }}
            .sidebar a {{
                color: white;
                text-decoration: none;
                display: flex;
                align-items: center;
                gap: 10px;
                padding: 8px 14px;
                margin: 2px 0;
                border-radius: 8px;
                transition: all 0.3s;
                font-size: 13px;
            }}
            .sidebar a:hover {{
                background: rgba(255,255,255,0.2);
                transform: translateX(5px);
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .sidebar a .nav-icon {{
                font-size: 16px;
                width: 24px;
                text-align: center;
            }}
            
            /* ===== MAIN CONTENT ===== */
            .main-content {{
                margin-left: 280px;
                padding: 30px 40px;
            }}
            
            /* ===== HEADER ===== */
            .header {{
                background: linear-gradient(135deg, {primary} 0%, {secondary} 100%);
                color: white;
                padding: 30px 40px;
                border-radius: 15px;
                margin-bottom: 30px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            }}
            .header-left {{
                display: flex;
                align-items: center;
                gap: 20px;
            }}
            .header-left img {{
                max-height: 130px;
                filter: drop-shadow(0 2px 4px rgba(0,0,0,0.2));
            }}
            .header h1 {{
                color: white;
                border-bottom: none;
                margin: 0;
                font-size: 28px;
                font-weight: 700;
                text-shadow: 0 2px 4px rgba(0,0,0,0.2);
                word-wrap: break-word;
            }}
            .header .subtitle {{
                opacity: 0.9;
                margin-top: 5px;
                font-size: 14px;
                text-shadow: 0 1px 2px rgba(0,0,0,0.15);
            }}
            
            /* ===== SECTIONS ===== */
            .section {{
                background: white;
                border-radius: 15px;
                padding: 25px 30px;
                margin-bottom: 25px;
                box-shadow: 0 2px 12px rgba(0,0,0,0.08);
                border: 1px solid #f0f0f0;
                transition: all 0.3s;
            }}
            .section:hover {{
                box-shadow: 0 4px 20px rgba(0,0,0,0.12);
            }}
            
            .section-header {{
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: space-between;
                user-select: none;
                padding: 5px 0;
            }}
            .section-header:hover .section-title {{
                color: {primary};
            }}
            .section-title {{
                font-size: 22px;
                font-weight: 700;
                margin-bottom: 0;
                padding-bottom: 0;
                border-bottom: none;
                display: flex;
                align-items: center;
                gap: 12px;
                color: #2C3E50;
                transition: color 0.3s;
            }}
            .section-title .icon {{
                font-size: 24px;
            }}
            .section-title .section-badge {{
                background: linear-gradient(135deg, {primary}, {secondary});
                color: white;
                padding: 2px 12px;
                border-radius: 20px;
                font-size: 13px;
                font-weight: 600;
                margin-left: 8px;
            }}
            .section-divider {{
                height: 3px;
                background: linear-gradient(90deg, {primary}, {secondary}, transparent);
                margin: 15px 0 20px 0;
                border-radius: 3px;
            }}
            .toggle-indicator {{
                font-size: 18px;
                transition: transform 0.3s;
                color: {primary};
                font-weight: 300;
            }}
            .toggle-indicator.collapsed {{
                transform: rotate(-90deg);
            }}
            .section-content {{
                display: block;
                transition: all 0.4s ease;
            }}
            .section-content.collapsed {{
                display: none;
            }}
            
            /* ===== METRICS GRID ===== */
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
                gap: 12px;
                margin: 15px 0;
            }}
            .metrics-grid-4 {{
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            }}
            .metric-card {{
                background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
                padding: 14px 18px;
                border-radius: 12px;
                border-left: 4px solid {primary};
                text-align: center;
                transition: all 0.3s;
                box-shadow: 0 2px 6px rgba(0,0,0,0.04);
                position: relative;
                overflow: hidden;
            }}
            .metric-card::after {{
                content: '';
                position: absolute;
                top: 0;
                right: 0;
                width: 60px;
                height: 60px;
                background: linear-gradient(135deg, transparent 50%, {primary}08 100%);
                border-radius: 0 12px 0 60px;
            }}
            .metric-card:hover {{
                transform: translateY(-4px);
                box-shadow: 0 6px 20px rgba(0,0,0,0.1);
                border-left-color: {secondary};
            }}
            .metric-card .metric-icon {{
                font-size: 20px;
                display: block;
                margin-bottom: 4px;
            }}
            .metric-value {{
                font-size: 26px;
                font-weight: 700;
                color: #2C3E50;
                font-family: 'Times New Roman', serif;
                background: linear-gradient(135deg, {primary}, {secondary});
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }}
            .metric-label {{
                font-size: 11px;
                color: #7F8C8D;
                margin-top: 4px;
                font-family: 'Times New Roman', serif;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.3px;
            }}
            
            /* ===== PROGRESS BARS ===== */
            .progress-bar-container {{
                width: 100%;
                background-color: #f0f0f0;
                border-radius: 8px;
                overflow: hidden;
                margin: 4px 0;
                height: 22px;
                position: relative;
                box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
            }}
            .progress-bar-fill {{
                height: 100%;
                border-radius: 8px;
                transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-size: 11px;
                font-weight: 700;
                text-shadow: 0 1px 2px rgba(0,0,0,0.3);
                position: relative;
                overflow: hidden;
                min-width: 30px;
            }}
            .progress-bar-fill.animate {{
                animation: shimmer 2s infinite linear;
                background-size: 200% 100%;
            }}
            @keyframes shimmer {{
                0% {{ background-position: -200% 0; }}
                100% {{ background-position: 200% 0; }}
            }}
            
            .progress-bar-label {{
                display: flex;
                justify-content: space-between;
                font-size: 12px;
                margin: 2px 0 1px 0;
                color: #555;
                font-weight: 500;
            }}
            .progress-bar-label .label-value {{
                font-weight: 700;
                color: #2C3E50;
            }}
            
            /* ===== OA BREAKDOWN ===== */
            .oa-breakdown {{
                display: flex;
                flex-wrap: wrap;
                gap: 12px;
                margin: 12px 0;
            }}
            .oa-item {{
                display: flex;
                align-items: center;
                gap: 10px;
                background: #f8f9fa;
                padding: 8px 16px 8px 12px;
                border-radius: 10px;
                border: 1px solid #e9ecef;
                flex: 1;
                min-width: 120px;
                transition: all 0.3s;
            }}
            .oa-item:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            }}
            .oa-item .color-dot {{
                width: 16px;
                height: 16px;
                border-radius: 50%;
                display: inline-block;
                flex-shrink: 0;
                border: 1px solid rgba(0,0,0,0.05);
            }}
            .oa-item .oa-info {{
                flex: 1;
            }}
            .oa-item .oa-name {{
                font-weight: 600;
                font-size: 13px;
            }}
            .oa-item .oa-count {{
                font-size: 12px;
                color: #666;
            }}
            .oa-item .oa-percent {{
                font-size: 14px;
                font-weight: 700;
                color: #2C3E50;
                margin-left: auto;
            }}
            
            /* ===== TABLES ===== */
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 12px 0;
                font-family: 'Times New Roman', serif;
                font-size: 13px;
            }}
            th {{
                background: linear-gradient(135deg, {primary} 0%, {secondary} 100%);
                color: white;
                padding: 10px 14px;
                text-align: left;
                font-weight: 600;
                position: sticky;
                top: 0;
                z-index: 10;
                white-space: nowrap;
            }}
            th.sortable {{
                cursor: pointer;
                user-select: none;
                position: relative;
            }}
            th.sortable:hover {{
                opacity: 0.9;
            }}
            th.sortable::after {{
                content: ' ↕';
                opacity: 0.5;
                font-size: 10px;
            }}
            th.sortable.asc::after {{
                content: ' ↑';
                opacity: 0.8;
            }}
            th.sortable.desc::after {{
                content: ' ↓';
                opacity: 0.8;
            }}
            td {{
                padding: 8px 14px;
                border-bottom: 1px solid #e9ecef;
                vertical-align: middle;
                transition: background 0.2s;
            }}
            tr:hover td {{
                background-color: #f8f9fa;
            }}
            .scrollable-table {{
                max-height: 500px;
                overflow-y: auto;
                border-radius: 8px;
                border: 1px solid #e9ecef;
            }}
            .scrollable-table thead {{
                position: sticky;
                top: 0;
                z-index: 10;
            }}
            
            .citation-count {{
                background: linear-gradient(135deg, {primary}15, {secondary}15);
                padding: 2px 10px;
                border-radius: 12px;
                font-weight: 700;
                color: {primary};
            }}
            
            .doi-link {{
                color: #2980B9;
                text-decoration: none;
                font-size: 11px;
                word-break: break-all;
                transition: color 0.2s;
            }}
            .doi-link:hover {{
                color: {primary};
                text-decoration: underline;
            }}
            
            .badge {{
                display: inline-block;
                padding: 2px 10px;
                border-radius: 12px;
                font-size: 11px;
                font-weight: 600;
                margin: 1px 2px;
            }}
            .badge-gold {{ background: #FFD700; color: #333; }}
            .badge-hybrid {{ background: #F1C40F; color: #333; }}
            .badge-green {{ background: #2ECC71; color: white; }}
            .badge-bronze {{ background: #CD7F32; color: white; }}
            .badge-closed {{ background: #95A5A6; color: white; }}
            .badge-unknown {{ background: #BDC3C7; color: #333; }}
            .badge-info {{ background: #3498DB; color: white; }}
            .badge-success {{ background: #2ECC71; color: white; }}
            .badge-warning {{ background: #F39C12; color: white; }}
            .badge-danger {{ background: #E74C3C; color: white; }}
            .badge-primary {{ background: {primary}; color: white; }}
            
            /* ===== HEATMAP ===== */
            .heatmap-cell {{
                text-align: center;
                padding: 6px 10px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: 600;
                transition: all 0.3s;
                min-width: 40px;
            }}
            .heatmap-cell:hover {{
                transform: scale(1.05);
                box-shadow: 0 2px 8px rgba(0,0,0,0.15);
                z-index: 5;
            }}
            
            /* ===== COLLAPSER (Detailed Citations) ===== */
            .collapser {{
                background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
                padding: 12px 18px;
                margin: 5px 0;
                border-radius: 10px;
                cursor: pointer;
                border-left: 4px solid {primary};
                transition: all 0.3s;
                display: flex;
                align-items: center;
                flex-wrap: wrap;
                gap: 8px;
                box-shadow: 0 1px 4px rgba(0,0,0,0.04);
            }}
            .collapser:hover {{
                background: #e9ecef;
                transform: translateX(5px);
                box-shadow: 0 2px 12px rgba(0,0,0,0.08);
            }}
            .collapser .citation-count-badge {{
                background: linear-gradient(135deg, {primary}, {secondary});
                color: white;
                padding: 2px 12px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: 700;
            }}
            .collapser .toggle-hint {{
                font-size: 11px;
                color: #999;
                margin-left: auto;
                font-weight: 400;
            }}
            .citation-detail {{
                background: #f8f9fa;
                padding: 12px 18px;
                margin: 4px 0 4px 24px;
                border-radius: 8px;
                border-left: 3px solid {secondary};
                font-size: 13px;
                transition: all 0.3s;
            }}
            .citation-detail:hover {{
                background: #f0f1f2;
                transform: translateX(3px);
            }}
            .citation-detail .cite-meta {{
                color: #555;
                font-size: 12px;
                margin-top: 4px;
                line-height: 1.6;
            }}
            .citation-detail .cite-title {{
                font-weight: 600;
                color: #2C3E50;
            }}
            
            /* ===== FILTER SECTION ===== */
            .filter-section {{
                background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
                padding: 15px 20px;
                border-radius: 10px;
                margin-bottom: 15px;
                border: 1px solid #e9ecef;
            }}
            .filter-row {{
                display: flex;
                flex-wrap: wrap;
                gap: 12px;
                align-items: center;
            }}
            .filter-row .filter-group {{
                display: flex;
                align-items: center;
                gap: 6px;
                background: white;
                padding: 4px 10px 4px 12px;
                border-radius: 8px;
                border: 1px solid #e9ecef;
            }}
            .filter-row label {{
                font-size: 11px;
                font-weight: 600;
                color: #555;
                white-space: nowrap;
                text-transform: uppercase;
                letter-spacing: 0.3px;
            }}
            .filter-row select, .filter-row input {{
                padding: 4px 8px;
                border: none;
                border-radius: 4px;
                font-size: 12px;
                font-family: 'Times New Roman', serif;
                background: transparent;
                outline: none;
            }}
            .filter-row select:focus, .filter-row input:focus {{
                box-shadow: 0 0 0 2px {primary}40;
            }}
            .filter-row input[type="text"] {{
                width: 130px;
            }}
            .filter-row input[type="number"] {{
                width: 70px;
            }}
            .filter-stats {{
                margin-top: 10px;
                font-size: 13px;
                color: #555;
                padding: 6px 12px;
                background: white;
                border-radius: 8px;
                border: 1px solid #e9ecef;
                display: inline-block;
            }}
            .filter-stats strong {{
                color: #2C3E50;
            }}
            
            /* ===== GEO GRID ===== */
            .geo-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin: 15px 0;
            }}
            .geo-card {{
                background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
                padding: 16px 20px;
                border-radius: 10px;
                border: 1px solid #e9ecef;
                transition: all 0.3s;
            }}
            .geo-card:hover {{
                box-shadow: 0 4px 16px rgba(0,0,0,0.06);
            }}
            .geo-card h4 {{
                color: {primary};
                margin-bottom: 8px;
                font-size: 14px;
            }}
            .geo-card .geo-value {{
                font-size: 18px;
                font-weight: 700;
                color: #2C3E50;
            }}
            .geo-card .geo-label {{
                font-size: 12px;
                color: #7F8C8D;
            }}
            
            /* ===== COLOR SCALE FOR NUMERIC VALUES ===== */
            .color-scale-value {{
                display: inline-block;
                padding: 2px 10px;
                border-radius: 8px;
                font-weight: 600;
                text-align: center;
                min-width: 30px;
                transition: all 0.2s;
            }}
            .color-scale-value:hover {{
                transform: scale(1.05);
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            
            /* ===== RESPONSIVE ===== */
            @media print {{
                .sidebar {{ display: none; }}
                .main-content {{ margin-left: 0; }}
                .section {{ box-shadow: none; border: 1px solid #ddd; }}
                .metric-card {{ box-shadow: none; }}
            }}
            @media (max-width: 768px) {{
                .sidebar {{ display: none; }}
                .main-content {{ margin-left: 0; padding: 15px; }}
                .header {{ flex-direction: column; text-align: center; padding: 20px; }}
                .header-left {{ flex-direction: column; }}
                .geo-grid {{ grid-template-columns: 1fr; }}
                .filter-row {{ flex-direction: column; align-items: stretch; }}
                .filter-row .filter-group {{ flex-wrap: wrap; }}
                .metrics-grid {{ grid-template-columns: repeat(2, 1fr); }}
                .metrics-grid-4 {{ grid-template-columns: 1fr 1fr; }}
                .oa-breakdown {{ flex-direction: column; }}
            }}
            
            /* ===== ANIMATIONS ===== */
            @keyframes fadeInUp {{
                from {{ opacity: 0; transform: translateY(20px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            .section {{
                animation: fadeInUp 0.6s ease forwards;
            }}
            .section:nth-child(2) {{ animation-delay: 0.1s; }}
            .section:nth-child(3) {{ animation-delay: 0.2s; }}
            .section:nth-child(4) {{ animation-delay: 0.3s; }}
            .section:nth-child(5) {{ animation-delay: 0.4s; }}
            .section:nth-child(6) {{ animation-delay: 0.5s; }}
            .section:nth-child(7) {{ animation-delay: 0.6s; }}
            
            .word-wrap {{
                word-wrap: break-word;
                max-width: 300px;
            }}
            
            .footer {{
                margin-top: 40px;
                padding-top: 20px;
                border-top: 2px solid #e9ecef;
                text-align: center;
                color: #7F8C8D;
                font-size: 12px;
            }}
            .footer a {{
                color: {primary};
                text-decoration: none;
            }}
            .footer a:hover {{
                text-decoration: underline;
            }}
            
            .orcid-full {{
                font-family: monospace;
                font-size: 12px;
                color: #1a1a1a;
            }}
            
            .dist-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin: 15px 0;
            }}
            .dist-card {{
                background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
                padding: 16px 20px;
                border-radius: 10px;
                border: 1px solid #e9ecef;
                transition: all 0.3s;
            }}
            .dist-card:hover {{
                box-shadow: 0 4px 16px rgba(0,0,0,0.06);
            }}
            .dist-card h4 {{
                color: {primary};
                margin-bottom: 12px;
                font-size: 14px;
                text-align: center;
            }}
            .dist-card .total-label {{
                text-align: center;
                font-size: 12px;
                color: #7F8C8D;
                margin-top: 10px;
                padding-top: 8px;
                border-top: 1px solid #e9ecef;
            }}
            
            .level-badge {{
                display: inline-block;
                padding: 2px 8px;
                border-radius: 10px;
                font-size: 10px;
                font-weight: 700;
                margin: 0 2px;
            }}
            .level-badge-I {{ background: #3498DB; color: white; }}
            .level-badge-II {{ background: #2ECC71; color: white; }}
            .level-badge-III {{ background: #E74C3C; color: white; }}
        </style>
    </head>
    <body>
        <div class="sidebar">
            <h3>DOI Analysis</h3>
            
            <div class="nav-section">
                <a href="#overview"><span class="nav-icon">{icon_img('overview', 'Overview', 20)}</span> {t('overview')}</a>
                <a href="#references"><span class="nav-icon">{icon_img('references', 'References', 20)}</span> {t('references')}</a>
                <a href="#analyzed_articles"><span class="nav-icon">{icon_img('analyzed', 'Analyzed Articles', 20)}</span> {t('analyzed_articles')}</a>
                <a href="#citation_analysis"><span class="nav-icon">{icon_img('citation', 'Citation Analysis', 20)}</span> {t('citation_analysis')}</a>
                <a href="#citing_works"><span class="nav-icon">{icon_img('citing', 'Citing Works', 20)}</span> {t('citing_works_analysis')}</a>
                <a href="#topics_analysis"><span class="nav-icon">{icon_img('topics', 'Topics Analysis', 20)}</span> {t('topics_analysis')}</a>
                <a href="#detailed_citations"><span class="nav-icon">{icon_img('detailed', 'Detailed Citations', 20)}</span> {t('detailed_citations')}</a>
                <a href="#multilevel"><span class="nav-icon">{icon_img('multilevel', 'Multilevel Relationships', 20)}</span> {t('multilevel_relationships')}</a>
                <a href="#title_keywords"><span class="nav-icon">{icon_img('keywords', 'Title Keywords', 20)}</span> {t('title_keywords_analysis')}</a>
                <a href="#temporal"><span class="nav-icon">{icon_img('temporal', 'Temporal Relationships', 20)}</span> {t('temporal_relationships')}</a>
            </div>
            </div>
            
            <div style="margin-top: 20px; padding-top: 15px; border-top: 1px solid rgba(255,255,255,0.2); font-size: 11px; opacity: 0.8; line-height: 1.6;">
                <div>Level II: {len(analyzer.level_II)} DOIs</div>
                <div>Level I: {len(analyzer.level_I)} unique references</div>
                <div>Level III: {len(analyzer.level_III)} unique citing works</div>
                <div style="margin-top: 4px; font-size: 10px; opacity: 0.6;">{t('generated_on')}: {datetime.now().strftime('%d.%m.%Y %H:%M')}</div>
            </div>
        </div>
        
        <div class="main-content">
            <!-- HEADER -->
            <div class="header">
                <div class="header-left">
                    {f'<img src="data:image/png;base64,{app_logo_base64}" alt="App Logo" style="max-height:105px;">' if app_logo_base64 else ''}
                    <div>
                        <div class="subtitle">
                            {len(analyzer.level_II)} Level II DOIs | 
                            {len(analyzer.level_I)} Level I references | 
                            {len(analyzer.level_III)} Level III citing works
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- ============================================================ -->
            <!-- SECTION 1: OVERVIEW -->
            <!-- ============================================================ -->
            <div id="overview" class="section">
                <div class="section-header" onclick="toggleSection('overview_content')">
                    <div class="section-title">
                        <span class="icon">{icon_img('overview', 'Overview', 28)}</span> {t('overview')}
                        <span class="section-badge">3 Levels</span>
                    </div>
                    <span class="toggle-indicator" id="overview_indicator">▼</span>
                </div>
                <div class="section-divider"></div>
                <div id="overview_content" class="section-content">
                    
                    <!-- Level I Metrics -->
                    <h3 style="color: #3498DB; font-size: 16px; margin: 10px 0;">{t('level_i')} - {t('references')}</h3>
                    <div class="metrics-grid">
                        <div class="metric-card"><div class="metric-value">{level_I_metrics.get('total_items', 0)}</div><div class="metric-label">{t('total_items')}</div></div>
                        <div class="metric-card"><div class="metric-value">{level_I_metrics.get('total_weighted', 0)}</div><div class="metric-label">{t('total_weighted')}</div></div>
                        <div class="metric-card"><div class="metric-value">{level_I_metrics.get('total_citations', 0):,}</div><div class="metric-label">{t('total_citations')}</div></div>
                        <div class="metric-card"><div class="metric-value">{level_I_metrics.get('avg_citations', 0):.1f}</div><div class="metric-label">{t('avg_citations')}</div></div>
                        <div class="metric-card"><div class="metric-value">{level_I_metrics.get('unique_authors', 0):,}</div><div class="metric-label">{t('unique_authors')}</div></div>
                        <div class="metric-card"><div class="metric-value">{level_I_metrics.get('oa_percentage', 0):.1f}%</div><div class="metric-label">{t('open_access')}</div></div>
                        <div class="metric-card"><div class="metric-value">{level_I_metrics.get('h_index', 0)}</div><div class="metric-label">{t('h_index')}</div></div>
                        <div class="metric-card"><div class="metric-value">{level_I_metrics.get('active_years', 0)}</div><div class="metric-label">{t('active_years')}</div></div>
                    </div>
                    
                    <!-- Level II Metrics -->
                    <h3 style="color: #2ECC71; font-size: 16px; margin: 15px 0 10px 0;">{t('level_ii')} - {t('analyzed_articles')}</h3>
                    <div class="metrics-grid">
                        <div class="metric-card"><div class="metric-value">{level_II_metrics.get('total_items', 0)}</div><div class="metric-label">{t('total_items')}</div></div>
                        <div class="metric-card"><div class="metric-value">{level_II_metrics.get('total_citations', 0):,}</div><div class="metric-label">{t('total_citations')}</div></div>
                        <div class="metric-card"><div class="metric-value">{level_II_metrics.get('avg_citations', 0):.1f}</div><div class="metric-label">{t('avg_citations')}</div></div>
                        <div class="metric-card"><div class="metric-value">{level_II_metrics.get('unique_authors', 0):,}</div><div class="metric-label">{t('unique_authors')}</div></div>
                        <div class="metric-card"><div class="metric-value">{level_II_metrics.get('oa_percentage', 0):.1f}%</div><div class="metric-label">{t('open_access')}</div></div>
                        <div class="metric-card"><div class="metric-value">{level_II_metrics.get('h_index', 0)}</div><div class="metric-label">{t('h_index')}</div></div>
                        <div class="metric-card"><div class="metric-value">{level_II_metrics.get('active_years', 0)}</div><div class="metric-label">{t('active_years')}</div></div>
                        <div class="metric-card"><div class="metric-value">{level_II_metrics.get('international_collaboration_rate', 0):.1f}%</div><div class="metric-label">{t('international_collaboration_rate')}</div></div>
                    </div>
                    
                    <!-- Level III Metrics -->
                    <h3 style="color: #E74C3C; font-size: 16px; margin: 15px 0 10px 0;">{t('level_iii')} - {t('citing_works_analysis')}</h3>
                    <div class="metrics-grid">
                        <div class="metric-card"><div class="metric-value">{level_III_metrics.get('total_items', 0)}</div><div class="metric-label">{t('total_items')}</div></div>
                        <div class="metric-card"><div class="metric-value">{level_III_metrics.get('total_weighted', 0)}</div><div class="metric-label">{t('total_weighted')}</div></div>
                        <div class="metric-card"><div class="metric-value">{level_III_metrics.get('total_citations', 0):,}</div><div class="metric-label">{t('total_citations')}</div></div>
                        <div class="metric-card"><div class="metric-value">{level_III_metrics.get('avg_citations', 0):.1f}</div><div class="metric-label">{t('avg_citations')}</div></div>
                        <div class="metric-card"><div class="metric-value">{level_III_metrics.get('unique_authors', 0):,}</div><div class="metric-label">{t('unique_authors')}</div></div>
                        <div class="metric-card"><div class="metric-value">{level_III_metrics.get('oa_percentage', 0):.1f}%</div><div class="metric-label">{t('open_access')}</div></div>
                        <div class="metric-card"><div class="metric-value">{level_III_metrics.get('h_index', 0)}</div><div class="metric-label">{t('h_index')}</div></div>
                        <div class="metric-card"><div class="metric-value">{level_III_metrics.get('active_years', 0)}</div><div class="metric-label">{t('active_years')}</div></div>
                    </div>
                    
                    <!-- Cross-level citations warning -->
                    {f'''
                    <div style="margin-top: 15px; padding: 12px 18px; background: #fff3cd; border-left: 4px solid #ffc107; border-radius: 8px;">
                        <strong>⚠️ {t('cross_level_citation', doi='', count='')}</strong>
                        {''.join([f'<div style="font-size: 12px; margin-top: 4px;">DOI: {c["doi"]} appears in Level {c["level"]} with count {c["count"]}</div>' for c in analyzer.cross_level_citations])}
                    </div>
                    ''' if analyzer.cross_level_citations else ''}
                    
                    <!-- Open Access Breakdown for all levels -->
                    <h3 style="color: {primary}; font-size: 16px; margin-top: 20px;">{t('open_access_breakdown')}</h3>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px;">
                        <!-- Level I -->
                        <div>
                            <h4 style="color: #3498DB; font-size: 13px; text-align: center;">Level I</h4>
                            {''.join([
                                f'''
                                <div style="margin: 4px 0;">
                                    <div class="progress-bar-label">
                                        <span><span class="color-dot" style="display:inline-block;width:12px;height:12px;border-radius:50%;background:{oa_colors.get(status, '#BDC3C7')};vertical-align:middle;margin-right:6px;"></span> {t(status)}</span>
                                        <span class="label-value">{count} ({count/level_I_metrics.get('total_items', 1)*100:.1f}%)</span>
                                    </div>
                                    <div class="progress-bar-container">
                                        <div class="progress-bar-fill animate" style="width: {count/level_I_metrics.get('total_items', 1)*100:.1f}%; background: {oa_colors.get(status, '#BDC3C7')};"></div>
                                    </div>
                                </div>
                                '''
                                for status, count in level_I_metrics.get('oa_breakdown', {}).items()
                                if count > 0 and status not in ['unknown']
                            ])}
                        </div>
                        
                        <!-- Level II -->
                        <div>
                            <h4 style="color: #2ECC71; font-size: 13px; text-align: center;">Level II</h4>
                            {''.join([
                                f'''
                                <div style="margin: 4px 0;">
                                    <div class="progress-bar-label">
                                        <span><span class="color-dot" style="display:inline-block;width:12px;height:12px;border-radius:50%;background:{oa_colors.get(status, '#BDC3C7')};vertical-align:middle;margin-right:6px;"></span> {t(status)}</span>
                                        <span class="label-value">{count} ({count/level_II_metrics.get('total_items', 1)*100:.1f}%)</span>
                                    </div>
                                    <div class="progress-bar-container">
                                        <div class="progress-bar-fill animate" style="width: {count/level_II_metrics.get('total_items', 1)*100:.1f}%; background: {oa_colors.get(status, '#BDC3C7')};"></div>
                                    </div>
                                </div>
                                '''
                                for status, count in level_II_metrics.get('oa_breakdown', {}).items()
                                if count > 0 and status not in ['unknown']
                            ])}
                        </div>
                        
                        <!-- Level III -->
                        <div>
                            <h4 style="color: #E74C3C; font-size: 13px; text-align: center;">Level III</h4>
                            {''.join([
                                f'''
                                <div style="margin: 4px 0;">
                                    <div class="progress-bar-label">
                                        <span><span class="color-dot" style="display:inline-block;width:12px;height:12px;border-radius:50%;background:{oa_colors.get(status, '#BDC3C7')};vertical-align:middle;margin-right:6px;"></span> {t(status)}</span>
                                        <span class="label-value">{count} ({count/level_III_metrics.get('total_items', 1)*100:.1f}%)</span>
                                    </div>
                                    <div class="progress-bar-container">
                                        <div class="progress-bar-fill animate" style="width: {count/level_III_metrics.get('total_items', 1)*100:.1f}%; background: {oa_colors.get(status, '#BDC3C7')};"></div>
                                    </div>
                                </div>
                                '''
                                for status, count in level_III_metrics.get('oa_breakdown', {}).items()
                                if count > 0 and status not in ['unknown']
                            ])}
                        </div>
                    </div>
                    
                </div>
            </div>
            
            <!-- ============================================================ -->
            <!-- SECTION 2: REFERENCES (Level I) -->
            <!-- ============================================================ -->
            <div id="references" class="section">
                <div class="section-header" onclick="toggleSection('references_content')">
                    <div class="section-title">
                        <span class="icon">{icon_img('references', 'References', 28)}</span> {t('references')} (Level I)
                        <span class="section-badge">{len(references_list)} {t('references')}</span>
                    </div>
                    <span class="toggle-indicator" id="references_indicator">▼</span>
                </div>
                <div class="section-divider"></div>
                <div id="references_content" class="section-content">
                    
                    <p style="color: #666; font-size: 13px; margin-bottom: 10px;">{t('level_i_description')}</p>
                    
                    <div class="scrollable-table" style="max-height: 500px;">
                        <table id="references_table">
                            <thead>
                                <tr>
                                    <th class="sortable" onclick="sortTable('references_table', 0)">#</th>
                                    <th class="sortable" onclick="sortTable('references_table', 1)">{t('doi')}</th>
                                    <th class="sortable" onclick="sortTable('references_table', 2)">{t('title')}</th>
                                    <th class="sortable" onclick="sortTable('references_table', 3)">{t('year')}</th>
                                    <th class="sortable" onclick="sortTable('references_table', 4)">{t('weighted_count')}</th>
                                    <th>{t('journal')}</th>
                                </tr>
                            </thead>
                            <tbody>
                                {''.join([
                                    f'''
                                    <tr>
                                        <td>{i+1}</td>
                                        <td><a href="https://doi.org/{html.escape(ref['doi'])}" target="_blank" class="doi-link">{html.escape(ref['doi'][:30])}...</a></td>
                                        <td class="word-wrap">{html.escape(ref['title'][:80])}{'...' if len(ref['title']) > 80 else ''}</td>
                                        <td>{ref['year'] or 'N/A'}</td>
                                        <td>{get_color_scale_html(ref['count'], references_list[0]['count'] if references_list else 1)}</td>
                                        <td>{html.escape(ref['journal'])}</td>
                                    </tr>
                                    '''
                                    for i, ref in enumerate(references_list)
                                ])}
                            </tbody>
                        </table>
                    </div>
                    <div style="margin-top: 8px; font-size: 12px; color: #999;">Total: {len(references_list)} references</div>
                    
                </div>
            </div>
            
            <!-- ============================================================ -->
            <!-- SECTION 3: ANALYZED ARTICLES (Level II) -->
            <!-- ============================================================ -->
            <div id="analyzed_articles" class="section">
                <div class="section-header" onclick="toggleSection('analyzed_content')">
                    <div class="section-title">
                        <span class="icon">{icon_img('analyzed', 'Analyzed Articles', 28)}</span> {t('analyzed_articles')} (Level II)
                        <span class="section-badge">{len(analyzer.level_II)} {t('articles')}</span>
                    </div>
                    <span class="toggle-indicator" id="analyzed_indicator">▼</span>
                </div>
                <div class="section-divider"></div>
                <div id="analyzed_content" class="section-content">
                    
                    <p style="color: #666; font-size: 13px; margin-bottom: 10px;">{t('level_ii_description')}</p>
                    
                    <!-- Author Distribution -->
                    <h3 style="color: {primary}; font-size: 16px;">{t('author_distribution')}</h3>
                    
                    <div class="dist-grid">
                        <div class="dist-card">
                            <h4>{t('author_distribution_analyzed')}</h4>
                            {''.join([
                                f'''
                                <div style="margin: 4px 0;">
                                    <div class="progress-bar-label">
                                        <span>{cat}</span>
                                        <span class="label-value">{count}</span>
                                    </div>
                                    <div class="progress-bar-container">
                                        <div class="progress-bar-fill animate" style="width: {count/max(author_distribution.get('level_II', {}).get('total', 1), 1)*100:.1f}%; background: {primary};">
                                            {count/max(author_distribution.get('level_II', {}).get('total', 1), 1)*100:.1f}%
                                        </div>
                                    </div>
                                </div>
                                '''
                                for cat, count in author_distribution.get('level_II', {}).get('distribution', {}).items()
                            ])}
                            <div class="total-label">Total: {author_distribution.get('level_II', {}).get('total', 0)} {t('publications')}</div>
                        </div>
                        
                        <div class="dist-card">
                            <h4>{t('author_distribution_citing')} (Level III)</h4>
                            {''.join([
                                f'''
                                <div style="margin: 4px 0;">
                                    <div class="progress-bar-label">
                                        <span>{cat}</span>
                                        <span class="label-value">{count}</span>
                                    </div>
                                    <div class="progress-bar-container">
                                        <div class="progress-bar-fill animate" style="width: {count/max(author_distribution.get('level_III', {}).get('total', 1), 1)*100:.1f}%; background: {secondary};">
                                            {count/max(author_distribution.get('level_III', {}).get('total', 1), 1)*100:.1f}%
                                        </div>
                                    </div>
                                </div>
                                '''
                                for cat, count in author_distribution.get('level_III', {}).get('distribution', {}).items()
                            ])}
                            <div class="total-label">Total: {author_distribution.get('level_III', {}).get('total', 0)} {t('publications')}</div>
                        </div>
                    </div>
                    
                    <!-- Author Analysis -->
                    <h3 style="color: {primary}; font-size: 16px; margin-top: 20px;">{t('author_analysis')}</h3>
                    <div class="scrollable-table" style="max-height: 400px;">
                        <table id="author_table">
                            <thead>
                                <tr>
                                    <th class="sortable" onclick="sortTable('author_table', 0)">{t('rank')}</th>
                                    <th class="sortable" onclick="sortTable('author_table', 1)">{t('authors')}</th>
                                    <th class="sortable" onclick="sortTable('author_table', 2)">ORCID</th>
                                    <th>{t('affiliations')}</th>
                                    <th>{t('countries')}</th>
                                    <th class="sortable" onclick="sortTable('author_table', 5)">{t('publications_count')}</th>
                                    <th class="sortable" onclick="sortTable('author_table', 6)">{t('citations_count')}</th>
                                </tr>
                            </thead>
                            <tbody>
                                {''.join([
                                    f'''
                                    <tr>
                                        <td>{i+1}</td>
                                        <td><strong>{html.escape(author['name'])}</strong></td>
                                        <td>{f'<a href="https://orcid.org/{author["orcid"]}" target="_blank" class="doi-link orcid-full">{author["orcid"]}</a>' if author.get('orcid') else '-'}</td>
                                        <td>{', '.join([html.escape(a) for a in author.get('affiliations', [])[:3]])}{' +' + str(len(author.get('affiliations', []))-3) if len(author.get('affiliations', [])) > 3 else ''}</td>
                                        <td>{', '.join(author.get('countries', [])[:3])}</td>
                                        <td>{get_color_scale_html(author.get('publications', 0), max([a.get('publications', 0) for a in author_analysis.get('top_authors', [])]) if author_analysis.get('top_authors') else 1)}</td>
                                        <td>{get_color_scale_html(author.get('citations', 0), max([a.get('citations', 0) for a in author_analysis.get('top_authors', [])]) if author_analysis.get('top_authors') else 1)}</td>
                                    </tr>
                                    '''
                                    for i, author in enumerate(author_analysis.get('top_authors', []))
                                ])}
                            </tbody>
                        </table>
                    </div>
                    
                    <!-- Top Affiliations -->
                    <h3 style="color: {primary}; font-size: 16px; margin-top: 20px;">{t('top_affiliations')}</h3>
                    <div class="scrollable-table" style="max-height: 300px;">
                        <table id="aff_table">
                            <thead>
                                <tr>
                                    <th class="sortable" onclick="sortTable('aff_table', 0)">{t('rank')}</th>
                                    <th class="sortable" onclick="sortTable('aff_table', 1)">{t('affiliations')}</th>
                                    <th class="sortable" onclick="sortTable('aff_table', 2)">{t('publications_count')}</th>
                                    <th>ROR ID</th>
                                </tr>
                            </thead>
                            <tbody>
                                {''.join([
                                    f'''
                                    <tr>
                                        <td>{i+1}</td>
                                        <td>{html.escape(aff['name'])}</td>
                                        <td>{get_color_scale_html(aff['count'], max([a.get('count', 0) for a in affiliation_analysis.get('top_affiliations', [])]) if affiliation_analysis.get('top_affiliations') else 1)}</td>
                                        <td>
                                            {f'<a href="https://colab.ws/organizations/{aff["ror_short"]}" target="_blank" class="doi-link" style="font-family: monospace; font-size: 11px;">{aff["ror_short"][:8]}...</a>' 
                                             if aff.get('ror_short') and aff['ror_short'] else '-'}
                                        </td>
                                    </tr>
                                    '''
                                    for i, aff in enumerate(affiliation_analysis.get('top_affiliations', []))
                                ])}
                            </tbody>
                        </table>
                    </div>
                    
                    <!-- ===== NEW: Analyzed Articles List ===== -->
                    <h3 style="color: {primary}; font-size: 16px; margin-top: 20px;">📄 {t('analyzed_articles_list')}</h3>
                    <p style="color: #666; font-size: 13px; margin-bottom: 10px;">
                        List of {len(analyzed_articles_list)} analyzed articles (Level II) with full metadata.
                    </p>
                    
                    <div class="scrollable-table" style="max-height: 600px;">
                        <table id="analyzed_articles_table">
                            <thead>
                                <tr>
                                    <th class="sortable" onclick="sortTable('analyzed_articles_table', 0)">#</th>
                                    <th class="sortable" onclick="sortTable('analyzed_articles_table', 1)">{t('doi')}</th>
                                    <th class="sortable" onclick="sortTable('analyzed_articles_table', 2)">{t('title')}</th>
                                    <th class="sortable" onclick="sortTable('analyzed_articles_table', 3)">{t('authors')}</th>
                                    <th class="sortable" onclick="sortTable('analyzed_articles_table', 4)">{t('year')}</th>
                                    <th class="sortable" onclick="sortTable('analyzed_articles_table', 5)">{t('journal')}</th>
                                    <th class="sortable" onclick="sortTable('analyzed_articles_table', 6)">{t('affiliations')}</th>
                                    <th class="sortable" onclick="sortTable('analyzed_articles_table', 7)">{t('countries')}</th>
                                    <th class="sortable" onclick="sortTable('analyzed_articles_table', 8)">{t('citations')}</th>
                                    <th class="sortable" onclick="sortTable('analyzed_articles_table', 9)">{t('open_access')}</th>
                                    <th>{t('topics')}</th>
                                </tr>
                            </thead>
                            <tbody>
                                {''.join([
                                    f'''
                                    <tr>
                                        <td>{i+1}</td>
                                        <td>
                                            <a href="https://doi.org/{html.escape(article['doi'])}" target="_blank" class="doi-link">
                                                {html.escape(article['doi'][:25])}{'...' if len(article['doi']) > 25 else ''}
                                            </a>
                                        </td>
                                        <td class="word-wrap" style="max-width: 250px;">
                                            <strong>{html.escape(article['title'][:80])}{'...' if len(article['title']) > 80 else ''}</strong>
                                        </td>
                                        <td style="font-size: 12px; max-width: 150px;">
                                            {html.escape(article['authors'])}
                                        </td>
                                        <td>{article['year'] or 'N/A'}</td>
                                        <td style="font-size: 12px; max-width: 120px;">
                                            {html.escape(article['journal'])}
                                        </td>
                                        <td style="font-size: 11px; max-width: 120px;">
                                            {html.escape(article['affiliations'])}
                                        </td>
                                        <td style="font-size: 11px;">
                                            {', '.join(article['countries_full'][:2]) if article['countries_full'] else 'N/A'}
                                            {f' +{len(article["countries_full"])-2}' if len(article.get("countries_full", [])) > 2 else ''}
                                        </td>
                                        <td>
                                            <span class="citation-count">{article['citations']}</span>
                                        </td>
                                        <td>
                                            {f'<span class="badge badge-success">✅ OA</span>' if article['is_oa'] else '<span class="badge badge-closed">🔒 Closed</span>'}
                                            <span style="font-size: 10px; color: #999;">{article['oa_status']}</span>
                                        </td>
                                        <td style="font-size: 11px; max-width: 120px;">
                                            {html.escape(article['topics'])}
                                        </td>
                                    </tr>
                                    '''
                                    for i, article in enumerate(analyzed_articles_list)
                                ])}
                            </tbody>
                        </table>
                    </div>
                    <div style="margin-top: 8px; font-size: 12px; color: #999;">
                        {len(analyzed_articles_list)} {t('analyzed_articles')} | 
                        {sum([1 for a in analyzed_articles_list if a['is_oa']])} Open Access | 
                        {len(set([a['journal'] for a in analyzed_articles_list]))} unique journals
                    </div>
                    
                    <!-- Geographic Analysis (existing) -->
                    <h3 style="color: {primary}; font-size: 16px; margin-top: 20px;">{t('geographic_analysis')}</h3>
                    
                    <div class="geo-grid">
                        <div class="geo-card">
                            <h4>{t('unique_countries_per_publication')}</h4>
                            <div>
                                <div><span class="geo-value">{geographic.get('unique_countries_per_publication', {}).get('avg', 0):.2f}</span> <span class="geo-label">Avg</span></div>
                                <div><span class="geo-value">{geographic.get('unique_countries_per_publication', {}).get('min', 0)}</span> <span class="geo-label">Min</span></div>
                                <div><span class="geo-value">{geographic.get('unique_countries_per_publication', {}).get('max', 0)}</span> <span class="geo-label">Max</span></div>
                                <div style="margin-top: 8px; font-size: 12px; color: #7F8C8D;">Based on {geographic.get('unique_countries_per_publication', {}).get('total_works', 0)} publications</div>
                            </div>
                        </div>
                        <div class="geo-card">
                            <h4>{t('collaboration_patterns')}</h4>
                            <div>
                                <div><span class="geo-value">{geographic.get('collaboration_patterns', {}).get('single_country', 0)}</span> <span class="geo-label">{t('single_country')} ({geographic.get('collaboration_patterns', {}).get('single_country_ratio', 0)*100:.1f}%)</span></div>
                                <div><span class="geo-value">{geographic.get('collaboration_patterns', {}).get('multi_country', 0)}</span> <span class="geo-label">{t('multi_country')} ({(1-geographic.get('collaboration_patterns', {}).get('single_country_ratio', 0))*100:.1f}%)</span></div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Countries -->
                    <h4 style="color: {primary}; margin-top: 15px; font-size: 14px;">{t('countries')}</h4>
                    <div class="scrollable-table" style="max-height: 300px;">
                        <table id="country_table">
                            <thead>
                                <tr>
                                    <th class="sortable" onclick="sortTable('country_table', 0)">{t('rank')}</th>
                                    <th class="sortable" onclick="sortTable('country_table', 1)">{t('countries')}</th>
                                    <th class="sortable" onclick="sortTable('country_table', 2)">{t('unique_works')}</th>
                                    <th class="sortable" onclick="sortTable('country_table', 3)">{t('authors_count')}</th>
                                </tr>
                            </thead>
                            <tbody>
                                {''.join([
                                    f'''
                                    <tr>
                                        <td>{i+1}</td>
                                        <td><strong>{html.escape(country['country'])}</strong></td>
                                        <td>{get_color_scale_html(country['unique_works'], max([c.get('unique_works', 0) for c in geographic.get('country_stats', [])]) if geographic.get('country_stats') else 1)}</td>
                                        <td>{get_color_scale_html(country['authors_count'], max([c.get('authors_count', 0) for c in geographic.get('country_stats', [])]) if geographic.get('country_stats') else 1)}</td>
                                    </tr>
                                    '''
                                    for i, country in enumerate(geographic.get('country_stats', []))
                                ])}
                            </tbody>
                        </table>
                    </div>
                    
                </div>
            </div>
            
            <!-- ============================================================ -->
            <!-- SECTION 4: CITATION ANALYSIS -->
            <!-- ============================================================ -->
            <div id="citation_analysis" class="section">
                <div class="section-header" onclick="toggleSection('citation_content')">
                    <div class="section-title">
                        <span class="icon">{icon_img('citation', 'Citation Analysis', 28)}</span> {t('citation_analysis')}
                        <span class="section-badge">{level_II_metrics.get('total_citations', 0):,} {t('citations')}</span>
                    </div>
                    <span class="toggle-indicator" id="citation_indicator">▼</span>
                </div>
                <div class="section-divider"></div>
                <div id="citation_content" class="section-content">
                    
                    <!-- Citation Dynamics -->
                    <h3 style="color: {primary}; font-size: 16px;">{t('citation_dynamics_by_year')}</h3>
                    <div class="scrollable-table" style="max-height: 400px;">
                        <table id="dynamics_table">
                            <thead>
                                <tr>
                                    <th class="sortable" onclick="sortTable('dynamics_table', 0)">{t('publication_year')}</th>
                                    <th class="sortable" onclick="sortTable('dynamics_table', 1)">{t('citation_year')}</th>
                                    <th class="sortable" onclick="sortTable('dynamics_table', 2)">{t('citations_count')}</th>
                                </tr>
                            </thead>
                            <tbody>
                                {''.join([
                                    f'<tr><td>{row["publication_year"]}</td><td>{row["citation_year"]}</td><td>{get_color_scale_html(row["citations_count"], max([r.get("citations_count", 0) for r in citation.get("dynamics", [])]) if citation.get("dynamics") else 1)}</td></tr>'
                                    for row in citation.get('dynamics', [])
                                ])}
                            </tbody>
                        </table>
                    </div>
                    
                    <!-- First Citation Analysis -->
                    <h3 style="color: {primary}; font-size: 16px; margin-top: 20px;">{t('first_citation_analysis')}</h3>
                    <div class="metrics-grid metrics-grid-4" style="grid-template-columns: repeat(4, 1fr);">
                        <div class="metric-card"><div class="metric-value">{citation.get('first_citation_stats', {}).get('min', 'N/A')}</div><div class="metric-label">{t('min_lag')}</div></div>
                        <div class="metric-card"><div class="metric-value">{citation.get('first_citation_stats', {}).get('max', 'N/A')}</div><div class="metric-label">{t('max_lag')}</div></div>
                        <div class="metric-card"><div class="metric-value">{citation.get('first_citation_stats', {}).get('avg', 0):.1f}</div><div class="metric-label">{t('avg_lag')}</div></div>
                        <div class="metric-card"><div class="metric-value">{citation.get('first_citation_stats', {}).get('median', 0):.1f}</div><div class="metric-label">{t('median_lag')}</div></div>
                    </div>
                    
                    <!-- Cumulative Citations -->
                    <h3 style="color: {primary}; font-size: 16px; margin-top: 20px;">{t('cumulative_citations')}</h3>
                    <div class="scrollable-table" style="max-height: 300px;">
                        <table id="cum_table">
                            <thead>
                                <tr>
                                    <th class="sortable" onclick="sortTable('cum_table', 0)">{t('year')}</th>
                                    <th class="sortable" onclick="sortTable('cum_table', 1)">{t('citations')}</th>
                                </tr>
                            </thead>
                            <tbody>
                                {''.join([
                                    f'<tr><td>{row["year"]}</td><td>{get_color_scale_html(row["citations"], citation.get("cumulative", [{}])[-1].get("citations", 1) if citation.get("cumulative") else 1)}</td></tr>'
                                    for row in citation.get('cumulative', [])
                                ])}
                            </tbody>
                        </table>
                    </div>
                    
                    <!-- Citation Network Heatmap -->
                    <h3 style="color: {primary}; font-size: 16px; margin-top: 20px;">{t('citation_network_heatmap')}</h3>
                    <div class="scrollable-table" style="max-height: 500px;">
                        <table id="heatmap_table">
                            <thead>
                                <tr>
                                    <th>{t('publication_year')} \ {t('citation_year')}</th>
                                    {''.join([f'<th>{year}</th>' for year in citation.get('heatmap_years', [])])}
                                </tr>
                            </thead>
                            <tbody>
                                {''.join([
                                    f'''
                                    <tr>
                                        <td><strong>{row.get("publication_year", "N/A")}</strong></td>
                                        {''.join([
                                            f'<td class="heatmap-cell" style="{f"background: {get_heatmap_cell_color(row.get(year, 0), heatmap_max)};" if row.get(year) is not None and row.get(year) > 0 else "background: transparent;"} color: {"#1a1a1a" if row.get(year) is not None and row.get(year) > 0 and row.get(year)/max(heatmap_max, 1) > 0.6 else "#333" if row.get(year) is not None and row.get(year) > 0 else "transparent"};">{row.get(year) if row.get(year) is not None and row.get(year) > 0 else ""}</td>'
                                            for year in citation.get('heatmap_years', [])
                                        ])}
                                    </tr>
                                    '''
                                    for row in citation.get('heatmap', [])
                                ])}
                            </tbody>
                        </table>
                    </div>
                    
                    <!-- Most Cited Publications -->
                    <h3 style="color: {primary}; font-size: 16px; margin-top: 20px;">{t('most_cited_publications')}</h3>
                    <div class="scrollable-table" style="max-height: 400px;">
                        <table id="mostcited_table">
                            <thead>
                                <tr>
                                    <th class="sortable" onclick="sortTable('mostcited_table', 0)">{t('rank')}</th>
                                    <th class="sortable" onclick="sortTable('mostcited_table', 1)">{t('title')}</th>
                                    <th class="sortable" onclick="sortTable('mostcited_table', 2)">{t('year')}</th>
                                    <th class="sortable" onclick="sortTable('mostcited_table', 3)">{t('citations')}</th>
                                    <th class="sortable" onclick="sortTable('mostcited_table', 4)">{t('citations_per_year_label')}</th>
                                    <th>{t('authors')}</th>
                                    <th>DOI</th>
                                </tr>
                            </thead>
                            <tbody>
                                {''.join([
                                    f'''
                                    <tr>
                                        <td><span class="badge badge-primary">{i+1}</span></td>
                                        <td class="word-wrap">{html.escape(pub['title'][:80])}{'...' if len(pub['title']) > 80 else ''}</td>
                                        <td>{pub.get('year', 'N/A')}</td>
                                        <td>{get_color_scale_html(pub['citations'], max([p.get('citations', 0) for p in citation.get('most_cited', [])]) if citation.get('most_cited') else 1)}</td>
                                        <td>{get_color_scale_html(round(pub.get('citations_per_year', 0), 1), max([p.get('citations_per_year', 0) for p in citation.get('most_cited', [])]) if citation.get('most_cited') else 1)}</td>
                                        <td>{html.escape(pub.get('authors', 'N/A'))}</td>
                                        <td><a href="https://doi.org/{html.escape(pub.get('doi', ''))}" target="_blank" class="doi-link">{html.escape(pub.get('doi', ''))[:20]}...</a></td>
                                    </tr>
                                    '''
                                    for i, pub in enumerate(citation.get('most_cited', [])[:10])
                                ])}
                            </tbody>
                        </table>
                    </div>
                    
                </div>
            </div>
            
            <!-- ============================================================ -->
            <!-- SECTION 5: CITING WORKS ANALYSIS (Level III) -->
            <!-- ============================================================ -->
            <div id="citing_works" class="section">
                <div class="section-header" onclick="toggleSection('citing_content')">
                    <div class="section-title">
                        <span class="icon">{icon_img('citing', 'Citing Works', 28)}</span> {t('citing_works_analysis')} (Level III)
                        <span class="section-badge">{citing.get('total_citing_works', 0):,} {t('citations')}</span>
                    </div>
                    <span class="toggle-indicator" id="citing_indicator">▼</span>
                </div>
                <div class="section-divider"></div>
                <div id="citing_content" class="section-content">
                    
                    <p style="color: #666; font-size: 13px; margin-bottom: 10px;">{t('level_iii_description')}</p>
                    
                    <div class="metrics-grid" style="grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));">
                        <div class="metric-card"><div class="metric-value">{citing.get('total_citing_works', 0):,}</div><div class="metric-label">{t('total_citing_works')}</div></div>
                        <div class="metric-card"><div class="metric-value">{citing.get('total_unique', 0)}</div><div class="metric-label">Unique</div></div>
                        <div class="metric-card"><div class="metric-value">{len(citing.get('citing_works_weighted', []))}</div><div class="metric-label">{t('citing_works_weighted')}</div></div>
                    </div>
                    
                    <!-- Top Citing Authors -->
                    <h3 style="color: {primary}; font-size: 16px;">{t('top_citing_authors')}</h3>
                    <div class="scrollable-table" style="max-height: 500px;">
                        <table id="citing_auth_table">
                            <thead>
                                <tr>
                                    <th class="sortable" onclick="sortTable('citing_auth_table', 0)">{t('rank')}</th>
                                    <th class="sortable" onclick="sortTable('citing_auth_table', 1)">{t('authors')}</th>
                                    <th class="sortable" onclick="sortTable('citing_auth_table', 2)">{t('citing_author_orcid')}</th>
                                    <th class="sortable" onclick="sortTable('citing_auth_table', 3)">{t('citations_count')}</th>
                                </tr>
                            </thead>
                            <tbody>
                                {''.join([
                                    f'''
                                    <tr>
                                        <td>{i+1}</td>
                                        <td>{html.escape(author['name'])}</td>
                                        <td>{f'<a href="https://orcid.org/{author["orcid"]}" target="_blank" class="doi-link orcid-full">{author["orcid"]}</a>' if author.get('orcid') else '-'}</td>
                                        <td>{get_color_scale_html(author['count'], max([a.get('count', 0) for a in citing.get('top_authors', [])]) if citing.get('top_authors') else 1)}</td>
                                    </tr>
                                    '''
                                    for i, author in enumerate(citing.get('top_authors', []))
                                ])}
                            </tbody>
                        </table>
                    </div>
                    
                    <!-- Top Citing Affiliations -->
                    <h3 style="color: {primary}; font-size: 16px; margin-top: 20px;">{t('top_citing_affiliations')}</h3>
                    <div class="scrollable-table" style="max-height: 500px;">
                        <table id="citing_aff_table">
                            <thead>
                                <tr>
                                    <th class="sortable" onclick="sortTable('citing_aff_table', 0)">{t('rank')}</th>
                                    <th class="sortable" onclick="sortTable('citing_aff_table', 1)">{t('affiliations')}</th>
                                    <th class="sortable" onclick="sortTable('citing_aff_table', 2)">{t('citations_count')}</th>
                                    <th>ROR ID</th>
                                </tr>
                            </thead>
                            <tbody>
                                {''.join([
                                    f'''
                                    <tr>
                                        <td>{i+1}</td>
                                        <td>{html.escape(aff['name'])}</td>
                                        <td>{get_color_scale_html(aff['count'], max([a.get('count', 0) for a in citing.get('top_affiliations', [])]) if citing.get('top_affiliations') else 1)}</td>
                                        <td>
                                            {f'<a href="https://colab.ws/organizations/{aff["ror_short"]}" target="_blank" class="doi-link" style="font-family: monospace; font-size: 11px;">{aff["ror_short"][:8]}...</a>' 
                                             if aff.get('ror_short') and aff['ror_short'] else '-'}
                                        </td>
                                    </tr>
                                    '''
                                    for i, aff in enumerate(citing.get('top_affiliations', []))
                                ])}
                            </tbody>
                        </table>
                    </div>
                    
                    <!-- Top Citing Countries -->
                    <h3 style="color: {primary}; font-size: 16px; margin-top: 20px;">{t('top_citing_countries')}</h3>
                    <div class="scrollable-table" style="max-height: 500px;">
                        <table id="citing_country_table">
                            <thead>
                                <tr>
                                    <th class="sortable" onclick="sortTable('citing_country_table', 0)">{t('rank')}</th>
                                    <th class="sortable" onclick="sortTable('citing_country_table', 1)">{t('countries')}</th>
                                    <th class="sortable" onclick="sortTable('citing_country_table', 2)">{t('citations_count')}</th>
                                </tr>
                            </thead>
                            <tbody>
                                {''.join([
                                    f'<tr><td>{i+1}</td><td>{html.escape(country["name"])}</td><td>{get_color_scale_html(country["count"], max([c.get("count", 0) for c in citing.get("top_countries", [])]) if citing.get("top_countries") else 1)}</td></tr>'
                                    for i, country in enumerate(citing.get('top_countries', []))
                                ])}
                            </tbody>
                        </table>
                    </div>
                    
                    <!-- Top Citing Journals -->
                    <h3 style="color: {primary}; font-size: 16px; margin-top: 20px;">{t('top_citing_journals')}</h3>
                    <div class="scrollable-table" style="max-height: 500px;">
                        <table id="citing_journal_table">
                            <thead>
                                <tr>
                                    <th class="sortable" onclick="sortTable('citing_journal_table', 0)">{t('rank')}</th>
                                    <th class="sortable" onclick="sortTable('citing_journal_table', 1)">{t('journal')}</th>
                                    <th class="sortable" onclick="sortTable('citing_journal_table', 2)">{t('citations_count')}</th>
                                </tr>
                            </thead>
                            <tbody>
                                {''.join([
                                    f'<tr><td>{i+1}</td><td>{html.escape(journal["name"])}</td><td>{get_color_scale_html(journal["count"], max([j.get("count", 0) for j in citing.get("top_journals", [])]) if citing.get("top_journals") else 1)}</td></tr>'
                                    for i, journal in enumerate(citing.get('top_journals', []))
                                ])}
                            </tbody>
                        </table>
                    </div>
                    
                    <!-- Top Citing Publishers -->
                    <h3 style="color: {primary}; font-size: 16px; margin-top: 20px;">{t('top_citing_publishers')}</h3>
                    <div class="scrollable-table" style="max-height: 500px;">
                        <table id="citing_pub_table">
                            <thead>
                                <tr>
                                    <th class="sortable" onclick="sortTable('citing_pub_table', 0)">{t('rank')}</th>
                                    <th class="sortable" onclick="sortTable('citing_pub_table', 1)">{t('publishers')}</th>
                                    <th class="sortable" onclick="sortTable('citing_pub_table', 2)">{t('citations_count')}</th>
                                </tr>
                            </thead>
                            <tbody>
                                {''.join([
                                    f'<tr><td>{i+1}</td><td>{html.escape(pub["name"])}</td><td>{get_color_scale_html(pub["count"], max([p.get("count", 0) for p in citing.get("top_publishers", [])]) if citing.get("top_publishers") else 1)}</td></tr>'
                                    for i, pub in enumerate(citing.get('top_publishers', []))
                                ])}
                            </tbody>
                        </table>
                    </div>
                    
                    <!-- ===== NEW: Citing Works with Weighted Counts ===== -->
                    <h3 style="color: {primary}; font-size: 16px; margin-top: 20px;">{t('citing_works_weighted')}</h3>
                    <p style="color: #666; font-size: 13px; margin-bottom: 10px;">{t('citing_weighted_count_desc')}</p>
                    <div class="scrollable-table" style="max-height: 500px;">
                        <table id="citing_weighted_table">
                            <thead>
                                <tr>
                                    <th class="sortable" onclick="sortTable('citing_weighted_table', 0)">{t('rank')}</th>
                                    <th class="sortable" onclick="sortTable('citing_weighted_table', 1)">{t('doi')}</th>
                                    <th class="sortable" onclick="sortTable('citing_weighted_table', 2)">{t('title')}</th>
                                    <th class="sortable" onclick="sortTable('citing_weighted_table', 3)">{t('year')}</th>
                                    <th class="sortable" onclick="sortTable('citing_weighted_table', 4)">{t('citing_weighted_count')}</th>
                                    <th>{t('journal')}</th>
                                    <th>{t('authors')}</th>
                                </tr>
                            </thead>
                            <tbody>
                                {''.join([
                                    f'''
                                    <tr>
                                        <td>{i+1}</td>
                                        <td><a href="https://doi.org/{html.escape(cite['doi'])}" target="_blank" class="doi-link">{html.escape(cite['doi'][:30])}...</a></td>
                                        <td class="word-wrap">{html.escape(cite['title'][:60])}{'...' if len(cite['title']) > 60 else ''}</td>
                                        <td>{cite.get('year') or 'N/A'}</td>
                                        <td>{get_color_scale_html(cite['weighted_count'], citing.get('max_weighted_count', 1))}</td>
                                        <td>{html.escape(cite.get('journal', 'Unknown'))}</td>
                                        <td>{', '.join([html.escape(a) for a in cite.get('authors', [])])}</td>
                                    </tr>
                                    '''
                                    for i, cite in enumerate(citing.get('citing_works_weighted', []))
                                ])}
                            </tbody>
                        </table>
                    </div>
                    {f'<div style="margin-top: 8px; font-size: 12px; color: #999;">Showing {len(citing.get("citing_works_weighted", []))} citing works with weighted counts (max: {citing.get("max_weighted_count", 0)})</div>' if citing.get('citing_works_weighted') else ''}
                    
                </div>
            </div>
            
            <!-- ============================================================ -->
            <!-- SECTION 6: TOPICS ANALYSIS (All 3 Levels) -->
            <!-- ============================================================ -->
            <div id="topics_analysis" class="section">
                <div class="section-header" onclick="toggleSection('topics_content')">
                    <div class="section-title">
                        <span class="icon">{icon_img('topics', 'Topics Analysis', 28)}</span> {t('topics_analysis')} (All Levels)
                        <span class="section-badge">{len(topics.get('topics', []))} {t('topics')}</span>
                    </div>
                    <span class="toggle-indicator" id="topics_indicator">▼</span>
                </div>
                <div class="section-divider"></div>
                <div id="topics_content" class="section-content">
                    
                    <h3 style="color: {primary}; font-size: 16px;">Topics</h3>
                    <div class="scrollable-table" style="max-height: 400px;">
                        <table id="topics_table">
                            <thead>
                                <tr>
                                    <th class="sortable" onclick="sortTable('topics_table', 0)">Topic</th>
                                    <th class="sortable" onclick="sortTable('topics_table', 1)">{t('count_level_i')}</th>
                                    <th class="sortable" onclick="sortTable('topics_table', 2)">{t('count_level_ii')}</th>
                                    <th class="sortable" onclick="sortTable('topics_table', 3)">{t('count_level_iii')}</th>
                                    <th class="sortable" onclick="sortTable('topics_table', 4)">{t('norm_level_i')}</th>
                                    <th class="sortable" onclick="sortTable('topics_table', 5)">{t('norm_level_ii')}</th>
                                    <th class="sortable" onclick="sortTable('topics_table', 6)">{t('norm_level_iii')}</th>
                                    <th class="sortable" onclick="sortTable('topics_table', 7)">{t('total_norm')}</th>
                                    <th>{t('first_year')}</th>
                                    <th>{t('peak_year')}</th>
                                </tr>
                            </thead>
                            <tbody>
                                {''.join([
                                    f'''
                                    <tr>
                                        <td class="word-wrap">{html.escape(topic['topic'][:50])}{'...' if len(topic['topic']) > 50 else ''}</td>
                                        <td>{get_color_scale_html(topic['count_I'], max_count_I)}</td>
                                        <td>{get_color_scale_html(topic['count_II'], max_count_II)}</td>
                                        <td>{get_color_scale_html(topic['count_III'], max_count_III)}</td>
                                        <td>{get_norm_scale_html(topic['norm_I'], max_norm_I, decimals=3)}</td>
                                        <td>{get_norm_scale_html(topic['norm_II'], max_norm_II, decimals=3)}</td>
                                        <td>{get_norm_scale_html(topic['norm_III'], max_norm_III, decimals=3)}</td>
                                        <td>{get_norm_scale_html(topic['total_norm'], max_total_norm, decimals=3)}</td>
                                        <td>{topic['first_year'] or 'N/A'}</td>
                                        <td>{topic['peak_year'] or 'N/A'}</td>
                                    </tr>
                                    '''
                                    for topic in topics.get('topics', [])[:30]
                                ])}
                            </tbody>
                        </table>
                    </div>
                    
                    <!-- Top Cited Topics -->
                    <h3 style="color: {primary}; font-size: 16px; margin-top: 20px;">{t('top_cited_topics')}</h3>
                    {''.join([
                        f'''
                        <div style="margin: 4px 0;">
                            <div class="progress-bar-label">
                                <span>{i+1}. {html.escape(topic[0][:50])}{'...' if len(topic[0]) > 50 else ''}</span>
                                <span class="label-value">{topic[1]} {t('publications')}</span>
                            </div>
                            <div class="progress-bar-container">
                                <div class="progress-bar-fill animate" style="width: {topic[1]/topics.get('top_cited_topics', [{}])[0][1]*100 if topics.get('top_cited_topics') and topics.get('top_cited_topics')[0][1] > 0 else 0:.1f}%; background: linear-gradient(90deg, {primary}, {secondary});">
                                    {topic[1]}
                                </div>
                            </div>
                        </div>
                        '''
                        for i, topic in enumerate(topics.get('top_cited_topics', [])[:10])
                    ])}
                    
                    <!-- Top Cited Subtopics -->
                    <h3 style="color: {primary}; font-size: 16px; margin-top: 20px;">{t('top_cited_subtopics')}</h3>
                    {''.join([
                        f'''
                        <div style="margin: 4px 0;">
                            <div class="progress-bar-label">
                                <span>{i+1}. {html.escape(subtopic[0][:50])}{'...' if len(subtopic[0]) > 50 else ''}</span>
                                <span class="label-value">{subtopic[1]} {t('publications')}</span>
                            </div>
                            <div class="progress-bar-container">
                                <div class="progress-bar-fill animate" style="width: {subtopic[1]/topics.get('top_cited_subtopics', [{}])[0][1]*100 if topics.get('top_cited_subtopics') and topics.get('top_cited_subtopics')[0][1] > 0 else 0:.1f}%; background: linear-gradient(90deg, {primary}, {secondary});">
                                    {subtopic[1]}
                                </div>
                            </div>
                        </div>
                        '''
                        for i, subtopic in enumerate(topics.get('top_cited_subtopics', [])[:10])
                    ])}
                    
                    <!-- Top Cited Fields -->
                    <h3 style="color: {primary}; font-size: 16px; margin-top: 20px;">{t('top_cited_fields')}</h3>
                    {''.join([
                        f'''
                        <div style="margin: 4px 0;">
                            <div class="progress-bar-label">
                                <span>{i+1}. {html.escape(field[0][:50])}{'...' if len(field[0]) > 50 else ''}</span>
                                <span class="label-value">{field[1]} {t('publications')}</span>
                            </div>
                            <div class="progress-bar-container">
                                <div class="progress-bar-fill animate" style="width: {field[1]/topics.get('top_cited_fields', [{}])[0][1]*100 if topics.get('top_cited_fields') and topics.get('top_cited_fields')[0][1] > 0 else 0:.1f}%; background: linear-gradient(90deg, {primary}, {secondary});">
                                    {field[1]}
                                </div>
                            </div>
                        </div>
                        '''
                        for i, field in enumerate(topics.get('top_cited_fields', [])[:10])
                    ])}
                    
                    <!-- Top Cited Domains -->
                    <h3 style="color: {primary}; font-size: 16px; margin-top: 20px;">{t('top_cited_domains')}</h3>
                    {''.join([
                        f'''
                        <div style="margin: 4px 0;">
                            <div class="progress-bar-label">
                                <span>{i+1}. {html.escape(domain[0][:50])}{'...' if len(domain[0]) > 50 else ''}</span>
                                <span class="label-value">{domain[1]} {t('publications')}</span>
                            </div>
                            <div class="progress-bar-container">
                                <div class="progress-bar-fill animate" style="width: {domain[1]/topics.get('top_cited_domains', [{}])[0][1]*100 if topics.get('top_cited_domains') and topics.get('top_cited_domains')[0][1] > 0 else 0:.1f}%; background: linear-gradient(90deg, {primary}, {secondary});">
                                    {domain[1]}
                                </div>
                            </div>
                        </div>
                        '''
                        for i, domain in enumerate(topics.get('top_cited_domains', [])[:10])
                    ])}
                    
                    <!-- Top Cited Concepts -->
                    <h3 style="color: {primary}; font-size: 16px; margin-top: 20px;">{t('top_cited_concepts')}</h3>
                    {''.join([
                        f'''
                        <div style="margin: 4px 0;">
                            <div class="progress-bar-label">
                                <span>{i+1}. {html.escape(concept[0][:50])}{'...' if len(concept[0]) > 50 else ''}</span>
                                <span class="label-value">{concept[1]} {t('publications')}</span>
                            </div>
                            <div class="progress-bar-container">
                                <div class="progress-bar-fill animate" style="width: {concept[1]/topics.get('top_cited_concepts', [{}])[0][1]*100 if topics.get('top_cited_concepts') and topics.get('top_cited_concepts')[0][1] > 0 else 0:.1f}%; background: linear-gradient(90deg, {primary}, {secondary});">
                                    {concept[1]}
                                </div>
                            </div>
                        </div>
                        '''
                        for i, concept in enumerate(topics.get('top_cited_concepts', [])[:10])
                    ])}
                    
                </div>
            </div>
            
            <!-- ============================================================ -->
            <!-- SECTION 7: DETAILED CITATIONS -->
            <!-- ============================================================ -->
            <div id="detailed_citations" class="section">
                <div class="section-header" onclick="toggleSection('detailed_content')">
                    <div class="section-title">
                        <span class="icon">{icon_img('detailed', 'Detailed Citations', 28)}</span> {t('detailed_citations')}
                        <span class="section-badge">{len(detailed_citations)} {t('publications')}</span>
                    </div>
                    <span class="toggle-indicator" id="detailed_indicator">▼</span>
                </div>
                <div class="section-divider"></div>
                <div id="detailed_content" class="section-content">
                    
                    {''.join([
                        f'''
                        <div class="collapser" onclick="toggleCitations('{html.escape(doi)}')">
                            <strong class="cite-title">{html.escape((data['title'] or 'No title')[:100])}{'...' if len(data['title'] or '') > 100 else ''}</strong>
                            <span class="badge badge-info">{data['year'] or 'N/A'}</span>
                            <span class="citation-count-badge">{data['total_citations']} {t('citations')}</span>
                            <span style="font-size: 12px; color: #999;">DOI: {data['doi'][:20] if data['doi'] else 'N/A'}...</span>
                            <span class="toggle-hint">{t('click_to_toggle')}</span>
                        </div>
                        <div id="citations_{html.escape(doi)}" style="display: none; margin-bottom: 8px;">
                            {''.join([
                                f'''
                                <div class="citation-detail">
                                    <div class="cite-title">{html.escape((cite['citing_title'] or 'No title')[:120])}{'...' if len(cite['citing_title'] or '') > 120 else ''}</div>
                                    <div class="cite-meta">
                                        <strong>{t('citing_journal')}:</strong> {html.escape(cite['citing_journal'] or 'Unknown')} | 
                                        <strong>{t('citing_year')}:</strong> {cite['citing_year'] or 'N/A'} | 
                                        <strong>{t('citing_date')}:</strong> {cite['citing_date'][:10] if cite['citing_date'] else 'N/A'} |
                                        <strong>{t('citation_lag')}:</strong> {cite['citation_lag'] or 'N/A'} {t('days') if cite['citation_lag'] else ''}
                                    </div>
                                    <div class="cite-meta">
                                        <strong>{t('authors')}:</strong> {', '.join([html.escape(a) for a in cite['citing_authors'][:5]]) if cite.get('citing_authors') else 'N/A'}{' +' + str(len(cite['citing_authors'])-5) if cite.get('citing_authors') and len(cite['citing_authors']) > 5 else ''} |
                                        <strong>{t('countries')}:</strong> {', '.join(cite['citing_countries'][:3]) if cite.get('citing_countries') else 'N/A'}
                                    </div>
                                    <div class="cite-meta">
                                        <a href="https://doi.org/{html.escape(cite['citing_doi'] or '')}" target="_blank" class="doi-link">DOI: {html.escape(cite['citing_doi'] or 'N/A')}</a>
                                    </div>
                                </div>
                                ''' for cite in sorted(data['citations'], key=lambda x: x.get('citation_lag') or 0, reverse=True)
                            ])}
                            {f'<div style="padding: 10px 18px; color: #999; font-style: italic;">{t("no_citations_found")}</div>' if not data['citations'] else ''}
                        </div>
                        ''' for doi, data in list(detailed_citations.items())
                    ])}
                    
                </div>
            </div>
            
            <!-- ============================================================ -->
            <!-- SECTION 8: MULTILEVEL RELATIONSHIPS -->
            <!-- ============================================================ -->
            <div id="multilevel" class="section">
                <div class="section-header" onclick="toggleSection('multilevel_content')">
                    <div class="section-title">
                        <span class="icon">{icon_img('multilevel', 'Multilevel Relationships', 28)}</span> {t('multilevel_relationships')}
                        <span class="section-badge">4 Matrices</span>
                    </div>
                    <span class="toggle-indicator" id="multilevel_indicator">▼</span>
                </div>
                <div class="section-divider"></div>
                <div id="multilevel_content" class="section-content">
                    
                    <p style="color: #666; font-size: 13px; margin-bottom: 15px;">
                        Matrices showing frequency of items across all three levels with normalized values.
                        <span class="level-badge level-badge-I">Level I</span>
                        <span class="level-badge level-badge-II">Level II</span>
                        <span class="level-badge level-badge-III">Level III</span>
                    </p>
                    
                    <!-- Author Matrix -->
                    <h3 style="color: {primary}; font-size: 16px;">{t('author_matrix')}</h3>
                    <div class="scrollable-table" style="max-height: 400px;">
                        <table id="author_matrix_table">
                            <thead>
                                <tr>
                                    <th class="sortable" onclick="sortTable('author_matrix_table', 0)">{t('rank')}</th>
                                    <th class="sortable" onclick="sortTable('author_matrix_table', 1)">{t('authors')}</th>
                                    <th>ORCID</th>
                                    <th class="sortable" onclick="sortTable('author_matrix_table', 3)">{t('count_level_i')}</th>
                                    <th class="sortable" onclick="sortTable('author_matrix_table', 4)">{t('count_level_ii')}</th>
                                    <th class="sortable" onclick="sortTable('author_matrix_table', 5)">{t('count_level_iii')}</th>
                                    <th class="sortable" onclick="sortTable('author_matrix_table', 6)">{t('norm_level_i')}</th>
                                    <th class="sortable" onclick="sortTable('author_matrix_table', 7)">{t('norm_level_ii')}</th>
                                    <th class="sortable" onclick="sortTable('author_matrix_table', 8)">{t('norm_level_iii')}</th>
                                    <th class="sortable" onclick="sortTable('author_matrix_table', 9)">{t('total_norm')}</th>
                                </tr>
                            </thead>
                            <tbody>
                                {''.join([
                                    f'''
                                    <tr>
                                        <td>{i+1}</td>
                                        <td><strong>{html.escape(author['name'])}</strong></td>
                                        <td>{f'<a href="https://orcid.org/{author["orcid"]}" target="_blank" class="doi-link orcid-full">{author["orcid"]}</a>' if author.get('orcid') else '-'}</td>
                                        <td>{get_color_scale_html(author['count_I'], max_author_count)}</td>
                                        <td>{get_color_scale_html(author['count_II'], max_author_count)}</td>
                                        <td>{get_color_scale_html(author['count_III'], max_author_count)}</td>
                                        <td>{get_norm_scale_html(author['norm_I'], max_author_norm, decimals=3)}</td>
                                        <td>{get_norm_scale_html(author['norm_II'], max_author_norm, decimals=3)}</td>
                                        <td>{get_norm_scale_html(author['norm_III'], max_author_norm, decimals=3)}</td>
                                        <td>{get_norm_scale_html(author['total_norm'], max_author_norm, decimals=3)}</td>
                                    </tr>
                                    '''
                                    for i, author in enumerate(author_matrix)
                                ])}
                            </tbody>
                        </table>
                    </div>
                    
                    <!-- Affiliation Matrix -->
                    <h3 style="color: {primary}; font-size: 16px; margin-top: 20px;">{t('affiliation_matrix')}</h3>
                    <div class="scrollable-table" style="max-height: 400px;">
                        <table id="aff_matrix_table">
                            <thead>
                                <tr>
                                    <th class="sortable" onclick="sortTable('aff_matrix_table', 0)">{t('rank')}</th>
                                    <th class="sortable" onclick="sortTable('aff_matrix_table', 1)">{t('affiliations')}</th>
                                    <th class="sortable" onclick="sortTable('aff_matrix_table', 2)">{t('count_level_i')}</th>
                                    <th class="sortable" onclick="sortTable('aff_matrix_table', 3)">{t('count_level_ii')}</th>
                                    <th class="sortable" onclick="sortTable('aff_matrix_table', 4)">{t('count_level_iii')}</th>
                                    <th class="sortable" onclick="sortTable('aff_matrix_table', 5)">{t('norm_level_i')}</th>
                                    <th class="sortable" onclick="sortTable('aff_matrix_table', 6)">{t('norm_level_ii')}</th>
                                    <th class="sortable" onclick="sortTable('aff_matrix_table', 7)">{t('norm_level_iii')}</th>
                                    <th class="sortable" onclick="sortTable('aff_matrix_table', 8)">{t('total_norm')}</th>
                                </tr>
                            </thead>
                            <tbody>
                                {''.join([
                                    f'''
                                    <tr>
                                        <td>{i+1}</td>
                                        <td>{html.escape(aff['name'])}</td>
                                        <td>{get_color_scale_html(aff['count_I'], max_aff_count)}</td>
                                        <td>{get_color_scale_html(aff['count_II'], max_aff_count)}</td>
                                        <td>{get_color_scale_html(aff['count_III'], max_aff_count)}</td>
                                        <td>{get_norm_scale_html(aff['norm_I'], max_aff_norm, decimals=3)}</td>
                                        <td>{get_norm_scale_html(aff['norm_II'], max_aff_norm, decimals=3)}</td>
                                        <td>{get_norm_scale_html(aff['norm_III'], max_aff_norm, decimals=3)}</td>
                                        <td>{get_norm_scale_html(aff['total_norm'], max_aff_norm, decimals=3)}</td>
                                    </tr>
                                    '''
                                    for i, aff in enumerate(aff_matrix)
                                ])}
                            </tbody>
                        </table>
                    </div>
                    
                    <!-- Journal Matrix -->
                    <h3 style="color: {primary}; font-size: 16px; margin-top: 20px;">{t('journal_matrix')}</h3>
                    <div class="scrollable-table" style="max-height: 400px;">
                        <table id="journal_matrix_table">
                            <thead>
                                <tr>
                                    <th class="sortable" onclick="sortTable('journal_matrix_table', 0)">{t('rank')}</th>
                                    <th class="sortable" onclick="sortTable('journal_matrix_table', 1)">{t('journal')}</th>
                                    <th class="sortable" onclick="sortTable('journal_matrix_table', 2)">{t('count_level_i')}</th>
                                    <th class="sortable" onclick="sortTable('journal_matrix_table', 3)">{t('count_level_ii')}</th>
                                    <th class="sortable" onclick="sortTable('journal_matrix_table', 4)">{t('count_level_iii')}</th>
                                    <th class="sortable" onclick="sortTable('journal_matrix_table', 5)">{t('norm_level_i')}</th>
                                    <th class="sortable" onclick="sortTable('journal_matrix_table', 6)">{t('norm_level_ii')}</th>
                                    <th class="sortable" onclick="sortTable('journal_matrix_table', 7)">{t('norm_level_iii')}</th>
                                    <th class="sortable" onclick="sortTable('journal_matrix_table', 8)">{t('total_norm')}</th>
                                </tr>
                            </thead>
                            <tbody>
                                {''.join([
                                    f'''
                                    <tr>
                                        <td>{i+1}</td>
                                        <td>{html.escape(journal['name'])}</td>
                                        <td>{get_color_scale_html(journal['count_I'], max_journal_count)}</td>
                                        <td>{get_color_scale_html(journal['count_II'], max_journal_count)}</td>
                                        <td>{get_color_scale_html(journal['count_III'], max_journal_count)}</td>
                                        <td>{get_norm_scale_html(journal['norm_I'], max_journal_norm, decimals=3)}</td>
                                        <td>{get_norm_scale_html(journal['norm_II'], max_journal_norm, decimals=3)}</td>
                                        <td>{get_norm_scale_html(journal['norm_III'], max_journal_norm, decimals=3)}</td>
                                        <td>{get_norm_scale_html(journal['total_norm'], max_journal_norm, decimals=3)}</td>
                                    </tr>
                                    '''
                                    for i, journal in enumerate(journal_matrix)
                                ])}
                            </tbody>
                        </table>
                    </div>
                    
                    <!-- Publisher Matrix -->
                    <h3 style="color: {primary}; font-size: 16px; margin-top: 20px;">{t('publisher_matrix')}</h3>
                    <div class="scrollable-table" style="max-height: 400px;">
                        <table id="pub_matrix_table">
                            <thead>
                                <tr>
                                    <th class="sortable" onclick="sortTable('pub_matrix_table', 0)">{t('rank')}</th>
                                    <th class="sortable" onclick="sortTable('pub_matrix_table', 1)">{t('publishers')}</th>
                                    <th class="sortable" onclick="sortTable('pub_matrix_table', 2)">{t('count_level_i')}</th>
                                    <th class="sortable" onclick="sortTable('pub_matrix_table', 3)">{t('count_level_ii')}</th>
                                    <th class="sortable" onclick="sortTable('pub_matrix_table', 4)">{t('count_level_iii')}</th>
                                    <th class="sortable" onclick="sortTable('pub_matrix_table', 5)">{t('norm_level_i')}</th>
                                    <th class="sortable" onclick="sortTable('pub_matrix_table', 6)">{t('norm_level_ii')}</th>
                                    <th class="sortable" onclick="sortTable('pub_matrix_table', 7)">{t('norm_level_iii')}</th>
                                    <th class="sortable" onclick="sortTable('pub_matrix_table', 8)">{t('total_norm')}</th>
                                </tr>
                            </thead>
                            <tbody>
                                {''.join([
                                    f'''
                                    <tr>
                                        <td>{i+1}</td>
                                        <td>{html.escape(pub['name'])}</td>
                                        <td>{get_color_scale_html(pub['count_I'], max_pub_count)}</td>
                                        <td>{get_color_scale_html(pub['count_II'], max_pub_count)}</td>
                                        <td>{get_color_scale_html(pub['count_III'], max_pub_count)}</td>
                                        <td>{get_norm_scale_html(pub['norm_I'], max_pub_norm, decimals=3)}</td>
                                        <td>{get_norm_scale_html(pub['norm_II'], max_pub_norm, decimals=3)}</td>
                                        <td>{get_norm_scale_html(pub['norm_III'], max_pub_norm, decimals=3)}</td>
                                        <td>{get_norm_scale_html(pub['total_norm'], max_pub_norm, decimals=3)}</td>
                                    </tr>
                                    '''
                                    for i, pub in enumerate(pub_matrix)
                                ])}
                            </tbody>
                        </table>
                    </div>
                    
                </div>
            </div>
            
            <!-- ============================================================ -->
            <!-- SECTION 9: TITLE KEYWORDS ANALYSIS (NEW!) -->
            <!-- ============================================================ -->
            <div id="title_keywords" class="section">
                <div class="section-header" onclick="toggleSection('title_keywords_content')">
                    <div class="section-title">
                        <span class="icon">{icon_img('keywords', 'Title Keywords', 28)}</span> {t('title_keywords_analysis')}
                        <span class="section-badge">{len(keywords_data)} {t('title_term')}</span>
                    </div>
                    <span class="toggle-indicator" id="title_keywords_indicator">▼</span>
                </div>
                <div class="section-divider"></div>
                <div id="title_keywords_content" class="section-content">
                    
                    <p style="color: #666; font-size: 13px; margin-bottom: 10px;">{t('title_keywords_desc')}</p>
                    
                    <div style="display: flex; gap: 20px; margin-bottom: 15px; flex-wrap: wrap;">
                        <div><strong>{t('total_titles')} Level I:</strong> {title_keywords.get('total_titles_I', 0)}</div>
                        <div><strong>{t('total_titles')} Level II:</strong> {title_keywords.get('total_titles_II', 0)}</div>
                        <div><strong>{t('total_titles')} Level III:</strong> {title_keywords.get('total_titles_III', 0)}</div>
                    </div>
                    
                    <div class="scrollable-table" style="max-height: 500px;">
                        <table id="keywords_table">
                            <thead>
                                <tr>
                                    <th class="sortable" onclick="sortTable('keywords_table', 0)">{t('rank')}</th>
                                    <th class="sortable" onclick="sortTable('keywords_table', 1)">{t('title_term')}</th>
                                    <th class="sortable" onclick="sortTable('keywords_table', 2)">{t('variants')}</th>
                                    <th class="sortable" onclick="sortTable('keywords_table', 3)">{t('term_type')}</th>
                                    <th class="sortable" onclick="sortTable('keywords_table', 4)">{t('level_i_count')}</th>
                                    <th class="sortable" onclick="sortTable('keywords_table', 5)">{t('level_ii_count')}</th>
                                    <th class="sortable" onclick="sortTable('keywords_table', 6)">{t('level_iii_count')}</th>
                                    <th class="sortable" onclick="sortTable('keywords_table', 7)">{t('norm_i')}</th>
                                    <th class="sortable" onclick="sortTable('keywords_table', 8)">{t('norm_ii')}</th>
                                    <th class="sortable" onclick="sortTable('keywords_table', 9)">{t('norm_iii')}</th>
                                    <th class="sortable" onclick="sortTable('keywords_table', 10)">{t('total_norm_keywords')}</th>
                                </tr>
                            </thead>
                            <tbody>
                                {''.join([
                                    f'''
                                    <tr>
                                        <td>{i+1}</td>
                                        <td><strong>{html.escape(kw['lemma'])}</strong></td>
                                        <td style="font-size: 11px; color: #666;">{html.escape(kw['variants'])}</td>
                                        <td><span class="badge badge-info">{html.escape(kw['type'])}</span></td>
                                        <td>{get_color_scale_html(kw['count_I'], max_keyword_count_I)}</td>
                                        <td>{get_color_scale_html(kw['count_II'], max_keyword_count_II)}</td>
                                        <td>{get_color_scale_html(kw['count_III'], max_keyword_count_III)}</td>
                                        <td>{get_norm_scale_html(kw['norm_I'], max_keyword_norm_I, decimals=4)}</td>
                                        <td>{get_norm_scale_html(kw['norm_II'], max_keyword_norm_II, decimals=4)}</td>
                                        <td>{get_norm_scale_html(kw['norm_III'], max_keyword_norm_III, decimals=4)}</td>
                                        <td>{get_norm_scale_html(kw['total_norm'], max_keyword_total_norm, decimals=4)}</td>
                                    </tr>
                                    '''
                                    for i, kw in enumerate(keywords_data[:50])
                                ])}
                            </tbody>
                        </table>
                    </div>
                    {f'<div style="margin-top: 8px; font-size: 12px; color: #999;">Showing {min(50, len(keywords_data))} of {len(keywords_data)} keywords</div>' if keywords_data else ''}
                    
                </div>
            </div>
            
            <!-- ============================================================ -->
            <!-- SECTION 10: TEMPORAL RELATIONSHIPS -->
            <!-- ============================================================ -->
            <div id="temporal" class="section">
                <div class="section-header" onclick="toggleSection('temporal_content')">
                    <div class="section-title">
                        <span class="icon">{icon_img('temporal', 'Temporal Relationships', 28)}</span> {t('temporal_relationships')}
                        <span class="section-badge">{temporal.get('ref_to_analyzed', {}).get('total_connections', 0) + temporal.get('analyzed_to_citing', {}).get('total_connections', 0)} {t('connections')}</span>
                    </div>
                    <span class="toggle-indicator" id="temporal_indicator">▼</span>
                </div>
                <div class="section-divider"></div>
                <div id="temporal_content" class="section-content">
                    
                    <p style="color: #666; font-size: 13px; margin-bottom: 10px;">{t('temporal_desc')}</p>
                    
                    <!-- Statistics Overview -->
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;">
                        <!-- Ref→Analyzed Stats -->
                        <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; border: 1px solid #e9ecef;">
                            <h4 style="color: #3498DB; margin-bottom: 8px;">📖 {t('reference_to_analyzed')}</h4>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
                                <div><strong>{t('total_connections')}:</strong> {temporal.get('ref_to_analyzed', {}).get('total_connections', 0)}</div>
                                <div><strong>{t('min_lag_days')}:</strong> {ref_analyzed_stats.get('min', 'N/A')}</div>
                                <div><strong>{t('max_lag_days')}:</strong> {ref_analyzed_stats.get('max', 'N/A')}</div>
                                <div><strong>{t('avg_lag_days')}:</strong> {ref_analyzed_stats.get('avg', 0):.1f}</div>
                                <div><strong>{t('median_lag_days')}:</strong> {ref_analyzed_stats.get('median', 0):.1f}</div>
                                <div><strong>Std dev:</strong> {ref_analyzed_stats.get('std', 0):.1f}</div>
                            </div>
                        </div>
                        
                        <!-- Analyzed→Citing Stats -->
                        <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; border: 1px solid #e9ecef;">
                            <h4 style="color: #E74C3C; margin-bottom: 8px;">📚 {t('analyzed_to_citing')}</h4>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
                                <div><strong>{t('total_connections')}:</strong> {temporal.get('analyzed_to_citing', {}).get('total_connections', 0)}</div>
                                <div><strong>{t('min_lag_days')}:</strong> {analyzed_citing_stats.get('min', 'N/A')}</div>
                                <div><strong>{t('max_lag_days')}:</strong> {analyzed_citing_stats.get('max', 'N/A')}</div>
                                <div><strong>{t('avg_lag_days')}:</strong> {analyzed_citing_stats.get('avg', 0):.1f}</div>
                                <div><strong>{t('median_lag_days')}:</strong> {analyzed_citing_stats.get('median', 0):.1f}</div>
                                <div><strong>Std dev:</strong> {analyzed_citing_stats.get('std', 0):.1f}</div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- REF→ANALYZED HEATMAP -->
                    <h3 style="color: #3498DB; font-size: 16px;">📊 {t('reference_to_analyzed')} - Temporal Heatmap</h3>
                    <p style="color: #666; font-size: 12px; margin-bottom: 8px;">
                        X-axis: Year of Analyzed article | Y-axis: Year of Reference article
                        <span style="margin-left: 15px; font-weight: 600;">Color intensity = Number of connections</span>
                    </p>
                    <div class="scrollable-table" style="max-height: 500px;">
                        <table id="ref_heatmap_table">
                            <thead>
                                <tr>
                                    <th>{t('publication_year')} (Ref) \ (Analyzed)</th>
                                    {''.join([f'<th>{year}</th>' for year in temporal.get('all_years', [])])}
                                </tr>
                            </thead>
                            <tbody>
                                {''.join([
                                    f'''
                                    <tr>
                                        <td><strong>{row.get("publication_year", "N/A")}</strong></td>
                                        {''.join([
                                            f'<td class="heatmap-cell" style="{f"background: {get_heatmap_cell_color(row.get(year, 0), heatmap_max_ref)};" if row.get(year) is not None and row.get(year) > 0 else "background: transparent;"} color: {"#1a1a1a" if row.get(year) is not None and row.get(year) > 0 and row.get(year)/max(heatmap_max_ref, 1) > 0.6 else "#333" if row.get(year) is not None and row.get(year) > 0 else "transparent"};">{row.get(year) if row.get(year) is not None and row.get(year) > 0 else ""}</td>'
                                            for year in temporal.get('all_years', [])
                                        ])}
                                    </tr>
                                    '''
                                    for row in temporal.get('ref_to_analyzed', {}).get('heatmap', [])
                                ])}
                            </tbody>
                        </table>
                    </div>
                    
                    <!-- ANALYZED→CITING HEATMAP -->
                    <h3 style="color: #E74C3C; font-size: 16px; margin-top: 25px;">📊 {t('analyzed_to_citing')} - Temporal Heatmap</h3>
                    <p style="color: #666; font-size: 12px; margin-bottom: 8px;">
                        X-axis: Year of Citing article | Y-axis: Year of Analyzed article
                        <span style="margin-left: 15px; font-weight: 600;">Color intensity = Number of connections</span>
                    </p>
                    <div class="scrollable-table" style="max-height: 500px;">
                        <table id="citing_heatmap_table">
                            <thead>
                                <tr>
                                    <th>{t('publication_year')} (Analyzed) \ (Citing)</th>
                                    {''.join([f'<th>{year}</th>' for year in temporal.get('all_years', [])])}
                                </tr>
                            </thead>
                            <tbody>
                                {''.join([
                                    f'''
                                    <tr>
                                        <td><strong>{row.get("publication_year", "N/A")}</strong></td>
                                        {''.join([
                                            f'<td class="heatmap-cell" style="{f"background: {get_heatmap_cell_color(row.get(year, 0), heatmap_max_citing)};" if row.get(year) is not None and row.get(year) > 0 else "background: transparent;"} color: {"#1a1a1a" if row.get(year) is not None and row.get(year) > 0 and row.get(year)/max(heatmap_max_citing, 1) > 0.6 else "#333" if row.get(year) is not None and row.get(year) > 0 else "transparent"};">{row.get(year) if row.get(year) is not None and row.get(year) > 0 else ""}</td>'
                                            for year in temporal.get('all_years', [])
                                        ])}
                                    </tr>
                                    '''
                                    for row in temporal.get('analyzed_to_citing', {}).get('heatmap', [])
                                ])}
                            </tbody>
                        </table>
                    </div>
                    
                    <!-- LAG DISTRIBUTION -->
                    <h3 style="color: {primary}; font-size: 16px; margin-top: 25px;">📈 Lag Distribution</h3>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                        <div>
                            <h4 style="color: #3498DB; font-size: 13px;">Reference → Analyzed</h4>
                            {''.join([
                                f'''
                                <div style="margin: 3px 0;">
                                    <div class="progress-bar-label">
                                        <span>{lag_range} days</span>
                                        <span class="label-value">{count}</span>
                                    </div>
                                    <div class="progress-bar-container">
                                        <div class="progress-bar-fill animate" style="width: {count/max([c for _, c in ref_lag_dist], default=1)*100:.1f}%; background: #3498DB;">
                                            {count}
                                        </div>
                                    </div>
                                </div>
                                '''
                                for lag_range, count in ref_lag_dist[:15]
                            ])}
                        </div>
                        <div>
                            <h4 style="color: #E74C3C; font-size: 13px;">Analyzed → Citing</h4>
                            {''.join([
                                f'''
                                <div style="margin: 3px 0;">
                                    <div class="progress-bar-label">
                                        <span>{lag_range} days</span>
                                        <span class="label-value">{count}</span>
                                    </div>
                                    <div class="progress-bar-container">
                                        <div class="progress-bar-fill animate" style="width: {count/max([c for _, c in citing_lag_dist], default=1)*100:.1f}%; background: #E74C3C;">
                                            {count}
                                        </div>
                                    </div>
                                </div>
                                '''
                                for lag_range, count in citing_lag_dist[:15]
                            ])}
                        </div>
                    </div>
                    
                    <!-- DETAILED CONNECTIONS TABLES (collapsible) -->
                    <div style="margin-top: 25px;">
                        <div class="collapser" onclick="toggleCitations('ref_connections')" style="border-left-color: #3498DB;">
                            <strong style="color: #3498DB;">📋 Show Reference → Analyzed Connections</strong>
                            <span class="citation-count-badge" style="background: #3498DB;">{len(ref_analyzed_connections)} connections</span>
                            <span class="toggle-hint">{t('click_to_toggle')}</span>
                        </div>
                        <div id="citations_ref_connections" style="display: none; margin-bottom: 10px;">
                            <div class="scrollable-table" style="max-height: 400px;">
                                <table>
                                    <thead>
                                        <tr>
                                            <th>#</th>
                                            <th>{t('ref_doi')}</th>
                                            <th>{t('ref_date')}</th>
                                            <th>{t('analyzed_doi')}</th>
                                            <th>{t('analyzed_date')}</th>
                                            <th>{t('time_lag_days')}</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {''.join([
                                            f'''
                                            <tr>
                                                <td>{i+1}</td>
                                                <td><a href="https://doi.org/{html.escape(conn['ref_doi'])}" target="_blank" class="doi-link">{html.escape(conn['ref_doi'][:20])}...</a></td>
                                                <td>{conn['ref_date']}</td>
                                                <td><a href="https://doi.org/{html.escape(conn['analyzed_doi'])}" target="_blank" class="doi-link">{html.escape(conn['analyzed_doi'][:20])}...</a></td>
                                                <td>{conn['analyzed_date']}</td>
                                                <td>{get_color_scale_html(conn['lag_days'], max([c['lag_days'] for c in ref_analyzed_connections]) if ref_analyzed_connections else 1)}</td>
                                            </tr>
                                            '''
                                            for i, conn in enumerate(ref_analyzed_connections)
                                        ])}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                        
                        <div class="collapser" onclick="toggleCitations('citing_connections')" style="border-left-color: #E74C3C;">
                            <strong style="color: #E74C3C;">📋 Show Analyzed → Citing Connections</strong>
                            <span class="citation-count-badge" style="background: #E74C3C;">{len(analyzed_citing_connections)} connections</span>
                            <span class="toggle-hint">{t('click_to_toggle')}</span>
                        </div>
                        <div id="citations_citing_connections" style="display: none; margin-bottom: 10px;">
                            <div class="scrollable-table" style="max-height: 400px;">
                                <table>
                                    <thead>
                                        <tr>
                                            <th>#</th>
                                            <th>{t('analyzed_doi')}</th>
                                            <th>{t('analyzed_date')}</th>
                                            <th>{t('citing_doi')}</th>
                                            <th>{t('citing_date')}</th>
                                            <th>{t('time_lag_days')}</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {''.join([
                                            f'''
                                            <tr>
                                                <td>{i+1}</td>
                                                <td><a href="https://doi.org/{html.escape(conn['analyzed_doi'])}" target="_blank" class="doi-link">{html.escape(conn['analyzed_doi'][:20])}...</a></td>
                                                <td>{conn['analyzed_date']}</td>
                                                <td><a href="https://doi.org/{html.escape(conn['citing_doi'])}" target="_blank" class="doi-link">{html.escape(conn['citing_doi'][:20])}...</a></td>
                                                <td>{conn['citing_date']}</td>
                                                <td>{get_color_scale_html(conn['lag_days'], max([c['lag_days'] for c in analyzed_citing_connections]) if analyzed_citing_connections else 1)}</td>
                                            </tr>
                                            '''
                                            for i, conn in enumerate(analyzed_citing_connections)
                                        ])}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                    
                </div>
            </div>
            
            <!-- ============================================================ -->
            <!-- FOOTER -->
            <!-- ============================================================ -->
            <div class="footer">
                <p>{t('footer')}</p>
                <p>{t('generated_on')}: {datetime.now().strftime('%d.%m.%Y %H:%M')}</p>
                <p><a href="{t('journal_url')}" target="_blank">{t('journal_url')}</a></p>
            </div>
            
        </div>
    </div>
    
    <script>
        // ===== TOGGLE SECTIONS =====
        function toggleSection(sectionId) {{
            var content = document.getElementById(sectionId);
            var indicator = document.getElementById(sectionId.replace('_content', '_indicator'));
            if (content) {{
                if (content.style.display === 'none' || content.style.display === '') {{
                    content.style.display = 'block';
                    if (indicator) indicator.textContent = '▼';
                    content.style.animation = 'fadeInUp 0.4s ease forwards';
                }} else {{
                    content.style.display = 'none';
                    if (indicator) indicator.textContent = '▶';
                }}
            }}
        }}
        
        // ===== TOGGLE CITATIONS =====
        function toggleCitations(id) {{
            var el = document.getElementById('citations_' + id);
            if (el) {{
                if (el.style.display === 'none' || el.style.display === '') {{
                    el.style.display = 'block';
                    el.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
                }} else {{
                    el.style.display = 'none';
                }}
            }}
        }}
        
        // ===== UNIVERSAL SORT FUNCTION =====
        function sortTable(tableId, colIndex) {{
            var table = document.getElementById(tableId);
            if (!table) return;
            var tbody = table.querySelector('tbody');
            if (!tbody) return;
            var rows = Array.from(tbody.querySelectorAll('tr'));
            
            var key = tableId + '_col_' + colIndex;
            if (!window.sortState) window.sortState = {{}};
            if (!window.sortState[key]) window.sortState[key] = 1;
            else window.sortState[key] *= -1;
            var direction = window.sortState[key];
            
            var headers = table.querySelectorAll('thead th');
            headers.forEach(function(th, idx) {{
                th.classList.remove('asc', 'desc');
                if (idx === colIndex) {{
                    th.classList.add(direction > 0 ? 'asc' : 'desc');
                }}
            }});
            
            rows.sort(function(a, b) {{
                var valA = a.cells[colIndex] ? a.cells[colIndex].textContent.trim() : '';
                var valB = b.cells[colIndex] ? b.cells[colIndex].textContent.trim() : '';
                
                var numA = parseFloat(valA.replace(/,/g, ''));
                var numB = parseFloat(valB.replace(/,/g, ''));
                if (!isNaN(numA) && !isNaN(numB)) {{
                    return (numA - numB) * direction;
                }}
                
                return valA.localeCompare(valB) * direction;
            }});
            
            rows.forEach(function(row) {{
                tbody.appendChild(row);
            }});
        }}
        
        // ===== AUTO-OPEN FIRST SECTION =====
        document.addEventListener('DOMContentLoaded', function() {{
            var sections = ['overview_content', 'references_content', 'analyzed_content', 'citation_content', 'citing_content', 'topics_content', 'detailed_content', 'multilevel_content', 'title_keywords_content', 'temporal_content'];
            sections.forEach(function(id) {{
                var el = document.getElementById(id);
                if (el) {{
                    el.style.display = 'none';
                }}
            }});
            var indicators = ['overview_indicator', 'references_indicator', 'analyzed_indicator', 'citation_indicator', 'citing_indicator', 'topics_indicator', 'detailed_indicator', 'multilevel_indicator', 'title_keywords_indicator', 'temporal_indicator'];
            indicators.forEach(function(id) {{
                var el = document.getElementById(id);
                if (el) {{
                    el.textContent = '▶';
                }}
            }});
        }});
    </script>
    
    </body>
    </html>
    """
    
    return html_content

# ============================================
# ОСНОВНАЯ ФУНКЦИЯ ЗАПУСКА ДЛЯ STREAMLIT
# ============================================

def run_multilevel_analysis(doi_input: str, max_workers: int = 6):
    """Run complete multi-level DOI analysis and save results to session state"""
    
    current_lang = st.session_state.get('language', 'en')
    def t(key: str, **kwargs) -> str:
        return translate(key, current_lang, **kwargs)
    
    if not doi_input or not doi_input.strip():
        st.error(t('no_doi'))
        return
    
    # Parse DOIs
    doi_list = parse_doi_input(doi_input)
    
    if not doi_list:
        st.error(t('no_doi'))
        return
    
    # Check duplicates
    unique_dois = list(dict.fromkeys(doi_list))
    if len(doi_list) != len(unique_dois):
        duplicates = len(doi_list) - len(unique_dois)
        st.warning(t('duplicate_dois', count=duplicates))
    
    # Check maximum
    if len(unique_dois) > 100:
        st.error(t('too_many_dois', count=len(unique_dois)))
        return
    
    st.info(f"🔍 {t('stage_fetch_level_ii')} - {len(unique_dois)} DOIs")
    
    progress_container = st.empty()
    status_container = st.empty()
    analysis_progress = st.progress(0, text=t('starting_analysis'))
    
    try:
        # Load app logo
        app_logo_base64 = None
        if os.path.exists("logo.png"):
            try:
                with open("logo.png", "rb") as f:
                    app_logo_base64 = base64.b64encode(f.read()).decode()
            except Exception as e:
                if SHOW_DEBUG_LOGS:
                    print(f"⚠️ Error loading app logo: {e}")
        
        # Stage weights
        stage_weights = {
            'fetch_level_ii': 0.25,
            'fetch_metadata': 0.35,
            'analyze_report': 0.40
        }
        
        # Initialize analyzer
        analyzer = DOIAnalyzer(unique_dois, max_workers)
        
        # Stage 1: Fetch Level II
        status_container.info(f"📡 {t('stage_fetch_level_ii')}")
        analysis_progress.progress(0.01, text=t('stage_fetch_level_ii'))
        
        def level2_progress(current, total):
            progress = 0.01 + (current / total) * stage_weights['fetch_level_ii']
            analysis_progress.progress(progress, text=f"{t('stage_fetch_level_ii')} - {t('stage_processing', current=current, total=total)}")
            status_container.info(f"📡 {t('stage_fetch_level_ii')} - {current}/{total}")
        
        metadata_II = analyzer.fetch_level_II(level2_progress)
        
        if not metadata_II:
            st.error(f"❌ {t('data_not_found')}")
            analysis_progress.empty()
            return
        
        st.success(f"✅ {t('stage_doi_found', count=len(metadata_II))}")
        st.success(f"✅ {t('stage_ref_found', count=analyzer.total_references, unique=len(analyzer.level_I))}")
        st.success(f"✅ {t('stage_citing_found', count=analyzer.total_citing, unique=len(analyzer.level_III))}")
        
        # Stage 2: Fetch all metadata
        status_container.info(f"📡 {t('stage_fetch_metadata')}")
        progress_base = stage_weights['fetch_level_ii']
        analysis_progress.progress(progress_base, text=t('stage_fetch_metadata'))
        
        def meta_progress(current, total):
            progress = progress_base + (current / total) * stage_weights['fetch_metadata']
            analysis_progress.progress(progress, text=f"{t('stage_fetch_metadata')} - {t('stage_processing', current=current, total=total)}")
            status_container.info(f"📡 {t('stage_fetch_metadata')} - {current}/{total}")
        
        all_metadata = analyzer.fetch_all_metadata(meta_progress)
        total_metadata = len(all_metadata.get('level_I', {})) + len(all_metadata.get('level_II', {})) + len(all_metadata.get('level_III', {}))
        st.success(f"✅ {t('stage_metadata_fetched', count=total_metadata)}")
        
        # Stage 3: Analyze and generate report
        status_container.info(f"📡 {t('stage_analyze_report')}")
        progress_base += stage_weights['fetch_metadata']
        analysis_progress.progress(progress_base, text=t('stage_analyze_report'))
        
        def analyze_progress(current, total):
            progress = progress_base + (current / total) * stage_weights['analyze_report']
            analysis_progress.progress(progress, text=f"{t('stage_analyze_report')} - {t('stage_processing', current=current, total=total)}")
            status_container.info(f"📡 {t('stage_analyze_report')} - {current}/{total}")
        
        results = analyzer.analyze_data(analyze_progress)
        
        # ====== SAVE TO SESSION STATE ======
        st.session_state['analyzer'] = analyzer
        st.session_state['results'] = results
        st.session_state['doi_list'] = unique_dois
        st.session_state['app_logo_base64'] = app_logo_base64
        st.session_state['analysis_complete'] = True
        
        analysis_progress.progress(1.0, text=f"✅ {t('analysis_complete_text')}!")
        
        st.success(t('analysis_complete', 
                    level1=len(analyzer.level_I), 
                    level2=len(analyzer.level_II), 
                    level3=len(analyzer.level_III), 
                    time=0))
        st.balloons()
        
        # ====== SHOW REPORT ======
        st.markdown("---")
        st.markdown(f"## {t('html_report')}")
        
        theme_colors = {
            'primary': st.session_state.primary_color,
            'secondary': st.session_state.secondary_color
        }
        
        with st.spinner(t('generating_report')):
            html_report = generate_multilevel_html_report(
                st.session_state.analyzer,
                st.session_state.app_logo_base64,
                theme_colors,
                current_lang
            )
        
        filename = f"doi_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        st.download_button(
            label="📥 " + t('download_report'),
            data=html_report.encode('utf-8'),
            file_name=filename,
            mime="text/html",
            type="primary",
            width='stretch'
        )
        
        st.markdown("---")
        st.markdown(f"### {t('report_preview')}")
        st.info(t('download_hint'))
        st.components.v1.html(html_report, height=800, scrolling=True)
        
        if st.button("🔄 " + t('reset_analysis'), type="secondary"):
            for key in ['analyzer', 'results', 'doi_list', 'analysis_complete']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        
    except Exception as e:
        st.error(f"❌ {t('error_occurred')}: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
    finally:
        analysis_progress.empty()

# ============================================
# СОЗДАНИЕ WIDGET-ИНТЕРФЕЙСА STREAMLIT
# ============================================

def main():
    # Page configuration
    st.set_page_config(
        page_title="Ref-Cit-Analysis",
        page_icon="logo.jpg",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'primary_color' not in st.session_state:
        st.session_state.primary_color = '#667eea'
    if 'secondary_color' not in st.session_state:
        st.session_state.secondary_color = '#f39c12'
    if 'language' not in st.session_state:
        st.session_state.language = 'en'
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'doi_list' not in st.session_state:
        st.session_state.doi_list = []
    if 'app_logo_base64' not in st.session_state:
        st.session_state.app_logo_base64 = None
    
    # Apply theme
    primary = st.session_state.primary_color
    secondary = st.session_state.secondary_color
    apply_theme_css(primary, secondary)
    
    # Get current language
    current_lang = st.session_state.language
    
    def t(key: str, **kwargs) -> str:
        return translate(key, current_lang, **kwargs)
    
    # Sidebar
    with st.sidebar:
        st.markdown(f"## {t('settings')}")
        
        # Language selector (only English)
        st.markdown(f"**{t('language')}:** English")
        
        st.markdown("---")
        
        # Color theme
        st.markdown(f"## {t('color_theme')}")
        
        preset_themes = {
            "Default (Blue-Purple)": {"primary": "#667eea", "secondary": "#a9019b"},
            "Emerald (Green-Teal)": {"primary": "#2ecc71", "secondary": "#27ae60"},
            "Sunset (Orange-Coral)": {"primary": "#e74c3c", "secondary": "#c0392b"},
            "Ocean (Deep Blue)": {"primary": "#3498db", "secondary": "#2980b9"},
            "Royal (Purple-Pink)": {"primary": "#9b59b6", "secondary": "#e84393"},
            "Forest (Dark Green)": {"primary": "#27ae60", "secondary": "#2ecc71"},
            "Cherry (Red-Pink)": {"primary": "#e84393", "secondary": "#9b59b6"},
            "Amber (Yellow-Orange)": {"primary": "#f39c12", "secondary": "#e67e22"},
        }
        
        theme_option = st.selectbox(
            t('preset_themes'),
            options=list(preset_themes.keys()),
            index=0
        )
        
        # Primary color picker
        selected_primary = st.color_picker(
            t('select_primary'),
            value=st.session_state.primary_color
        )
        
        # Secondary color picker
        selected_secondary = st.color_picker(
            t('select_secondary'),
            value=st.session_state.secondary_color
        )
        
        st.session_state.primary_color = selected_primary
        st.session_state.secondary_color = selected_secondary
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f'<div style="text-align: center;">'
                f'<div class="color-preview" style="background: {st.session_state.primary_color};"></div>'
                f'<div style="font-size: 11px; margin-top: 5px;">{t("primary")}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f'<div style="text-align: center;">'
                f'<div class="color-preview" style="background: {st.session_state.secondary_color};"></div>'
                f'<div style="font-size: 11px; margin-top: 5px;">{t("secondary")}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        
        st.markdown("---")
        
        st.markdown(f"## {t('analysis_params')}")
        
        global USE_CACHE
        use_cache = st.checkbox(t('use_cache'), value=USE_CACHE)
        USE_CACHE = use_cache
        
        if st.button(t('clear_cache')):
            import shutil
            if os.path.exists('cache_doi'):
                shutil.rmtree('cache_doi')
                st.cache_data.clear()
                st.success(t('cache_cleared'))
        
        st.markdown("---")
        
        st.markdown(f"""
        <div style="font-size: 11px; color: #666; text-align: center;">
            © daM / Chimica Techno Acta / {t('journal_url')}
        </div>
        """, unsafe_allow_html=True)
    
    # Main area - logo
    if os.path.exists("logo.png"):
        col_logo, col_spacer = st.columns([1, 3])
        with col_logo:
            st.image("logo.png", width=400)
    st.markdown("---")
    
    # ====== CHECK: Data in session state? ======
    if st.session_state.analysis_complete and st.session_state.analyzer:
        st.info(f"📦 {t('analysis_data_from_cache')}")
        st.markdown(f"**Level II DOIs:** {len(st.session_state.doi_list)}")
        
        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("🗑️ " + t('reset_analysis'), type="secondary"):
                for key in ['analyzer', 'results', 'doi_list', 'analysis_complete']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        
        st.markdown("---")
        
        st.markdown(f"## {t('html_report')}")
        
        theme_colors = {
            'primary': st.session_state.primary_color,
            'secondary': st.session_state.secondary_color
        }
        
        with st.spinner(t('generating_report')):
            html_report = generate_multilevel_html_report(
                st.session_state.analyzer,
                st.session_state.app_logo_base64,
                theme_colors,
                current_lang
            )
        
        filename = f"doi_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        st.download_button(
            label="📥 " + t('download_report'),
            data=html_report.encode('utf-8'),
            file_name=filename,
            mime="text/html",
            type="primary",
            width='stretch'
        )
        
        st.markdown("---")
        st.markdown(f"### {t('report_preview')}")
        st.info(t('download_hint'))
        st.components.v1.html(html_report, height=800, scrolling=True)
        
    else:
        # Input section for new analysis
        st.markdown(f"## {t('load_data')}")
        
        st.markdown(f"**{t('doi_input')}**")
        
        doi_textarea = st.text_area(
            label="",
            placeholder=t('doi_placeholder'),
            help=t('doi_help'),
            height=150
        )
        
        col3, col4 = st.columns([1, 3])
        
        with col3:
            workers = st.slider(
                t('workers'),
                min_value=4,
                max_value=10,
                value=6,
                step=1,
                help=t('workers_help')
            )
        
        with col4:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button(t('analyze_button'), type="primary", width='stretch'):
                run_multilevel_analysis(doi_textarea, workers)

if __name__ == "__main__":
    main()
