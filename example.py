import pandas as pd
import numpy as np
import re
import logging
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Optional
import json
from dataclasses import dataclass, asdict
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
from pathlib import Path
from difflib import SequenceMatcher
import Levenshtein

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot_analysis.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class CategoryConfig:
    """Configuration for a category with weighted keywords and patterns"""
    keywords: Dict[str, float]  # keyword: weight
    patterns: List[Tuple[str, float]]  # (regex_pattern, weight)
    min_score: float = 0.3  # minimum score to classify into this category
    priority: int = 1  # higher priority categories are checked first
    fuzzy_threshold: float = 0.8  # minimum similarity for fuzzy matching
    fuzzy_weight_multiplier: float = 0.7  # weight multiplier for fuzzy matches

class EnhancedChatbotAnalyzer:
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.categories = self._initialize_categories()
        self.compiled_patterns = self._compile_patterns()
        self.session_cache = {}
        self.keyword_stats = defaultdict(int)
        self.fuzzy_cache = {}  # Cache for fuzzy matching results
        
    def _initialize_categories(self) -> Dict[str, CategoryConfig]:
        """Initialize categories with weighted keywords and patterns"""
        return {
            # Technical Categories (Priority 1 - Check First)
            "DevOps/Infrastructure": CategoryConfig(
                keywords={
                    # CI/CD & Automation
                    "jenkins": 2.0, "ci/cd": 2.5, "pipeline": 1.8, "github actions": 2.2,
                    "gitlab ci": 2.0, "circleci": 2.0, "travis": 1.8, "bamboo": 1.8,
                    
                    # Containerization & Orchestration
                    "docker": 2.5, "kubernetes": 2.5, "k8s": 2.5, "container": 1.5,
                    "helm": 2.0, "istio": 2.0, "openshift": 2.0, "rancher": 1.8,
                    "podman": 1.8, "containerd": 1.8, "docker-compose": 2.0,
                    
                    # IaC & Configuration
                    "terraform": 2.5, "ansible": 2.5, "puppet": 2.0, "chef": 2.0,
                    "pulumi": 2.0, "cloudformation": 2.2, "arm template": 2.0,
                    
                    # Cloud Platforms
                    "aws": 2.0, "azure": 2.0, "gcp": 2.0, "google cloud": 2.0,
                    "ec2": 2.0, "s3": 1.8, "lambda": 1.8, "eks": 2.0, "aks": 2.0,
                    
                    # Monitoring & Logging
                    "prometheus": 2.0, "grafana": 2.0, "elk": 2.0, "elasticsearch": 2.0,
                    "logstash": 1.8, "kibana": 1.8, "datadog": 2.0, "new relic": 2.0,
                    "splunk": 2.0, "cloudwatch": 1.8, "stackdriver": 1.8,
                },
                patterns=[
                    (r'\b(?:deploy|deployment)\s+(?:to|on|in)\s+(?:prod|production|staging)\b', 2.0),
                    (r'\b(?:scale|scaling|autoscale|autoscaling)\b', 1.5),
                    (r'\b(?:load\s*balanc(?:er|ing)|lb|alb|nlb|elb)\b', 1.8),
                    (r'\b(?:infra|infrastructure)\s+as\s+code\b', 2.5),
                    (r'\b(?:micro)?services?\s+(?:architecture|deployment)\b', 2.0),
                ],
                min_score=0.5,
                priority=1
            ),
            
            "Backend Development": CategoryConfig(
                keywords={
                    # Languages
                    "python": 1.5, "java": 1.5, "c#": 1.5, "golang": 2.0, "go": 1.2,
                    "rust": 2.0, "scala": 1.8, "kotlin": 1.8, "php": 1.2, "ruby": 1.5,
                    
                    # Frameworks
                    "spring": 2.0, "django": 2.0, "flask": 1.8, "fastapi": 2.0,
                    "express": 1.8, "nestjs": 2.0, "rails": 1.8, "laravel": 1.8,
                    ".net": 2.0, "asp.net": 2.0, "gin": 1.8, "echo": 1.8,
                    
                    # API & Backend Concepts
                    "rest api": 2.0, "graphql": 2.2, "grpc": 2.2, "websocket": 2.0,
                    "microservice": 2.0, "serverless": 2.0, "crud": 1.5,
                    "authentication": 1.8, "authorization": 1.8, "jwt": 2.0,
                    "oauth": 2.0, "middleware": 1.8, "orm": 1.8,
                },
                patterns=[
                    (r'\bapi\s+(?:endpoint|design|development|integration)\b', 2.0),
                    (r'\b(?:rest|restful)\s+(?:api|service|endpoint)\b', 2.0),
                    (r'\b(?:backend|back-end|server-side)\s+(?:code|logic|development)\b', 2.0),
                    (r'\b(?:database|db)\s+(?:query|connection|pool|optimization)\b', 1.8),
                ],
                min_score=0.5,
                priority=1
            ),
            
            "Frontend Development": CategoryConfig(
                keywords={
                    # Frameworks & Libraries
                    "react": 2.5, "angular": 2.5, "vue": 2.5, "svelte": 2.2,
                    "nextjs": 2.5, "next.js": 2.5, "nuxt": 2.2, "gatsby": 2.0,
                    "webpack": 1.8, "vite": 2.0, "parcel": 1.8, "rollup": 1.8,
                    
                    # Web Technologies
                    "javascript": 1.5, "typescript": 2.0, "html": 1.0, "css": 1.0,
                    "sass": 1.5, "scss": 1.5, "tailwind": 2.0, "bootstrap": 1.5,
                    "material-ui": 1.8, "mui": 1.8, "chakra": 1.8,
                    
                    # Frontend Concepts
                    "responsive": 1.5, "spa": 2.0, "pwa": 2.0, "ssr": 2.0,
                    "state management": 2.0, "redux": 2.0, "mobx": 1.8,
                    "zustand": 1.8, "recoil": 1.8, "context api": 1.8,
                },
                patterns=[
                    (r'\b(?:front-?end|frontend|client-?side)\s+(?:development|code|framework)\b', 2.0),
                    (r'\b(?:ui|ux|user\s+interface)\s+(?:design|development|component)\b', 1.8),
                    (r'\bcomponent\s+(?:library|design|development)\b', 1.5),
                    (r'\b(?:web|mobile)\s+app(?:lication)?\s+development\b', 1.8),
                ],
                min_score=0.5,
                priority=1
            ),
            
            "Data Engineering & Analytics": CategoryConfig(
                keywords={
                    # Data Processing
                    "spark": 2.5, "hadoop": 2.0, "kafka": 2.5, "airflow": 2.5,
                    "flink": 2.2, "beam": 2.0, "etl": 2.0, "elt": 2.0,
                    "databricks": 2.5, "snowflake": 2.5, "redshift": 2.2,
                    "bigquery": 2.2, "athena": 2.0, "presto": 2.0, "trino": 2.0,
                    
                    # Databases
                    "sql": 1.5, "nosql": 1.8, "postgresql": 2.0, "mysql": 1.8,
                    "mongodb": 2.0, "cassandra": 2.0, "redis": 2.0, "elasticsearch": 2.2,
                    "dynamodb": 2.0, "cosmos db": 2.0, "neo4j": 2.0,
                    
                    # Analytics & BI
                    "tableau": 2.0, "power bi": 2.0, "looker": 2.0, "metabase": 1.8,
                    "superset": 1.8, "quicksight": 1.8, "data pipeline": 2.0,
                },
                patterns=[
                    (r'\bdata\s+(?:pipeline|warehouse|lake|mart|mesh)\b', 2.2),
                    (r'\b(?:batch|stream|real-?time)\s+processing\b', 2.0),
                    (r'\b(?:data|database)\s+(?:migration|replication|sync)\b', 1.8),
                    (r'\bquery\s+(?:optimization|performance|tuning)\b', 2.0),
                ],
                min_score=0.5,
                priority=1
            ),
            
            "AI/ML & Data Science": CategoryConfig(
                keywords={
                    # ML Frameworks
                    "tensorflow": 2.5, "pytorch": 2.5, "scikit-learn": 2.2, "sklearn": 2.2,
                    "keras": 2.0, "xgboost": 2.0, "lightgbm": 2.0, "catboost": 2.0,
                    
                    # ML Concepts
                    "machine learning": 2.5, "deep learning": 2.5, "neural network": 2.5,
                    "nlp": 2.2, "computer vision": 2.2, "reinforcement learning": 2.2,
                    "transformer": 2.0, "bert": 2.0, "gpt": 2.0, "llm": 2.5,
                    
                    # Data Science Tools
                    "pandas": 2.0, "numpy": 1.8, "jupyter": 1.8, "matplotlib": 1.5,
                    "seaborn": 1.5, "plotly": 1.8, "streamlit": 2.0, "gradio": 2.0,
                    
                    # MLOps
                    "mlflow": 2.2, "kubeflow": 2.2, "sagemaker": 2.2, "vertex ai": 2.2,
                    "wandb": 2.0, "dvc": 2.0, "feast": 2.0,
                },
                patterns=[
                    (r'\b(?:train|training|fine-?tun(?:e|ing))\s+(?:a\s+)?model\b', 2.2),
                    (r'\b(?:model|algorithm)\s+(?:deployment|serving|inference)\b', 2.0),
                    (r'\b(?:feature|data)\s+engineering\b', 2.0),
                    (r'\b(?:classification|regression|clustering|prediction)\s+(?:model|algorithm)\b', 2.0),
                ],
                min_score=0.5,
                priority=1
            ),
            
            "Security & Compliance": CategoryConfig(
                keywords={
                    # Security Tools
                    "owasp": 2.5, "burp": 2.0, "metasploit": 2.0, "nmap": 1.8,
                    "wireshark": 1.8, "snort": 2.0, "vault": 2.2, "hashicorp vault": 2.5,
                    
                    # Security Concepts
                    "vulnerability": 2.0, "penetration testing": 2.5, "pentest": 2.5,
                    "security scan": 2.0, "encryption": 2.0, "tls": 1.8, "ssl": 1.8,
                    "firewall": 1.8, "waf": 2.0, "ddos": 2.0, "xss": 2.0, "csrf": 2.0,
                    "sql injection": 2.2, "authentication": 1.8, "authorization": 1.8,
                    
                    # Compliance
                    "gdpr": 2.2, "hipaa": 2.2, "pci": 2.0, "sox": 2.0, "iso 27001": 2.2,
                    "compliance": 1.8, "audit": 1.5,
                },
                patterns=[
                    (r'\bsecurity\s+(?:vulnerability|audit|assessment|scan)\b', 2.2),
                    (r'\b(?:secure|security)\s+(?:coding|development|configuration)\b', 2.0),
                    (r'\b(?:data|information)\s+(?:security|protection|privacy)\b', 1.8),
                    (r'\b(?:zero\s+trust|least\s+privilege|defense\s+in\s+depth)\b', 2.0),
                ],
                min_score=0.5,
                priority=1
            ),
            
            # Business Categories (Priority 2)
            "Project Management": CategoryConfig(
                keywords={
                    "jira": 2.0, "confluence": 2.0, "trello": 1.8, "asana": 1.8,
                    "monday": 1.5, "clickup": 1.5, "notion": 1.5, "linear": 1.8,
                    "sprint": 1.8, "scrum": 2.0, "agile": 2.0, "kanban": 1.8,
                    "epic": 1.5, "user story": 2.0, "backlog": 1.5, "roadmap": 1.8,
                    "retrospective": 1.8, "standup": 1.5, "grooming": 1.5,
                },
                patterns=[
                    (r'\b(?:project|product)\s+(?:management|planning|timeline)\b', 1.8),
                    (r'\b(?:sprint|iteration)\s+(?:planning|review|retrospective)\b', 2.0),
                    (r'\b(?:story|task)\s+(?:point|estimation|sizing)\b', 1.8),
                ],
                min_score=0.4,
                priority=2
            ),
            
            "Documentation & Knowledge": CategoryConfig(
                keywords={
                    "documentation": 2.0, "readme": 1.8, "wiki": 1.5, "knowledge base": 2.0,
                    "tutorial": 1.5, "guide": 1.2, "manual": 1.5, "api docs": 2.2,
                    "swagger": 2.0, "openapi": 2.0, "postman": 1.8,
                },
                patterns=[
                    (r'\b(?:write|create|update)\s+(?:documentation|docs|readme)\b', 2.0),
                    (r'\b(?:technical|api|user)\s+documentation\b', 2.0),
                    (r'\bdocument(?:ing|ation)\s+(?:code|api|process)\b', 1.8),
                ],
                min_score=0.4,
                priority=2
            ),
            
            "Testing & QA": CategoryConfig(
                keywords={
                    "unit test": 2.2, "integration test": 2.2, "e2e test": 2.2,
                    "selenium": 2.0, "cypress": 2.2, "jest": 2.0, "pytest": 2.0,
                    "mocha": 1.8, "jasmine": 1.8, "testing": 1.5, "qa": 1.8,
                    "test automation": 2.2, "tdd": 2.0, "bdd": 2.0,
                    "cucumber": 1.8, "postman": 1.8, "jmeter": 2.0,
                },
                patterns=[
                    (r'\b(?:write|create|implement)\s+(?:unit|integration|e2e)\s+test\b', 2.2),
                    (r'\btest\s+(?:coverage|automation|strategy|plan)\b', 2.0),
                    (r'\b(?:bug|defect|issue)\s+(?:report|tracking|fixing)\b', 1.8),
                ],
                min_score=0.4,
                priority=1
            ),
            
            # General Business Categories (Priority 3)
            "Communication & Writing": CategoryConfig(
                keywords={
                    "email": 1.5, "message": 1.0, "slack": 1.5, "teams": 1.2,
                    "presentation": 1.5, "slides": 1.3, "powerpoint": 1.5,
                    "report": 1.3, "proposal": 1.5, "memo": 1.3,
                },
                patterns=[
                    (r'\b(?:write|draft|compose)\s+(?:an?\s+)?(?:email|message|letter)\b', 1.8),
                    (r'\b(?:create|prepare)\s+(?:a\s+)?(?:presentation|report|proposal)\b', 1.5),
                ],
                min_score=0.3,
                priority=3
            ),
            
            "Analysis & Research": CategoryConfig(
                keywords={
                    "analyze": 1.5, "analysis": 1.5, "research": 1.5, "investigate": 1.3,
                    "compare": 1.2, "evaluate": 1.3, "assess": 1.3, "review": 1.2,
                    "metrics": 1.5, "kpi": 1.8, "dashboard": 1.8, "report": 1.3,
                },
                patterns=[
                    (r'\b(?:data|business|market)\s+analysis\b', 1.8),
                    (r'\b(?:research|investigate|explore)\s+(?:options|solutions|approaches)\b', 1.5),
                    (r'\b(?:competitive|market|industry)\s+(?:analysis|research)\b', 1.8),
                ],
                min_score=0.3,
                priority=3
            ),
        }
    
    def _compile_patterns(self) -> Dict[str, List[Tuple[re.Pattern, float]]]:
        """Pre-compile regex patterns for better performance"""
        compiled = {}
        for cat_name, cat_config in self.categories.items():
            compiled[cat_name] = [
                (re.compile(pattern, re.IGNORECASE), weight)
                for pattern, weight in cat_config.patterns
            ]
        return compiled
    
    def _fuzzy_match_keyword(self, text: str, keyword: str, threshold: float = 0.8) -> Tuple[bool, float]:
        """Check if a keyword fuzzy matches in text using multiple algorithms"""
        text_lower = text.lower()
        keyword_lower = keyword.lower()
        
        # Check for exact match first (fastest)
        if keyword_lower in text_lower:
            return True, 1.0
        
        # Check for word boundary matches
        words = text_lower.split()
        for word in words:
            # Levenshtein distance for word-level matching
            if len(word) >= 3:  # Only check words with 3+ characters
                distance = Levenshtein.distance(word, keyword_lower)
                max_len = max(len(word), len(keyword_lower))
                similarity = 1 - (distance / max_len)
                
                if similarity >= threshold:
                    return True, similarity
        
        # Check for substring similarity using SequenceMatcher
        for word in words:
            if len(word) >= 3:
                matcher = SequenceMatcher(None, word, keyword_lower)
                similarity = matcher.ratio()
                
                if similarity >= threshold:
                    return True, similarity
        
        # Check for partial matches (e.g., "docker" in "docker-compose")
        if keyword_lower in text_lower.replace('-', ' ').replace('_', ' '):
            return True, 0.9
        
        return False, 0.0
    
    def calculate_category_score(self, text: str, category_name: str) -> float:
        """Calculate weighted score for a category based on text with fuzzy matching"""
        text_lower = text.lower()
        score = 0.0
        matched_keywords = []
        fuzzy_matches = []
        
        category = self.categories[category_name]
        
        # Check keywords with exact and fuzzy matching
        for keyword, weight in category.keywords.items():
            # Exact match
            if keyword in text_lower:
                score += weight
                matched_keywords.append(keyword)
                self.keyword_stats[keyword] += 1
            else:
                # Fuzzy match
                is_match, similarity = self._fuzzy_match_keyword(
                    text, keyword, category.fuzzy_threshold
                )
                if is_match:
                    fuzzy_weight = weight * category.fuzzy_weight_multiplier * similarity
                    score += fuzzy_weight
                    fuzzy_matches.append(f"{keyword} (fuzzy: {similarity:.2f})")
                    self.keyword_stats[f"{keyword}_fuzzy"] += 1
        
        # Check regex patterns
        for pattern, weight in self.compiled_patterns[category_name]:
            if pattern.search(text):
                score += weight
        
        # Apply length normalization (longer texts shouldn't automatically score higher)
        text_length = len(text.split())
        if text_length > 0:
            score = score / np.log(max(text_length, 10))
        
        return score
    
    def classify_message(self, text: str) -> Tuple[str, float, List[str], Dict[str, List[str]]]:
        """Classify a message into categories with confidence score and match details"""
        if not text or len(text.strip()) < 3:
            return "Uncategorized", 0.0, [], {}
        
        # Calculate scores for all categories
        scores = {}
        match_details = {}
        
        for cat_name in self.categories:
            score, exact_matches, fuzzy_matches = self._calculate_category_score_with_details(text, cat_name)
            scores[cat_name] = score
            match_details[cat_name] = {
                'exact_matches': exact_matches,
                'fuzzy_matches': fuzzy_matches
            }
        
        # Sort categories by priority and score
        sorted_categories = sorted(
            scores.items(),
            key=lambda x: (self.categories[x[0]].priority, -x[1])
        )
        
        # Get best category
        best_category, best_score = sorted_categories[0]
        
        # Check if score meets minimum threshold
        if best_score < self.categories[best_category].min_score:
            return "Uncategorized", best_score, [], match_details
        
        # Get secondary categories (multi-label classification)
        secondary_categories = []
        for cat, score in sorted_categories[1:]:
            if score >= self.categories[cat].min_score * 0.7:  # 70% of min threshold
                secondary_categories.append(cat)
        
        return best_category, best_score, secondary_categories, match_details
    
    def _calculate_category_score_with_details(self, text: str, category_name: str) -> Tuple[float, List[str], List[str]]:
        """Calculate category score with detailed match information"""
        text_lower = text.lower()
        score = 0.0
        exact_matches = []
        fuzzy_matches = []
        
        category = self.categories[category_name]
        
        # Check keywords with exact and fuzzy matching
        for keyword, weight in category.keywords.items():
            # Exact match
            if keyword in text_lower:
                score += weight
                exact_matches.append(keyword)
                self.keyword_stats[keyword] += 1
            else:
                # Fuzzy match
                is_match, similarity = self._fuzzy_match_keyword(
                    text, keyword, category.fuzzy_threshold
                )
                if is_match:
                    fuzzy_weight = weight * category.fuzzy_weight_multiplier * similarity
                    score += fuzzy_weight
                    fuzzy_matches.append(f"{keyword} (similarity: {similarity:.2f})")
                    self.keyword_stats[f"{keyword}_fuzzy"] += 1
        
        # Check regex patterns
        for pattern, weight in self.compiled_patterns[category_name]:
            if pattern.search(text):
                score += weight
        
        # Apply length normalization (longer texts shouldn't automatically score higher)
        text_length = len(text.split())
        if text_length > 0:
            score = score / np.log(max(text_length, 10))
        
        return score, exact_matches, fuzzy_matches
    
    def analyze_session_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze user session patterns"""
        session_stats = {
            'total_sessions': df['session_id'].nunique(),
            'avg_messages_per_session': df.groupby('session_id').size().mean(),
            'session_category_transitions': defaultdict(int),
            'user_journey_patterns': []
        }
        
        # Analyze category transitions within sessions
        for session_id, group in df.groupby('session_id'):
            if len(group) > 1:
                categories = group['primary_category'].tolist()
                for i in range(len(categories) - 1):
                    transition = f"{categories[i]} → {categories[i+1]}"
                    session_stats['session_category_transitions'][transition] += 1
        
        # Find common user journeys
        session_journeys = df.groupby('session_id')['primary_category'].apply(
            lambda x: ' → '.join(x) if len(x) > 1 else x.iloc[0]
        ).value_counts().head(10)
        
        session_stats['user_journey_patterns'] = session_journeys.to_dict()
        
        return session_stats
    
    def detect_emerging_topics(self, df: pd.DataFrame) -> Dict:
        """Detect emerging topics using time-based analysis"""
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        
        # Calculate keyword frequency over time
        keyword_timeline = defaultdict(lambda: defaultdict(int))
        
        for _, row in df.iterrows():
            text_lower = str(row['input']).lower()
            date = row['date']
            
            # Check all keywords
            for category in self.categories.values():
                for keyword in category.keywords:
                    if keyword in text_lower:
                        keyword_timeline[keyword][date] += 1
        
        # Identify trending keywords (increasing frequency)
        trending_keywords = []
        for keyword, timeline in keyword_timeline.items():
            if len(timeline) >= 3:  # Need at least 3 data points
                dates = sorted(timeline.keys())
                recent_avg = np.mean([timeline[d] for d in dates[-3:]])
                older_avg = np.mean([timeline[d] for d in dates[:-3]]) if len(dates) > 3 else 0
                
                if recent_avg > older_avg * 1.5:  # 50% increase
                    trending_keywords.append({
                        'keyword': keyword,
                        'growth_rate': (recent_avg - older_avg) / max(older_avg, 1),
                        'recent_frequency': recent_avg
                    })
        
        trending_keywords.sort(key=lambda x: x['growth_rate'], reverse=True)
        
        return {
            'trending_topics': trending_keywords[:20],
            'keyword_timeline': dict(keyword_timeline)
        }
    
    def generate_insights(self, df: pd.DataFrame) -> Dict:
        """Generate actionable insights from the analysis"""
        insights = {
            'recommendations': [],
            'alerts': [],
            'opportunities': []
        }
        
        # Category distribution insights
        category_dist = df['primary_category'].value_counts()
        total = len(df)
        
        # Check for imbalanced usage
        for category, count in category_dist.items():
            percentage = (count / total) * 100
            if percentage > 40:
                insights['alerts'].append(
                    f"High concentration in {category} ({percentage:.1f}%). "
                    f"Consider specialized tooling or training for this area."
                )
            elif percentage < 1 and category in ['Security & Compliance', 'Testing & QA']:
                insights['alerts'].append(
                    f"Low {category} usage ({percentage:.1f}%). "
                    f"This might indicate a gap in practices."
                )
        
        # Session pattern insights
        avg_confidence = df['confidence_score'].mean()
        if avg_confidence < 0.5:
            insights['recommendations'].append(
                "Low average classification confidence. Consider refining keyword definitions "
                "or implementing ML-based classification for better accuracy."
            )
        
        # Uncategorized messages
        uncategorized_pct = (len(df[df['primary_category'] == 'Uncategorized']) / total) * 100
        if uncategorized_pct > 20:
            insights['alerts'].append(
                f"{uncategorized_pct:.1f}% of messages are uncategorized. "
                f"Review these messages to identify new use cases or improve classification."
            )
        
        # Multi-category messages (complex queries)
        multi_category = df[df['secondary_categories'].apply(len) > 0]
        if len(multi_category) / total > 0.15:
            insights['opportunities'].append(
                "15%+ queries span multiple categories. Consider creating integrated "
                "solutions or documentation that addresses cross-functional needs."
            )
        
        return insights
    
    def analyze_from_database(self, conn, time_window_days: int = 30, 
                             batch_size: int = 10000) -> Dict:
        """Main analysis function with comprehensive reporting"""
        
        # Query with time window
        query = f"""
        SELECT session_id, message_id, input, timestamp
        FROM chat_messages
        WHERE input IS NOT NULL
        AND timestamp >= CURRENT_DATE - INTERVAL '{time_window_days} days'
        ORDER BY timestamp
        """
        
        logging.info(f"Starting analysis for last {time_window_days} days...")
        
        # Process in chunks for memory efficiency
        chunks = pd.read_sql(query, conn, chunksize=batch_size)
        
        all_results = []
        total_processed = 0
        
        # Process chunks with progress tracking
        for chunk_num, chunk in enumerate(chunks, 1):
            logging.info(f"Processing chunk {chunk_num} ({len(chunk)} messages)...")
            
            # Clean and prepare data
            chunk = chunk.dropna(subset=['input'])
            chunk['input'] = chunk['input'].astype(str)
            
            # Classify messages
            classifications = []
            for _, row in chunk.iterrows():
                primary_cat, confidence, secondary_cats, match_details = self.classify_message(row['input'])
                classifications.append({
                    'session_id': row['session_id'],
                    'message_id': row['message_id'],
                    'timestamp': row['timestamp'],
                    'input': row['input'],
                    'primary_category': primary_cat,
                    'confidence_score': confidence,
                    'secondary_categories': secondary_cats,
                    'match_details': match_details
                })
            
            chunk_df = pd.DataFrame(classifications)
            all_results.append(chunk_df)
            total_processed += len(chunk)
            
            # Cache intermediate results
            if chunk_num % 5 == 0:
                self._save_checkpoint(all_results, chunk_num)
        
        # Combine all results
        logging.info("Combining results...")
        df_results = pd.concat(all_results, ignore_index=True)
        
        # Generate comprehensive analysis
        analysis_results = {
            'summary': {
                'total_messages': total_processed,
                'time_period': f"Last {time_window_days} days",
                'analysis_timestamp': datetime.now().isoformat()
            },
            'category_distribution': self._calculate_category_distribution(df_results),
            'technical_vs_business': self._calculate_tech_vs_business_split(df_results),
            'session_patterns': self.analyze_session_patterns(df_results),
            'emerging_topics': self.detect_emerging_topics(df_results),
            'time_analysis': self._analyze_temporal_patterns(df_results),
            'insights': self.generate_insights(df_results),
            'keyword_statistics': dict(Counter(self.keyword_stats).most_common(50)),
            'fuzzy_match_analysis': self._analyze_fuzzy_matches(df_results)
        }
        
        # Save detailed results
        self._save_results(analysis_results, df_results)
        
        return analysis_results
    
    def _calculate_category_distribution(self, df: pd.DataFrame) -> Dict:
        """Calculate detailed category distribution"""
        primary_dist = df['primary_category'].value_counts()
        
        # Calculate with percentages
        total = len(df)
        distribution = {}
        for category, count in primary_dist.items():
            distribution[category] = {
                'count': int(count),
                'percentage': round((count / total) * 100, 2),
                'avg_confidence': round(
                    df[df['primary_category'] == category]['confidence_score'].mean(), 3
                )
            }
        
        return distribution
    
    def _analyze_fuzzy_matches(self, df: pd.DataFrame) -> Dict:
        """Analyze fuzzy matching performance and patterns"""
        fuzzy_stats = {
            'total_fuzzy_matches': 0,
            'fuzzy_match_rate': 0.0,
            'top_fuzzy_keywords': [],
            'fuzzy_match_by_category': {},
            'improvement_opportunities': []
        }
        
        total_messages = len(df)
        fuzzy_matches_by_category = defaultdict(int)
        fuzzy_keyword_counts = defaultdict(int)
        
        for _, row in df.iterrows():
            match_details = row.get('match_details', {})
            for category, details in match_details.items():
                fuzzy_matches = details.get('fuzzy_matches', [])
                if fuzzy_matches:
                    fuzzy_stats['total_fuzzy_matches'] += len(fuzzy_matches)
                    fuzzy_matches_by_category[category] += len(fuzzy_matches)
                    
                    # Extract keywords from fuzzy matches
                    for match in fuzzy_matches:
                        keyword = match.split(' (similarity:')[0]
                        fuzzy_keyword_counts[keyword] += 1
        
        # Calculate fuzzy match rate
        if total_messages > 0:
            fuzzy_stats['fuzzy_match_rate'] = (fuzzy_stats['total_fuzzy_matches'] / total_messages) * 100
        
        # Top fuzzy keywords
        fuzzy_stats['top_fuzzy_keywords'] = [
            {'keyword': kw, 'count': count} 
            for kw, count in sorted(fuzzy_keyword_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        ]
        
        # Fuzzy matches by category
        fuzzy_stats['fuzzy_match_by_category'] = dict(fuzzy_matches_by_category)
        
        # Identify improvement opportunities
        high_fuzzy_categories = [
            cat for cat, count in fuzzy_matches_by_category.items() 
            if count > 10  # Categories with many fuzzy matches
        ]
        
        for category in high_fuzzy_categories:
            fuzzy_stats['improvement_opportunities'].append(
                f"Category '{category}' has many fuzzy matches. Consider adding common variations "
                f"or adjusting fuzzy threshold for better accuracy."
            )
        
        return fuzzy_stats
    
    def _calculate_tech_vs_business_split(self, df: pd.DataFrame) -> Dict:
        """Calculate technical vs business split"""
        tech_categories = {
            'DevOps/Infrastructure', 'Backend Development', 'Frontend Development',
            'Data Engineering & Analytics', 'AI/ML & Data Science', 
            'Security & Compliance', 'Testing & QA', 'Documentation & Knowledge'
        }
        
        tech_messages = df[df['primary_category'].isin(tech_categories)]
        business_messages = df[~df['primary_category'].isin(tech_categories)]
        
        return {
            'technical': {
                'count': len(tech_messages),
                'percentage': round((len(tech_messages) / len(df)) * 100, 2),
                'categories': tech_messages['primary_category'].value_counts().to_dict()
            },
            'business': {
                'count': len(business_messages),
                'percentage': round((len(business_messages) / len(df)) * 100, 2),
                'categories': business_messages['primary_category'].value_counts().to_dict()
            }
        }
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze temporal patterns in usage"""
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()
        df['date'] = df['timestamp'].dt.date
        
        return {
            'hourly_distribution': df.groupby('hour')['message_id'].count().to_dict(),
            'daily_distribution': df.groupby('day_of_week')['message_id'].count().to_dict(),
            'daily_trend': df.groupby('date')['message_id'].count().to_dict(),
            'category_by_hour': df.groupby(['hour', 'primary_category']).size().unstack(fill_value=0).to_dict(),
            'peak_hours': df.groupby('hour')['message_id'].count().nlargest(3).to_dict(),
            'peak_days': df.groupby('day_of_week')['message_id'].count().nlargest(3).to_dict()
        }
    
    def _save_checkpoint(self, results: List[pd.DataFrame], chunk_num: int):
        """Save intermediate results for recovery"""
        checkpoint_file = self.cache_dir / f"checkpoint_{chunk_num}.pkl"
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(results, f)
        logging.info(f"Checkpoint saved: {checkpoint_file}")
    
    def _save_results(self, analysis_results: Dict, df_results: pd.DataFrame):
        """Save analysis results in multiple formats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON report
        json_file = f"chatbot_analysis_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        logging.info(f"JSON report saved: {json_file}")
        
        # Save detailed CSV
        csv_file = f"chatbot_messages_classified_{timestamp}.csv"
        df_results.to_csv(csv_file, index=False)
        logging.info(f"Detailed CSV saved: {csv_file}")
        
        # Generate HTML report
        html_report = self._generate_html_report(analysis_results)
        html_file = f"chatbot_analysis_report_{timestamp}.html"
        with open(html_file, 'w') as f:
            f.write(html_report)
        logging.info(f"HTML report saved: {html_file}")
        
        # Generate executive summary
        exec_summary = self._generate_executive_summary(analysis_results)
        summary_file = f"executive_summary_{timestamp}.md"
        with open(summary_file, 'w') as f:
            f.write(exec_summary)
        logging.info(f"Executive summary saved: {summary_file}")
    
    def _generate_html_report(self, results: Dict) -> str:
        """Generate an HTML report with visualizations"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Chatbot Usage Analysis Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }
                h1 { color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }
                h2 { color: #555; margin-top: 30px; }
                .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }
                .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; }
                .metric-value { font-size: 2em; font-weight: bold; }
                .metric-label { opacity: 0.9; margin-top: 5px; }
                .chart { margin: 20px 0; }
                .insight-box { background: #f0f8ff; border-left: 4px solid #2196F3; padding: 15px; margin: 10px 0; }
                .alert-box { background: #fff3cd; border-left: 4px solid #ff9800; padding: 15px; margin: 10px 0; }
                .table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                .table th, .table td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
                .table th { background: #4CAF50; color: white; }
                .table tr:hover { background: #f5f5f5; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🤖 Chatbot Usage Analysis Report</h1>
                <p><strong>Analysis Period:</strong> {time_period} | <strong>Generated:</strong> {timestamp}</p>
                
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-value">{total_messages:,}</div>
                        <div class="metric-label">Total Messages</div>
                    </div>
                    <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                        <div class="metric-value">{tech_percentage:.1f}%</div>
                        <div class="metric-label">Technical Queries</div>
                    </div>
                    <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                        <div class="metric-value">{business_percentage:.1f}%</div>
                        <div class="metric-label">Business Queries</div>
                    </div>
                    <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
                        <div class="metric-value">{total_sessions:,}</div>
                        <div class="metric-label">Unique Sessions</div>
                    </div>
                </div>
                
                <h2>📊 Category Distribution</h2>
                <div id="categoryChart" class="chart"></div>
                
                <h2>📈 Usage Trends Over Time</h2>
                <div id="timelineChart" class="chart"></div>
                
                <h2>🔥 Top Categories</h2>
                <table class="table">
                    <thead>
                        <tr>
                            <th>Category</th>
                            <th>Messages</th>
                            <th>Percentage</th>
                            <th>Avg Confidence</th>
                        </tr>
                    </thead>
                    <tbody>
                        {category_rows}
                    </tbody>
                </table>
                
                <h2>💡 Key Insights</h2>
                {insights_html}
                
                <h2>⚠️ Alerts</h2>
                {alerts_html}
                
                <h2>📈 Trending Topics</h2>
                {trending_html}
                
                <h2>🔍 Fuzzy Match Analysis</h2>
                <div class="metric-grid">
                    <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                        <div class="metric-value">{fuzzy_match_rate:.1f}%</div>
                        <div class="metric-label">Fuzzy Match Rate</div>
                    </div>
                    <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                        <div class="metric-value">{total_fuzzy_matches:,}</div>
                        <div class="metric-label">Total Fuzzy Matches</div>
                    </div>
                </div>
                
                <h3>Top Fuzzy Keywords</h3>
                <table class="table">
                    <thead>
                        <tr>
                            <th>Keyword</th>
                            <th>Fuzzy Match Count</th>
                        </tr>
                    </thead>
                    <tbody>
                        {fuzzy_keywords_rows}
                    </tbody>
                </table>
            </div>
            
            <script>
                {charts_javascript}
            </script>
        </body>
        </html>
        """
        
        # Fill in the template with actual data
        tech_vs_business = results.get('technical_vs_business', {})
        
        # Prepare category rows for the table
        category_rows = []
        for cat, data in results['category_distribution'].items():
            category_rows.append(f"""
                <tr>
                    <td>{cat}</td>
                    <td>{data['count']:,}</td>
                    <td>{data['percentage']:.1f}%</td>
                    <td>{data['avg_confidence']:.2f}</td>
                </tr>
            """)
        
        # Prepare insights and alerts
        insights = results.get('insights', {})
        insights_html = ""
        for rec in insights.get('recommendations', []):
            insights_html += f'<div class="insight-box">💡 {rec}</div>'
        
        alerts_html = ""
        for alert in insights.get('alerts', []):
            alerts_html += f'<div class="alert-box">⚠️ {alert}</div>'
        
        # Trending topics
        trending_html = "<ul>"
        for topic in results.get('emerging_topics', {}).get('trending_topics', [])[:10]:
            trending_html += f"<li><strong>{topic['keyword']}</strong> - {topic['growth_rate']:.1%} growth</li>"
        trending_html += "</ul>"
        
        # Fuzzy match analysis
        fuzzy_analysis = results.get('fuzzy_match_analysis', {})
        fuzzy_match_rate = fuzzy_analysis.get('fuzzy_match_rate', 0)
        total_fuzzy_matches = fuzzy_analysis.get('total_fuzzy_matches', 0)
        
        # Prepare fuzzy keywords rows
        fuzzy_keywords_rows = ""
        for item in fuzzy_analysis.get('top_fuzzy_keywords', [])[:10]:
            fuzzy_keywords_rows += f"""
                <tr>
                    <td>{item['keyword']}</td>
                    <td>{item['count']:,}</td>
                </tr>
            """
        
        # Generate JavaScript for charts
        charts_js = self._generate_charts_javascript(results)
        
        return html.format(
            time_period=results['summary']['time_period'],
            timestamp=results['summary']['analysis_timestamp'],
            total_messages=results['summary']['total_messages'],
            tech_percentage=tech_vs_business.get('technical', {}).get('percentage', 0),
            business_percentage=tech_vs_business.get('business', {}).get('percentage', 0),
            total_sessions=results.get('session_patterns', {}).get('total_sessions', 0),
            category_rows=''.join(category_rows[:10]),
            insights_html=insights_html,
            alerts_html=alerts_html,
            trending_html=trending_html,
            fuzzy_match_rate=fuzzy_match_rate,
            total_fuzzy_matches=total_fuzzy_matches,
            fuzzy_keywords_rows=fuzzy_keywords_rows,
            charts_javascript=charts_js
        )
    
    def _generate_charts_javascript(self, results: Dict) -> str:
        """Generate Plotly charts JavaScript"""
        # Category distribution chart
        categories = list(results['category_distribution'].keys())[:15]
        values = [results['category_distribution'][cat]['count'] for cat in categories]
        
        # Time series data
        time_data = results.get('time_analysis', {}).get('daily_trend', {})
        dates = list(time_data.keys())
        counts = list(time_data.values())
        
        js_code = f"""
        // Category Distribution Pie Chart
        var categoryData = [{{
            type: 'pie',
            labels: {json.dumps(categories)},
            values: {json.dumps(values)},
            hole: 0.4,
            marker: {{
                colors: ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', 
                        '#00f2fe', '#43e97b', '#38f9d7', '#fa709a', '#fee140']
            }}
        }}];
        
        var categoryLayout = {{
            title: 'Message Distribution by Category',
            height: 400
        }};
        
        Plotly.newPlot('categoryChart', categoryData, categoryLayout);
        
        // Timeline Chart
        var timelineData = [{{
            x: {json.dumps([str(d) for d in dates])},
            y: {json.dumps(counts)},
            type: 'scatter',
            mode: 'lines+markers',
            line: {{
                color: '#667eea',
                width: 2
            }},
            marker: {{
                size: 6,
                color: '#764ba2'
            }}
        }}];
        
        var timelineLayout = {{
            title: 'Daily Message Volume',
            xaxis: {{ title: 'Date' }},
            yaxis: {{ title: 'Number of Messages' }},
            height: 400
        }};
        
        Plotly.newPlot('timelineChart', timelineData, timelineLayout);
        """
        
        return js_code
    
    def _generate_executive_summary(self, results: Dict) -> str:
        """Generate executive summary in Markdown"""
        tech_vs_business = results.get('technical_vs_business', {})
        insights = results.get('insights', {})
        
        summary = f"""
# Executive Summary: Chatbot Usage Analysis

**Report Generated:** {results['summary']['analysis_timestamp']}  
**Analysis Period:** {results['summary']['time_period']}  
**Total Messages Analyzed:** {results['summary']['total_messages']:,}

## Key Findings

### Usage Split
- **Technical Queries:** {tech_vs_business.get('technical', {}).get('percentage', 0):.1f}%
- **Business Queries:** {tech_vs_business.get('business', {}).get('percentage', 0):.1f}%

### Top Technical Categories
"""
        
        # Add top technical categories
        for cat, data in list(results['category_distribution'].items())[:5]:
            if cat in ['DevOps/Infrastructure', 'Backend Development', 'Frontend Development', 
                      'Data Engineering & Analytics', 'AI/ML & Data Science']:
                summary += f"- **{cat}:** {data['count']:,} messages ({data['percentage']:.1f}%)\n"
        
        summary += "\n## Critical Insights\n\n"
        
        # Add alerts
        if insights.get('alerts'):
            summary += "### ⚠️ Alerts\n"
            for alert in insights['alerts']:
                summary += f"- {alert}\n"
        
        # Add recommendations
        if insights.get('recommendations'):
            summary += "\n### 💡 Recommendations\n"
            for rec in insights['recommendations']:
                summary += f"- {rec}\n"
        
        # Add trending topics
        trending = results.get('emerging_topics', {}).get('trending_topics', [])
        if trending:
            summary += "\n### 📈 Trending Topics\n"
            for topic in trending[:5]:
                summary += f"- **{topic['keyword']}**: {topic['growth_rate']:.1%} growth\n"
        
        summary += """
## Next Steps

1. **Review Uncategorized Messages**: Identify patterns in uncategorized messages to improve classification
2. **Focus on High-Volume Categories**: Provide specialized resources and documentation for top categories
3. **Monitor Trending Topics**: Stay ahead of emerging needs by tracking trending keywords
4. **Optimize for Peak Usage**: Review temporal patterns to ensure chatbot availability during peak hours
5. **Analyze Fuzzy Matches**: Review fuzzy match patterns to improve keyword definitions and thresholds
6. **Regular Analysis**: Schedule monthly analysis to track usage evolution and identify new patterns

---
*This report provides data-driven insights into chatbot usage patterns. For detailed analysis, refer to the full HTML report.*
"""
        
        return summary


# Usage Example
def main():
    """Main execution function"""
    # Initialize analyzer
    analyzer = EnhancedChatbotAnalyzer(cache_dir="./analysis_cache")
    
    # Connect to database
    conn = get_db_connection()  # Your existing connection function
    
    if not conn:
        logging.error("Failed to connect to database")
        return
    
    try:
        # Run analysis
        results = analyzer.analyze_from_database(
            conn=conn,
            time_window_days=30,
            batch_size=10000
        )
        
        # Print summary to console
        print("\n" + "="*60)
        print("CHATBOT USAGE ANALYSIS COMPLETE")
        print("="*60)
        print(f"Total Messages Analyzed: {results['summary']['total_messages']:,}")
        print(f"Analysis Period: {results['summary']['time_period']}")
        
        print("\n📊 Top Categories:")
        for cat, data in list(results['category_distribution'].items())[:5]:
            print(f"  {cat}: {data['count']:,} ({data['percentage']:.1f}%)")
        
        print("\n📈 Technical vs Business Split:")
        tech_pct = results['technical_vs_business']['technical']['percentage']
        bus_pct = results['technical_vs_business']['business']['percentage']
        print(f"  Technical: {tech_pct:.1f}%")
        print(f"  Business: {bus_pct:.1f}%")
        
        print("\n✅ Analysis reports saved successfully!")
        print("  - JSON report: chatbot_analysis_*.json")
        print("  - HTML report: chatbot_analysis_report_*.html")
        print("  - Executive summary: executive_summary_*.md")
        print("  - Detailed CSV: chatbot_messages_classified_*.csv")
        
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        raise
    finally:
        conn.close()
        logging.info("Database connection closed")


if __name__ == "__main__":
    main()