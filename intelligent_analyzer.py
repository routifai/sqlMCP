import pandas as pd
import numpy as np
import re
import logging
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Optional, NamedTuple
import json
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import psycopg2
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot_analysis.log'),
        logging.StreamHandler()
    ]
)

class MatchType(Enum):
    """Types of pattern matching"""
    EXACT_PHRASE = "exact_phrase"      # Multi-word exact phrases
    WORD_BOUNDARY = "word_boundary"    # Single words with boundaries
    CONTEXT_PATTERN = "context_pattern" # Regex with context
    NEGATIVE_PATTERN = "negative"      # Patterns to exclude

@dataclass
class SmartPattern:
    """Smart pattern with context awareness"""
    pattern: str
    weight: float
    match_type: MatchType
    requires_context: List[str] = field(default_factory=list)  # Must have one of these
    excludes_context: List[str] = field(default_factory=list)  # Must NOT have these
    min_word_length: int = 3  # Minimum word length to consider

class SmartCategoryConfig:
    """Enhanced category configuration with smart matching"""
    def __init__(self, 
                 exact_phrases: Dict[str, float],
                 word_patterns: Dict[str, float],
                 context_patterns: List[Tuple[str, float]],
                 negative_indicators: List[str] = None,
                 min_score: float = 1.0,
                 priority: int = 1):
        
        self.exact_phrases = exact_phrases
        self.word_patterns = word_patterns
        self.context_patterns = context_patterns
        self.negative_indicators = negative_indicators or []
        self.min_score = min_score
        self.priority = priority
        
        # Pre-compile patterns for efficiency
        self.compiled_exact = self._compile_exact_phrases()
        self.compiled_words = self._compile_word_patterns()
        self.compiled_context = self._compile_context_patterns()
        self.compiled_negative = self._compile_negative_patterns()
    
    def _compile_exact_phrases(self):
        """Compile exact phrase patterns"""
        patterns = []
        for phrase, weight in self.exact_phrases.items():
            # Escape special regex characters and create exact phrase pattern
            escaped = re.escape(phrase)
            pattern = re.compile(r'\b' + escaped + r'\b', re.IGNORECASE)
            patterns.append((pattern, weight, phrase))
        return patterns
    
    def _compile_word_patterns(self):
        """Compile single word patterns with boundaries"""
        patterns = []
        for word, weight in self.word_patterns.items():
            if len(word) >= 3:  # Only words 3+ characters
                escaped = re.escape(word)
                # Ensure word boundaries - no partial matches!
                pattern = re.compile(r'\b' + escaped + r'(?:\b|s\b|ing\b|ed\b)', re.IGNORECASE)
                patterns.append((pattern, weight, word))
        return patterns
    
    def _compile_context_patterns(self):
        """Compile context-aware patterns"""
        patterns = []
        for pattern_str, weight in self.context_patterns:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            patterns.append((pattern, weight))
        return patterns
    
    def _compile_negative_patterns(self):
        """Compile negative indicator patterns"""
        if not self.negative_indicators:
            return []
        
        patterns = []
        for indicator in self.negative_indicators:
            pattern = re.compile(r'\b' + re.escape(indicator) + r'\b', re.IGNORECASE)
            patterns.append(pattern)
        return patterns

class IntelligentChatbotAnalyzer:
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.categories = self._initialize_smart_categories()
        self.classification_cache = {}
        self.phrase_frequency = defaultdict(int)
        self.context_stats = defaultdict(lambda: defaultdict(int))
        
    def _initialize_smart_categories(self) -> Dict[str, SmartCategoryConfig]:
        """Initialize categories with intelligent pattern matching"""
        return {
            "DevOps/Infrastructure": SmartCategoryConfig(
                exact_phrases={
                    # CI/CD - Only meaningful phrases
                    "continuous integration": 3.0,
                    "continuous deployment": 3.0,
                    "continuous delivery": 3.0,
                    "github actions": 3.0,
                    "gitlab ci": 3.0,
                    "azure devops": 3.0,
                    "bitbucket pipelines": 3.0,
                    "jenkins pipeline": 3.0,
                    "build pipeline": 2.5,
                    "deployment pipeline": 2.5,
                    
                    # Container & Orchestration
                    "docker container": 3.0,
                    "docker compose": 3.0,
                    "docker image": 3.0,
                    "kubernetes cluster": 3.5,
                    "kubernetes deployment": 3.5,
                    "kubernetes service": 3.0,
                    "container orchestration": 3.0,
                    "service mesh": 3.0,
                    "helm chart": 3.0,
                    "ingress controller": 3.0,
                    
                    # Cloud specific
                    "aws ec2": 3.0,
                    "aws lambda": 3.0,
                    "aws s3": 3.0,
                    "azure functions": 3.0,
                    "google cloud": 3.0,
                    "cloud formation": 3.0,
                    "infrastructure as code": 3.5,
                    "terraform module": 3.0,
                    "ansible playbook": 3.0,
                    
                    # Monitoring
                    "prometheus monitoring": 3.0,
                    "grafana dashboard": 3.0,
                    "elk stack": 3.0,
                    "log aggregation": 2.5,
                    "metric collection": 2.5,
                    "distributed tracing": 3.0,
                },
                word_patterns={
                    # Only unambiguous tech terms
                    "kubernetes": 3.5,
                    "docker": 3.0,
                    "terraform": 3.0,
                    "ansible": 3.0,
                    "jenkins": 3.0,
                    "prometheus": 3.0,
                    "grafana": 3.0,
                    "elasticsearch": 3.0,
                    "kibana": 2.5,
                    "logstash": 2.5,
                    "datadog": 3.0,
                    "cloudwatch": 2.5,
                    "kubectl": 3.0,
                    "dockerfile": 3.0,
                    "containerization": 3.0,
                    "orchestration": 2.5,
                    "microservices": 2.5,
                },
                context_patterns=[
                    # Context-aware patterns
                    (r'\b(?:deploy|deploying|deployed|deployment)\s+to\s+(?:production|staging|prod|stage|dev)\b', 3.5),
                    (r'\b(?:scale|scaling|autoscale|autoscaling)\s+(?:pods?|containers?|instances?|services?)\b', 3.0),
                    (r'\b(?:ci/cd|cicd)\s+(?:pipeline|workflow|process|setup)\b', 3.5),
                    (r'\b(?:container|docker|kubernetes|k8s)\s+(?:orchestration|management|deployment)\b', 3.5),
                    (r'\b(?:infrastructure|infra)\s+(?:automation|provisioning|management)\b', 3.0),
                    (r'\b(?:monitoring|observability|logging)\s+(?:setup|configuration|stack|solution)\b', 3.0),
                    (r'\b(?:aws|azure|gcp|cloud)\s+(?:infrastructure|architecture|deployment|resources)\b', 3.0),
                    (r'\bkubernetes\s+(?:cluster|pod|service|deployment|namespace|configmap|secret)\b', 3.5),
                    (r'\bdocker\s+(?:build|run|push|pull|tag|compose|swarm)\b', 3.0),
                    (r'\b(?:helm|chart|charts)\s+(?:install|upgrade|rollback|deployment)\b', 3.0),
                ],
                negative_indicators=[
                    "docker desktop",  # Too basic, probably not DevOps
                    "getting started with docker",  # Tutorial level
                    "what is kubernetes",  # Basic question
                ],
                min_score=2.0,
                priority=1
            ),
            
            "Backend Development": SmartCategoryConfig(
                exact_phrases={
                    # Frameworks with context
                    "spring boot": 3.5,
                    "spring framework": 3.5,
                    "spring mvc": 3.0,
                    "spring security": 3.0,
                    "django rest framework": 3.5,
                    "django orm": 3.0,
                    "flask api": 3.0,
                    "fastapi": 3.5,
                    "express.js": 3.5,
                    "express middleware": 3.0,
                    "nest.js": 3.5,
                    "ruby on rails": 3.5,
                    "asp.net core": 3.5,
                    "asp.net mvc": 3.0,
                    
                    # API & Architecture
                    "rest api": 3.5,
                    "restful api": 3.5,
                    "graphql api": 3.5,
                    "grpc service": 3.5,
                    "api gateway": 3.0,
                    "api endpoint": 3.0,
                    "microservice architecture": 3.5,
                    "service oriented architecture": 3.0,
                    "event driven architecture": 3.5,
                    "clean architecture": 3.5,
                    "hexagonal architecture": 3.5,
                    "domain driven design": 3.5,
                    
                    # Database specific
                    "database connection": 2.5,
                    "connection pool": 3.0,
                    "database migration": 3.0,
                    "database schema": 2.5,
                    "sql query": 2.5,
                    "nosql database": 3.0,
                    "orm mapping": 3.0,
                    "database transaction": 3.0,
                    
                    # Authentication
                    "jwt token": 3.5,
                    "jwt authentication": 3.5,
                    "oauth2 flow": 3.5,
                    "session management": 3.0,
                    "password hashing": 3.0,
                    "api authentication": 3.0,
                },
                word_patterns={
                    # Only unambiguous backend terms
                    "springframework": 3.5,
                    "django": 3.0,
                    "flask": 2.5,  # Could be ambiguous
                    "fastapi": 3.5,
                    "nestjs": 3.5,
                    "laravel": 3.0,
                    "symfony": 3.0,
                    "postgresql": 3.0,
                    "mysql": 2.5,
                    "mongodb": 3.0,
                    "redis": 3.0,
                    "rabbitmq": 3.0,
                    "kafka": 3.0,
                    "elasticsearch": 3.0,
                    "graphql": 3.5,
                    "grpc": 3.5,
                    "websocket": 3.0,
                    "webhook": 3.0,
                    "middleware": 2.5,
                    "serialization": 2.5,
                    "deserialization": 2.5,
                },
                context_patterns=[
                    (r'\b(?:build|building|create|creating)\s+(?:a\s+)?(?:rest|restful|graphql|grpc)\s+api\b', 3.5),
                    (r'\b(?:spring|django|flask|express|nestjs)\s+(?:application|backend|server|api|service)\b', 3.5),
                    (r'\b(?:implement|implementing|implemented)\s+(?:authentication|authorization|jwt|oauth)\b', 3.0),
                    (r'\b(?:database|db)\s+(?:query|queries|connection|transaction|migration)\b', 2.5),
                    (r'\b(?:api|endpoint|route|controller)\s+(?:design|development|implementation|testing)\b', 3.0),
                    (r'\b(?:microservice|service)\s+(?:communication|architecture|pattern|design)\b', 3.0),
                    (r'\b(?:backend|server-side|server)\s+(?:development|code|logic|architecture)\b', 3.0),
                    (r'\b(?:orm|object.?relational|hibernate|sequelize|prisma)\s+(?:mapping|query|model)\b', 3.0),
                    (r'\b(?:message|event)\s+(?:queue|broker|streaming|processing)\b', 3.0),
                    (r'\b(?:caching|cache)\s+(?:strategy|implementation|layer|redis|memcached)\b', 2.5),
                ],
                negative_indicators=[
                    "html form",  # Probably frontend
                    "css styling",  # Definitely frontend
                    "react component",  # Frontend
                ],
                min_score=2.0,
                priority=1
            ),
            
            "Frontend Development": SmartCategoryConfig(
                exact_phrases={
                    # Frameworks & Libraries
                    "react component": 3.5,
                    "react hooks": 3.5,
                    "react router": 3.0,
                    "react state": 3.0,
                    "angular component": 3.5,
                    "angular service": 3.0,
                    "angular directive": 3.0,
                    "vue component": 3.5,
                    "vue router": 3.0,
                    "vue composition api": 3.5,
                    "next.js": 3.5,
                    "server side rendering": 3.0,
                    "static site generation": 3.0,
                    
                    # Styling
                    "css grid": 3.0,
                    "css flexbox": 3.0,
                    "tailwind css": 3.5,
                    "styled components": 3.0,
                    "sass variables": 2.5,
                    "css modules": 3.0,
                    "responsive design": 3.0,
                    "mobile first": 3.0,
                    
                    # State Management
                    "redux store": 3.5,
                    "redux action": 3.0,
                    "redux reducer": 3.0,
                    "state management": 3.0,
                    "context api": 3.0,
                    
                    # Performance
                    "lazy loading": 3.0,
                    "code splitting": 3.0,
                    "bundle size": 3.0,
                    "tree shaking": 3.0,
                    "webpack config": 3.0,
                    "vite config": 3.0,
                },
                word_patterns={
                    # Frontend-specific terms
                    "react": 3.0,
                    "angular": 3.0,
                    "vue": 2.5,  # Could be ambiguous (vue = view in French)
                    "svelte": 3.5,
                    "nextjs": 3.5,
                    "nuxtjs": 3.5,
                    "gatsby": 3.0,
                    "webpack": 3.0,
                    "vite": 3.0,
                    "tailwind": 3.5,
                    "bootstrap": 2.5,
                    "sass": 2.5,
                    "scss": 2.5,
                    "typescript": 2.5,  # Used in backend too
                    "jsx": 3.5,
                    "tsx": 3.5,
                    "redux": 3.0,
                    "mobx": 3.0,
                    "zustand": 3.5,
                    "recoil": 3.0,
                    "storybook": 3.0,
                    "cypress": 3.0,
                    "playwright": 3.0,
                },
                context_patterns=[
                    (r'\b(?:react|angular|vue|svelte)\s+(?:component|hook|application|app|project)\b', 3.5),
                    (r'\b(?:frontend|front-end|client-side|ui)\s+(?:development|framework|application|code)\b', 3.0),
                    (r'\b(?:component|components)\s+(?:library|design|development|testing)\b', 2.5),
                    (r'\b(?:responsive|mobile|adaptive)\s+(?:design|layout|ui|interface)\b', 3.0),
                    (r'\b(?:css|scss|sass|styled)\s+(?:styling|animation|layout|framework)\b', 3.0),
                    (r'\b(?:state|redux|mobx|zustand)\s+(?:management|store|action|reducer)\b', 3.0),
                    (r'\b(?:spa|single.?page)\s+application\b', 3.5),
                    (r'\b(?:pwa|progressive.?web)\s+app\b', 3.5),
                    (r'\b(?:webpack|vite|parcel|rollup)\s+(?:config|configuration|bundle|build)\b', 3.0),
                    (r'\b(?:user|ui|ux)\s+(?:interface|experience|design|interaction)\b', 2.5),
                ],
                negative_indicators=[
                    "backend api",
                    "database query",
                    "server configuration",
                ],
                min_score=2.0,
                priority=1
            ),
            
            "Data Engineering": SmartCategoryConfig(
                exact_phrases={
                    # Data Processing
                    "apache spark": 3.5,
                    "spark sql": 3.0,
                    "spark streaming": 3.5,
                    "apache kafka": 3.5,
                    "kafka streaming": 3.5,
                    "apache airflow": 3.5,
                    "airflow dag": 3.5,
                    "etl pipeline": 3.5,
                    "elt pipeline": 3.5,
                    "data pipeline": 3.5,
                    "batch processing": 3.0,
                    "stream processing": 3.0,
                    "real-time processing": 3.0,
                    
                    # Data Storage
                    "data warehouse": 3.5,
                    "data lake": 3.5,
                    "data lakehouse": 3.5,
                    "delta lake": 3.5,
                    "data mart": 3.0,
                    "data mesh": 3.5,
                    
                    # Platforms
                    "databricks": 3.5,
                    "snowflake": 3.5,
                    "amazon redshift": 3.5,
                    "google bigquery": 3.5,
                    "azure synapse": 3.5,
                },
                word_patterns={
                    # Data engineering specific
                    "spark": 2.5,  # Could be ambiguous
                    "kafka": 3.0,
                    "airflow": 3.5,
                    "databricks": 3.5,
                    "snowflake": 3.0,
                    "redshift": 3.0,
                    "bigquery": 3.5,
                    "athena": 2.5,
                    "presto": 3.0,
                    "trino": 3.5,
                    "hadoop": 3.0,
                    "hive": 2.5,
                    "flink": 3.0,
                    "beam": 2.0,  # Too ambiguous
                },
                context_patterns=[
                    (r'\b(?:data|etl|elt)\s+(?:pipeline|pipelines|workflow|orchestration)\b', 3.5),
                    (r'\b(?:batch|stream|real-time)\s+(?:processing|ingestion|data)\b', 3.0),
                    (r'\b(?:data)\s+(?:warehouse|lake|lakehouse|mart|mesh)\s+(?:design|architecture|implementation)\b', 3.5),
                    (r'\b(?:spark|kafka|flink)\s+(?:job|streaming|processing|cluster)\b', 3.5),
                    (r'\b(?:airflow|luigi|dagster)\s+(?:dag|workflow|pipeline|orchestration)\b', 3.5),
                    (r'\b(?:data)\s+(?:transformation|cleansing|validation|quality|governance)\b', 2.5),
                    (r'\b(?:sql|query)\s+(?:optimization|performance|tuning)\s+(?:for|in)\s+(?:redshift|snowflake|bigquery)\b', 3.5),
                ],
                negative_indicators=[
                    "data science",
                    "machine learning",
                    "data analysis",  # More analytics than engineering
                ],
                min_score=2.0,
                priority=1
            ),
            
            "Machine Learning/AI": SmartCategoryConfig(
                exact_phrases={
                    # ML Frameworks
                    "machine learning": 3.5,
                    "deep learning": 3.5,
                    "neural network": 3.5,
                    "neural networks": 3.5,
                    "tensorflow model": 3.5,
                    "pytorch model": 3.5,
                    "scikit-learn": 3.5,
                    "sklearn pipeline": 3.5,
                    
                    # ML Concepts
                    "supervised learning": 3.5,
                    "unsupervised learning": 3.5,
                    "reinforcement learning": 3.5,
                    "transfer learning": 3.5,
                    "feature engineering": 3.5,
                    "model training": 3.5,
                    "model evaluation": 3.0,
                    "hyperparameter tuning": 3.5,
                    "cross validation": 3.5,
                    
                    # Specific Algorithms
                    "random forest": 3.5,
                    "gradient boosting": 3.5,
                    "support vector machine": 3.5,
                    "k-means clustering": 3.5,
                    "linear regression": 3.0,
                    "logistic regression": 3.0,
                    
                    # Deep Learning
                    "convolutional neural network": 3.5,
                    "recurrent neural network": 3.5,
                    "transformer model": 3.5,
                    "attention mechanism": 3.5,
                    
                    # NLP
                    "natural language processing": 3.5,
                    "text classification": 3.5,
                    "named entity recognition": 3.5,
                    "sentiment analysis": 3.5,
                    
                    # Computer Vision
                    "computer vision": 3.5,
                    "image classification": 3.5,
                    "object detection": 3.5,
                    "image segmentation": 3.5,
                    
                    # LLMs
                    "large language model": 3.5,
                    "prompt engineering": 3.5,
                    "fine tuning": 3.5,
                    "rag pipeline": 3.5,
                },
                word_patterns={
                    # ML specific terms
                    "tensorflow": 3.5,
                    "pytorch": 3.5,
                    "keras": 3.0,
                    "sklearn": 3.5,
                    "xgboost": 3.5,
                    "lightgbm": 3.5,
                    "catboost": 3.5,
                    "mlflow": 3.5,
                    "kubeflow": 3.5,
                    "sagemaker": 3.5,
                    "huggingface": 3.5,
                    "transformers": 3.0,
                    "bert": 3.5,
                    "gpt": 3.0,  # Be careful with this
                    "lstm": 3.5,
                    "cnn": 2.5,  # Could mean CNN news
                    "rnn": 3.5,
                    "gan": 3.0,
                    "autoencoder": 3.5,
                    "embedding": 3.0,
                    "tokenization": 3.0,
                },
                context_patterns=[
                    (r'\b(?:train|training|fine-tune|fine-tuning)\s+(?:a\s+)?(?:model|algorithm|network)\b', 3.5),
                    (r'\b(?:machine|deep)\s+learning\s+(?:model|algorithm|pipeline|project)\b', 3.5),
                    (r'\b(?:classification|regression|clustering|prediction)\s+(?:model|task|problem)\b', 3.0),
                    (r'\b(?:neural|convolutional|recurrent)\s+network\b', 3.5),
                    (r'\b(?:model)\s+(?:deployment|serving|inference|monitoring|evaluation)\b', 3.0),
                    (r'\b(?:feature)\s+(?:engineering|extraction|selection|importance)\b', 3.5),
                    (r'\b(?:accuracy|precision|recall|f1|auc|roc)\s+(?:score|metric|curve)\b', 3.5),
                    (r'\b(?:overfitting|underfitting|regularization|dropout)\b', 3.5),
                    (r'\b(?:epoch|batch|gradient|optimizer|loss)\s+(?:size|descent|function)\b', 3.0),
                ],
                negative_indicators=[
                    "business intelligence",
                    "data visualization",
                    "reporting dashboard",
                ],
                min_score=2.0,
                priority=1
            ),
            
            "Testing/QA": SmartCategoryConfig(
                exact_phrases={
                    "unit test": 3.5,
                    "unit testing": 3.5,
                    "integration test": 3.5,
                    "integration testing": 3.5,
                    "end to end test": 3.5,
                    "e2e testing": 3.5,
                    "test automation": 3.5,
                    "automated testing": 3.5,
                    "test coverage": 3.5,
                    "code coverage": 3.5,
                    "test driven development": 3.5,
                    "behavior driven development": 3.5,
                    "acceptance testing": 3.0,
                    "regression testing": 3.0,
                    "performance testing": 3.0,
                    "load testing": 3.0,
                    "stress testing": 3.0,
                    "security testing": 3.0,
                    "penetration testing": 3.5,
                },
                word_patterns={
                    "jest": 3.5,
                    "mocha": 3.0,
                    "jasmine": 3.0,
                    "pytest": 3.5,
                    "unittest": 3.0,
                    "selenium": 3.5,
                    "cypress": 3.5,
                    "playwright": 3.5,
                    "puppeteer": 3.0,
                    "testng": 3.0,
                    "junit": 3.0,
                    "cucumber": 3.0,
                    "postman": 2.5,
                    "jmeter": 3.5,
                    "gatling": 3.5,
                },
                context_patterns=[
                    (r'\b(?:write|writing|create|creating)\s+(?:unit|integration|e2e|automated)\s+tests?\b', 3.5),
                    (r'\b(?:test|testing)\s+(?:strategy|plan|framework|automation|coverage)\b', 3.0),
                    (r'\b(?:bug|defect|issue)\s+(?:tracking|reporting|fixing|resolution)\b', 2.5),
                    (r'\b(?:qa|quality)\s+(?:assurance|testing|automation|process)\b', 3.0),
                    (r'\b(?:test)\s+(?:case|cases|scenario|scenarios|suite|suites)\b', 3.0),
                    (r'\b(?:mock|mocking|stub|stubbing)\s+(?:data|service|api|function)\b', 3.0),
                ],
                negative_indicators=[
                    "test environment",  # More about DevOps
                    "test data",  # Could be data engineering
                ],
                min_score=2.0,
                priority=1
            ),
        }
    
    def calculate_smart_score(self, text: str, category: SmartCategoryConfig) -> Tuple[float, Dict[str, List[str]]]:
        """Calculate score using intelligent matching"""
        score = 0.0
        matches = {
            'exact_phrases': [],
            'word_patterns': [],
            'context_patterns': [],
            'negative_matches': []
        }
        
        text_lower = text.lower()
        
        # Check for negative indicators first
        for pattern in category.compiled_negative:
            if pattern.search(text):
                matches['negative_matches'].append(pattern.pattern)
                score -= 1.0  # Penalize negative matches
        
        # Check exact phrases (highest confidence)
        for pattern, weight, phrase in category.compiled_exact:
            if pattern.search(text):
                score += weight
                matches['exact_phrases'].append(phrase)
                self.phrase_frequency[phrase] += 1
        
        # Check word patterns (medium confidence)
        for pattern, weight, word in category.compiled_words:
            if pattern.search(text):
                score += weight
                matches['word_patterns'].append(word)
        
        # Check context patterns (context-aware)
        for pattern, weight in category.compiled_context:
            if pattern.search(text):
                score += weight * 1.2  # Bonus for context matches
                matches['context_patterns'].append(pattern.pattern[:50] + "...")
        
        # Apply length normalization (but don't over-penalize)
        text_length = len(text.split())
        if text_length > 0:
            # Smoother normalization that doesn't penalize short texts as much
            score = score * (1 + np.log10(text_length)) / np.log10(max(text_length, 10))
        
        return score, matches
    
    def classify_message(self, text: str) -> Tuple[str, float, List[str], Dict[str, List[str]]]:
        """Classify a message using intelligent matching"""
        if not text or len(text.strip()) < 3:
            return "Uncategorized", 0.0, [], {}
        
        # Check cache first
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.classification_cache:
            return self.classification_cache[text_hash]
        
        # Calculate scores for all categories
        scores = {}
        all_matches = {}
        
        for cat_name, category in self.categories.items():
            score, matches = self.calculate_smart_score(text, category)
            scores[cat_name] = score
            all_matches[cat_name] = matches
        
        # Sort categories by priority and score
        sorted_categories = sorted(
            scores.items(),
            key=lambda x: (category.priority, -x[1])
        )
        
        # Get best category
        best_category, best_score = sorted_categories[0]
        
        # Check if score meets minimum threshold
        if best_score < self.categories[best_category].min_score:
            result = ("Uncategorized", best_score, [], all_matches)
            self.classification_cache[text_hash] = result
            return result
        
        # Get secondary categories (multi-label classification)
        secondary_categories = []
        for cat, score in sorted_categories[1:]:
            if score >= self.categories[cat].min_score * 0.7:  # 70% of min threshold
                secondary_categories.append(cat)
        
        result = (best_category, best_score, secondary_categories, all_matches)
        self.classification_cache[text_hash] = result
        return result
    
    def get_smart_stats(self) -> Dict:
        """Get intelligent statistics"""
        return {
            'classification_cache_size': len(self.classification_cache),
            'phrase_frequency': dict(self.phrase_frequency),
            'categories_processed': len(self.categories),
            'total_exact_phrases': sum(len(cat.exact_phrases) for cat in self.categories.values()),
            'total_word_patterns': sum(len(cat.word_patterns) for cat in self.categories.values()),
            'total_context_patterns': sum(len(cat.context_patterns) for cat in self.categories.values()),
        }

def main():
    """Main execution function"""
    analyzer = IntelligentChatbotAnalyzer(cache_dir="./intelligent_cache")
    
    # Test the intelligent classification
    test_messages = [
        "How do I deploy a Docker container to Kubernetes?",
        "I need help with React component state management",
        "What's the best way to implement JWT authentication in Spring Boot?",
        "How to create a data pipeline with Apache Spark?",
        "Can you help me write unit tests for my Python code?",
        "I want to build a machine learning model with TensorFlow",
        "How to set up CI/CD pipeline with GitHub Actions?",
        "What's the difference between REST and GraphQL APIs?",
        "I need to optimize my database queries in PostgreSQL",
        "How to implement responsive design with CSS Grid?",
    ]
    
    print("🧠 Intelligent Chatbot Analyzer Test Results:")
    print("=" * 60)
    
    for message in test_messages:
        category, score, secondary, matches = analyzer.classify_message(message)
        print(f"\n📝 Message: {message}")
        print(f"🏷️  Category: {category} (Score: {score:.2f})")
        if secondary:
            print(f"🔗 Secondary: {', '.join(secondary)}")
        
        # Show match details
        if matches.get(category):
            cat_matches = matches[category]
            if cat_matches['exact_phrases']:
                print(f"✅ Exact phrases: {', '.join(cat_matches['exact_phrases'])}")
            if cat_matches['word_patterns']:
                print(f"🔤 Word patterns: {', '.join(cat_matches['word_patterns'])}")
            if cat_matches['context_patterns']:
                print(f"🎯 Context patterns: {len(cat_matches['context_patterns'])} found")
    
    print(f"\n📊 Smart Stats: {analyzer.get_smart_stats()}")

if __name__ == "__main__":
    main() 