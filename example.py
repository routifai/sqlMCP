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
import psycopg2  # or your preferred database library

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

class EnhancedChatbotAnalyzer:
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.categories = self._initialize_categories()
        self.compiled_patterns = self._compile_patterns()
        self.session_cache = {}
        self.keyword_stats = defaultdict(int)
        self._text_cache = {}  # Cache for lowercase text
        self.classification_cache = {}  # Cache for classification results
        self.match_details_cache = {}  # Cache for match details
        
    def _initialize_categories(self) -> Dict[str, CategoryConfig]:
        """Initialize categories with weighted keywords and patterns"""
        return {
            # Technical Categories (Priority 1 - Check First)
            "DevOps/Infrastructure": CategoryConfig(
                keywords={
                    # CI/CD & Automation
                    "jenkins": 2.0, "ci/cd": 2.5, "pipeline": 1.8, "github actions": 2.2,
                    "gitlab ci": 2.0, "circleci": 2.0, "travis": 1.8, "bamboo": 1.8,
                    "teamcity": 2.0, "azure devops": 2.2, "bitbucket pipelines": 2.0,
                    "build": 1.5, "deploy": 1.8, "deployment": 2.0, "release": 1.5,
                    "automation": 2.0, "continuous integration": 2.5, "continuous deployment": 2.5,
                    
                    # Containerization & Orchestration
                    "docker": 2.5, "kubernetes": 2.5, "k8s": 2.5, "container": 1.5,
                    "helm": 2.0, "istio": 2.0, "openshift": 2.0, "rancher": 1.8,
                    "podman": 1.8, "containerd": 1.8, "docker-compose": 2.0, "dockerfile": 2.0,
                    "containerization": 2.0, "orchestration": 2.0, "microservices": 2.0,
                    "service mesh": 2.0, "ingress": 1.8, "namespace": 1.8, "pod": 1.8,
                    "service": 1.5, "configmap": 1.8, "secret": 1.8,
                    
                    # IaC & Configuration
                    "terraform": 2.5, "ansible": 2.5, "puppet": 2.0, "chef": 2.0,
                    "pulumi": 2.0, "cloudformation": 2.2, "arm template": 2.0,
                    "infrastructure as code": 2.5, "iac": 2.0, "provisioning": 2.0,
                    "configuration management": 2.0, "state management": 1.8,
                    
                    # Cloud Platforms
                    "aws": 2.0, "azure": 2.0, "gcp": 2.0, "google cloud": 2.0,
                    "ec2": 2.0, "s3": 1.8, "lambda": 1.8, "eks": 2.0, "aks": 2.0,
                    "ecs": 2.0, "fargate": 2.0, "vpc": 1.8, "subnet": 1.8,
                    "route53": 1.8, "cloudfront": 1.8, "rds": 1.8, "dynamodb": 2.0,
                    "cloud storage": 1.8, "compute": 1.5, "serverless": 2.0,
                    
                    # Monitoring & Logging
                    "prometheus": 2.0, "grafana": 2.0, "elk": 2.0, "elasticsearch": 2.0,
                    "logstash": 1.8, "kibana": 1.8, "datadog": 2.0, "new relic": 2.0,
                    "splunk": 2.0, "cloudwatch": 1.8, "stackdriver": 1.8,
                    "monitoring": 2.0, "logging": 2.0, "alerting": 1.8, "metrics": 1.8,
                    "dashboard": 1.8, "observability": 2.0, "tracing": 1.8,
                    "jaeger": 1.8, "zipkin": 1.8, "opentelemetry": 2.0,
                    
                    # Networking & Security
                    "security group": 1.8, "firewall": 1.8, "load balancer": 1.8, 
                    "alb": 1.8, "nlb": 1.8, "elb": 1.8, "ssl": 1.8, "tls": 1.8, 
                    "certificate": 1.8, "vpn": 1.8, "bastion": 1.8, "jump server": 1.8, 
                    "nat gateway": 1.8,
                },
                patterns=[
                    (r'\b(?:deploy|deployment)\s+(?:to|on|in)\s+(?:prod|production|staging)\b', 2.0),
                    (r'\b(?:scale|scaling|autoscale|autoscaling)\b', 1.5),
                    (r'\b(?:load\s*balanc(?:er|ing)|lb|alb|nlb|elb)\b', 1.8),
                    (r'\b(?:infra|infrastructure)\s+as\s+code\b', 2.5),
                    (r'\b(?:micro)?services?\s+(?:architecture|deployment)\b', 2.0),
                    (r'\b(?:container|docker|kubernetes)\s+(?:orchestration|management)\b', 2.0),
                    (r'\b(?:cloud|aws|azure|gcp)\s+(?:infrastructure|setup|configuration)\b', 2.0),
                    (r'\b(?:monitoring|logging|observability)\s+(?:setup|configuration)\b', 2.0),
                    (r'\b(?:ci/cd|continuous\s+integration|continuous\s+deployment)\s+(?:pipeline|setup)\b', 2.5),
                    (r'\b(?:terraform|ansible|puppet|chef)\s+(?:configuration|deployment)\b', 2.0),
                    (r'\b(?:kubernetes|k8s)\s+(?:cluster|deployment|service)\b', 2.0),
                    (r'\b(?:docker|container)\s+(?:build|image|registry)\b', 2.0),
                    (r'\b(?:aws|azure|gcp)\s+(?:service|resource|instance)\b', 2.0),
                    (r'\b(?:prometheus|grafana|elk|datadog)\s+(?:monitoring|dashboard)\b', 2.0),
                ],
                min_score=0.5,
                priority=1
            ),
            
            "Backend Development": CategoryConfig(
                keywords={
                    # Backend Frameworks & Technologies
                    "spring boot": 2.5, "spring framework": 2.5, "django": 2.5, "flask": 2.0, 
                    "fastapi": 2.5, "express.js": 2.5, "nestjs": 2.5, "rails": 2.5, 
                    "laravel": 2.5, "asp.net core": 2.5, "gin": 2.0, "echo": 2.0,
                    "koa": 2.0, "hapi": 2.0, "strapi": 2.5, "symfony": 2.5,
                    
                    # API Technologies
                    "rest api": 2.5, "graphql": 2.5, "grpc": 2.5, "websocket": 2.5,
                    "openapi": 2.0, "swagger": 2.0, "api gateway": 2.5,
                    
                    # Backend Architecture
                    "microservice": 2.5, "serverless": 2.5, "monolith": 2.0,
                    "clean architecture": 2.5, "hexagonal architecture": 2.5, "ddd": 2.5,
                    "event sourcing": 2.5, "cqrs": 2.5, "saga pattern": 2.5,
                    
                    # Authentication & Security
                    "jwt": 2.5, "oauth2": 2.5, "openid connect": 2.5, "saml": 2.5,
                    "bcrypt": 2.0, "passport": 2.0, "auth0": 2.5,
                    
                    # Database Technologies
                    "postgresql": 2.5, "mysql": 2.5, "mongodb": 2.5, "redis": 2.5,
                    "elasticsearch": 2.5, "cassandra": 2.5, "dynamodb": 2.5,
                    "prisma": 2.5, "sequelize": 2.0, "typeorm": 2.0, "hibernate": 2.5,
                    
                    # Backend Patterns
                    "repository pattern": 2.5, "factory pattern": 2.0, "singleton pattern": 2.0,
                    "dependency injection": 2.5, "inversion of control": 2.5,
                    "middleware": 2.0, "interceptor": 2.0, "decorator": 2.0,
                    
                    # Data Processing
                    "serialization": 2.0, "deserialization": 2.0, "marshalling": 2.0,
                    "data transfer object": 2.0, "dto": 2.0, "entity": 2.0,
                    "migration": 2.0, "schema migration": 2.5,
                },
                patterns=[
                    (r'\b(?:spring|django|flask|fastapi|express|nestjs|rails|laravel)\s+(?:backend|api|service)\b', 2.5),
                    (r'\b(?:rest|graphql|grpc)\s+(?:api|endpoint|service)\s+(?:development|design|implementation)\b', 2.5),
                    (r'\b(?:microservice|serverless)\s+(?:architecture|design|deployment|development)\b', 2.5),
                    (r'\b(?:postgresql|mysql|mongodb|redis|elasticsearch)\s+(?:database|connection|query|migration)\b', 2.5),
                    (r'\b(?:jwt|oauth|authentication)\s+(?:token|flow|system|implementation)\b', 2.5),
                    (r'\b(?:clean|hexagonal|ddd)\s+(?:architecture|design|pattern)\b', 2.5),
                    (r'\b(?:repository|factory|singleton)\s+pattern\s+(?:implementation|design)\b', 2.5),
                    (r'\b(?:dependency\s+injection|inversion\s+of\s+control)\s+(?:container|setup)\b', 2.5),
                    (r'\b(?:event\s+sourcing|cqrs|saga)\s+(?:pattern|architecture|implementation)\b', 2.5),
                    (r'\b(?:prisma|sequelize|typeorm|hibernate)\s+(?:orm|database|migration)\b', 2.5),
                    (r'\b(?:middleware|interceptor|decorator)\s+(?:implementation|setup|configuration)\b', 2.0),
                    (r'\b(?:serialization|deserialization|marshalling)\s+(?:data|object|format)\b', 2.0),
                    (r'\b(?:api\s+gateway|service\s+mesh)\s+(?:configuration|deployment)\b', 2.5),
                ],
                min_score=1.0,
                priority=1
            ),
            
            "Frontend Development": CategoryConfig(
                keywords={
                    # Frameworks & Libraries
                    "react": 2.5, "angular": 2.5, "vue": 2.5, "svelte": 2.2,
                    "nextjs": 2.5, "next.js": 2.5, "nuxt": 2.2, "gatsby": 2.0,
                    "webpack": 1.8, "vite": 2.0, "parcel": 1.8, "rollup": 1.8,
                    "remix": 2.2, "astro": 2.0, "solid": 2.2, "preact": 2.0,
                    "ember": 1.8, "backbone": 1.5, "jquery": 1.2, "lodash": 1.5,
                    
                    # Web Technologies
                    "javascript": 1.5, "typescript": 2.0, "html": 1.0, "css": 1.0,
                    "sass": 1.5, "scss": 1.5, "tailwind": 2.0, "bootstrap": 1.5,
                    "material-ui": 1.8, "mui": 1.8, "chakra": 1.8, "ant design": 1.8,
                    "styled-components": 2.0, "emotion": 1.8, "framer motion": 2.0,
                    "three.js": 2.0, "d3.js": 2.0, "chart.js": 1.8, "canvas": 1.5,
                    "webgl": 2.0, "webassembly": 2.0, "wasm": 2.0,
                    
                    # Frontend Concepts
                    "responsive": 1.5, "spa": 2.0, "pwa": 2.0, "ssr": 2.0,
                    "state management": 2.0, "redux": 2.0, "mobx": 1.8,
                    "zustand": 1.8, "recoil": 1.8, "context api": 1.8,
                    "component": 1.5, "hook": 1.8, "lifecycle": 1.8, "routing": 1.8,
                    "virtual dom": 2.0, "reconciliation": 1.8, "hydration": 1.8,
                    "code splitting": 2.0, "lazy loading": 1.8, "tree shaking": 1.8,
                    
                    # UI/UX & Design
                    "ui": 1.5, "ux": 1.5, "user interface": 1.8, "user experience": 1.8,
                    "design system": 2.0, "component library": 2.0, "design tokens": 1.8,
                    "accessibility": 1.8, "a11y": 1.8, "semantic": 1.5, "aria": 1.8,
                    "responsive design": 2.0, "mobile first": 1.8, "progressive enhancement": 1.8,
                    
                    # Performance & Optimization
                    "performance": 1.8, "optimization": 1.8, "bundle size": 1.8,
                    "lighthouse": 1.8, "core web vitals": 2.0, "lcp": 1.8, "fid": 1.8,
                    "cls": 1.8, "caching": 1.5, "cdn": 1.8, "minification": 1.5,
                    
                    # Testing & Tools
                    "jest": 2.0, "cypress": 2.2, "playwright": 2.0, "selenium": 1.8,
                    "storybook": 2.0, "chromatic": 1.8, "testing library": 1.8,
                    "eslint": 1.5, "prettier": 1.5, "husky": 1.8, "lint-staged": 1.8,
                },
                patterns=[
                    (r'\b(?:front-?end|frontend|client-?side)\s+(?:development|code|framework)\b', 2.0),
                    (r'\b(?:ui|ux|user\s+interface)\s+(?:design|development|component)\b', 1.8),
                    (r'\bcomponent\s+(?:library|design|development)\b', 1.5),
                    (r'\b(?:web|mobile)\s+app(?:lication)?\s+development\b', 1.8),
                    (r'\b(?:react|angular|vue|svelte)\s+(?:component|hook|lifecycle)\b', 2.0),
                    (r'\b(?:responsive|mobile)\s+(?:design|layout|development)\b', 1.8),
                    (r'\b(?:state|redux|mobx)\s+(?:management|store|action)\b', 2.0),
                    (r'\b(?:css|scss|sass|styled)\s+(?:component|module|framework)\b', 1.8),
                    (r'\b(?:webpack|vite|parcel)\s+(?:configuration|build|bundle)\b', 1.8),
                    (r'\b(?:performance|optimization)\s+(?:frontend|web|app)\b', 1.8),
                    (r'\b(?:accessibility|a11y)\s+(?:compliance|testing|implementation)\b', 1.8),
                    (r'\b(?:pwa|progressive\s+web\s+app)\s+(?:development|implementation)\b', 2.0),
                    (r'\b(?:ssr|server\s+side\s+rendering)\s+(?:setup|configuration)\b', 2.0),
                    (r'\b(?:testing|jest|cypress)\s+(?:frontend|component|unit)\b', 2.0),
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
                    "data processing": 2.0, "data transformation": 2.0, "data ingestion": 2.0,
                    "stream processing": 2.0, "batch processing": 2.0, "real-time processing": 2.0,
                    
                    # Databases
                    "sql": 1.5, "nosql": 1.8, "postgresql": 2.0, "mysql": 1.8,
                    "mongodb": 2.0, "cassandra": 2.0, "redis": 2.0, "elasticsearch": 2.2,
                    "dynamodb": 2.0, "cosmos db": 2.0, "neo4j": 2.0,
                    "database": 1.8, "rdbms": 1.8, "data warehouse": 2.0, "data lake": 2.0,
                    "data mart": 1.8, "data mesh": 2.0, "data fabric": 2.0,
                    
                    # Analytics & BI
                    "tableau": 2.0, "power bi": 2.0, "looker": 2.0, "metabase": 1.8,
                    "superset": 1.8, "quicksight": 1.8, "data pipeline": 2.0,
                    "analytics": 1.8, "business intelligence": 2.0, "bi": 1.5,
                    "dashboard": 1.8, "reporting": 1.8, "data visualization": 2.0,
                    "kpi": 1.8, "metrics": 1.5, "measurement": 1.5,
                    
                    # Data Science Tools
                    "pandas": 2.0, "numpy": 1.8, "jupyter": 1.8, "matplotlib": 1.5,
                    "seaborn": 1.5, "plotly": 1.8, "streamlit": 2.0, "gradio": 2.0,
                    "r": 1.8, "rstudio": 1.8, "sas": 1.8, "spss": 1.8,
                    
                    # Data Quality & Governance
                    "data quality": 2.0, "data governance": 2.0, "data lineage": 1.8,
                    "data catalog": 1.8, "data dictionary": 1.8, "metadata": 1.8,
                    "data validation": 1.8, "data profiling": 1.8, "data cleansing": 1.8,
                    
                    # Big Data & Cloud
                    "big data": 2.0, "distributed computing": 2.0, "mapreduce": 1.8,
                    "hive": 1.8, "hbase": 1.8, "zookeeper": 1.8, "yarn": 1.8,
                    "cloud data": 1.8, "data lakehouse": 2.0, "delta lake": 2.0,
                    "iceberg": 1.8, "hudi": 1.8,
                },
                patterns=[
                    (r'\bdata\s+(?:pipeline|warehouse|lake|mart|mesh)\b', 2.2),
                    (r'\b(?:batch|stream|real-?time)\s+processing\b', 2.0),
                    (r'\b(?:data|database)\s+(?:migration|replication|sync)\b', 1.8),
                    (r'\bquery\s+(?:optimization|performance|tuning)\b', 2.0),
                    (r'\b(?:etl|elt)\s+(?:pipeline|process|workflow)\b', 2.0),
                    (r'\b(?:spark|hadoop|kafka)\s+(?:configuration|setup|deployment)\b', 2.0),
                    (r'\b(?:data|analytics)\s+(?:platform|tool|solution)\b', 2.0),
                    (r'\b(?:business\s+intelligence|bi)\s+(?:dashboard|report|analysis)\b', 2.0),
                    (r'\b(?:data\s+quality|governance)\s+(?:management|framework)\b', 2.0),
                    (r'\b(?:big\s+data|distributed)\s+(?:processing|computing)\b', 2.0),
                    (r'\b(?:data\s+visualization|dashboard)\s+(?:creation|design)\b', 1.8),
                    (r'\b(?:sql|query)\s+(?:optimization|performance|tuning)\b', 2.0),
                    (r'\b(?:data\s+warehouse|lake)\s+(?:design|architecture)\b', 2.0),
                    (r'\b(?:stream|batch)\s+(?:processing|ingestion|transformation)\b', 2.0),
                ],
                min_score=0.5,
                priority=1
            ),
            
            "AI/ML & Data Science": CategoryConfig(
                keywords={
                    # ML Frameworks
                    "tensorflow": 2.5, "pytorch": 2.5, "scikit-learn": 2.2, "sklearn": 2.2,
                    "keras": 2.0, "xgboost": 2.0, "lightgbm": 2.0, "catboost": 2.0,
                    "hugging face": 2.5, "transformers": 2.5, "spacy": 2.0, "nltk": 1.8,
                    "opencv": 2.0, "pillow": 1.5, "scipy": 1.8, "statsmodels": 1.8,
                    
                    # ML Concepts
                    "machine learning": 2.5, "deep learning": 2.5, "neural network": 2.5,
                    "nlp": 2.2, "computer vision": 2.2, "reinforcement learning": 2.2,
                    "transformer": 2.0, "bert": 2.0, "gpt": 2.0, "llm": 2.5,
                    "supervised learning": 2.0, "unsupervised learning": 2.0, "semi-supervised": 2.0,
                    "transfer learning": 2.0, "ensemble": 1.8, "gradient boosting": 2.0,
                    "random forest": 1.8, "svm": 1.8, "k-means": 1.8, "clustering": 1.8,
                    
                    # AI/ML Applications
                    "natural language processing": 2.2, "speech recognition": 2.0,
                    "recommendation system": 2.0, "chatbot": 2.0, "virtual assistant": 2.0,
                    "autonomous vehicle": 2.0, "robotics": 2.0, "expert system": 1.8,
                    "knowledge graph": 2.0, "semantic analysis": 2.0, "sentiment analysis": 2.0,
                    
                    # Data Science Tools
                    "pandas": 2.0, "numpy": 1.8, "jupyter": 1.8, "matplotlib": 1.5,
                    "seaborn": 1.5, "plotly": 1.8, "streamlit": 2.0, "gradio": 2.0,
                    "r": 1.8, "rstudio": 1.8, "sas": 1.8, "spss": 1.8,
                    "weka": 1.8, "rapidminer": 1.8, "knime": 1.8,
                    
                    # MLOps
                    "mlflow": 2.2, "kubeflow": 2.2, "sagemaker": 2.2, "vertex ai": 2.2,
                    "wandb": 2.0, "dvc": 2.0, "feast": 2.0, "mlops": 2.0,
                    "model serving": 2.0, "model deployment": 2.0, "model monitoring": 2.0,
                    "feature store": 2.0, "model registry": 2.0, "experiment tracking": 2.0,
                    
                    # Advanced ML
                    "generative ai": 2.5, "large language model": 2.5, "foundation model": 2.5,
                    "multimodal": 2.0, "object detection": 2.0,
                    "image classification": 2.0, "semantic segmentation": 2.0, "instance segmentation": 2.0,
                    "time series": 1.8, "forecasting": 1.8, "anomaly detection": 2.0,
                    "q-learning": 2.0, "policy gradient": 2.0,
                    
                    # Data Science Process
                    "data science": 2.0, "statistical analysis": 1.8, "hypothesis testing": 1.8,
                    "a/b testing": 1.8, "experiment design": 1.8, "feature engineering": 2.0,
                    "data preprocessing": 1.8, "data cleaning": 1.8, "data wrangling": 1.8,
                    "exploratory data analysis": 2.0, "eda": 1.8, "data visualization": 1.8,
                },
                patterns=[
                    (r'\b(?:train|training|fine-?tun(?:e|ing))\s+(?:a\s+)?model\b', 2.2),
                    (r'\b(?:model|algorithm)\s+(?:deployment|serving|inference)\b', 2.0),
                    (r'\b(?:feature|data)\s+engineering\b', 2.0),
                    (r'\b(?:classification|regression|clustering|prediction)\s+(?:model|algorithm)\b', 2.0),
                    (r'\b(?:machine\s+learning|deep\s+learning)\s+(?:model|algorithm|system)\b', 2.5),
                    (r'\b(?:natural\s+language\s+processing|nlp)\s+(?:model|system|application)\b', 2.2),
                    (r'\b(?:computer\s+vision|image\s+processing)\s+(?:model|system|application)\b', 2.2),
                    (r'\b(?:data\s+science|ml|ai)\s+(?:project|experiment|analysis)\b', 2.0),
                    (r'\b(?:model|algorithm)\s+(?:evaluation|validation|testing)\b', 2.0),
                    (r'\b(?:feature|data)\s+(?:selection|extraction|transformation)\b', 2.0),
                    (r'\b(?:mlops|machine\s+learning\s+operations)\s+(?:pipeline|workflow)\b', 2.0),
                    (r'\b(?:experiment|trial)\s+(?:tracking|management|monitoring)\b', 1.8),
                    (r'\b(?:generative\s+ai|llm|large\s+language\s+model)\s+(?:development|deployment)\b', 2.5),
                    (r'\b(?:reinforcement\s+learning|rl)\s+(?:agent|environment|policy)\b', 2.2),
                    (r'\b(?:time\s+series|forecasting)\s+(?:analysis|prediction|model)\b', 1.8),
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
    
    def _get_lowercase_text(self, text: str) -> str:
        """Cache lowercase conversions for performance"""
        if text not in self._text_cache:
            self._text_cache[text] = text.lower()
        return self._text_cache[text]
    
    def calculate_category_score(self, text: str, category_name: str) -> Tuple[float, List[str], List[str]]:
        """Calculate weighted score for a category based on text with match details"""
        text_lower = self._get_lowercase_text(text)  # FIX: Use cached lowercase
        score = 0.0
        matched_keywords = []
        matched_patterns = []
        
        category = self.categories[category_name]
        
        # Check keywords with weights
        for keyword, weight in category.keywords.items():
            if keyword in text_lower:
                score += weight
                matched_keywords.append(keyword)
                self.keyword_stats[keyword] += 1
        
        # Check regex patterns
        for pattern, weight in self.compiled_patterns[category_name]:
            if pattern.search(text):
                score += weight
                matched_patterns.append(pattern.pattern)
        
        # Apply length normalization (FIX: Improved normalization)
        text_length = len(text.split())
        if text_length > 0:
            # Use smoother normalization that doesn't penalize short texts as much
            score = score * (1 + np.log10(text_length)) / np.log10(max(text_length, 10))
        
        return score, matched_keywords, matched_patterns
    
    def classify_message(self, text: str) -> Tuple[str, float, List[str], Dict[str, List[str]]]:
        """Classify a message into categories with confidence score and match details"""
        if not text or len(text.strip()) < 3:
            return "Uncategorized", 0.0, [], {}
        
        # Check cache first
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.classification_cache:
            return self.classification_cache[text_hash]
        
        # Memory management: Clear caches if they get too large
        if len(self._text_cache) > 10000:
            self._text_cache.clear()
        if len(self.classification_cache) > 5000:
            self.classification_cache.clear()
        
        # Calculate scores for all categories with match details
        scores = {}
        match_details = {}
        
        for cat_name in self.categories:
            score, matched_keywords, matched_patterns = self.calculate_category_score(text, cat_name)
            scores[cat_name] = score
            match_details[cat_name] = {
                'keywords': matched_keywords,
                'patterns': matched_patterns
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
            result = ("Uncategorized", best_score, [], match_details)
            self.classification_cache[text_hash] = result
            return result
        
        # Get secondary categories (multi-label classification)
        secondary_categories = []
        for cat, score in sorted_categories[1:]:
            if score >= self.categories[cat].min_score * 0.7:  # 70% of min threshold
                secondary_categories.append(cat)
        
        result = (best_category, best_score, secondary_categories, match_details)
        self.classification_cache[text_hash] = result
        return result
    
    def classify_messages_batch(self, texts: List[str]) -> List[Tuple[str, float, List[str], Dict[str, List[str]]]]:
        """Batch classify messages for better performance"""
        results = []
        for text in texts:
            result = self.classify_message(text)
            results.append(result)
        return results
    
    def analyze_session_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze user session patterns"""
        session_stats = {
            'total_sessions': int(df['session_id'].nunique()),  # FIX: Ensure JSON serializable
            'avg_messages_per_session': float(df.groupby('session_id').size().mean()),
            'session_category_transitions': {},
            'user_journey_patterns': {}
        }
        
        # Analyze category transitions within sessions
        transition_counts = defaultdict(int)
        for session_id, group in df.groupby('session_id'):
            if len(group) > 1:
                categories = group['primary_category'].tolist()
                for i in range(len(categories) - 1):
                    transition = f"{categories[i]} → {categories[i+1]}"
                    transition_counts[transition] += 1
        
        session_stats['session_category_transitions'] = dict(transition_counts)
        
        # Find common user journeys
        session_journeys = df.groupby('session_id')['primary_category'].apply(
            lambda x: ' → '.join(x) if len(x) > 1 else x.iloc[0]
        ).value_counts().head(10)
        
        session_stats['user_journey_patterns'] = {k: int(v) for k, v in session_journeys.items()}
        
        return session_stats
    
    def detect_emerging_topics(self, df: pd.DataFrame) -> Dict:
        """Detect emerging topics using time-based analysis"""
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        
        # Calculate keyword frequency over time
        keyword_timeline = defaultdict(lambda: defaultdict(int))
        
        for _, row in df.iterrows():
            text_lower = self._get_lowercase_text(str(row['input']))
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
                
                if older_avg > 0 and recent_avg > older_avg * 1.5:  # FIX: Check for zero
                    trending_keywords.append({
                        'keyword': keyword,
                        'growth_rate': float((recent_avg - older_avg) / older_avg),
                        'recent_frequency': float(recent_avg)
                    })
        
        trending_keywords.sort(key=lambda x: x['growth_rate'], reverse=True)
        
        # FIX: Convert timeline dates to strings
        keyword_timeline_serializable = {
            keyword: {str(date): count for date, count in dates.items()}
            for keyword, dates in keyword_timeline.items()
        }
        
        return {
            'trending_topics': trending_keywords[:20],
            'keyword_timeline': keyword_timeline_serializable
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
        
        if total == 0:  # FIX: Handle empty dataframe
            return insights
        
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
            
            # FIX: Use vectorized approach for better performance
            chunk_results = self._process_chunk_vectorized(chunk)
            all_results.append(chunk_results)
            total_processed += len(chunk)
            
            # Clear text cache periodically to manage memory
            if len(self._text_cache) > 10000:
                self._text_cache.clear()
            
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
            'keyword_statistics': dict(Counter(self.keyword_stats).most_common(50))
        }
        
        # Save detailed results
        self._save_results(analysis_results, df_results)
        
        # Save classification cache for future use
        self.save_classification_cache()
        
        return analysis_results
    
    def _process_chunk_vectorized(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Process chunk with better performance using apply"""
        # FIX: More efficient processing
        def classify_row(row):
            primary_cat, confidence, secondary_cats, match_details = self.classify_message(row['input'])
            
            # Get the best category's match details
            best_matches = match_details.get(primary_cat, {})
            matched_keywords = best_matches.get('keywords', [])
            matched_patterns = best_matches.get('patterns', [])
            
            # Create match summary
            match_summary = []
            if matched_keywords:
                match_summary.append(f"Keywords: {', '.join(matched_keywords)}")
            if matched_patterns:
                match_summary.append(f"Patterns: {', '.join(matched_patterns)}")
            
            return pd.Series({
                'session_id': row['session_id'],
                'message_id': row['message_id'],
                'timestamp': row['timestamp'],
                'input': row['input'],
                'primary_category': primary_cat,
                'confidence_score': confidence,
                'secondary_categories': secondary_cats,
                'matched_keywords': '; '.join(matched_keywords) if matched_keywords else '',
                'matched_patterns': '; '.join(matched_patterns) if matched_patterns else '',
                'match_summary': ' | '.join(match_summary) if match_summary else 'No matches'
            })
        
        return chunk.apply(classify_row, axis=1)
    
    def _calculate_category_distribution(self, df: pd.DataFrame) -> Dict:
        """Calculate detailed category distribution"""
        primary_dist = df['primary_category'].value_counts()
        
        # Calculate with percentages
        total = len(df)
        distribution = {}
        
        if total == 0:  # FIX: Handle empty dataframe
            return distribution
            
        for category, count in primary_dist.items():
            distribution[category] = {
                'count': int(count),
                'percentage': round((count / total) * 100, 2),
                'avg_confidence': round(
                    df[df['primary_category'] == category]['confidence_score'].mean(), 3
                )
            }
        
        return distribution
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics for the analyzer"""
        return {
            'text_cache_size': len(self._text_cache),
            'classification_cache_size': len(self.classification_cache),
            'session_cache_size': len(self.session_cache),
            'keyword_stats': dict(self.keyword_stats),
            'categories_processed': len(self.categories),
            'total_keywords': sum(len(cat.keywords) for cat in self.categories.values()),
            'total_patterns': sum(len(cat.patterns) for cat in self.categories.values()),
            'keywords_hit_count': sum(self.keyword_stats.values()),
            'cache_hit_rate': f"{(len(self.classification_cache) / max(sum(self.keyword_stats.values()), 1)) * 100:.1f}%"
        }
    
    def _calculate_tech_vs_business_split(self, df: pd.DataFrame) -> Dict:
        """Calculate technical vs business split"""
        tech_categories = {
            'DevOps/Infrastructure', 'Backend Development', 'Frontend Development',
            'Data Engineering & Analytics', 'AI/ML & Data Science', 
            'Security & Compliance', 'Testing & QA', 'Documentation & Knowledge'
        }
        
        tech_messages = df[df['primary_category'].isin(tech_categories)]
        business_messages = df[~df['primary_category'].isin(tech_categories)]
        
        total = len(df)
        if total == 0:  # FIX: Handle empty dataframe
            return {
                'technical': {'count': 0, 'percentage': 0, 'categories': {}},
                'business': {'count': 0, 'percentage': 0, 'categories': {}}
            }
        
        return {
            'technical': {
                'count': len(tech_messages),
                'percentage': round((len(tech_messages) / total) * 100, 2),
                'categories': tech_messages['primary_category'].value_counts().to_dict()
            },
            'business': {
                'count': len(business_messages),
                'percentage': round((len(business_messages) / total) * 100, 2),
                'categories': business_messages['primary_category'].value_counts().to_dict()
            }
        }
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze temporal patterns in usage"""
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()
        df['date'] = df['timestamp'].dt.date
        
        # FIX: Proper day ordering
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_dist = df.groupby('day_of_week')['message_id'].count()
        daily_dist_ordered = {day: daily_dist.get(day, 0) for day in day_order}
        
        # FIX: Convert dates to strings for JSON serialization
        daily_trend = df.groupby('date')['message_id'].count()
        daily_trend_serializable = {str(date): int(count) for date, count in daily_trend.items()}
        
        return {
            'hourly_distribution': df.groupby('hour')['message_id'].count().to_dict(),
            'daily_distribution': daily_dist_ordered,
            'daily_trend': daily_trend_serializable,
            'category_by_hour': df.groupby(['hour', 'primary_category']).size().unstack(fill_value=0).to_dict(),
            'peak_hours': df.groupby('hour')['message_id'].count().nlargest(3).to_dict(),
            'peak_days': {day: int(count) for day, count in 
                         df.groupby('day_of_week')['message_id'].count().nlargest(3).items()}
        }
    
    def _save_checkpoint(self, results: List[pd.DataFrame], chunk_num: int):
        """Save intermediate results for recovery"""
        checkpoint_file = self.cache_dir / f"checkpoint_{chunk_num}.pkl"
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(results, f)
        logging.info(f"Checkpoint saved: {checkpoint_file}")
    
    def save_classification_cache(self, filename: str = "classification_cache.pkl"):
        """Save classification cache for reuse"""
        cache_file = self.cache_dir / filename
        cache_data = {
            'classification_cache': self.classification_cache,
            'keyword_stats': dict(self.keyword_stats),
            'timestamp': datetime.now().isoformat()
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        logging.info(f"Classification cache saved: {cache_file}")
    
    def load_classification_cache(self, filename: str = "classification_cache.pkl"):
        """Load classification cache from file"""
        cache_file = self.cache_dir / filename
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            self.classification_cache = cache_data.get('classification_cache', {})
            # Update keyword stats
            for keyword, count in cache_data.get('keyword_stats', {}).items():
                self.keyword_stats[keyword] = count
            logging.info(f"Classification cache loaded: {cache_file} ({len(self.classification_cache)} entries)")
        else:
            logging.info(f"No existing cache found: {cache_file}")
    
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
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }}
                h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
                h2 {{ color: #555; margin-top: 30px; }}
                .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
                .metric-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; }}
                .metric-value {{ font-size: 2em; font-weight: bold; }}
                .metric-label {{ opacity: 0.9; margin-top: 5px; }}
                .chart {{ margin: 20px 0; }}
                .insight-box {{ background: #f0f8ff; border-left: 4px solid #2196F3; padding: 15px; margin: 10px 0; }}
                .alert-box {{ background: #fff3cd; border-left: 4px solid #ff9800; padding: 15px; margin: 10px 0; }}
                .table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                .table th, .table td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                .table th {{ background: #4CAF50; color: white; }}
                .table tr:hover {{ background: #f5f5f5; }}
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
5. **Regular Analysis**: Schedule monthly analysis to track usage evolution and identify new patterns

---
*This report provides data-driven insights into chatbot usage patterns. For detailed analysis, refer to the full HTML report.*
"""
        
        return summary


# Usage Example
def get_db_connection():
    """FIX: Add actual database connection function"""
    try:
        # Example PostgreSQL connection
        conn = psycopg2.connect(
            host="localhost",
            database="your_database",
            user="your_user",
            password="your_password"
        )
        return conn
    except Exception as e:
        logging.error(f"Database connection failed: {e}")
        return None


def main():
    """Main execution function"""
    # Initialize analyzer
    analyzer = EnhancedChatbotAnalyzer(cache_dir="./analysis_cache")
    
    # Load existing classification cache if available
    analyzer.load_classification_cache()
    
    # Connect to database
    conn = get_db_connection()
    
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