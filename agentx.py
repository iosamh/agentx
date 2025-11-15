"""
🏥 MEDIBOT - Advanced Medical Training Assistant System
=======================================================
A production-ready AI-powered medical training assistant with Google Sheets integration
Features NLP classification, session management, analytics, and comprehensive training modules
Version: 2.0.0 - Production Ready
"""

import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import uuid
import json
import hashlib
import re
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import random
from collections import defaultdict, Counter

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import BaseMessage

# For advanced NLP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data (one-time setup)
try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

# Page Configuration - MUST BE FIRST
st.set_page_config(
    page_title="MediBot - Medical Training Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "MediBot v2.0 - Advanced Medical Training Assistant"
    }
)

# System Constants
SYSTEM_NAME = "MediBot"
VERSION = "2.0.0"
MAX_RETRIES = 3
TIMEOUT_SECONDS = 30
SESSION_TIMEOUT_MINUTES = 30
MAX_MESSAGES_PER_SESSION = 100
CONFIDENCE_THRESHOLD = 0.75
ESCALATION_THRESHOLD = 0.5

# Google Sheets Configuration
SHEETS_CONFIG = {
    "main_sheet": "medibot_sessions",
    "chat_sheet": "chat_history",
    "analytics_sheet": "analytics",
    "knowledge_sheet": "knowledge_base",
    "user_sheet": "user_profiles",
    "quiz_sheet": "quiz_results",
    "feedback_sheet": "feedback"
}

# Training Modules
class TrainingModule(Enum):
    ANATOMY = "Anatomy & Physiology"
    PHARMACOLOGY = "Pharmacology"
    CLINICAL_SKILLS = "Clinical Skills"
    PATIENT_CARE = "Patient Care"
    EMERGENCY = "Emergency Medicine"
    DIAGNOSIS = "Diagnosis & Treatment"
    ETHICS = "Medical Ethics"
    RESEARCH = "Medical Research"

# Medical Specialties
class MedicalSpecialty(Enum):
    GENERAL = "General Medicine"
    CARDIOLOGY = "Cardiology"
    NEUROLOGY = "Neurology"
    PEDIATRICS = "Pediatrics"
    SURGERY = "Surgery"
    PSYCHIATRY = "Psychiatry"
    EMERGENCY = "Emergency Medicine"
    RADIOLOGY = "Radiology"

# Intent Categories for Medical Training
class MedicalIntent(Enum):
    CONCEPT_EXPLANATION = "concept_explanation"
    CASE_STUDY = "case_study"
    DIAGNOSIS_HELP = "diagnosis_help"
    TREATMENT_GUIDANCE = "treatment_guidance"
    DRUG_INFORMATION = "drug_information"
    PROCEDURE_STEPS = "procedure_steps"
    QUIZ_REQUEST = "quiz_request"
    ANATOMY_QUERY = "anatomy_query"
    LAB_INTERPRETATION = "lab_interpretation"
    EMERGENCY_PROTOCOL = "emergency_protocol"
    RESEARCH_QUERY = "research_query"
    ETHICAL_SCENARIO = "ethical_scenario"

# Difficulty Levels
class DifficultyLevel(Enum):
    BEGINNER = "Beginner"
    INTERMEDIATE = "Intermediate"
    ADVANCED = "Advanced"
    EXPERT = "Expert"

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class UserProfile:
    """User profile for personalized learning"""
    user_id: str
    name: str
    email: str
    role: str  # student, resident, physician, nurse
    specialty: str
    experience_level: str
    total_sessions: int = 0
    total_messages: int = 0
    quiz_scores: List[float] = field(default_factory=list)
    completed_modules: List[str] = field(default_factory=list)
    weak_areas: List[str] = field(default_factory=list)
    strong_areas: List[str] = field(default_factory=list)
    last_active: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)
    preferences: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self):
        """Convert to dictionary for storage"""
        return {
            "user_id": self.user_id,
            "name": self.name,
            "email": self.email,
            "role": self.role,
            "specialty": self.specialty,
            "experience_level": self.experience_level,
            "total_sessions": self.total_sessions,
            "total_messages": self.total_messages,
            "avg_quiz_score": np.mean(self.quiz_scores) if self.quiz_scores else 0,
            "completed_modules": json.dumps(self.completed_modules),
            "weak_areas": json.dumps(self.weak_areas),
            "strong_areas": json.dumps(self.strong_areas),
            "last_active": self.last_active.isoformat(),
            "created_at": self.created_at.isoformat(),
            "preferences": json.dumps(self.preferences)
        }

@dataclass
class ChatSession:
    """Chat session information"""
    session_id: str
    user_id: str
    module: str
    specialty: str
    difficulty: str
    start_time: datetime
    end_time: Optional[datetime] = None
    messages: List[Dict] = field(default_factory=list)
    intents_detected: List[str] = field(default_factory=list)
    topics_covered: List[str] = field(default_factory=list)
    quiz_results: List[Dict] = field(default_factory=list)
    notes: str = ""
    attempts: int = 0
    satisfaction_score: Optional[float] = None
    
    def to_dict(self):
        """Convert to dictionary for Google Sheets"""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "module": self.module,
            "specialty": self.specialty,
            "difficulty": self.difficulty,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "message_count": len(self.messages),
            "chat_history": json.dumps(self.messages),
            "intents": json.dumps(self.intents_detected),
            "topics": json.dumps(self.topics_covered),
            "quiz_results": json.dumps(self.quiz_results),
            "notes": self.notes,
            "attempts": self.attempts,
            "satisfaction_score": self.satisfaction_score
        }

@dataclass
class MedicalCase:
    """Medical case study structure"""
    case_id: str
    title: str
    specialty: str
    difficulty: str
    patient_info: Dict[str, Any]
    symptoms: List[str]
    vital_signs: Dict[str, Any]
    lab_results: Dict[str, Any]
    imaging: Optional[str]
    diagnosis: str
    treatment_plan: List[str]
    learning_points: List[str]
    
@dataclass
class Analytics:
    """Session analytics"""
    total_sessions: int = 0
    avg_session_duration: float = 0
    total_messages: int = 0
    avg_messages_per_session: float = 0
    intent_distribution: Dict[str, int] = field(default_factory=dict)
    module_usage: Dict[str, int] = field(default_factory=dict)
    sentiment_scores: List[float] = field(default_factory=list)
    confidence_scores: List[float] = field(default_factory=list)
    quiz_performance: Dict[str, float] = field(default_factory=dict)
    peak_usage_hours: List[int] = field(default_factory=list)
    user_retention_rate: float = 0
    feature_usage: Dict[str, int] = field(default_factory=dict)

# ============================================================================
# NLP CLASSIFICATION SYSTEM
# ============================================================================

class MedicalNLPClassifier:
    """Advanced NLP system for medical intent classification"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 3))
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Medical keywords for intent detection
        self.intent_keywords = {
            MedicalIntent.CONCEPT_EXPLANATION: [
                "what is", "explain", "define", "meaning", "tell me about",
                "how does", "mechanism", "pathophysiology", "etiology"
            ],
            MedicalIntent.CASE_STUDY: [
                "case", "patient", "scenario", "clinical", "presentation",
                "history", "examination", "findings"
            ],
            MedicalIntent.DIAGNOSIS_HELP: [
                "diagnose", "differential", "symptoms", "signs", "tests",
                "workup", "investigation", "rule out"
            ],
            MedicalIntent.TREATMENT_GUIDANCE: [
                "treatment", "therapy", "management", "medication", "dose",
                "intervention", "protocol", "guidelines"
            ],
            MedicalIntent.DRUG_INFORMATION: [
                "drug", "medication", "dose", "contraindication", "interaction",
                "side effect", "pharmacology", "prescription"
            ],
            MedicalIntent.PROCEDURE_STEPS: [
                "procedure", "steps", "technique", "how to", "perform",
                "protocol", "method", "approach"
            ],
            MedicalIntent.QUIZ_REQUEST: [
                "quiz", "test", "question", "assess", "evaluate",
                "practice", "exam", "mcq"
            ],
            MedicalIntent.ANATOMY_QUERY: [
                "anatomy", "structure", "location", "organ", "system",
                "physiology", "function", "innervation"
            ],
            MedicalIntent.LAB_INTERPRETATION: [
                "lab", "result", "value", "normal", "abnormal",
                "interpret", "reading", "range"
            ],
            MedicalIntent.EMERGENCY_PROTOCOL: [
                "emergency", "acute", "urgent", "critical", "resuscitation",
                "trauma", "shock", "arrest"
            ]
        }
        
        # Medical entity patterns
        self.medical_entities = {
            "symptoms": r"\b(pain|fever|cough|fatigue|nausea|vomiting|dizziness|weakness|headache|dyspnea)\b",
            "conditions": r"\b(diabetes|hypertension|pneumonia|asthma|cancer|infection|syndrome|disease)\b",
            "medications": r"\b(aspirin|ibuprofen|antibiotic|insulin|statin|beta-blocker|ace inhibitor)\b",
            "procedures": r"\b(surgery|biopsy|endoscopy|mri|ct scan|x-ray|ultrasound|ecg|blood test)\b",
            "anatomy": r"\b(heart|lung|liver|kidney|brain|bone|muscle|nerve|artery|vein)\b"
        }
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for NLP analysis"""
        # Convert to lowercase
        text = text.lower()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) 
                 for token in tokens 
                 if token not in self.stop_words and token.isalnum()]
        
        return ' '.join(tokens)
    
    def classify_intent(self, text: str) -> Tuple[MedicalIntent, float, Dict[str, Any]]:
        """Classify medical intent with confidence score"""
        text_lower = text.lower()
        preprocessed = self.preprocess_text(text)
        
        # Score each intent
        intent_scores = {}
        for intent, keywords in self.intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            intent_scores[intent] = score / len(keywords) if keywords else 0
        
        # Get best intent
        best_intent = max(intent_scores, key=intent_scores.get)
        confidence = min(1.0, intent_scores[best_intent] * 2)
        
        # Extract entities
        entities = self.extract_medical_entities(text)
        
        # Additional context
        context = {
            "entities": entities,
            "preprocessed_text": preprocessed,
            "all_scores": intent_scores,
            "word_count": len(text.split()),
            "complexity": self.assess_complexity(text)
        }
        
        return best_intent, confidence, context
    
    def extract_medical_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract medical entities from text"""
        entities = {}
        
        for entity_type, pattern in self.medical_entities.items():
            matches = re.findall(pattern, text.lower())
            if matches:
                entities[entity_type] = list(set(matches))
        
        return entities
    
    def assess_complexity(self, text: str) -> str:
        """Assess query complexity"""
        word_count = len(text.split())
        entity_count = sum(len(v) for v in self.extract_medical_entities(text).values())
        
        if word_count < 10 and entity_count <= 1:
            return "simple"
        elif word_count < 25 and entity_count <= 3:
            return "moderate"
        else:
            return "complex"
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between texts"""
        try:
            # Preprocess texts
            texts = [self.preprocess_text(text1), self.preprocess_text(text2)]
            
            # Create TF-IDF vectors
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return similarity
        except:
            return 0.0

# ============================================================================
# SENTIMENT ANALYZER
# ============================================================================

class MedicalSentimentAnalyzer:
    """Sentiment analysis for medical training context"""
    
    POSITIVE_INDICATORS = [
        "understand", "clear", "helpful", "thanks", "great", "excellent",
        "good", "perfect", "awesome", "useful", "informative"
    ]
    
    NEGATIVE_INDICATORS = [
        "confused", "difficult", "hard", "unclear", "complex", "lost",
        "don't understand", "complicated", "frustrated", "wrong"
    ]
    
    NEUTRAL_INDICATORS = [
        "okay", "fine", "alright", "next", "continue", "more"
    ]
    
    @classmethod
    def analyze(cls, text: str) -> Tuple[str, float]:
        """Analyze sentiment with score"""
        text_lower = text.lower()
        
        # Count indicators
        positive = sum(1 for word in cls.POSITIVE_INDICATORS if word in text_lower)
        negative = sum(1 for word in cls.NEGATIVE_INDICATORS if word in text_lower)
        neutral = sum(1 for word in cls.NEUTRAL_INDICATORS if word in text_lower)
        
        # Calculate sentiment
        total = positive + negative + neutral
        if total == 0:
            return "neutral", 0.0
        
        if positive > negative:
            score = positive / (positive + negative) if (positive + negative) > 0 else 0.5
            if score >= 0.7:
                return "very_positive", score
            else:
                return "positive", score
        elif negative > positive:
            score = -negative / (positive + negative) if (positive + negative) > 0 else -0.5
            if score <= -0.7:
                return "very_negative", score
            else:
                return "negative", score
        else:
            return "neutral", 0.0

# ============================================================================
# GOOGLE SHEETS DATABASE MANAGER
# ============================================================================

class GoogleSheetsManager:
    """Manages all Google Sheets operations"""
    
    def __init__(self, connection: GSheetsConnection):
        self.conn = connection
        self.cache = {}
        self.last_sync = {}
        
    def read_sheet(self, worksheet_name: str, use_cache: bool = True) -> pd.DataFrame:
        """Read data from Google Sheets with caching"""
        try:
            # Check cache
            if use_cache and worksheet_name in self.cache:
                last_sync = self.last_sync.get(worksheet_name, datetime.min)
                if datetime.now() - last_sync < timedelta(minutes=5):
                    return self.cache[worksheet_name]
            
            # Read from sheets
            df = self.conn.read(worksheet=worksheet_name)
            
            # Update cache
            self.cache[worksheet_name] = df
            self.last_sync[worksheet_name] = datetime.now()
            
            return df
        except Exception as e:
            st.error(f"Error reading {worksheet_name}: {e}")
            return pd.DataFrame()
    
    def append_to_sheet(self, worksheet_name: str, data: Union[Dict, List[Dict]], create_if_missing: bool = True) -> bool:
        """Append data to Google Sheets"""
        try:
            # Convert to DataFrame
            if isinstance(data, dict):
                data = [data]
            new_df = pd.DataFrame(data)
            
            # Read existing data
            try:
                existing_df = self.read_sheet(worksheet_name, use_cache=False)
            except:
                if create_if_missing:
                    existing_df = pd.DataFrame()
                else:
                    return False
            
            # Combine data
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            
            # Update sheet
            self.conn.update(worksheet=worksheet_name, data=combined_df)
            
            # Clear cache for this sheet
            if worksheet_name in self.cache:
                del self.cache[worksheet_name]
            
            return True
        except Exception as e:
            st.error(f"Error appending to {worksheet_name}: {e}")
            return False
    
    def update_sheet(self, worksheet_name: str, data: pd.DataFrame) -> bool:
        """Update entire sheet"""
        try:
            self.conn.update(worksheet=worksheet_name, data=data)
            
            # Clear cache
            if worksheet_name in self.cache:
                del self.cache[worksheet_name]
            
            return True
        except Exception as e:
            st.error(f"Error updating {worksheet_name}: {e}")
            return False
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile from sheets"""
        try:
            df = self.read_sheet(SHEETS_CONFIG["user_sheet"])
            
            if not df.empty and "user_id" in df.columns:
                user_data = df[df["user_id"] == user_id]
                if not user_data.empty:
                    row = user_data.iloc[0].to_dict()
                    return UserProfile(
                        user_id=row["user_id"],
                        name=row["name"],
                        email=row["email"],
                        role=row["role"],
                        specialty=row["specialty"],
                        experience_level=row["experience_level"],
                        total_sessions=int(row.get("total_sessions", 0)),
                        total_messages=int(row.get("total_messages", 0)),
                        quiz_scores=json.loads(row.get("quiz_scores", "[]")),
                        completed_modules=json.loads(row.get("completed_modules", "[]")),
                        weak_areas=json.loads(row.get("weak_areas", "[]")),
                        strong_areas=json.loads(row.get("strong_areas", "[]")),
                        last_active=datetime.fromisoformat(row.get("last_active", datetime.now().isoformat())),
                        created_at=datetime.fromisoformat(row.get("created_at", datetime.now().isoformat())),
                        preferences=json.loads(row.get("preferences", "{}"))
                    )
            return None
        except Exception as e:
            st.error(f"Error getting user profile: {e}")
            return None
    
    def save_user_profile(self, profile: UserProfile) -> bool:
        """Save or update user profile"""
        try:
            df = self.read_sheet(SHEETS_CONFIG["user_sheet"])
            profile_dict = profile.to_dict()
            
            if not df.empty and "user_id" in df.columns:
                # Update existing
                mask = df["user_id"] == profile.user_id
                if mask.any():
                    for key, value in profile_dict.items():
                        df.loc[mask, key] = value
                else:
                    # Add new
                    df = pd.concat([df, pd.DataFrame([profile_dict])], ignore_index=True)
            else:
                # Create new
                df = pd.DataFrame([profile_dict])
            
            return self.update_sheet(SHEETS_CONFIG["user_sheet"], df)
        except Exception as e:
            st.error(f"Error saving user profile: {e}")
            return False
    
    def save_session(self, session: ChatSession) -> bool:
        """Save chat session to sheets"""
        try:
            # Save to main session sheet
            session_dict = session.to_dict()
            success = self.append_to_sheet(SHEETS_CONFIG["main_sheet"], session_dict)
            
            # Save detailed chat history
            if success and session.messages:
                chat_data = []
                for i, msg in enumerate(session.messages):
                    chat_data.append({
                        "session_id": session.session_id,
                        "user_id": session.user_id,
                        "message_index": i,
                        "role": msg.get("role"),
                        "content": msg.get("content"),
                        "timestamp": msg.get("timestamp", datetime.now().isoformat()),
                        "metadata": json.dumps(msg.get("metadata", {}))
                    })
                self.append_to_sheet(SHEETS_CONFIG["chat_sheet"], chat_data)
            
            return success
        except Exception as e:
            st.error(f"Error saving session: {e}")
            return False
    
    def get_analytics(self, user_id: Optional[str] = None, days: int = 30) -> Analytics:
        """Get analytics from sheets"""
        try:
            analytics = Analytics()
            
            # Read sessions
            df = self.read_sheet(SHEETS_CONFIG["main_sheet"])
            if df.empty:
                return analytics
            
            # Filter by date
            cutoff_date = datetime.now() - timedelta(days=days)
            df["start_time"] = pd.to_datetime(df["start_time"])
            df = df[df["start_time"] >= cutoff_date]
            
            # Filter by user if specified
            if user_id:
                df = df[df["user_id"] == user_id]
            
            if df.empty:
                return analytics
            
            # Calculate metrics
            analytics.total_sessions = len(df)
            analytics.total_messages = df["message_count"].sum() if "message_count" in df.columns else 0
            analytics.avg_messages_per_session = analytics.total_messages / analytics.total_sessions if analytics.total_sessions > 0 else 0
            
            # Module usage
            if "module" in df.columns:
                analytics.module_usage = df["module"].value_counts().to_dict()
            
            # Intent distribution
            if "intents" in df.columns:
                all_intents = []
                for intents_str in df["intents"].dropna():
                    try:
                        intents = json.loads(intents_str)
                        all_intents.extend(intents)
                    except:
                        pass
                analytics.intent_distribution = dict(Counter(all_intents))
            
            # Quiz performance
            if "quiz_results" in df.columns:
                quiz_scores = []
                for results_str in df["quiz_results"].dropna():
                    try:
                        results = json.loads(results_str)
                        for result in results:
                            if "score" in result:
                                quiz_scores.append(result["score"])
                    except:
                        pass
                if quiz_scores:
                    analytics.quiz_performance = {
                        "avg_score": np.mean(quiz_scores),
                        "min_score": min(quiz_scores),
                        "max_score": max(quiz_scores),
                        "total_quizzes": len(quiz_scores)
                    }
            
            return analytics
        except Exception as e:
            st.error(f"Error getting analytics: {e}")
            return Analytics()

# ============================================================================
# KNOWLEDGE BASE MANAGER
# ============================================================================

class MedicalKnowledgeBase:
    """Medical knowledge base with RAG capabilities"""
    
    def __init__(self, api_key: str):
        self.embeddings = OpenAIEmbeddings(api_key=api_key)
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        self.knowledge_docs = []
    
    def initialize_knowledge(self):
        """Initialize medical knowledge base"""
        # Medical knowledge documents
        medical_knowledge = [
            # Anatomy
            Document(
                page_content="The cardiovascular system consists of the heart, blood vessels, and blood. The heart has four chambers: two atria and two ventricles. The right side pumps deoxygenated blood to the lungs, while the left side pumps oxygenated blood to the body.",
                metadata={"category": "anatomy", "topic": "cardiovascular", "difficulty": "beginner"}
            ),
            Document(
                page_content="The nervous system is divided into the central nervous system (CNS) and peripheral nervous system (PNS). The CNS includes the brain and spinal cord, while the PNS includes all nerves outside the CNS.",
                metadata={"category": "anatomy", "topic": "nervous_system", "difficulty": "beginner"}
            ),
            
            # Pharmacology
            Document(
                page_content="Beta-blockers work by blocking beta-adrenergic receptors, reducing heart rate and blood pressure. Common examples include metoprolol, atenolol, and propranolol. Contraindications include asthma and severe bradycardia.",
                metadata={"category": "pharmacology", "topic": "cardiovascular_drugs", "difficulty": "intermediate"}
            ),
            Document(
                page_content="Antibiotics are classified by mechanism: beta-lactams inhibit cell wall synthesis, aminoglycosides inhibit protein synthesis, and fluoroquinolones inhibit DNA synthesis. Always consider resistance patterns and allergies.",
                metadata={"category": "pharmacology", "topic": "antibiotics", "difficulty": "intermediate"}
            ),
            
            # Clinical Skills
            Document(
                page_content="Physical examination should follow a systematic approach: inspection, palpation, percussion, and auscultation. Always explain procedures to patients and ensure comfort throughout the examination.",
                metadata={"category": "clinical_skills", "topic": "physical_exam", "difficulty": "beginner"}
            ),
            Document(
                page_content="ECG interpretation: Check rhythm, rate, axis, intervals (PR, QRS, QT), and morphology. Look for ST elevation/depression, T wave changes, and Q waves. Always correlate with clinical presentation.",
                metadata={"category": "clinical_skills", "topic": "ecg_interpretation", "difficulty": "advanced"}
            ),
            
            # Emergency Medicine
            Document(
                page_content="ACLS protocol for cardiac arrest: Start CPR, attach defibrillator, analyze rhythm. For VF/VT: defibrillate, continue CPR, give epinephrine every 3-5 minutes. For asystole/PEA: CPR, epinephrine, search for reversible causes (H's and T's).",
                metadata={"category": "emergency", "topic": "acls", "difficulty": "advanced"}
            ),
            Document(
                page_content="Trauma assessment follows ATLS principles: Primary survey (ABCDE), resuscitation, secondary survey, definitive care. Always maintain cervical spine immobilization until cleared.",
                metadata={"category": "emergency", "topic": "trauma", "difficulty": "intermediate"}
            ),
            
            # Common Conditions
            Document(
                page_content="Pneumonia diagnosis requires clinical symptoms (fever, cough, dyspnea) plus radiographic evidence. Treatment depends on setting: CAP typically treated with macrolides or fluoroquinolones, HAP requires broader coverage.",
                metadata={"category": "diagnosis", "topic": "respiratory", "difficulty": "intermediate"}
            ),
            Document(
                page_content="Diabetes management involves lifestyle modifications, glucose monitoring, and medications. Type 2 DM: start with metformin, add agents based on patient factors. Always screen for complications: retinopathy, nephropathy, neuropathy.",
                metadata={"category": "treatment", "topic": "endocrine", "difficulty": "intermediate"}
            ),
        ]
        
        # Create vector store
        self.knowledge_docs = medical_knowledge
        self.vector_store = FAISS.from_documents(
            documents=medical_knowledge,
            embedding=self.embeddings
        )
    
    def search(self, query: str, k: int = 3, filters: Optional[Dict] = None) -> List[Document]:
        """Search knowledge base"""
        if not self.vector_store:
            self.initialize_knowledge()
        
        try:
            # Basic similarity search
            docs = self.vector_store.similarity_search(query, k=k)
            
            # Apply filters if provided
            if filters:
                filtered_docs = []
                for doc in docs:
                    match = True
                    for key, value in filters.items():
                        if doc.metadata.get(key) != value:
                            match = False
                            break
                    if match:
                        filtered_docs.append(doc)
                return filtered_docs
            
            return docs
        except Exception as e:
            st.error(f"Knowledge search error: {e}")
            return []
    
    def get_related_topics(self, topic: str) -> List[str]:
        """Get related medical topics"""
        # Simple related topics mapping
        related_map = {
            "cardiovascular": ["heart", "blood pressure", "ecg", "cardiac drugs"],
            "respiratory": ["lungs", "breathing", "pneumonia", "asthma"],
            "nervous": ["brain", "neurology", "stroke", "seizures"],
            "pharmacology": ["drugs", "medications", "dosing", "interactions"],
            "emergency": ["trauma", "resuscitation", "critical care", "acls"]
        }
        
        for key, values in related_map.items():
            if key in topic.lower():
                return values
        
        return []

# ============================================================================
# QUIZ GENERATOR
# ============================================================================

class MedicalQuizGenerator:
    """Generates medical quizzes based on topics"""
    
    def __init__(self):
        self.quiz_database = self._create_quiz_database()
    
    def _create_quiz_database(self) -> Dict[str, List[Dict]]:
        """Create quiz question database"""
        return {
            "anatomy": [
                {
                    "question": "Which chamber of the heart pumps oxygenated blood to the body?",
                    "options": ["Right atrium", "Right ventricle", "Left atrium", "Left ventricle"],
                    "correct": 3,
                    "explanation": "The left ventricle pumps oxygenated blood through the aorta to the systemic circulation.",
                    "difficulty": "beginner"
                },
                {
                    "question": "What is the largest artery in the human body?",
                    "options": ["Carotid artery", "Femoral artery", "Aorta", "Pulmonary artery"],
                    "correct": 2,
                    "explanation": "The aorta is the largest artery, carrying oxygenated blood from the left ventricle to the body.",
                    "difficulty": "beginner"
                }
            ],
            "pharmacology": [
                {
                    "question": "Which class of antibiotics inhibits bacterial cell wall synthesis?",
                    "options": ["Aminoglycosides", "Beta-lactams", "Fluoroquinolones", "Macrolides"],
                    "correct": 1,
                    "explanation": "Beta-lactam antibiotics (penicillins, cephalosporins) inhibit cell wall synthesis by blocking peptidoglycan cross-linking.",
                    "difficulty": "intermediate"
                },
                {
                    "question": "What is the antidote for opioid overdose?",
                    "options": ["Flumazenil", "Naloxone", "Atropine", "Glucagon"],
                    "correct": 1,
                    "explanation": "Naloxone is an opioid antagonist that rapidly reverses opioid effects, including respiratory depression.",
                    "difficulty": "intermediate"
                }
            ],
            "emergency": [
                {
                    "question": "In the ABCDE approach, what does 'D' stand for?",
                    "options": ["Defibrillation", "Disability", "Diagnosis", "Documentation"],
                    "correct": 1,
                    "explanation": "D stands for Disability, focusing on neurological assessment including GCS and pupil response.",
                    "difficulty": "beginner"
                },
                {
                    "question": "What is the first medication given in cardiac arrest?",
                    "options": ["Atropine", "Amiodarone", "Epinephrine", "Lidocaine"],
                    "correct": 2,
                    "explanation": "Epinephrine 1mg IV/IO is given every 3-5 minutes during cardiac arrest to increase coronary perfusion pressure.",
                    "difficulty": "intermediate"
                }
            ]
        }
    
    def generate_quiz(self, topic: str, difficulty: str = "all", num_questions: int = 5) -> List[Dict]:
        """Generate a quiz based on topic and difficulty"""
        questions = []
        
        # Get questions for topic
        topic_questions = self.quiz_database.get(topic.lower(), [])
        
        # Filter by difficulty if specified
        if difficulty != "all":
            topic_questions = [q for q in topic_questions if q["difficulty"] == difficulty]
        
        # Random sample
        if topic_questions:
            num_available = min(num_questions, len(topic_questions))
            questions = random.sample(topic_questions, num_available)
        
        return questions
    
    def evaluate_answer(self, question: Dict, answer_index: int) -> Tuple[bool, str]:
        """Evaluate quiz answer"""
        is_correct = answer_index == question["correct"]
        feedback = "Correct! " if is_correct else f"Incorrect. The correct answer is: {question['options'][question['correct']]}. "
        feedback += question["explanation"]
        
        return is_correct, feedback

# ============================================================================
# MEDICAL CASE GENERATOR
# ============================================================================

class MedicalCaseGenerator:
    """Generates medical case studies for training"""
    
    def __init__(self):
        self.case_templates = self._create_case_templates()
    
    def _create_case_templates(self) -> List[MedicalCase]:
        """Create medical case templates"""
        return [
            MedicalCase(
                case_id="CASE001",
                title="Acute Myocardial Infarction",
                specialty="Cardiology",
                difficulty="intermediate",
                patient_info={
                    "age": 58,
                    "gender": "Male",
                    "history": "Hypertension, Type 2 DM, Smoker"
                },
                symptoms=["Chest pain", "Dyspnea", "Diaphoresis", "Nausea"],
                vital_signs={
                    "BP": "150/90",
                    "HR": "110",
                    "RR": "22",
                    "O2Sat": "94%",
                    "Temp": "37.2°C"
                },
                lab_results={
                    "Troponin": "2.5 ng/mL (elevated)",
                    "CK-MB": "45 U/L (elevated)",
                    "ECG": "ST elevation in leads II, III, aVF"
                },
                imaging="Chest X-ray: Normal",
                diagnosis="Inferior STEMI",
                treatment_plan=[
                    "Aspirin 325mg",
                    "Clopidogrel 600mg",
                    "Heparin bolus and infusion",
                    "Urgent cardiac catheterization",
                    "Beta-blocker",
                    "Statin"
                ],
                learning_points=[
                    "Recognize STEMI criteria on ECG",
                    "Time is muscle - door to balloon time",
                    "Dual antiplatelet therapy importance",
                    "Secondary prevention measures"
                ]
            ),
            MedicalCase(
                case_id="CASE002",
                title="Community-Acquired Pneumonia",
                specialty="Pulmonology",
                difficulty="beginner",
                patient_info={
                    "age": 72,
                    "gender": "Female",
                    "history": "COPD, Former smoker"
                },
                symptoms=["Productive cough", "Fever", "Pleuritic chest pain", "Fatigue"],
                vital_signs={
                    "BP": "110/70",
                    "HR": "105",
                    "RR": "26",
                    "O2Sat": "88% on RA",
                    "Temp": "38.9°C"
                },
                lab_results={
                    "WBC": "15,000/μL",
                    "CRP": "150 mg/L",
                    "Procalcitonin": "2.5 ng/mL"
                },
                imaging="Chest X-ray: Right lower lobe consolidation",
                diagnosis="Community-acquired pneumonia",
                treatment_plan=[
                    "Oxygen supplementation",
                    "Ceftriaxone 1g IV daily",
                    "Azithromycin 500mg PO daily",
                    "IV fluids",
                    "Antipyretics"
                ],
                learning_points=[
                    "CURB-65 score for severity assessment",
                    "Empiric antibiotic selection",
                    "Importance of early treatment",
                    "Prevention with vaccination"
                ]
            )
        ]
    
    def get_case(self, specialty: Optional[str] = None, difficulty: Optional[str] = None) -> Optional[MedicalCase]:
        """Get a medical case based on criteria"""
        cases = self.case_templates
        
        # Filter by specialty
        if specialty:
            cases = [c for c in cases if c.specialty.lower() == specialty.lower()]
        
        # Filter by difficulty
        if difficulty:
            cases = [c for c in cases if c.difficulty.lower() == difficulty.lower()]
        
        if cases:
            return random.choice(cases)
        return None
    
    def format_case_presentation(self, case: MedicalCase) -> str:
        """Format case for presentation"""
        presentation = f"""
**{case.title}**

**Patient Information:**
- Age: {case.patient_info['age']} years
- Gender: {case.patient_info['gender']}
- Past Medical History: {case.patient_info['history']}

**Presentation:**
- Symptoms: {', '.join(case.symptoms)}

**Vital Signs:**
- Blood Pressure: {case.vital_signs['BP']} mmHg
- Heart Rate: {case.vital_signs['HR']} bpm
- Respiratory Rate: {case.vital_signs['RR']}/min
- O2 Saturation: {case.vital_signs['O2Sat']}
- Temperature: {case.vital_signs['Temp']}

**Laboratory Results:**
"""
        for test, result in case.lab_results.items():
            presentation += f"- {test}: {result}\n"
        
        presentation += f"\n**Imaging:** {case.imaging}"
        
        return presentation

# ============================================================================
# MAIN MEDIBOT AGENT
# ============================================================================

class MediBotAgent:
    """Main orchestrator for MediBot system"""
    
    def __init__(self, api_key: str, sheets_manager: GoogleSheetsManager):
        self.api_key = api_key
        self.sheets_manager = sheets_manager
        self.llm = None
        self.memory = None
        self.knowledge_base = None
        self.nlp_classifier = MedicalNLPClassifier()
        self.sentiment_analyzer = MedicalSentimentAnalyzer()
        self.quiz_generator = MedicalQuizGenerator()
        self.case_generator = MedicalCaseGenerator()
        
        # Initialize components
        self._initialize_llm()
        self._initialize_knowledge_base()
    
    def _initialize_llm(self):
        """Initialize language model"""
        try:
            self.llm = ChatOpenAI(
                api_key=self.api_key,
                model="gpt-4o-mini",
                temperature=0.7,
                max_tokens=1200
            )
            
            # Initialize memory
            self.memory = ConversationSummaryBufferMemory(
                llm=self.llm,
                max_token_limit=2000,
                return_messages=True
            )
        except Exception as e:
            st.error(f"Failed to initialize LLM: {e}")
            # Fallback
            self.llm = ChatOpenAI(
                api_key=self.api_key,
                model="gpt-3.5-turbo",
                temperature=0.7
            )
    
    def _initialize_knowledge_base(self):
        """Initialize medical knowledge base"""
        self.knowledge_base = MedicalKnowledgeBase(self.api_key)
        self.knowledge_base.initialize_knowledge()
    
    def process_message(self, message: str, session: ChatSession, user_profile: UserProfile) -> Tuple[str, Dict[str, Any]]:
        """Process user message and generate response"""
        metadata = {
            "intent": None,
            "confidence": 0.0,
            "sentiment": None,
            "entities": {},
            "knowledge_used": [],
            "quiz_generated": False,
            "case_presented": False
        }
        
        try:
            # Classify intent
            intent, confidence, context = self.nlp_classifier.classify_intent(message)
            metadata["intent"] = intent.value
            metadata["confidence"] = confidence
            metadata["entities"] = context.get("entities", {})
            
            # Analyze sentiment
            sentiment, score = self.sentiment_analyzer.analyze(message)
            metadata["sentiment"] = sentiment
            metadata["sentiment_score"] = score
            
            # Search knowledge base
            knowledge_context = ""
            if intent in [MedicalIntent.CONCEPT_EXPLANATION, MedicalIntent.DIAGNOSIS_HELP, 
                         MedicalIntent.TREATMENT_GUIDANCE, MedicalIntent.DRUG_INFORMATION]:
                
                # Search with filters based on user level
                filters = {"difficulty": user_profile.experience_level.lower()} if user_profile.experience_level else None
                docs = self.knowledge_base.search(message, k=3, filters=filters)
                
                if docs:
                    knowledge_context = "\n\nRelevant medical information:\n"
                    for doc in docs:
                        knowledge_context += f"- {doc.page_content}\n"
                        metadata["knowledge_used"].append(doc.metadata.get("topic", "general"))
            
            # Handle specific intents
            if intent == MedicalIntent.QUIZ_REQUEST:
                response = self._handle_quiz_request(message, session, user_profile)
                metadata["quiz_generated"] = True
            elif intent == MedicalIntent.CASE_STUDY:
                response = self._handle_case_request(message, session, user_profile)
                metadata["case_presented"] = True
            else:
                # Generate contextual response
                response = self._generate_response(message, knowledge_context, session, user_profile)
            
            # Add to session
            session.intents_detected.append(intent.value)
            if metadata["entities"]:
                for entity_type, entities in metadata["entities"].items():
                    session.topics_covered.extend(entities)
            
            return response, metadata
            
        except Exception as e:
            st.error(f"Processing error: {e}")
            return "I apologize for the technical issue. Let me help you with your medical training question. Could you please rephrase your question?", metadata
    
    def _generate_response(self, message: str, knowledge_context: str, session: ChatSession, user_profile: UserProfile) -> str:
        """Generate AI response"""
        # Build context
        system_prompt = f"""You are {SYSTEM_NAME}, an advanced medical training AI assistant.
        
        Student Profile:
        - Name: {user_profile.name}
        - Role: {user_profile.role}
        - Specialty: {user_profile.specialty}
        - Experience: {user_profile.experience_level}
        - Weak Areas: {', '.join(user_profile.weak_areas) if user_profile.weak_areas else 'Not identified'}
        
        Current Module: {session.module}
        Difficulty: {session.difficulty}
        
        Guidelines:
        - Provide accurate medical information
        - Adapt explanations to the student's level
        - Use clinical examples when relevant
        - Encourage critical thinking
        - Correct misconceptions gently
        - Reference evidence-based medicine
        
        {knowledge_context}
        """
        
        # Create messages
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=message)
        ]
        
        # Add conversation history if available
        if hasattr(self.memory, 'chat_memory') and self.memory.chat_memory.messages:
            messages = [SystemMessage(content=system_prompt)] + self.memory.chat_memory.messages[-4:] + [HumanMessage(content=message)]
        
        # Generate response
        response = self.llm.invoke(messages)
        
        # Save to memory
        self.memory.save_context({"input": message}, {"output": response.content})
        
        return response.content
    
    def _handle_quiz_request(self, message: str, session: ChatSession, user_profile: UserProfile) -> str:
        """Handle quiz generation request"""
        # Determine topic from message or session
        topic = "anatomy"  # Default
        for module in ["anatomy", "pharmacology", "emergency"]:
            if module in message.lower() or module in session.module.lower():
                topic = module
                break
        
        # Generate quiz
        questions = self.quiz_generator.generate_quiz(
            topic=topic,
            difficulty=session.difficulty.lower(),
            num_questions=3
        )
        
        if not questions:
            return "I'll prepare a custom quiz for you. Let me know which specific topic you'd like to focus on."
        
        # Format quiz
        response = f"📝 **Medical Quiz - {topic.capitalize()}**\n\n"
        
        for i, q in enumerate(questions, 1):
            response += f"**Question {i}:** {q['question']}\n"
            for j, option in enumerate(q['options']):
                response += f"   {chr(65+j)}. {option}\n"
            response += "\n"
        
        # Store quiz in session
        session.quiz_results.append({
            "topic": topic,
            "questions": len(questions),
            "timestamp": datetime.now().isoformat()
        })
        
        response += "\nPlease provide your answers (e.g., 'A, B, C') and I'll review them with explanations."
        
        # Store quiz in session state for evaluation
        st.session_state.current_quiz = questions
        
        return response
    
    def _handle_case_request(self, message: str, session: ChatSession, user_profile: UserProfile) -> str:
        """Handle case study request"""
        # Get appropriate case
        specialty = user_profile.specialty if user_profile.specialty != "General" else None
        case = self.case_generator.get_case(
            specialty=specialty,
            difficulty=session.difficulty.lower()
        )
        
        if not case:
            return "Let me prepare a clinical case for you. What type of case would you like to review?"
        
        # Format case presentation
        presentation = self.case_generator.format_case_presentation(case)
        
        response = f"📋 **Clinical Case Study**\n\n{presentation}\n\n"
        response += "**Questions to consider:**\n"
        response += "1. What is your differential diagnosis?\n"
        response += "2. What additional tests would you order?\n"
        response += "3. What is your management plan?\n\n"
        response += "Please share your clinical reasoning and I'll provide feedback."
        
        # Store case in session
        st.session_state.current_case = case
        session.topics_covered.append(case.title)
        
        return response

# ============================================================================
# UI COMPONENTS
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables"""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.user_id = None
        st.session_state.user_profile = None
        st.session_state.current_session = None
        st.session_state.agent = None
        st.session_state.sheets_manager = None
        st.session_state.current_quiz = None
        st.session_state.current_case = None
        st.session_state.analytics = Analytics()
        st.session_state.start_time = datetime.now()

def render_sidebar():
    """Render sidebar with configuration"""
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/4CAF50/FFFFFF?text=MediBot+2.0", use_container_width=True)
        st.markdown(f"### {SYSTEM_NAME} v{VERSION}")
        st.markdown("Advanced Medical Training Assistant")
        st.markdown("---")
        
        # API Configuration
        st.markdown("#### 🔑 Configuration")
        api_key = st.text_input("OpenAI API Key", type="password", key="api_key",
                               help="Your OpenAI API key for AI capabilities")
        
        # User Profile Section
        st.markdown("#### 👤 User Profile")
        
        if not st.session_state.user_profile:
            # Registration form
            with st.form("registration_form"):
                st.markdown("**New User Registration**")
                name = st.text_input("Full Name", placeholder="Dr. John Smith")
                email = st.text_input("Email", placeholder="john.smith@hospital.com")
                role = st.selectbox("Role", ["Student", "Resident", "Physician", "Nurse"])
                specialty = st.selectbox("Specialty", [s.value for s in MedicalSpecialty])
                experience = st.selectbox("Experience Level", [d.value for d in DifficultyLevel])
                
                if st.form_submit_button("Register", type="primary"):
                    if name and email:
                        # Create user profile
                        user_id = hashlib.md5(email.encode()).hexdigest()[:8]
                        profile = UserProfile(
                            user_id=user_id,
                            name=name,
                            email=email,
                            role=role,
                            specialty=specialty,
                            experience_level=experience
                        )
                        st.session_state.user_profile = profile
                        st.session_state.user_id = user_id
                        
                        # Save to sheets if manager available
                        if st.session_state.sheets_manager:
                            st.session_state.sheets_manager.save_user_profile(profile)
                        
                        st.success(f"Welcome, {name}!")
                        st.rerun()
                    else:
                        st.error("Please fill in all required fields")
        else:
            # Display user info
            profile = st.session_state.user_profile
            st.info(f"👤 **{profile.name}**")
            st.caption(f"Role: {profile.role}")
            st.caption(f"Specialty: {profile.specialty}")
            st.caption(f"Level: {profile.experience_level}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Sessions", profile.total_sessions)
            with col2:
                st.metric("Messages", profile.total_messages)
            
            if profile.quiz_scores:
                avg_score = np.mean(profile.quiz_scores)
                st.metric("Avg Quiz Score", f"{avg_score:.1%}")
            
            # Logout button
            if st.button("Switch User", key="logout"):
                st.session_state.user_profile = None
                st.session_state.user_id = None
                st.session_state.messages = []
                st.rerun()
        
        # Training Configuration
        if st.session_state.user_profile:
            st.markdown("---")
            st.markdown("#### 🎓 Training Settings")
            
            # Module selection
            module = st.selectbox(
                "Training Module",
                [m.value for m in TrainingModule],
                key="module_select"
            )
            
            # Difficulty
            difficulty = st.selectbox(
                "Difficulty Level",
                [d.value for d in DifficultyLevel],
                key="difficulty_select"
            )
            
            # Learning Mode
            st.markdown("**Learning Modes**")
            col1, col2 = st.columns(2)
            with col1:
                interactive_mode = st.checkbox("Interactive", value=True)
                quiz_mode = st.checkbox("Quiz Mode", value=False)
            with col2:
                case_mode = st.checkbox("Case Studies", value=True)
                research_mode = st.checkbox("Research", value=False)
            
            # Initialize Session
            if api_key and not st.session_state.current_session:
                with st.spinner("Initializing training session..."):
                    try:
                        # Initialize Google Sheets connection
                        conn = st.connection("gsheets", type=GSheetsConnection)
                        st.session_state.sheets_manager = GoogleSheetsManager(conn)
                        
                        # Initialize agent
                        st.session_state.agent = MediBotAgent(api_key, st.session_state.sheets_manager)
                        
                        # Create session
                        st.session_state.current_session = ChatSession(
                            session_id=st.session_state.session_id,
                            user_id=st.session_state.user_id or "anonymous",
                            module=module,
                            specialty=st.session_state.user_profile.specialty,
                            difficulty=difficulty,
                            start_time=datetime.now()
                        )
                        
                        st.success("✅ Session ready!")
                    except Exception as e:
                        st.error(f"Initialization failed: {e}")
        
        # Quick Actions
        if st.session_state.current_session:
            st.markdown("---")
            st.markdown("#### ⚡ Quick Actions")
            
            if st.button("📝 Generate Quiz", use_container_width=True):
                st.session_state.pending_action = "quiz"
            
            if st.button("📋 Present Case", use_container_width=True):
                st.session_state.pending_action = "case"
            
            if st.button("📚 Study Guide", use_container_width=True):
                st.session_state.pending_action = "guide"
            
            if st.button("🔄 Reset Session", type="secondary", use_container_width=True):
                # Save current session
                if st.session_state.sheets_manager and st.session_state.current_session:
                    st.session_state.current_session.end_time = datetime.now()
                    st.session_state.sheets_manager.save_session(st.session_state.current_session)
                
                # Reset
                st.session_state.messages = []
                st.session_state.session_id = str(uuid.uuid4())
                st.session_state.current_session = None
                st.session_state.current_quiz = None
                st.session_state.current_case = None
                st.rerun()

def render_analytics_dashboard():
    """Render analytics dashboard"""
    if not st.session_state.sheets_manager:
        return
    
    st.markdown("### 📊 Learning Analytics")
    
    # Get analytics
    analytics = st.session_state.sheets_manager.get_analytics(
        user_id=st.session_state.user_id,
        days=30
    )
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Sessions", analytics.total_sessions)
        st.caption("Last 30 days")
    
    with col2:
        st.metric("Total Messages", analytics.total_messages)
        st.caption("Interactions")
    
    with col3:
        if analytics.quiz_performance:
            st.metric("Quiz Average", f"{analytics.quiz_performance.get('avg_score', 0):.1%}")
            st.caption("Performance")
        else:
            st.metric("Quiz Average", "No data")
    
    with col4:
        session_duration = (datetime.now() - st.session_state.start_time).seconds
        st.metric("Session Time", f"{session_duration // 60}m")
        st.caption("Current")
    
    # Module usage
    if analytics.module_usage:
        st.markdown("#### Module Usage")
        for module, count in analytics.module_usage.items():
            progress = count / sum(analytics.module_usage.values())
            st.progress(progress, text=f"{module}: {count} sessions")
    
    # Intent distribution
    if analytics.intent_distribution:
        st.markdown("#### Learning Focus Areas")
        intents_sorted = sorted(analytics.intent_distribution.items(), key=lambda x: x[1], reverse=True)[:5]
        for intent, count in intents_sorted:
            st.caption(f"• {intent.replace('_', ' ').title()}: {count} times")

def render_chat_interface():
    """Render main chat interface"""
    st.title(f"🏥 {SYSTEM_NAME} - Medical Training Assistant")
    
    # Display user and session info
    if st.session_state.user_profile and st.session_state.current_session:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Student", st.session_state.user_profile.name)
            st.caption(st.session_state.user_profile.role)
        
        with col2:
            st.metric("Module", st.session_state.current_session.module)
            st.caption(st.session_state.current_session.difficulty)
        
        with col3:
            message_count = len(st.session_state.messages)
            st.metric("Messages", message_count)
            st.caption("This session")
        
        with col4:
            # Sentiment indicator
            if st.session_state.current_session.messages:
                last_sentiment = "😐 Neutral"
                for msg in reversed(st.session_state.current_session.messages):
                    if msg.get("metadata", {}).get("sentiment"):
                        sentiment = msg["metadata"]["sentiment"]
                        if sentiment == "very_positive":
                            last_sentiment = "😄 Great"
                        elif sentiment == "positive":
                            last_sentiment = "😊 Good"
                        elif sentiment == "negative":
                            last_sentiment = "😟 Struggling"
                        elif sentiment == "very_negative":
                            last_sentiment = "😰 Difficult"
                        break
                st.metric("Understanding", last_sentiment)
                st.caption("Current level")
        
        st.markdown("---")
    
    # Chat messages container
    chat_container = st.container()
    
    with chat_container:
        # Display messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar="🏥" if message["role"] == "assistant" else "👨‍⚕️"):
                st.markdown(message["content"])
                
                # Show metadata if available
                if message.get("metadata"):
                    meta = message["metadata"]
                    meta_cols = st.columns(6)
                    
                    if meta.get("intent"):
                        with meta_cols[0]:
                            st.caption(f"📍 {meta['intent'].replace('_', ' ').title()}")
                    
                    if meta.get("confidence"):
                        with meta_cols[1]:
                            conf_emoji = "🟢" if meta["confidence"] > 0.7 else "🟡" if meta["confidence"] > 0.4 else "🔴"
                            st.caption(f"{conf_emoji} {meta['confidence']:.0%}")
                    
                    if meta.get("knowledge_used"):
                        with meta_cols[2]:
                            st.caption(f"📚 {len(meta['knowledge_used'])} refs")
                    
                    if meta.get("quiz_generated"):
                        with meta_cols[3]:
                            st.caption("📝 Quiz")
                    
                    if meta.get("case_presented"):
                        with meta_cols[4]:
                            st.caption("📋 Case")
                    
                    if meta.get("entities"):
                        entity_count = sum(len(v) for v in meta["entities"].values())
                        if entity_count > 0:
                            with meta_cols[5]:
                                st.caption(f"🏷️ {entity_count} terms")
    
    # Input area
    if st.session_state.current_session and st.session_state.agent:
        # Check for pending actions
        if "pending_action" in st.session_state:
            if st.session_state.pending_action == "quiz":
                user_input = "Generate a quiz on the current topic"
            elif st.session_state.pending_action == "case":
                user_input = "Present a clinical case for practice"
            elif st.session_state.pending_action == "guide":
                user_input = "Create a study guide for this module"
            else:
                user_input = None
            
            del st.session_state.pending_action
        else:
            user_input = st.chat_input("Ask your medical question or request a quiz/case study...")
        
        if user_input:
            # Add user message
            user_message = {
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.messages.append(user_message)
            st.session_state.current_session.messages.append(user_message)
            
            # Display user message
            with st.chat_message("user", avatar="👨‍⚕️"):
                st.markdown(user_input)
            
            # Generate response
            with st.chat_message("assistant", avatar="🏥"):
                with st.spinner("Analyzing your question..."):
                    try:
                        # Process message
                        response, metadata = st.session_state.agent.process_message(
                            user_input,
                            st.session_state.current_session,
                            st.session_state.user_profile
                        )
                        
                        # Display response
                        st.markdown(response)
                        
                        # Add assistant message
                        assistant_message = {
                            "role": "assistant",
                            "content": response,
                            "metadata": metadata,
                            "timestamp": datetime.now().isoformat()
                        }
                        st.session_state.messages.append(assistant_message)
                        st.session_state.current_session.messages.append(assistant_message)
                        
                        # Update user profile stats
                        st.session_state.user_profile.total_messages += 2
                        
                        # Update session
                        st.session_state.current_session.attempts += 1
                        
                        # Auto-save periodically
                        if len(st.session_state.messages) % 10 == 0:
                            if st.session_state.sheets_manager:
                                st.session_state.sheets_manager.save_session(st.session_state.current_session)
                                st.session_state.sheets_manager.save_user_profile(st.session_state.user_profile)
                        
                    except Exception as e:
                        st.error(f"Error: {e}")
                        error_message = {
                            "role": "assistant",
                            "content": "I apologize for the technical issue. Please try rephrasing your question.",
                            "timestamp": datetime.now().isoformat()
                        }
                        st.session_state.messages.append(error_message)
                        st.markdown(error_message["content"])
            
            # Force rerun for UI update
            st.rerun()
    else:
        st.info("👆 Please complete your profile and enter your API key in the sidebar to start learning!")

def main():
    """Main application"""
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Main layout
    tab1, tab2 = st.tabs(["💬 Training Chat", "📊 Analytics"])
    
    with tab1:
        render_chat_interface()
    
    with tab2:
        if st.session_state.user_profile:
            render_analytics_dashboard()
        else:
            st.info("Please register to view your analytics")
    
    # Footer
    st.markdown("---")
    st.caption(f"{SYSTEM_NAME} v{VERSION} | Medical Training AI Assistant | Powered by OpenAI & Google Sheets")

if __name__ == "__main__":
    main()