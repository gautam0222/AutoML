import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import re
import string
from collections import Counter
from datetime import datetime
import time
import pickle
import io
import base64

if "current_analysis" not in st.session_state:
    st.session_state["current_analysis"] = {}
if "test_size" not in st.session_state:
    st.session_state["test_size"] = {}
if "cleaning_level" not in st.session_state:
    st.session_state["cleaning_level"] = {}


try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from textblob import TextBlob
    NLTK_AVAILABLE = True
    
    nltk_downloads = ['punkt', 'stopwords', 'vader_lexicon', 'wordnet', 'omw-1.4']
    for resource in nltk_downloads:
        try:
            nltk.download(resource, quiet=True)
        except:
            pass
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    analyzer = SentimentIntensityAnalyzer()
    
except ImportError:
    NLTK_AVAILABLE = False
    stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
    lemmatizer = None
    analyzer = None

st.set_page_config(
    page_title="E-commerce AutoML Analyzer",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { padding-top: 2rem; }
    .custom-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem; border-radius: 10px; color: white;
        text-align: center; margin-bottom: 2rem;
    }
    .metric-card {
        background: white; padding: 1rem; border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .stButton > button {
        width: 100%; border-radius: 20px; border: none;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)

APP_CONFIG = {
    "app_name": "E-commerce AutoML Analyzer",
    "version": "1.0.0",
    "max_file_size": 200,
    "supported_formats": [".csv", ".xlsx", ".json"],
    "default_test_size": 0.2,
    "random_state": 42
}

ML_CONFIG = {
    "classification_models": {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SVM": SVC(random_state=42, probability=True)
    },
    "regression_models": {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeRegressor(random_state=42)
    }
}

# Enhanced ML Configuration
ENHANCED_ML_CONFIG = {
    "classification_models": {
        "Logistic Regression": {
            "model": LogisticRegression,
            "params": {"random_state": 42, "max_iter": 1000},
            "tuning_params": {
                "C": [0.1, 1, 10],
                "solver": ["liblinear", "lbfgs"]
            }
        },
        "Random Forest": {
            "model": RandomForestClassifier,
            "params": {"random_state": 42},
            "tuning_params": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5, 10]
            }
        },
    }
}

class EnhancedAutoMLPipeline:
    def __init__(self):
        self.best_models = {}
        self.cv_scores = {}
        self.feature_importance = {}
        
    def auto_feature_engineering(self, df, text_column):
        """Automatic feature engineering for text data"""
        features = {}
        
        # Text length features
        features['text_length'] = df[text_column].str.len()
        features['word_count'] = df[text_column].str.split().str.len()
        features['char_count'] = df[text_column].str.len()
        
        # Text complexity features
        features['exclamation_count'] = df[text_column].str.count('!')
        features['question_count'] = df[text_column].str.count('\?')
        # avoid division by zero
        text_length = features['text_length'].replace(0, np.nan)
        features['uppercase_ratio'] = (df[text_column].str.count('[A-Z]') / text_length).fillna(0)

        
        return pd.DataFrame(features).fillna(0)
    
    def automated_hyperparameter_tuning(self, X_train, y_train, model_config, cv_folds=3):
        """Automated hyperparameter tuning using GridSearchCV"""
        from sklearn.model_selection import GridSearchCV
        
        model_class = model_config["model"]
        base_params = model_config["params"]
        tuning_params = model_config["tuning_params"]
        
        model = model_class(**base_params)
        
        grid_search = GridSearchCV(
            model, 
            tuning_params, 
            cv=cv_folds, 
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        return {
            "best_model": grid_search.best_estimator_,
            "best_params": grid_search.best_params_,
            "best_cv_score": grid_search.best_score_,
            "cv_results": grid_search.cv_results_
        }
    
    def ensemble_prediction(self, models, X_test):
        """Create ensemble predictions from multiple models"""
        predictions = []
        for model_name, model_info in models.items():
            pred = model_info["model"].predict(X_test)
            predictions.append(pred)
        
        # Simple voting ensemble
        ensemble_pred = np.round(np.mean(predictions, axis=0))
        return ensemble_pred
    
    def auto_model_selection(self, X_train, X_test, y_train, y_test, task_type):
        """Automated model selection with cross-validation"""
        from sklearn.model_selection import cross_val_score
        
        results = {}
        
        config = ENHANCED_ML_CONFIG["classification_models"] if task_type == "classification" else ML_CONFIG["regression_models"]
        
        for model_name, model_config in config.items():
            try:
                if model_name == "XGBoost":
                    # Skip XGBoost if not available
                    continue
                    
                # Hyperparameter tuning
                tuning_result = self.automated_hyperparameter_tuning(
                    X_train, y_train, model_config
                )
                
                best_model = tuning_result["best_model"]
                
                # Cross-validation scores
                cv_scores = cross_val_score(best_model, X_train, y_train, cv=3)
                
                # Final evaluation
                best_model.fit(X_train, y_train)
                y_pred = best_model.predict(X_test)
                
                if task_type == "classification":
                    accuracy = accuracy_score(y_test, y_pred)
                    metrics = {"accuracy": accuracy, "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std()}
                else:
                    r2 = r2_score(y_test, y_pred)
                    metrics = {"r2_score": r2, "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std()}
                
                results[model_name] = {
                    "model": best_model,
                    "metrics": metrics,
                    "best_params": tuning_result["best_params"],
                    "predictions": y_pred
                }
                
            except Exception as e:
                st.warning(f"Failed to train {model_name}: {str(e)}")
                continue
        
        return results

TEXT_CONFIG = {
    "max_features": 5000,
    "min_df": 2,
    "max_df": 0.95,
    "ngram_range": (1, 2),
    "stop_words": list(stop_words) if stop_words else None
}

class DataValidator:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.suggestions = []
    
    def validate_dataset(self, df):
        self.errors = []
        self.warnings = []
        self.suggestions = []
        
        self._validate_structure(df)
        self._validate_columns(df)
        self._validate_data_quality(df)
        
        return {
            "is_valid": len(self.errors) == 0,
            "errors": self.errors,
            "warnings": self.warnings,
            "suggestions": self.suggestions
        }
    
    def _validate_structure(self, df):
        if df is None or df.empty:
            self.errors.append("Dataset is empty or could not be loaded")
            return
        if len(df.columns) < 2:
            self.errors.append("Dataset must have at least 2 columns")
        if len(df) < 10:
            self.warnings.append("Dataset has very few rows (<10). Results may not be reliable.")
    
    def _validate_columns(self, df):
        text_columns = []
        for col in df.columns:
            if df[col].dtype == 'object':
                sample_values = df[col].dropna().astype(str).head(100)
                if len(sample_values) > 0:
                    avg_length = sample_values.str.len().mean()
                    if avg_length > 20:
                        text_columns.append(col)
        
        if not text_columns:
            self.errors.append("No text columns detected. At least one text column is required.")
        else:
            self.suggestions.append(f"Detected text columns: {', '.join(text_columns)}")
    
    def _validate_data_quality(self, df):
        missing_counts = df.isnull().sum()
        high_missing = missing_counts[missing_counts > len(df) * 0.5]
        if len(high_missing) > 0:
            self.warnings.append(f"Columns with >50% missing values: {list(high_missing.index)}")
        
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            self.warnings.append(f"Found {duplicate_count} duplicate rows")

class TextPreprocessor:
    def __init__(self):
        self.stop_words = stop_words
        self.lemmatizer = lemmatizer
    
    def clean_text(self, text, level="basic"):
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        text = str(text).lower().strip()
        
        if level in ["moderate", "aggressive"]:
            text = re.sub(r'http\S+|www\S+', '', text)
            text = re.sub(r'\S+@\S+', '', text)
            text = re.sub(r'\s+', ' ', text)
        
        if level == "aggressive":
            text = text.translate(str.maketrans('', '', string.punctuation))
            text = re.sub(r'\d+', '', text)
            
            if self.stop_words:
                words = text.split()
                text = ' '.join([word for word in words if word not in self.stop_words])
            
            if self.lemmatizer:
                words = text.split()
                text = ' '.join([self.lemmatizer.lemmatize(word) for word in words])
        
        return text.strip()
    
    def batch_clean_text(self, texts, level="basic"):
        return [self.clean_text(text, level) for text in texts]

class GenAIAssistant:
    def __init__(self):
        self.sentiment_analyzer = analyzer
    
    def analyze_dataset_characteristics(self, df):
        analysis = {
            "dataset_size": {"rows": len(df), "columns": len(df.columns)},
            "text_columns": [],
            "numeric_columns": [],
            "potential_targets": [],
            "recommendations": []
        }
        
        for col in df.columns:
            if df[col].dtype == 'object':
                sample_text = df[col].dropna().astype(str).head(100)
                if len(sample_text) > 0:
                    avg_length = sample_text.str.len().mean()
                    if avg_length > 20:
                        analysis["text_columns"].append({
                            "name": col,
                            "avg_length": avg_length
                        })
                    else:
                        unique_count = df[col].nunique()
                        analysis["potential_targets"].append({
                            "name": col,
                            "type": "categorical",
                            "unique_values": unique_count,
                            "task_type": "classification"
                        })
            else:
                analysis["numeric_columns"].append({"name": col, "type": str(df[col].dtype)})
                unique_ratio = df[col].nunique() / len(df)
                task_type = "regression" if unique_ratio > 0.1 else "classification"
                analysis["potential_targets"].append({
                    "name": col,
                    "type": "numeric",
                    "unique_count": df[col].nunique(),
                    "task_type": task_type
                })
        
        if analysis["dataset_size"]["rows"] < 100:
            analysis["recommendations"].append("Small dataset - consider collecting more data")
        if len(analysis["text_columns"]) == 0:
            analysis["recommendations"].append("No text columns detected")
        
        return analysis
    
    def analyze_text_sentiment(self, texts):
        if not self.sentiment_analyzer:
            return [{"compound": 0, "pos": 0, "neu": 1, "neg": 0} for _ in texts]
        
        sentiments = []
        for text in texts:
            if pd.isna(text) or not isinstance(text, str):
                sentiments.append({"compound": 0, "pos": 0, "neu": 1, "neg": 0})
            else:
                score = self.sentiment_analyzer.polarity_scores(str(text))
                sentiments.append(score)
        return sentiments
    
    def suggest_ml_approach(self, df, target_column):
        if target_column not in df.columns:
            return {"error": "Target column not found in dataset"}
        
        target_data = df[target_column].dropna()
        unique_count = target_data.nunique()
        unique_ratio = unique_count / len(target_data) if len(target_data) > 0 else 0
        is_numeric = pd.api.types.is_numeric_dtype(target_data)
        
        if is_numeric and (unique_ratio > 0.1 or unique_count > 20):
            return {
                "recommended_approach": "regression",
                "confidence": 0.9,
                "reasoning": [f"Numeric target with {unique_count} unique values"],
                "models": ["Random Forest Regressor", "Decision Tree Regressor"]
            }
        else:
            return {
                "recommended_approach": "classification", 
                "confidence": 0.8,
                "reasoning": [f"Categorical target with {unique_count} classes"],
                "models": ["Random Forest Classifier", "Logistic Regression", "SVM"]
            }

class ComprehensiveEDA:
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
    
    def create_data_profiling_report(self, df):
        """Generate comprehensive data profiling visualizations"""
        
        # Missing data heatmap
        fig_missing = px.imshow(
            df.isnull().values,
            title="Missing Data Pattern",
            labels=dict(x="Columns", y="Rows", color="Missing"),
            color_continuous_scale="Viridis"
        )
        
        # Correlation matrix for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            fig_corr = px.imshow(
                corr_matrix,
                title="Correlation Matrix",
                color_continuous_scale="RdBu",
                aspect="auto"
            )
        else:
            fig_corr = None
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            fig_dist = None
        else:
            rows = min(3, max(1, len(numeric_cols)))
            # create subplot with at least 1 row
            fig_dist = make_subplots(
                rows=min(3, len(numeric_cols)),
                cols=2,
                subplot_titles=[f"Distribution of {col}" for col in numeric_cols[:6]]
            )
        
        for i, col in enumerate(numeric_cols[:6]):
            row = (i // 2) + 1
            col_pos = (i % 2) + 1
            fig_dist.add_trace(
                go.Histogram(x=df[col], name=col, showlegend=False),
                row=row, col=col_pos
            )
        
        return {
            "missing_data": fig_missing,
            "correlation": fig_corr,
            "distributions": fig_dist
        }
    
    def text_analytics_dashboard(self, df, text_column):
        """Create comprehensive text analytics visualizations"""
        
        if text_column not in df.columns:
            return None
        
        text_data = df[text_column].dropna().astype(str)
        
        # Text length distribution
        text_lengths = text_data.str.len()
        fig_length = px.histogram(
            x=text_lengths,
            title="Text Length Distribution",
            nbins=30
        )
        
        # Word count distribution
        word_counts = text_data.str.split().str.len()
        fig_words = px.box(
            y=word_counts,
            title="Word Count Distribution"
        )
        
        # Most common words
        all_words = ' '.join(text_data).lower().split()
        word_freq = Counter(all_words)
        top_words = dict(word_freq.most_common(20))
        
        fig_wordcloud = px.bar(
            x=list(top_words.values()),
            y=list(top_words.keys()),
            orientation='h',
            title="Top 20 Most Frequent Words"
        )
        
        # Character analysis
        total_chars = ''.join(text_data)
        char_freq = Counter(total_chars.lower())
        common_chars = dict(sorted(char_freq.items(), key=lambda x: x[1], reverse=True)[:15])
        
        fig_chars = px.bar(
            x=list(common_chars.keys()),
            y=list(common_chars.values()),
            title="Most Common Characters"
        )
        
        return {
            "text_length": fig_length,
            "word_count": fig_words,
            "word_frequency": fig_wordcloud,
            "char_frequency": fig_chars
        }
    
    def model_performance_dashboard(self, results, task_type):
        """Create comprehensive model performance visualizations"""
        
        if not results:
            return None
        
        results_df = pd.DataFrame({
            model_name: result["metrics"] 
            for model_name, result in results.items()
        }).T
        
        # Performance comparison radar chart
        if task_type == "classification":
            metrics = ["accuracy", "precision", "recall", "f1_score"]
        else:
            metrics = ["r2_score", "mae", "mse"]
        
        available_metrics = [m for m in metrics if m in results_df.columns]
        
        fig_radar = go.Figure()
        
        for model_name in results_df.index:
            fig_radar.add_trace(go.Scatterpolar(
                r=[results_df.loc[model_name, m] for m in available_metrics],
                theta=available_metrics,
                fill='toself',
                name=model_name
            ))
        
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Model Performance Comparison (Radar Chart)",
            showlegend=True
        )
        
        # Training time vs performance
        if "training_time" in results_df.columns and available_metrics:
            primary_metric = available_metrics[0]
            fig_scatter = px.scatter(
                results_df,
                x="training_time",
                y=primary_metric,
                text=results_df.index,
                title=f"Training Time vs {primary_metric.title()}",
                size_max=60
            )
            fig_scatter.update_traces(textposition="top center")
        else:
            fig_scatter = None
        
        return {
            "radar_chart": fig_radar,
            "time_vs_performance": fig_scatter,
            "detailed_metrics": results_df
        }
        
class GenAIChatbot:
    def __init__(self):
        self.conversation_history = []
        self.context_data = {}
    
    def set_context(self, df, models_info=None, current_analysis=None):
        """Set context for the chatbot based on current data and models"""
        self.context_data = {
            "dataset_shape": df.shape if df is not None else None,
            "columns": list(df.columns) if df is not None else [],
            "data_types": df.dtypes.to_dict() if df is not None else {},
            "models_trained": list(models_info.keys()) if models_info else [],
            "current_analysis": current_analysis
        }
    
    def generate_data_insights(self, df):
        """Generate automatic insights about the dataset"""
        insights = []
        
        if df is None:
            return ["No dataset loaded. Please upload data first."]
        
        # Dataset size insights
        if len(df) < 100:
            insights.append(f"⚠️ Small dataset ({len(df)} rows) - consider collecting more data for better model performance.")
        elif len(df) > 10000:
            insights.append(f"📊 Large dataset ({len(df)} rows) - good for training robust models.")
        
        # Missing data insights
        missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
        high_missing = missing_pct[missing_pct > 20]
        if len(high_missing) > 0:
            insights.append(f"🔍 High missing values detected in: {', '.join(high_missing.index)}")
        
        # Text column insights
        text_cols = []
        for col in df.columns:
            if df[col].dtype == 'object':
                avg_length = df[col].astype(str).str.len().mean()
                if avg_length > 20:
                    text_cols.append(col)
        
        if text_cols:
            insights.append(f"📝 Text columns identified: {', '.join(text_cols)}")
        
        # Target column suggestions
        potential_targets = []
        for col in df.columns:
            col_lower = col.lower()
            if any(word in col_lower for word in ['rating', 'score', 'sentiment', 'class', 'label']):
                potential_targets.append(col)
        
        if potential_targets:
            insights.append(f"🎯 Potential target columns: {', '.join(potential_targets)}")
        
        return insights
    
    def answer_question(self, question, df=None):
        """Answer user questions about the data and models"""
        question_lower = question.lower()
        
        # Dataset-related questions
        if any(word in question_lower for word in ['shape', 'size', 'rows', 'columns']):
            if df is not None:
                return f"Your dataset has {len(df)} rows and {len(df.columns)} columns. Columns: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}"
            else:
                return "No dataset loaded yet. Please upload your data first."
        
        elif any(word in question_lower for word in ['missing', 'null', 'empty']):
            if df is not None:
                missing_info = df.isnull().sum()
                total_missing = missing_info.sum()
                return f"Total missing values: {total_missing}. Columns with missing data: {missing_info[missing_info > 0].to_dict()}"
            else:
                return "No dataset loaded to check for missing values."
        
        elif any(word in question_lower for word in ['columns', 'features', 'variables']):
            if df is not None:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                text_cols = df.select_dtypes(include=['object']).columns.tolist()
                return f"Numeric columns: {numeric_cols}\nText columns: {text_cols}"
            else:
                return "No dataset loaded to analyze columns."
        
        # Model-related questions
        elif any(word in question_lower for word in ['model', 'algorithm', 'accuracy', 'performance']):
            if self.context_data.get("models_trained"):
                models = self.context_data["models_trained"]
                return f"Trained models: {', '.join(models)}. You can view detailed performance metrics in the Results page."
            else:
                return "No models trained yet. Go to the AutoML Training page to train models."
        
        elif any(word in question_lower for word in ['best', 'recommend', 'which']):
            return "The best model is automatically selected based on performance metrics. Check the Results page for the top-performing model and detailed comparison."
        
        # General help
        elif any(word in question_lower for word in ['help', 'how', 'what']):
            return """Here's how to use this application:
            1. Upload your dataset (CSV/Excel/JSON) in the Data Upload page
            2. Configure text and target columns in AutoML Training
            3. Train multiple models automatically
            4. View results and insights in the Results page
            
            Ask me about your data, model performance, or specific features!"""
        
        # Default response
        else:
            return "I can help you with questions about your dataset, model training, performance metrics, and general usage. Try asking about data shape, missing values, model performance, or recommendations!"
    
    def get_model_recommendations(self, df, task_type, current_results=None):
        """Provide AI-powered model recommendations"""
        recommendations = []
        
        if df is None:
            return ["Upload a dataset first to get recommendations."]
        
        dataset_size = len(df)
        
        # Size-based recommendations
        if dataset_size < 500:
            recommendations.append("🔸 For small datasets, try Naive Bayes or Logistic Regression - they work well with limited data.")
        elif dataset_size > 5000:
            recommendations.append("🔸 With your large dataset, Random Forest or SVM should perform well.")
        
        # Task-specific recommendations
        if task_type == "classification":
            unique_classes = df[self.context_data.get("target_column", "")].nunique() if "target_column" in self.context_data else 0
            if unique_classes > 10:
                recommendations.append("🔸 Many classes detected - consider Random Forest for multi-class classification.")
            elif unique_classes == 2:
                recommendations.append("🔸 Binary classification - Logistic Regression or SVM often work well.")
        
        # Performance-based recommendations
        if current_results:
            best_model = max(current_results.keys(), 
                           key=lambda x: current_results[x]["metrics"].get("accuracy", 
                                                                        current_results[x]["metrics"].get("r2_score", 0)))
            recommendations.append(f"🔸 Based on current results, {best_model} is performing best for your data.")
        
        return recommendations if recommendations else ["Train some models first to get personalized recommendations!"]

def load_sample_data():
    sample_reviews = [
        "This product is amazing! Great quality and fast delivery.",
        "Terrible experience. Product broke after one day.", 
        "Average product, nothing special but does the job.",
        "Excellent customer service and high quality product.",
        "Disappointing quality, not worth the price.",
        "Perfect! Exactly what I was looking for.",
        "Poor packaging, item was damaged during shipping.",
        "Outstanding product, highly recommend to everyone!",
        "Mediocre at best, there are better alternatives.",
        "Fantastic value for money, will buy again!"
    ]
    
    sample_ratings = [5, 1, 3, 5, 2, 5, 2, 5, 3, 4]
    sample_products = ["Product A", "Product B", "Product A", "Product C", 
                      "Product B", "Product A", "Product C", "Product A", 
                      "Product B", "Product C"]
    
    df = pd.DataFrame({
        "review_text": sample_reviews * 20,
        "rating": sample_ratings * 20,
        "product_name": sample_products * 20,
    })
    
    return df

def detect_target_column_type(df, target_col):
    if target_col not in df.columns:
        return None
    
    col_data = df[target_col].dropna()
    unique_values = col_data.nunique()
    total_values = len(col_data)
    
    if pd.api.types.is_numeric_dtype(col_data):
        unique_ratio = unique_values / total_values
        return "regression" if unique_ratio > 0.1 or unique_values > 20 else "classification"
    else:
        return "classification"

def get_smart_column_suggestions(df):
    suggestions = {"text_columns": [], "target_columns": []}
    
    for col in df.columns:
        col_lower = col.lower()
        
        text_indicators = ['review', 'comment', 'feedback', 'text', 'description', 'content']
        if any(indicator in col_lower for indicator in text_indicators):
            suggestions["text_columns"].append(col)
        elif df[col].dtype == 'object':
            sample_text = df[col].dropna().astype(str).head(10)
            if len(sample_text) > 0 and sample_text.str.len().mean() > 20:
                suggestions["text_columns"].append(col)
        
        target_indicators = ['rating', 'score', 'sentiment', 'label', 'class', 'target']
        if any(indicator in col_lower for indicator in target_indicators):
            suggestions["target_columns"].append(col)
    
    return suggestions

def create_visualization(df, viz_type, **kwargs):
    if viz_type == "data_overview":
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=('Dataset Shape', 'Missing Values', 'Data Types', 'Column Info'))
        
        fig.add_trace(go.Bar(x=['Rows', 'Columns'], y=[len(df), len(df.columns)], name="Shape"), row=1, col=1)
        
        missing_data = df.isnull().sum().sort_values(ascending=False)[:10]
        fig.add_trace(go.Bar(x=missing_data.index, y=missing_data.values, name="Missing"), row=1, col=2)
        
        type_counts = df.dtypes.value_counts()
        fig.add_trace(go.Pie(labels=type_counts.index.astype(str), values=type_counts.values, name="Types"), row=2, col=1)
        
        fig.update_layout(height=600, showlegend=False, title_text="Dataset Overview")
        return fig
    
    elif viz_type == "sentiment_distribution":
        sentiments = kwargs.get('sentiments', [])
        if not sentiments:
            return None
        
        compound_scores = [s.get('compound', 0) for s in sentiments]
        sentiment_labels = ['Negative' if score < -0.1 else 'Positive' if score > 0.1 else 'Neutral' 
                           for score in compound_scores]
        
        fig = px.histogram(x=sentiment_labels, title="Sentiment Distribution")
        return fig
    
    elif viz_type == "target_distribution":
        target_col = kwargs.get('target_column')
        if target_col and target_col in df.columns:
            fig = px.histogram(df, x=target_col, title=f"Distribution of {target_col}")
            return fig
    
    return None

def train_ml_model(X_train, X_test, y_train, y_test, model_name, task_type):
    if task_type == "classification":
        model = ML_CONFIG["classification_models"].get(model_name)
    else:
        model = ML_CONFIG["regression_models"].get(model_name)

    if model is None:
        return None

    start_time = time.time()
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        st.warning(f"Training failed for {model_name}: {e}")
        return None
    training_time = time.time() - start_time

    try:
        y_pred = model.predict(X_test)
    except Exception as e:
        st.warning(f"Prediction failed for {model_name}: {e}")
        return {"model": model, "metrics": {"training_time": training_time, "model_name": model_name}, "predictions": None}

    metrics = {"training_time": training_time, "model_name": model_name}

    if task_type == "classification":
        try:
            accuracy = accuracy_score(y_test, y_pred)
            metrics.update({"accuracy": accuracy})
            report = classification_report(y_test, y_pred, output_dict=True)
            metrics["precision"] = report.get('weighted avg', {}).get('precision', None)
            metrics["recall"] = report.get('weighted avg', {}).get('recall', None)
            metrics["f1_score"] = report.get('weighted avg', {}).get('f1-score', None)
        except Exception:
            pass
    else:
        try:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            metrics.update({"mse": mse, "r2_score": r2, "mae": mae})
        except Exception:
            pass

    return {"model": model, "metrics": metrics, "predictions": y_pred}


def main():
    st.markdown('<div class="custom-header"><h1>🛒 E-commerce AutoML Customer Feedback Analyzer</h1></div>', 
                unsafe_allow_html=True)
    
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", 
                           ["📊 Data Upload & Analysis", "🔍 Advanced EDA", "🤖 AutoML Training", "💬 AI Assistant", "📈 Results & Insights"])
    
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = {}
    
    validator = DataValidator()
    preprocessor = TextPreprocessor()
    genai_assistant = GenAIAssistant()
    
    if page == "📊 Data Upload & Analysis":
        st.header("Data Upload and Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Upload Your Dataset")
            uploaded_file = st.file_uploader(
                "Choose a CSV, Excel, or JSON file",
                type=['csv', 'xlsx', 'json'],
                help="Upload your e-commerce review/feedback dataset"
            )
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith('.xlsx'):
                        df = pd.read_excel(uploaded_file)
                    elif uploaded_file.name.endswith('.json'):
                        df = pd.read_json(uploaded_file)
                    
                    st.session_state.df = df
                    st.success(f"Dataset loaded successfully! Shape: {df.shape}")
                    
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
                    df = None
        
        with col2:
            st.subheader("Or Use Sample Data")
            if st.button("Load Sample Dataset", type="primary"):
                st.session_state.df = load_sample_data()
                st.success("Sample dataset loaded!")
        
        if st.session_state.df is not None:
            df = st.session_state.df
            
            st.subheader("Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Rows", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            with col4:
                st.metric("Duplicates", df.duplicated().sum())
            
            st.subheader("Data Validation")
            validation_results = validator.validate_dataset(df)
            
            if validation_results["is_valid"]:
                st.success("✅ Dataset validation passed!")
            else:
                st.error("❌ Dataset validation failed!")
                for error in validation_results["errors"]:
                    st.error(f"• {error}")
            
            if validation_results["warnings"]:
                st.warning("⚠️ Warnings:")
                for warning in validation_results["warnings"]:
                    st.warning(f"• {warning}")
            
            if validation_results["suggestions"]:
                st.info("💡 Suggestions:")
                for suggestion in validation_results["suggestions"]:
                    st.info(f"• {suggestion}")
            
            st.subheader("Column Analysis")
            analysis = genai_assistant.analyze_dataset_characteristics(df)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Text Columns:**")
                for text_col in analysis["text_columns"]:
                    st.write(f"• {text_col['name']} (avg length: {text_col['avg_length']:.1f})")
            
            with col2:
                st.write("**Potential Target Columns:**")
                for target in analysis["potential_targets"]:
                    st.write(f"• {target['name']} ({target['task_type']}, {target['unique_count']} unique)")
            
            st.subheader("Data Preview")
            st.dataframe(df.head(10))
            
            fig = create_visualization(df, "data_overview")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    elif page == "🤖 AutoML Training":
        st.header("AutoML Model Training")
        
        if st.session_state.df is None:
            st.warning("Please upload a dataset first in the 'Data Upload & Analysis' page.")
            return
        
        df = st.session_state.df
        
        st.subheader("Configure Training")
        
        col1, col2 = st.columns(2)
        
        with col1:
            suggestions = get_smart_column_suggestions(df)
            
            text_column = st.selectbox(
                "Select Text Column",
                df.columns,
                index=df.columns.get_loc(suggestions["text_columns"][0]) if suggestions["text_columns"] else 0,
                help="Choose the column containing text data for analysis"
            )
            
            target_column = st.selectbox(
                "Select Target Column",
                df.columns,
                index=df.columns.get_loc(suggestions["target_columns"][0]) if suggestions["target_columns"] else 1,
                help="Choose the column you want to predict"
            )
            
            task_type = detect_target_column_type(df, target_column)
            st.info(f"Detected task type: **{task_type}**")
        
        with col2:
            cleaning_level = st.selectbox(
                "Text Preprocessing Level",
                ["basic", "moderate", "aggressive"],
                index=1,
                help="Choose the level of text preprocessing"
            )
            
            test_size = st.slider(
                "Test Set Size",
                min_value=0.1,
                max_value=0.5,
                value=0.2,
                step=0.05,
                help="Proportion of data to use for testing"
            )
        
        if st.button("Start AutoML Training", type="primary"):
            with st.spinner("Training models... This may take a few minutes."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Preprocessing text data...")
                progress_bar.progress(0.1)
                
                clean_texts = preprocessor.batch_clean_text(df[text_column].fillna("").tolist(), cleaning_level)
                
                status_text.text("Extracting features...")
                progress_bar.progress(0.3)
                
                vectorizer = TfidfVectorizer(**TEXT_CONFIG)
                X = vectorizer.fit_transform(clean_texts)
                y = df[target_column].fillna(0 if task_type == "regression" else "unknown")
                
                if task_type == "classification":
                    # always convert target to numeric categorical labels
                    try:
                        le = LabelEncoder()
                        y = le.fit_transform(y.astype(str))
                        st.session_state['label_encoder'] = le  # save for later decoding if needed
                    except Exception as e:
                        st.warning(f"Label encoding failed: {e}")

                
                if task_type == "classification":
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42, stratify=y
                    )
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )

                
                models_to_train = (ML_CONFIG["classification_models"] if task_type == "classification" 
                                 else ML_CONFIG["regression_models"])
                
                results = {}
                for i, model_name in enumerate(models_to_train.keys()):
                    status_text.text(f"Training {model_name}...")
                    progress_bar.progress(0.4 + (i * 0.5 / len(models_to_train)))
                    
                    result = train_ml_model(X_train, X_test, y_train, y_test, model_name, task_type)
                    if result:
                        results[model_name] = result
                
                progress_bar.progress(1.0)
                status_text.text("Training completed!")
                
                st.session_state.models_trained = results
                st.session_state.task_type = task_type
                st.session_state.text_column = text_column
                st.session_state.target_column = target_column
                st.session_state.vectorizer = vectorizer
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
                
                st.success(f"✅ Successfully trained {len(results)} models!")
                
                st.subheader("Training Results")
                results_df = pd.DataFrame({
                    model_name: result["metrics"] 
                    for model_name, result in results.items()
                }).T
                
                st.dataframe(results_df.style.highlight_max(axis=0))
    elif page == "🔍 Advanced EDA":
        st.header("Advanced Exploratory Data Analysis")
        
        if st.session_state.df is None:
            st.warning("Please upload a dataset first.")
            return
        
        df = st.session_state.df
        eda_analyzer = ComprehensiveEDA()
        
        st.subheader("Comprehensive Data Profiling")
        
        if st.button("Generate Full EDA Report"):
            with st.spinner("Generating comprehensive analysis..."):
                
                # Data profiling
                profiling_results = eda_analyzer.create_data_profiling_report(df)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(profiling_results["missing_data"], use_container_width=True)
                    if profiling_results["distributions"]:
                        st.plotly_chart(profiling_results["distributions"], use_container_width=True)
                
                with col2:
                    if profiling_results["correlation"]:
                        st.plotly_chart(profiling_results["correlation"], use_container_width=True)
        
        # Text Analytics Section
        st.subheader("Text Analytics Deep Dive")
        text_columns = [col for col in df.columns if df[col].dtype == 'object']
        
        if text_columns:
            selected_text_col = st.selectbox("Select text column for analysis", text_columns)
            
            if st.button("Analyze Text Data"):
                with st.spinner("Performing text analysis..."):
                    text_results = eda_analyzer.text_analytics_dashboard(df, selected_text_col)
                    
                    if text_results:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.plotly_chart(text_results["text_length"], use_container_width=True)
                            st.plotly_chart(text_results["char_frequency"], use_container_width=True)
                        
                        with col2:
                            st.plotly_chart(text_results["word_count"], use_container_width=True)
                            st.plotly_chart(text_results["word_frequency"], use_container_width=True)

    elif page == "💬 AI Assistant":
        st.header("AI Assistant & Recommendations")
        
        # Initialize chatbot
        if 'chatbot' not in st.session_state:
            st.session_state.chatbot = GenAIChatbot()
        
        chatbot = st.session_state.chatbot
        
        # Set context for chatbot
        if st.session_state.df is not None:
            chatbot.set_context(
                st.session_state.df,
                st.session_state.models_trained,
                st.session_state.get('current_analysis')
            )
        
        # Auto-generated insights
        st.subheader("🔍 Automatic Data Insights")
        
        if st.session_state.df is not None:
            insights = chatbot.generate_data_insights(st.session_state.df)
            for insight in insights:
                st.info(insight)
        
        # Model recommendations
        st.subheader("🎯 AI Recommendations")
        
        if st.button("Get AI Recommendations"):
            task_type = st.session_state.get('task_type', 'classification')
            recommendations = chatbot.get_model_recommendations(
                st.session_state.df,
                task_type,
                st.session_state.models_trained
            )
            
            for rec in recommendations:
                st.success(rec)
        
        # Interactive chat
        st.subheader("💬 Ask the AI Assistant")
        
        # Chat interface
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        for i, (user_msg, bot_response) in enumerate(st.session_state.chat_history):
            st.text_area(f"You:", value=user_msg, height=50, disabled=True, key=f"user_{i}")
            st.text_area(f"AI Assistant:", value=bot_response, height=100, disabled=True, key=f"bot_{i}")
        
        # Chat input
        user_question = st.text_input("Ask a question about your data or models:", key="chat_input")
        
        if st.button("Send") and user_question:
            with st.spinner("Thinking..."):
                response = chatbot.answer_question(user_question, st.session_state.df)
                st.session_state.chat_history.append((user_question, response))
                st.experimental_rerun()
        
        # Quick action buttons
        st.subheader("🚀 Quick Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Analyze My Data"):
                if st.session_state.df is not None:
                    response = chatbot.answer_question("Tell me about my dataset", st.session_state.df)
                    st.info(response)
        
        with col2:
            if st.button("Model Suggestions"):
                response = "Based on your data characteristics, I recommend starting with Random Forest for its balance of performance and interpretability, followed by Logistic Regression for faster training."
                st.info(response)
        
        with col3:
            if st.button("Optimization Tips"):
                response = "To improve model performance: 1) Try different text preprocessing levels, 2) Ensure balanced target classes, 3) Consider feature engineering, 4) Use cross-validation for reliable metrics."
                st.info(response)
    
    elif page == "📈 Results & Insights":
        st.header("Results and Insights")
        
        if not st.session_state.models_trained:
            st.warning("No trained models found. Please train models first in the 'AutoML Training' page.")
            return
        
        results = st.session_state.models_trained
        task_type = st.session_state.get('task_type', 'classification')
        
        st.subheader("Model Performance Comparison")
        
        results_df = pd.DataFrame({
            model_name: result["metrics"] 
            for model_name, result in results.items()
        }).T
        
        if task_type == "classification":
            best_metric = "accuracy"
        else:
            best_metric = "r2_score"
        
        best_model = results_df[best_metric].idxmax()
        st.success(f"🏆 Best performing model: **{best_model}**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                x=results_df.index,
                y=results_df[best_metric],
                title=f"Model Comparison - {best_metric.title()}"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                x=results_df.index,
                y=results_df["training_time"],
                title="Training Time Comparison"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Detailed Metrics")
        st.dataframe(results_df.style.highlight_max(axis=0))
        
        if NLTK_AVAILABLE and st.session_state.df is not None:
            st.subheader("Sentiment Analysis")
            df = st.session_state.df
            text_column = st.session_state.get('text_column')
            
            if text_column and st.button("Analyze Sentiment"):
                with st.spinner("Analyzing sentiment..."):
                    sample_texts = df[text_column].dropna().head(100).tolist()
                    sentiments = genai_assistant.analyze_text_sentiment(sample_texts)
                    
                    fig = create_visualization(df, "sentiment_distribution", sentiments=sentiments)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Model Insights")
        
        selected_model = st.selectbox("Select model for detailed analysis", list(results.keys()))
        
        if selected_model:
            model_info = results[selected_model]
            metrics = model_info["metrics"]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                primary_metric = metrics.get("accuracy", metrics.get("r2_score", 0))
                st.metric("Primary Score", f"{primary_metric:.4f}")
            
            with col2:
                st.metric("Training Time", f"{metrics['training_time']:.2f}s")
            
            with col3:
                if "f1_score" in metrics:
                    st.metric("F1 Score", f"{metrics['f1_score']:.4f}")
                elif "mae" in metrics:
                    st.metric("MAE", f"{metrics['mae']:.4f}")
        
        st.subheader("Export Results")
        
        if st.button("Download Model Report"):
            report = {
                "timestamp": datetime.now().isoformat(),
                "task_type": task_type,
                "best_model": best_model,
                "all_results": {k: v["metrics"] for k, v in results.items()}
            }
            report_str = str(report)
            b64 = base64.b64encode(report_str.encode()).decode()
            href = f'<a href="data:file/txt;base64,{b64}" download="model_report.txt">Download Report</a>'
            st.markdown(href, unsafe_allow_html=True)  
        if st.button("Download Best Model"):
            best_model_obj = results[best_model]["model"]
            vectorizer = st.session_state.get('vectorizer')
            model_package = {
                "model": best_model_obj,
                "vectorizer": vectorizer,
                "task_type": task_type,
                "text_column": st.session_state.get('text_column'),
                "target_column": st.session_state.get('target_column')
            }
            buffer = io.BytesIO()
            pickle.dump(model_package, buffer)
            buffer.seek(0)
            b64 = base64.b64encode(buffer.read()).decode()
            href = f'<a href="data:file/pickle;base64,{b64}" download="best_model.pkl">Download Best Model</a>'
            st.markdown(href, unsafe_allow_html=True)

        # Download report as JSON
        if st.button("Download Report (JSON)"):
            import json
            report = {
                "timestamp": datetime.now().isoformat(),
                "task_type": task_type,
                "best_model": best_model,
                "all_results": {k: v["metrics"] for k, v in results.items()},
                "configuration": {
                    "text_column": st.session_state.get('text_column'),
                    "target_column": st.session_state.get('target_column'),
                    "test_size": st.session_state.get('test_size', 0.2),
                    "cleaning_level": st.session_state.get('cleaning_level', "moderate")
                }
            }
            report_json = json.dumps(report, indent=2, default=str)
            st.download_button(
                label="📄 Download Report (JSON)",
                data=report_json,
                file_name=f"automl_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        # Model Predictions Viewer
        if st.button("Show Sample Predictions"):
            if 'X_test' in st.session_state and 'y_test' in st.session_state:
                X_test = st.session_state.X_test
                y_test = st.session_state.y_test
                
                # Get predictions from best model
                best_model_obj = results[best_model]["model"]
                predictions = best_model_obj.predict(X_test[:10])  # Show first 10 predictions
                
                # Create predictions dataframe
                pred_df = pd.DataFrame({
                    "Actual": y_test[:10] if hasattr(y_test, '__getitem__') else list(y_test)[:10],
                    "Predicted": predictions,
                    "Model": [best_model] * len(predictions)
                })
                
                st.subheader("Sample Predictions")
                st.dataframe(pred_df)
                
                # Prediction accuracy visualization
                if task_type == "classification":
                    correct = (pred_df["Actual"] == pred_df["Predicted"]).sum()
                    total = len(pred_df)
                    st.metric("Sample Accuracy", f"{correct}/{total} ({correct/total*100:.1f}%)")
                else:
                    mae_sample = np.mean(np.abs(pred_df["Actual"] - pred_df["Predicted"]))
                    st.metric("Sample MAE", f"{mae_sample:.4f}")

        # Feature Importance (for tree-based models)
        st.subheader("Feature Analysis")
        
        if selected_model in ["Random Forest", "Decision Tree"]:
            model_obj = results[selected_model]["model"]
            
            if hasattr(model_obj, 'feature_importances_'):
                vectorizer = st.session_state.get('vectorizer')
                if vectorizer:
                    try:
                        feature_names = vectorizer.get_feature_names_out()
                    except AttributeError:
                        feature_names = vectorizer.get_feature_names()

                    importances = model_obj.feature_importances_
                    
                    n = min(20, len(importances))
                    top_indices = np.argsort(importances)[-n:][::-1]  # descending
                    top_features = [feature_names[i] for i in top_indices]
                    top_importances = importances[top_indices]

                    
                    fig = px.bar(
                        x=top_importances,
                        y=top_features,
                        orientation='h',
                        title="Top 20 Feature Importances",
                        labels={'x': 'Importance', 'y': 'Features'}
                    )
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Advanced Analytics
        st.subheader("Advanced Analytics")
        
        analytics_option = st.selectbox(
            "Choose Analysis Type",
            ["Confusion Matrix", "Learning Curves", "Model Comparison", "Error Analysis"]
        )
        
        if analytics_option == "Confusion Matrix" and task_type == "classification":
            if st.button("Generate Confusion Matrix"):
                y_test = st.session_state.get('y_test')
                if y_test is not None:
                    y_pred = results[selected_model]["predictions"]
                    
                    # Create confusion matrix
                    cm = confusion_matrix(y_test, y_pred)
                    
                    fig = px.imshow(
                        cm,
                        text_auto=True,
                        aspect="auto",
                        title=f"Confusion Matrix - {selected_model}",
                        labels=dict(x="Predicted", y="Actual")
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        elif analytics_option == "Model Comparison":
            if len(results) > 1:
                comparison_metric = st.selectbox(
                    "Select metric for comparison",
                    list(results_df.columns)
                )
                
                fig = px.bar(
                    x=results_df.index,
                    y=results_df[comparison_metric],
                    title=f"Model Comparison - {comparison_metric}",
                    color=results_df[comparison_metric],
                    color_continuous_scale="viridis"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif analytics_option == "Error Analysis" and task_type == "regression":
            if st.button("Analyze Prediction Errors"):
                y_test = st.session_state.get('y_test')
                if y_test is not None:
                    y_pred = results[selected_model]["predictions"]
                    
                    errors = y_test - y_pred
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.histogram(errors, title="Prediction Error Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.scatter(
                            x=y_test, 
                            y=y_pred,
                            title="Actual vs Predicted",
                            labels={'x': 'Actual', 'y': 'Predicted'}
                        )
                        # Add perfect prediction line
                        min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
                        fig.add_trace(go.Scatter(
                            x=[min_val, max_val], 
                            y=[min_val, max_val],
                            mode='lines',
                            name='Perfect Prediction',
                            line=dict(dash='dash', color='red')
                        ))
                        st.plotly_chart(fig, use_container_width=True)

    # Footer and Information
    st.sidebar.markdown("---")
    st.sidebar.info(
        f"**{APP_CONFIG['app_name']}**\n"
        f"Version: {APP_CONFIG['version']}\n"
        f"Max file size: {APP_CONFIG['max_file_size']}MB\n"
        f"Supported formats: {', '.join(APP_CONFIG['supported_formats'])}"
    )
    
    # Help section
    with st.sidebar.expander("❓ Help & Tips"):
        st.markdown("""
        **Getting Started:**
        1. Upload your e-commerce dataset (CSV, Excel, or JSON)
        2. Configure text and target columns
        3. Train multiple ML models automatically
        4. Compare results and get insights
        
        **Data Requirements:**
        - At least one text column (reviews, comments, etc.)
        - One target column (ratings, sentiment, categories)
        - Minimum 10 rows recommended
        
        **Tips:**
        - Use 'aggressive' preprocessing for noisy text
        - Try different test sizes for small datasets
        - Check feature importance for interpretability
        """)
    
    # Performance monitoring
    if st.sidebar.button("🔧 System Info"):
        st.sidebar.json({
            "Python Libraries": {
                "Pandas": pd.__version__,
                "Scikit-learn": "Available",
                "NLTK": "Available" if NLTK_AVAILABLE else "Not Available",
                "Plotly": "Available"
            },
            "Session State": {
                "Dataset Loaded": st.session_state.df is not None,
                "Models Trained": len(st.session_state.models_trained),
                "Task Type": st.session_state.get('task_type', 'None')
            }
        })

# Additional utility functions for future enhancements
def export_model_pickle(model, filename):
    """Export trained model as pickle file"""
    try:
        model_bytes = pickle.dumps(model)
        return model_bytes
    except Exception as e:
        st.error(f"Error exporting model: {str(e)}")
        return None

def load_model_pickle(uploaded_file):
    """Load model from pickle file"""
    try:
        model = pickle.load(uploaded_file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def create_model_card(model_info, task_type):
    """Create a model card with metadata"""
    return {
        "model_name": model_info["metrics"]["model_name"],
        "task_type": task_type,
        "performance": model_info["metrics"],
        "created_at": datetime.now().isoformat(),
        "framework": "scikit-learn",
        "preprocessing": "TF-IDF Vectorization"
    }

# Run the application
if __name__ == "__main__":
    main()