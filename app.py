import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Dry Bean Classification", layout="wide")

st.title("🫘 Dry Bean Species Classification Dashboard")
st.markdown("Interactive machine learning platform for evaluating multiclass dry bean classification models.")

@st.cache_resource
def load_artifacts():
    scaler = joblib.load('model/scaler.pkl')
    le = joblib.load('model/label_encoder.pkl')
    models = {
        'Logistic Regression': joblib.load('model/logistic_regression.pkl'),
        'Decision Tree': joblib.load('model/decision_tree.pkl'),
        'kNN': joblib.load('model/knn.pkl'),
        'Naive Bayes': joblib.load('model/naive_bayes.pkl'),
        'Random Forest (Ensemble)': joblib.load('model/random_forest.pkl'),
        'Gradient Boosting (Ensemble)': joblib.load('model/gradient_boosting.pkl')
    }
    return scaler, le, models

scaler, le, models = load_artifacts()

st.sidebar.header("⚙️ Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Test CSV File", type=["csv"])
selected_model_name = st.sidebar.selectbox("Select ML Model", list(models.keys()))

if uploaded_file is not None:
    df_test = pd.read_csv(uploaded_file)
    st.subheader("📋 Uploaded Dataset Preview")
    st.dataframe(df_test.head(5), use_container_width=True)
    
    if 'Class' in df_test.columns:
        X_test = df_test.drop(columns=['Class'])
        y_true_str = df_test['Class']
        y_true = le.transform(y_true_str)
        
        model = models[selected_model_name]
        
        if selected_model_name in ['Logistic Regression', 'kNN', 'Naive Bayes']:
            X_eval = scaler.transform(X_test)
        else:
            X_eval = X_test.values
            
        y_pred = model.predict(X_eval)
        y_prob = model.predict_proba(X_eval)
        
        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        prec = precision_score(y_true, y_pred, average='macro')
        rec = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        mcc = matthews_corrcoef(y_true, y_pred)
        
        st.subheader(f"📊 Evaluation Metrics: {selected_model_name}")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Accuracy", f"{acc:.4f}")
        c2.metric("AUC Score", f"{auc:.4f}")
        c3.metric("Precision", f"{prec:.4f}")
        c4.metric("Recall", f"{rec:.4f}")
        c5.metric("F1 Score", f"{f1:.4f}")
        c6.metric("MCC Score", f"{mcc:.4f}")
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("🎯 Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
            
        with col_right:
            st.subheader("📜 Classification Report")
            report_dict = classification_report(y_true_str, le.inverse_transform(y_pred), output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()
            st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)
    else:
        st.error("Uploaded CSV must contain the 'Class' target column for evaluation.")
else:
    st.info("👈 Upload `test_data.csv` in the sidebar to run model evaluations.")