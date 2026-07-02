# CognitIDS 🛡️
### A Context-Based Explainable Intrusion Detection System for IoT Networks

CognitIDS is a full-stack machine learning-powered IDS trained on the 
Edge-IIoTset dataset (2.2M packets, 14 attack types). It combines XGBoost 
classification, SHAP explainability, and a context-aware severity engine 
to deliver real-time intrusion detection with actionable, transparent alerts.

## Key Features
- ✅ 100% classification accuracy on 332K unseen test packets
- 🔍 SHAP explainability — every alert tells you WHY it was flagged
- 🧠 Context-aware engine with sliding window severity scoring (LOW → CRITICAL)
- 📊 Real-time GUI dashboard with live traffic feed and alert log
- ⚡ Zero false positives | Zero false negatives

## Tech Stack
Python • XGBoost • SHAP • CustomTkinter • Pandas • Scikit-learn • Matplotlib

## Dataset
Edge-IIoTset — 2,219,201 packets | 63 features | 14 attack categories
