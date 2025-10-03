# 📁 BudgetWise AI - Directory Structure

This document outlines the organized directory structure of the BudgetWise AI project.

## 🏗️ Root Directory Structure

```
BudgetWise-AI-based-Expense-Forecasting-Tool/
├── 📄 README.md                     # Main project documentation
├── 📄 requirements.txt              # Python dependencies
├── 📄 setup.py                      # Project setup script
├── 📄 launch_app.py                 # Application launcher
├── 📄 streamlit_app.py              # Streamlit Cloud entry point
├── 🚫 .gitignore                    # Git ignore rules
├── 🔧 .streamlit/                   # Streamlit configuration
│
├── 📊 **DATA PIPELINE**
│   ├── 📁 data/                     # Data storage
│   │   ├── raw/                     # Original datasets
│   │   ├── processed/               # Cleaned & processed data
│   │   └── features/                # Feature-engineered data
│   │
│   ├── 📁 src/                      # Source code
│   │   ├── data_preprocessing.py    # Data cleaning pipeline
│   │   └── feature_engineering.py  # Feature creation
│   │
│   └── 📁 scripts/                  # Training scripts
│       ├── train_models.py          # Main training pipeline
│       ├── ml_training.py           # ML model training
│       ├── deep_learning_training.py # DL model training
│       └── transformer_training.py  # Transformer training
│
├── 🤖 **MODELS & CONFIG**
│   ├── 📁 models/                   # Trained models
│   │   ├── baseline/                # Statistical models
│   │   ├── ml/                      # Machine learning models
│   │   ├── deep_learning/           # Neural network models
│   │   └── transformer/             # Transformer models
│   │
│   └── 📁 config/                   # Configuration files
│       └── config.yaml              # Main configuration
│
├── 🌐 **APPLICATION**
│   ├── 📁 app/                      # Streamlit application
│   │   ├── budgetwise_app.py        # Main application
│   │   ├── USER_MANUAL.md           # User guide
│   │   └── DEPLOYMENT_GUIDE.md      # Deployment instructions
│   │
│   └── 📁 notebooks/                # Jupyter notebooks
│       └── data_Preprocessing.ipynb # EDA notebook
│
├── 📋 **DOCUMENTATION**
│   ├── 📁 docs/                     # All documentation
│   │   ├── project_specs/           # Project specifications
│   │   │   ├── Project Description.md
│   │   │   └── Roadmap.md
│   │   │
│   │   ├── technical_reports/       # Technical documentation
│   │   │   ├── PROJECT_SUMMARY.md
│   │   │   ├── ML_PIPELINE_RESULTS.md
│   │   │   ├── DATA_PIPELINE_CONFIRMATION.md
│   │   │   ├── ENHANCED_PREPROCESSING_SUMMARY.md
│   │   │   ├── CURRENCY_OUTLIER_FIXES.md
│   │   │   ├── OUTLIER_FIX_DOCUMENTATION.md
│   │   │   ├── OUTLIER_CURRENCY_FIX_SUMMARY.md
│   │   │   ├── ML_PIPELINE_EXECUTION_COMPLETE.md
│   │   │   └── TEST_VERIFICATION_PLAN.md
│   │   │
│   │   └── deployment/              # Deployment guides
│   │       ├── STREAMLIT_CLOUD_FIX.md
│   │       └── STREAMLIT_FIX_SUMMARY.md
│   │
│   └── 📁 reports/                  # Generated reports
│
├── 🔧 **UTILITIES & TESTING**
│   ├── 📁 utils/                    # Utility scripts
│   │   ├── analyze_amounts.py       # Data analysis utilities
│   │   ├── analyze_daily_aggregation.py
│   │   ├── analyze_ml_results.py
│   │   ├── check_all_models.py      # Model verification
│   │   ├── check_data_stats.py      # Data statistics
│   │   ├── test_capping.py          # Testing scripts
│   │   ├── test_data_loading.py
│   │   ├── test_streamlit_models.py
│   │   ├── verify_accuracy_metrics.py
│   │   ├── enhancement_summary.py   # Enhancement utilities
│   │   ├── final_verification.py
│   │   ├── capping_analysis.py
│   │   └── get_model_results.py
│   │
│   └── 📁 tests/                    # Test suite
│
├── 📝 **LOGS & TEMPORARY FILES**
│   ├── 📁 logs/                     # Log files
│   │   ├── training.log             # Training logs
│   │   └── transformer_training.log # Transformer logs
│   │
│   ├── 📁 temp/                     # Temporary files
│   │   ├── COMPREHENSIVE_ML_RESULTS.csv
│   │   └── sample_expense_data.csv  # Sample data for cloud
│   │
│   └── 📁 myvenv/                   # Virtual environment
```

## 📂 Directory Descriptions

### **📊 Data Pipeline**
- **`data/`**: All datasets (raw, processed, features)
- **`src/`**: Core data processing modules
- **`scripts/`**: Model training and pipeline scripts

### **🤖 Models & Configuration**
- **`models/`**: Trained model artifacts organized by type
- **`config/`**: Configuration files for the entire system

### **🌐 Application**
- **`app/`**: Streamlit web application with user guides
- **`notebooks/`**: Jupyter notebooks for analysis and experimentation

### **📋 Documentation**
- **`docs/project_specs/`**: Original project requirements and roadmap
- **`docs/technical_reports/`**: Technical documentation and analysis reports
- **`docs/deployment/`**: Deployment guides and cloud setup instructions
- **`reports/`**: Generated reports and benchmarks

### **🔧 Utilities & Testing**
- **`utils/`**: Analysis scripts, verification tools, and utilities
- **`tests/`**: Test suite for quality assurance

### **📝 Support Files**
- **`logs/`**: Training and application logs
- **`temp/`**: Temporary files and sample data
- **`myvenv/`**: Python virtual environment

## 🎯 Benefits of This Organization

### **✅ Improved Navigation**
- Clear separation of concerns
- Logical grouping of related files
- Easy to find specific components

### **✅ Better Maintainability**
- Modular structure supports easier updates
- Clear documentation hierarchy
- Separated utilities from core code

### **✅ Enhanced Collaboration**
- Intuitive structure for team members
- Comprehensive documentation organization
- Clear distinction between development and production files

### **✅ Production Ready**
- Clean root directory
- Proper separation of temporary and permanent files
- Organized deployment documentation

## 🚀 Quick Navigation

| **Need** | **Go To** |
|----------|-----------|
| Run the app | `python launch_app.py` |
| View models | `models/` directory |
| Read docs | `docs/` directory |
| Check logs | `logs/` directory |
| Find utilities | `utils/` directory |
| View data | `data/` directory |

---

**Last Updated**: October 2, 2025  
**Structure Version**: 2.0 (Organized)