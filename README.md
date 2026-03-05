# CSV Insight Bot 🤖📊

CSV Insight Bot is an **Agentic AI web application** that automatically analyzes CSV datasets and generates insights, visualizations, and machine learning models.

Upload a dataset, choose a target column, and the system will automatically perform data cleaning, exploratory data analysis, and machine learning to extract actionable insights.

> This project demonstrates how **AI agents can automate a complete data science workflow**.

🌐 **Live Demo:** [csv-insight-bot.onrender.com](https://csv-insight-bot.onrender.com)

---

## 🚀 Features

### 🔍 1. Schema Agent
Analyzes the dataset structure.
- Detects numeric and categorical columns
- Identifies missing values
- Displays dataset size and schema summary

### 🧹 2. Cleaning Agent
Prepares the dataset for analysis.
- Fills missing numeric values using the median
- Fills categorical values using the mode
- Generates a cleaning report

### 📊 3. EDA Agent *(Exploratory Data Analysis)*
Automatically generates visualizations including:
- Target distribution
- Correlation heatmap
- Feature relationships

### 🤖 4. Model Agent
Builds a baseline machine learning model.

**Supported models:**
- Random Forest
- Logistic Regression

**Automatically performs:**
- Train/test split
- Feature encoding
- Model training & accuracy evaluation
- Feature importance calculation

### 💡 5. Insight Agent
Generates human-readable insights explaining:
- Key features influencing predictions
- Data trends
- Model results

### 📄 6. Report Agent
Creates a downloadable `.txt` report summarizing:
- Dataset information
- Cleaning steps
- Model performance
- Generated insights

---

## 🧠 Example Use Case

Using the **Titanic dataset**, the system automatically discovers insights such as:

- Survival differences between genders
- Impact of passenger class on survival
- Importance of age and fare as predictive features
- Key features ranked by the ML model

---

## 🛠 Tech Stack

| Layer | Technologies |
|---|---|
| **Backend** | Python, FastAPI |
| **Data Science** | Pandas, Scikit-learn, Matplotlib, Seaborn |
| **Frontend** | HTML, Jinja2 Templates, CSS |
| **Deployment** | Render |

---

## 📂 Project Structure

```
csv-insight-bot/
│
├── app/
│   ├── agents/
│   │   ├── schema_agent.py
│   │   ├── cleaning_agent.py
│   │   ├── eda_agent.py
│   │   ├── model_agent.py
│   │   ├── insight_agent.py
│   │   ├── feature_plot_agent.py
│   │   └── report_download_agent.py
│   │
│   ├── templates/
│   │   ├── base.html
│   │   ├── index.html
│   │   ├── configure.html
│   │   └── results.html
│   │
│   ├── static/
│   │   └── styles.css
│   │
│   └── storage/
│       ├── uploads/
│       └── outputs/
│
├── main.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation (Run Locally)

**1. Clone the repository**
```bash
git clone https://github.com/Tharumini03/csv-insight-bot.git
cd csv-insight-bot
```

**2. Create a virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the application**
```bash
uvicorn main:app --reload
```

**5. Open in browser**
```
http://127.0.0.1:8000
```

---

## 📈 Future Improvements

- [ ] Interactive charts using Plotly
- [ ] Additional ML models (XGBoost, SVM)
- [ ] Automatic feature engineering
- [ ] LLM-based natural language insight generation
- [ ] More advanced UI/UX

---

## 🎯 Learning Objectives

This project demonstrates:
- Building **agent-based AI pipelines**
- Creating **FastAPI** web applications
- Automating **end-to-end data science workflows**
- Deploying Python applications to the cloud

---

## 👩‍💻 Author

**Tharumini Gamage**  
Computer Science Undergraduate  
University of Moratuwa  
[GitHub](https://github.com/Tharumini03)
