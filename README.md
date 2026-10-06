# 📦 Supply Chain Demand Forecasting & Anomaly Detection

A machine-learning system that predicts future product demand using historical Walmart sales data, and flags unusual demand patterns (sudden spikes or drops) that could signal supply chain problems.

## 🎯 What It Does

- **Demand Forecasting** — Uses Facebook Prophet to predict weekly sales for any store/department combination
- **Anomaly Detection** — Uses Isolation Forest to flag unusual sales patterns automatically
- **Interactive Dashboard** — Streamlit-based dark-mode dashboard for exploring data, forecasts, and anomalies
- **SQL Analytics** — SQLite database with pre-built analytical queries

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **Python** | Data cleaning, feature engineering, ML models |
| **SQL (SQLite)** | Store and query historical sales data |
| **Facebook Prophet** | Time-series demand forecasting |
| **scikit-learn** | Anomaly detection (Isolation Forest) |
| **Streamlit** | Interactive dashboard |
| **Plotly** | Data visualizations |

## 📁 Project Structure

```
Supply-chain-demand-forecasting/
├── data/
│   ├── raw/                    ← Original Walmart dataset files
│   └── processed/
│       └── cleaned_data.csv    ← Cleaned & merged dataset
├── sql/
│   ├── schema.sql              ← Table definitions
│   ├── queries.sql             ← Business analytics queries
│   └── walmart.db              ← SQLite database
├── src/
│   ├── preprocessing.py        ← Data cleaning & feature engineering
│   ├── create_database.py      ← Load CSV into SQLite
│   ├── model.py                ← Prophet forecasting functions
│   └── anomaly.py              ← Isolation Forest anomaly detection
├── notebooks/
│   └── model_training.ipynb    ← Exploratory analysis & model experiments
├── dashboard/
│   └── app.py                  ← Streamlit dashboard
├── .streamlit/
│   └── config.toml             ← Dark theme configuration
├── report/
│   ├── project_report.docx
│   └── presentation.pptx
├── README.md
└── requirements.txt
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/Supply-chain-demand-forecasting.git
   cd Supply-chain-demand-forecasting
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up the data** (if starting from scratch)
   ```bash
   # Step 1: Place Walmart dataset files in data/raw/
   # Step 2: Run preprocessing
   python src/preprocessing.py
   # Step 3: Create SQLite database
   python src/create_database.py
   ```

### Run the Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard will open at `http://localhost:8501` with a dark-mode interface where you can:
- Select any Store and Department to analyze
- View historical sales trends and monthly breakdowns
- Generate Prophet forecasts with adjustable horizons
- Detect sales anomalies with configurable sensitivity
- Compare store performance across the entire dataset

## 📊 Dashboard Features

| Tab | Description |
|---|---|
| **Sales Overview** | Weekly trend line, monthly bar chart, holiday vs non-holiday comparison |
| **Forecast** | Prophet predictions with confidence bands, MAE/RMSE metrics, forecast data table |
| **Anomaly Detection** | Sales timeline with anomaly markers, anomaly rate metrics, detailed anomaly table |
| **Store Analytics** | Top stores ranking, sales by store type, store × month heatmap |

## 📂 Dataset

This project uses the [Walmart Recruiting – Store Sales Forecasting](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting) dataset from Kaggle.

- **train.csv** — Historical weekly sales for 45 stores × 99 departments
- **features.csv** — Store, date, temperature, fuel price, markdowns, CPI, unemployment
- **stores.csv** — Store type and size

## 📄 License

This project is for educational purposes as part of a university course project.
