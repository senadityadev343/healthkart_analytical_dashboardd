# HealthKart Influencer Marketing Analytics Dashboard

A Streamlit-based interactive dashboard for synthetic influencer marketing campaign analytics for HealthKart. Generate and explore fake campaign data—covering influencers, posts, conversions, payouts, competitor benchmarking, advanced time-series & cohort analyses, Monte Carlo risk simulations, and AI-powered insights.

## 🚀 Features

- **Synthetic Data Generation**  
  - 150 influencers with realistic tiers, platforms, demographics & engagement  
  - 1,000 posts across Instagram, YouTube, Twitter, Facebook, TikTok  
  - 8,000+ conversion events with order-level revenue tracking  
  - Competitor spend & performance simulation  

- **Interactive Dashboard Tabs**  
  1. **Campaign Performance**: Sunburst, Trends, Funnel, Budget Simulation  
  2. **Influencer Insights**: Top-performers, radar & treemap charts, audience demographics  
  3. **Payout Tracking & Risk**: GST breakdown, Monte Carlo risk gauge  
  4. **Advanced Analytics**: Incremental ROAS (CausalImpact), Seasonality, Anomaly Detection  
  5. **AI Advisor**: Gemini-powered natural-language insights (requires Google API key)  
  6. **Settings**: Data filters by date, platform, category, tier, brand, product  
  7. **Export Data**: CSV/PDF report generation  
  8. **Boardroom Insights**: Executive-style summary for presentations  

- **Analytics Modules**  
  - ROAS, ROI, CPA calculators  
  - Budget-impact simulation  
  - Cohort & retention analysis  
  - Shapley, Markov & custom attribution modeling  
  - Customer LTV survival curves (Kaplan–Meier)  
  - Prophet-based revenue forecasting  
  - Competitor benchmarking  

## 🛠️ Tech Stack & Dependencies

- Python 3.10+  
- Streamlit  
- Pandas, NumPy  
- Plotly (express & graph_objects)  
- Faker (synthetic data)  
- scikit-learn (IsolationForest, StandardScaler)  
- Prophet (time-series forecasting)  
- lifelines (Kaplan–Meier)  
- statsmodels (ARIMA)  
- reportlab (PDF export)  
- Google Generative AI (`google-generativeai`)  
- Optional: `causalimpact` for CausalImpact analysis  

## ⚙️ Installation & Setup

1. **Clone the repo**  
   ```bash
   git clone https://github.com/senadityadev343/healthkart_analytical_dashboardd.git
   cd healthkart_analytical_dashboardd
   ```

2. **Create & activate a Python virtual environment**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate    # Linux/Mac
   venv\Scripts\activate       # Windows
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure secrets**  
   - Create a `.streamlit/secrets.toml` file:  
     ```toml
     [google]
     api_key = "YOUR_GEMINI_API_KEY_HERE"
     ```
   - If you don't need AI Advisor and Boardroom Insights, you can skip adding your Google API key.

5. **Running the App**  
   ```bash
   streamlit run main.py
   ```
