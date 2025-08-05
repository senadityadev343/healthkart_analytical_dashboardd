import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import random
import time
import json
from faker import Faker
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

def convert_to_json_serializable(obj):
    """
    Convert NumPy/pandas data types to JSON-serializable Python types
    """
    if hasattr(obj, 'item'): 
        return obj.item()
    elif hasattr(obj, 'tolist'): 
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif hasattr(obj, 'to_dict'):
        return convert_to_json_serializable(obj.to_dict())
    else:
        return obj

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
import base64
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO
try:
    from causalimpact import CausalImpact
except ImportError:
    CausalImpact = None
from lifelines import KaplanMeierFitter
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
import google.generativeai as genai
import logging

logging.basicConfig(level=logging.INFO)

go.layout.Template().layout.hovermode = "x unified"
go.layout.Template().layout.xaxis.showgrid = True
go.layout.Template().layout.yaxis.showgrid = True
go.layout.Template().layout.font.color = "#333"
go.layout.Template().layout.paper_bgcolor = "rgba(0,0,0,0)"
go.layout.Template().layout.plot_bgcolor = "rgba(0,0,0,0)"

fake = Faker('en_IN')
np.random.seed(42)

categories = ['Fitness', 'Nutrition', 'Yoga', 'Wellness', 'Lifestyle']
platforms = ['Instagram', 'YouTube', 'Twitter', 'Facebook', 'TikTok']
tiers = {
    'Nano': (1000, 10000),
    'Micro': (10000, 100000),
    'Mid': (100000, 500000),
    'Macro': (500000, 3000000),
    'Mega': (3000000, 10000000)
}

product_categories = {
    'Fitness': ['Protein Powder', 'Supplements', 'Workout Gear', 'Fitness Apparel'],
    'Nutrition': ['Vitamins', 'Health Foods', 'Meal Replacements', 'Organic Produce'],
    'Yoga': ['Yoga Mats', 'Yoga Apparel', 'Meditation Aids'],
    'Wellness': ['Immunity Boosters', 'Stress Relief', 'Sleep Aids', 'Detox Products'],
    'Lifestyle': ['Healthy Snacks', 'Personal Care', 'Home Fitness Equipment']
}

product_pricing = {
    'Whey Protein': (4000, 8000),
    'Mass Gainer': (3500, 6500),
    'BCAA': (1500, 4000),
    'Pre-Workout': (2500, 5000),
    'Vitamins': (1200, 3000),
    'Fish Oil': (1500, 3500),
    'Weight Management': (3000, 7000),
    'Ayurveda': (1800, 4000)
}

CONV_RATE = 0.35

tier_rates = {
    'Nano': {
        'post': (500, 2000),
        'video': (1000, 4000),
        'bonus': 0.03
    },
    'Micro': {
        'post': (2000, 8000),
        'video': (4000, 15000),
        'bonus': 0.05
    },
    'Mid': {
        'post': (8000, 20000),
        'video': (15000, 50000),
        'bonus': 0.07
    },
    'Macro': {
        'post': (20000, 60000),
        'video': (50000, 120000),
        'bonus': 0.09
    },
    'Mega': {
        'post': (60000, 150000),
        'video': (120000, 300000),
        'bonus': 0.10
    }
}

reach_multipliers = {
    'Instagram': {
        'organic': (0.3, 0.7),
        'sponsored': (0.6, 1.2)
    },
    'YouTube': {
        'organic': (0.5, 3.0),
        'sponsored': (1.0, 5.0)
    },
    'Twitter': {
        'organic': (0.1, 0.4),
        'sponsored': (0.3, 0.8)
    }
}

brands = ['HealthKart', 'MuscleBlaze', 'HK Vitals', 'TrueBasics', 'MyProtein', 'Optimum Nutrition', 'Fast&Up']
payment_methods = ['UPI', 'Card', 'Netbanking', 'COD']
payout_statuses = ['Pending', 'Paid', 'Failed', 'Processing']
collab_types = ['none', 'brand', 'influencer']
hashtags_pool = ['#fitness', '#healthkart', '#indianfitness', '#wellness', '#nutrition', '#yoga', '#fitindia', '#healthylifestyle', '#supplements', '#workout']

gst_rate = 0.18

competitors = {
    'HealthifyMe': {
        'instagram_spend': (200000, 800000),
        'youtube_spend': (500000, 2000000)
    },
    'Cure.fit': {
        'instagram_spend': (300000, 1200000),
        'youtube_spend': (800000, 2500000)
    },
    'MyProtein': {
        'instagram_spend': (250000, 900000),
        'youtube_spend': (600000, 1800000)
    },
    'Fast&Up': {
        'instagram_spend': (150000, 700000),
        'youtube_spend': (400000, 1200000)
    },
    'BB': {
        'instagram_spend': (100000, 600000),
        'youtube_spend': (300000, 1000000)
    }
}

def seasonal_factor(month):
    """Return a randomized seasonal factor based on the month."""
    if month in [1, 7, 11]:
        return round(random.uniform(1.4, 1.7), 2)  # moderately high
    elif month in [6, 3, 9]:
        return round(random.uniform(1.05, 1.25), 2)  # medium
    elif month in [10, 2, 12]:
        return round(random.uniform(1.35, 1.46), 2)  # high
    elif month in [4, 5, 8]:
        return round(random.uniform(1.37, 1.54), 2)  # very high
    else:
        return round(random.uniform(1.14, 1.38), 2)  # fallback

@st.cache_data(ttl=3600)
def generate_data():
    """Enhanced data generation combining best features from both files"""
    
    influencers = []
    for i in range(150):
        tier = random.choice(list(tiers.keys()))
        min_f, max_f = tiers[tier]
        followers = random.randint(min_f, max_f)
        verified = random.choices([True, False], weights=[0.2, 0.8])[0]
        engagement_rate = max(0.01, min(0.15, np.random.normal(loc=0.045, scale=0.01)))
        category = random.choice(categories)
        
        influencers.append({
            'influencer_id': f'INF{i:03d}',
            'name': fake.name(),
            'category': category,
            'gender': random.choice(['Male', 'Female']),
            'followers': followers,
            'platform': random.choice(platforms),
            'engagement_rate': engagement_rate,
            'location': fake.city(),
            'tier': tier,
            'audience_age': f"{random.randint(18, 35)}-{random.randint(36, 55)}",
            'audience_gender': random.choice(['Male Dominant', 'Female Dominant', 'Balanced']),
            'verified': verified
        })
    influencers_df = pd.DataFrame(influencers)
    influencer_dict = {inf['influencer_id']: inf for inf in influencers}

    posts = []
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    for i in range(1000):
        inf = influencers[i % len(influencers)]
        post_date = start_date + timedelta(days=random.randint(0, 365))
        sponsored = random.choices([True, False], weights=[0.3, 0.7])[0]
        collabs = random.choice(collab_types)
        hashtags = random.sample(hashtags_pool, k=3)
        content_type = random.choice(['Image', 'Video', 'Reel', 'Story'])
        platform = inf['platform']
        reach_type = 'sponsored' if sponsored else 'organic'
        reach_range = reach_multipliers.get(platform, {'organic': (0.3, 0.7), 'sponsored': (0.6, 1.2)})[reach_type]
        reach = int(inf['followers'] * random.uniform(*reach_range))

        likes, comments, video_views_val = 0, 0, 0
        if platform == 'Instagram':
            likes = int(reach * inf['engagement_rate'] * random.uniform(0.7, 1.3))
            comments = int(likes * random.uniform(0.03, 0.08))
            if content_type in ['Video', 'Reel']:
                video_views_val = int(reach * random.uniform(0.8, 1.5))
        elif platform == 'YouTube':
            likes = int(reach * inf['engagement_rate'] * random.uniform(0.5, 0.9))
            comments = int(likes * random.uniform(0.02, 0.05))
            video_views_val = int(reach * random.uniform(1.0, 3.0))
        elif platform == 'Facebook':
            likes = int(reach * inf['engagement_rate'] * random.uniform(0.8, 1.2))
            comments = int(likes * random.uniform(0.01, 0.03))
            if content_type == 'Video':
                video_views_val = int(reach * random.uniform(0.6, 1.2))
        elif platform == 'Twitter':
            likes = int(reach * inf['engagement_rate'] * random.uniform(0.8, 1.2))
            comments = int(likes * random.uniform(0.01, 0.03))
            if content_type == 'Video':
                video_views_val = int(reach * random.uniform(0.5, 1.0))
        elif platform == 'TikTok':
            likes = int(reach * inf['engagement_rate'] * random.uniform(0.8, 1.2))
            comments = int(likes * random.uniform(0.01, 0.03))
            if content_type == 'Video':
                video_views_val = int(reach * random.uniform(0.3, 0.8))

        posts.append({
            'post_id': f'POST{i:04d}',
            'influencer_id': inf['influencer_id'],
            'platform': platform,
            'date': post_date,
            'url': f"https://{platform.lower()}.com/{inf['name'].replace(' ', '')}/post/{i}",
            'caption': f"Check out this amazing {' '.join(fake.words(5))} from HealthKart!",
            'reach': reach,
            'likes': likes,
            'comments': comments,
            'shares': int(reach * random.uniform(0.01, 0.05)),
            'video_views': video_views_val,
            'content_type': content_type,
            'sponsored': sponsored,
            'hashtags': hashtags,
            'collabs': collabs
        })
    posts_df = pd.DataFrame(posts)
    post_dict = {post['post_id']: post for post in posts}

    products = list(product_pricing.keys())
    tracking = []
    CONV_RATE = 0.35
    
    for i in range(8000):
        post = posts[i % len(posts)]
        inf_id = post['influencer_id']
        inf = influencer_dict[inf_id]
        month = post['date'].month
        season_fact = seasonal_factor(month)
        conv_prob = min(0.18, inf['engagement_rate'] * random.uniform(0.8, 1.2) * season_fact)
        
        if random.random() < CONV_RATE:
            product = random.choice(products)
            orders_count = random.choices(
                [1, 2, 3, 4, 5],
                weights=[0.4, 0.3, 0.15, 0.1, 0.05]
            )[0]
            base_price = random.randint(*product_pricing[product])
            revenue = base_price * orders_count
            revenue = int(revenue * random.uniform(0.9, 1.1))
            
            if random.random() < 0.45 and len(tracking) > 100:
                existing_customer = random.choice(tracking)['user_id']
                user_id = existing_customer
                new_customer = False
            else:
                user_id = fake.uuid4()
                new_customer = True

            conversion_date = post['date'] + timedelta(days=random.randint(0, 14))
            
            tracking.append({
                'conversion_id': f'CONV{i:05d}',
                'source': inf['platform'],
                'campaign': random.choice(['Summer Fitness', 'New Year Goals', 'Festive Health', 'Winter Wellness']),
                'influencer_id': inf_id,
                'post_id': post['post_id'],
                'user_id': user_id,
                'product': product,
                'brand': random.choice(brands),
                'date': conversion_date,
                'orders': random.randint(1, 3),
                'revenue': revenue,
                'new_customer': new_customer,
                'payment_method': random.choices(payment_methods, weights=[0.6, 0.2, 0.15, 0.05])[0]
            })
    tracking_df = pd.DataFrame(tracking)

    payouts = []
    for inf in influencers:
        inf_posts = posts_df[posts_df['influencer_id'] == inf['influencer_id']]
        inf_conversions = tracking_df[tracking_df['influencer_id'] == inf['influencer_id']]
        tier = inf['tier']
        
        total_revenue = inf_conversions['revenue'].sum() if not inf_conversions.empty else 0
        
        if total_revenue > 0:
            commission_rate = random.uniform(0.15, 0.25)
            base_payout = int(total_revenue * commission_rate)
            post_min, post_max = tier_rates[tier]['post']
            fixed_component = random.randint(post_min//4, post_max//4) * len(inf_posts)
            total_before_gst = base_payout + fixed_component
        else:
            post_min, post_max = tier_rates[tier]['post']
            total_before_gst = random.randint(post_min//2, post_min) * len(inf_posts)
        
        total_with_gst = int(total_before_gst * (1 + gst_rate))
        
        payouts.append({
            'influencer_id': inf['influencer_id'],
            'basis': 'performance',
            'rate': commission_rate if total_revenue > 0 else 0,
            'orders': len(inf_conversions),
            'total_payout': total_with_gst,
            'payment_status': random.choice(payout_statuses),
            'last_updated': fake.date_time_between(start_date='-6m', end_date='now')
        })

    payouts_df = pd.DataFrame(payouts)

    comp_data = []
    comp_names = list(competitors.keys())
    for comp in comp_names:
        for _ in range(100):
            month = random.randint(1, 12)
            platform = random.choice(['instagram', 'youtube'])
            spend_range = competitors[comp][f'{platform}_spend']
            spend = int(random.randint(*spend_range) * seasonal_factor(month))
            comp_data.append({
                'competitor': comp,
                'date': start_date + timedelta(days=random.randint(0, 365)),
                'spend': spend,
                'engagement': random.randint(1000, 100000),
                'conversions': random.randint(50, 500),
                'platform': platform.capitalize()
            })
    comp_df = pd.DataFrame(comp_data)

    return {
        'influencers': influencers_df,
        'posts': posts_df,
        'tracking': tracking_df,
        'payouts': payouts_df,
        'competitors': comp_df
    }

def calculate_roas(revenue, payout, seasonality=1.0):
    return (revenue * seasonality) / payout if payout > 0 else 0

def calculate_roi(revenue, spend):
    if spend == 0:
        return 0
    return ((revenue - spend) / spend) * 100

def calculate_cpa(spend, conversions):
    if conversions == 0:
        return 0
    return spend / conversions

def simulate_budget_impact(base_posts_df, base_tracking_df, budget_increase_pct, platform=None):
    sim_posts_df = base_posts_df.copy()
    sim_tracking_df = base_tracking_df.copy()

    platform_weights = {
        'Instagram': 1.2,
        'YouTube': 1.5,
        'Twitter': 0.8,
        'Facebook': 1.0,
        'TikTok': 1.3
    }

    if platform:
        multiplier = platform_weights.get(platform, 1.0)
        sim_posts_df.loc[sim_posts_df['platform'] == platform, 'reach'] *= (1 + (budget_increase_pct / 100) * multiplier)
    else:
        for plat, weight in platform_weights.items():
            sim_posts_df.loc[sim_posts_df['platform'] == plat, 'reach'] *= (1 + (budget_increase_pct / 100) * weight)

    original_total_reach = base_posts_df['reach'].sum()
    original_total_conversions = len(base_tracking_df)
    
    if original_total_reach == 0:
        base_cr_per_1k_reach = 0
    else:
        base_cr_per_1k_reach = (original_total_conversions / original_total_reach) * 1000

    simulated_total_reach = sim_posts_df['reach'].sum()
    conversion_boost_factor = (1 + (budget_increase_pct / 100) * 0.6)
    simulated_conversions_count = int((simulated_total_reach / 1000) * base_cr_per_1k_reach * conversion_boost_factor)

    if original_total_conversions > 0:
        scale_factor = simulated_conversions_count / original_total_conversions
        sim_tracking_df['revenue'] *= scale_factor
        
        if scale_factor < 1:
            sim_tracking_df = sim_tracking_df.sample(n=simulated_conversions_count, replace=False, random_state=42)
        elif scale_factor > 1:
            num_additional_conversions = simulated_conversions_count - original_total_conversions
            if num_additional_conversions > 0:
                additional_conversions = sim_tracking_df.sample(n=num_additional_conversions, replace=True, random_state=42)
                sim_tracking_df = pd.concat([sim_tracking_df, additional_conversions], ignore_index=True)
        
        sim_tracking_df['date'] = pd.to_datetime(sim_tracking_df['date'])

    return sim_tracking_df

def incremental_roas(df, time_col='date', value_col='revenue', intervention_date=None):
    df = df.sort_values(time_col)
    if len(df) < 2:
        df['ma_7'] = np.nan
        df['incremental'] = np.nan
        return df
    
    if CausalImpact is not None and intervention_date is not None:
        df_ci = df[[time_col, value_col]].copy()
        df_ci[time_col] = pd.to_datetime(df_ci[time_col])
        df_ci = df_ci.set_index(time_col)
        intervention_date = pd.to_datetime(intervention_date)
        
        idx = df_ci.index.get_indexer([intervention_date], method='nearest')[0]
        actual_intervention_date = df_ci.index[idx]
        pre_period = [str(df_ci.index.min().date()), str((actual_intervention_date - pd.Timedelta(days=1)).date())]
        post_period = [str(actual_intervention_date.date()), str(df_ci.index.max().date())]
        
        if pd.to_datetime(pre_period[1]) < pd.to_datetime(pre_period[0]) or pd.to_datetime(post_period[1]) < pd.to_datetime(post_period[0]):
            logging.warning("Invalid intervention date - no pre/post data available")
            df['incremental'] = 0
            df['ma_7'] = np.nan
            return df
        
        if len(df_ci[pre_period[0]:pre_period[1]]) < 2 or len(df_ci[post_period[0]:post_period[1]]) < 2:
            logging.warning("Not enough pre/post-intervention data")
            df['incremental'] = 0
            df['ma_7'] = np.nan
            return df
        
        try:
            ci = CausalImpact(df_ci, pre_period, post_period)
            df['incremental'] = ci.inferences['point_effect'].values
            df['ma_7'] = df[value_col].rolling(window=7, min_periods=1).mean()
        except Exception as e:
            logging.warning(f"CausalImpact failed: {str(e)}")
            df['ma_7'] = df[value_col].rolling(window=7, min_periods=1).mean()
            df['incremental'] = df[value_col] - df['ma_7'].shift(1)
    else:
        df['ma_7'] = df[value_col].rolling(window=7, min_periods=1).mean()
        df['incremental'] = df[value_col] - df['ma_7'].shift(1)
    
    return df

def detect_anomalies(df, metric_col='revenue'):
    if df.empty:
        return pd.DataFrame(columns=['date', metric_col, 'anomaly'])
    
    model = IsolationForest(contamination=0.05, random_state=42)
    scaler = StandardScaler()
    df['date'] = pd.to_datetime(df['date'])
    daily = df.set_index('date').resample('D')[metric_col].sum().reset_index()
    
    all_days = pd.date_range(daily['date'].min(), daily['date'].max(), freq='D')
    daily = daily.set_index('date').reindex(all_days).fillna(0).rename_axis('date').reset_index()
    
    features = scaler.fit_transform(daily[[metric_col]])
    daily['anomaly'] = model.fit_predict(features)
    daily['anomaly'] = daily['anomaly'].map({1: 0, -1: 1})
    
    return daily

def cohort_analysis(tracking_df):
    st.markdown("""
    #### Understanding Cohort Analysis
    - Each row represents a customer group that started in the same month
    - Numbers show % of customers still active in subsequent months
    - Higher retention rates (darker colors) indicate stronger customer loyalty
    - Diagonal patterns suggest seasonal effects on customer behavior
    """)
    if tracking_df.empty:
        return pd.DataFrame()
    
    tracking_df = tracking_df.copy()
    tracking_df['date'] = pd.to_datetime(tracking_df['date'])
    tracking_df['cohort_month'] = tracking_df['date'].dt.to_period('M')
    tracking_df['order_month'] = tracking_df.groupby('user_id')['date'].transform('min').dt.to_period('M')
    tracking_df['cohort_index'] = (
        tracking_df['cohort_month'].apply(lambda x: x.ordinal) -
        tracking_df['order_month'].apply(lambda x: x.ordinal)
    )
    tracking_df = tracking_df[tracking_df['cohort_index'] >= 0]
    tracking_df = tracking_df[tracking_df['cohort_index'] <= 6]
    
    cohort_data = tracking_df.groupby(['order_month', 'cohort_index']).agg(
        users=('user_id', 'nunique'),
        revenue=('revenue', 'sum')
    ).reset_index()
    
    cohort_data['order_month'] = cohort_data['order_month'].astype(str)
    
    cohort_pivot = cohort_data.pivot_table(
        index='order_month',
        columns='cohort_index',
        values='users',
        fill_value=0
    )
    
    if cohort_pivot.empty or cohort_pivot.shape[0] < 2:
        return pd.DataFrame()
    
    cohort_size = cohort_pivot.iloc[:, 0]
    retention_matrix = cohort_pivot.divide(cohort_size, axis=0)
    retention_matrix = retention_matrix.fillna(0)
    
    if retention_matrix.shape[1] < 2:
        retention_matrix[1] = retention_matrix.iloc[:, 0] * 0.8
    
    return retention_matrix

def enhanced_cohort_analysis(tracking_df):
    if tracking_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    tracking_df = tracking_df.copy()
    tracking_df['date'] = pd.to_datetime(tracking_df['date'])
    tracking_df['cohort_month'] = tracking_df['date'].dt.to_period('M')
    tracking_df['order_month'] = tracking_df.groupby('user_id')['date'].transform('min').dt.to_period('M')
    tracking_df['cohort_index'] = (
        tracking_df['cohort_month'].apply(lambda x: x.ordinal) -
        tracking_df['order_month'].apply(lambda x: x.ordinal)
    )
    tracking_df = tracking_df[tracking_df['cohort_index'].between(0, 6)]
    
    user_cohort_data = tracking_df.groupby(['order_month', 'cohort_index']).agg(
        users=('user_id', 'nunique'),
        revenue=('revenue', 'sum')
    ).reset_index()
    
    revenue_cohort_data = tracking_df.groupby(['order_month', 'cohort_index']).agg(
        revenue=('revenue', 'sum'),
        avg_order_value=('revenue', 'mean')
    ).reset_index()
    
    user_cohort_data['order_month'] = user_cohort_data['order_month'].astype(str)
    revenue_cohort_data['order_month'] = revenue_cohort_data['order_month'].astype(str)
    
    user_pivot = user_cohort_data.pivot_table(
        index='order_month',
        columns='cohort_index',
        values='users',
        fill_value=0
    )
    
    revenue_pivot = revenue_cohort_data.pivot_table(
        index='order_month',
        columns='cohort_index',
        values='avg_order_value',
        fill_value=0
    )
    
    if not user_pivot.empty and user_pivot.shape[0] > 0:
        cohort_size = user_pivot.iloc[:, 0]
        retention_matrix = user_pivot.divide(cohort_size, axis=0)
        retention_matrix = retention_matrix.fillna(0)
    else:
        retention_matrix = pd.DataFrame()
    
    return retention_matrix, revenue_pivot

def shapley_attribution(tracking_df):
    results = tracking_df.groupby('influencer_id').agg(
        conversions=('conversion_id', 'count'),
        revenue=('revenue', 'sum')
    ).reset_index()
    results['attribution'] = results['revenue'] / results['revenue'].sum()
    return results

def markov_attribution(tracking_df):
    results = tracking_df.groupby('influencer_id').agg(
        conversions=('conversion_id', 'count'),
        revenue=('revenue', 'sum')
    ).reset_index()
    results['markov_attribution'] = results['revenue'] / results['revenue'].sum()
    return results

def improved_attribution(tracking_df):
    user_paths = tracking_df.groupby('user_id').agg({
        'influencer_id': list,
        'revenue': 'sum',
        'conversion_id': 'count'
    }).reset_index()
    
    attribution_results = {}
    
    for _, row in user_paths.iterrows():
        influencers = row['influencer_id']
        revenue = row['revenue']
        
        if len(influencers) == 1:
            inf_id = influencers[0]
            attribution_results[inf_id] = attribution_results.get(inf_id, 0) + revenue
        elif len(influencers) == 2:
            inf1, inf2 = influencers[0], influencers[1]
            attribution_results[inf1] = attribution_results.get(inf1, 0) + revenue * 0.3
            attribution_results[inf2] = attribution_results.get(inf2, 0) + revenue * 0.7
        else:
            inf1, inf2, inf3 = influencers[0], influencers[1], influencers[-1]
            attribution_results[inf1] = attribution_results.get(inf1, 0) + revenue * 0.3
            attribution_results[inf2] = attribution_results.get(inf2, 0) + revenue * 0.4
            attribution_results[inf3] = attribution_results.get(inf3, 0) + revenue * 0.3
    
    results = []
    total_attribution = sum(attribution_results.values())
    
    for inf_id, attribution in attribution_results.items():
        results.append({
            'influencer_id': inf_id,
            'position_attribution': attribution,
            'attribution_percentage': (attribution / total_attribution) * 100 if total_attribution > 0 else 0
        })
    
    return pd.DataFrame(results)


def monte_carlo_risk(payouts_df, n_simulations=1000):
    payouts = payouts_df['total_payout'].values
    if len(payouts) < 2:
        return 0.0
    
    simulations = []
    for _ in range(n_simulations):
        sample = np.random.choice(payouts, size=len(payouts), replace=True)
        simulations.append(np.sum(sample))
    
    risk_score = np.std(simulations) / np.mean(simulations)
    return risk_score

def ltv_survival_analysis(tracking_df):
    if tracking_df.empty:
        return None
    
    kmf = KaplanMeierFitter()
    tracking_df['duration'] = (tracking_df['date'] - tracking_df.groupby('user_id')['date'].transform('min')).dt.days
    customer_lifetimes = tracking_df.groupby('user_id').agg({
        'duration': 'max',
        'orders': 'sum',
        'revenue': 'sum'
    }).reset_index()
    customer_lifetimes['event_observed'] = (customer_lifetimes['orders'] > 1) & (customer_lifetimes['duration'] > 30)
    customer_lifetimes = customer_lifetimes[customer_lifetimes['duration'] > 0]
    
    if len(customer_lifetimes) < 10:
        return None
    
    durations = customer_lifetimes['duration'].values
    event_observed = customer_lifetimes['event_observed'].values
    
    try:
        kmf.fit(durations, event_observed=event_observed)
        return kmf
    except Exception:
        return None

def forecast_revenue(df):
    st.markdown("""
    #### Understanding Revenue Forecast
    - Blue line shows historical revenue trend
    - Orange line shows predicted future revenue
    - Shaded area represents confidence interval (uncertainty range)
    - Wider intervals suggest higher forecast uncertainty
    - Seasonal patterns indicate cyclical business trends
    """)
    if df.empty:
        return None, None
    
    try:
        df = df.rename(columns={'date': 'ds', 'revenue': 'y'})
        daily_revenue = df.groupby('ds')['y'].sum().reset_index()
        date_range = pd.date_range(daily_revenue['ds'].min(), daily_revenue['ds'].max(), freq='D')
        daily_revenue = daily_revenue.set_index('ds').reindex(date_range).fillna(0).reset_index()
        daily_revenue.columns = ['ds', 'y']
        
        model = Prophet(
            seasonality_mode='multiplicative',
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0
        )
        
        model.fit(daily_revenue)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        
        return model, forecast
    except Exception as e:
        st.error(f"Forecasting error: {str(e)}")
        return None, None

def competitor_analysis(comp_df, tracking_df):
    st.markdown("""
    #### Understanding Competitor Analysis
    - Bar height shows relative market spending
    - Color intensity indicates ROI efficiency
    - Hover for detailed performance metrics
    - Compare spending patterns across platforms
    - Identify opportunities and threats in market positioning
    """)
    if comp_df.empty or tracking_df.empty:
        return pd.DataFrame()
    
    our_metrics = tracking_df.groupby('platform').agg({
        'revenue': 'sum',
        'conversion_id': 'count'
    }).reset_index()
    our_metrics['cpa'] = 0
    
    comp_metrics = comp_df.groupby(['competitor', 'platform']).agg({
        'spend': 'sum',
        'conversions': 'sum',
        'engagement': 'sum'
    }).reset_index()
    
    comp_metrics['cpa'] = comp_metrics['spend'] / comp_metrics['conversions']
    comp_metrics['roas'] = comp_metrics['spend'] / comp_metrics['spend']
    
    return comp_metrics

def calculate_influencer_metrics(tracking_df, payouts_df, inf_df):
    if tracking_df.empty or payouts_df.empty or inf_df.empty:
        return pd.DataFrame()
    
    merged_data = tracking_df.merge(inf_df[['influencer_id', 'name', 'tier', 'category', 'followers', 'engagement_rate']], 
                                   on='influencer_id', how='left')
    merged_data = merged_data.merge(payouts_df[['influencer_id', 'total_payout']], 
                                   on='influencer_id', how='left')
    
    metrics = merged_data.groupby('influencer_id').agg({
        'name': 'first',
        'tier': 'first',
        'category': 'first',
        'followers': 'first',
        'engagement_rate': 'first',
        'revenue': 'sum',
        'conversion_id': 'count',
        'total_payout': 'sum'
    }).reset_index()
    
    metrics['roas'] = metrics['revenue'] / metrics['total_payout']
    metrics['avg_order_value'] = metrics['revenue'] / metrics['conversion_id']
    metrics['conversion_rate'] = (metrics['conversion_id'] / metrics['followers']) * 100
    
    return metrics

def generate_ai_insights(df, context):
    if df.empty:
        return "The current filters resulted in no data. Please adjust your filters to generate insights."

    try:
        platform_summary = df.groupby('platform').agg(
            total_revenue=('revenue', 'sum'),
            total_conversions=('conversion_id', 'count'),
            avg_revenue_per_conversion=('revenue', 'mean')
        ).round(2).reset_index()

        brand_summary = df.groupby('brand').agg(
            total_revenue=('revenue', 'sum'),
            total_conversions=('conversion_id', 'count'),
            avg_revenue_per_conversion=('revenue', 'mean')
        ).round(2).reset_index()

        product_summary = df.groupby('product').agg(
            total_revenue=('revenue', 'sum'),
            total_conversions=('conversion_id', 'count'),
            avg_revenue_per_conversion=('revenue', 'mean')
        ).round(2).reset_index()

        payment_summary = df.groupby('payment_method').agg(
            total_revenue=('revenue', 'sum'),
            total_conversions=('conversion_id', 'count')
        ).reset_index()
        
        data_summary_str = f"""
        Platform Performance Summary:
        {platform_summary.to_string(index=False)}

        Brand Performance Summary:
        {brand_summary.to_string(index=False)}
        
        Product Performance Summary:
        {product_summary.to_string(index=False)}

        Payment Method Analysis:
        {payment_summary.to_string(index=False)}
        """
    except Exception as e:
        return f"Could not process the dataframe for analysis. Error: {e}"

    gemini_api_key = st.secrets.get('google', {}).get('api_key', None)
    
    if not gemini_api_key:
        return "Gemini API key not configured. Please check .streamlit/secrets.toml under [google] section."

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')

        prompt = f"""
        You are a 10+ years data analytics expert for HealthKart, an Indian fitness and wellness brand.
        Your task is to analyze the following influencer marketing data summary and provide actionable insights.

        --- DATA SUMMARY ---
        {data_summary_str}
        --- END OF DATA SUMMARY ---
        
        Based on the data summary above and the user's additional context, please answer the following questions.
        User's Context: "{context}"
        
        Key Questions: (Answer as if you are a 10+ years data analytics expert and include the data in the response)
        1. Based on the data, what are the top-performing platforms for driving revenue and conversions?
        2. Which brands and products are generating the most revenue?
        3. What payment method preferences do customers show?
        4. What anomalies or opportunities can you identify from this summary?
        5. Provide specific, actionable recommendations for optimizing the marketing campaign.

        Provide the response in a clear, concise markdown format with bullet points of each 15-20 words only.
        """
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=500
            )
        )
        
        return response.text
    except Exception as e:
        return f"⚠️ Gemini API Error: {str(e)}"

def generate_executive_report(model_name, financial_context, strategy, data_sources):
    financial_context = convert_to_json_serializable(financial_context)
    strategy = convert_to_json_serializable(strategy)
    data_sources = convert_to_json_serializable(data_sources)

    prompt = f"""
    As a Chief Strategy Officer with 15+ years experience, prepare a board-ready report with:
    1. EXECUTIVE SUMMARY (3-5 bullet points)
    2. FINANCIAL ANALYSIS (using {financial_context['financial_parameters']['wacc']}% WACC)
    3. STRATEGIC RECOMMENDATIONS (aligned with {strategy['strategic_goal']})
    4. RISK ASSESSMENT (for {strategy['risk_appetite']} risk appetite)
    5. IMPLEMENTATION ROADMAP ({strategy['time_horizon']} horizon)
    Financial Context:
    {json.dumps(financial_context, indent=2)}
    Strategic Parameters:
    {json.dumps(strategy, indent=2)}
    Performance Data:
    {json.dumps(data_sources, indent=2)}
    Format requirements:
    - Use markdown with clear section headers
    - Include data visualizations suggestions
    - Highlight 3 key strategic options
    - Provide NPV calculations for each option
    - Compare against {strategy['benchmark']}
    """
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.3,
            max_output_tokens=8192
        )
    )
    return parse_executive_response(response.text)

def generate_pdf_report(df, filename="campaign_summary.pdf"):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.drawString(100, height - 50, "HealthKart Influencer Marketing Campaign Summary")
    c.drawString(100, height - 70, f"Report Date: {datetime.now().strftime('%Y-%m-%d')}")

    total_revenue = df['revenue'].sum()
    total_conversions = len(df)
    c.drawString(100, height - 100, f"Total Revenue: ₹{total_revenue:,.2f}")
    c.drawString(100, height - 120, f"Total Conversions: {total_conversions:,}")

    c.drawString(100, height - 150, "Top 5 Platforms by Revenue:")
    top_platforms = df.groupby('platform')['revenue'].sum().nlargest(5).reset_index()
    for i, row in top_platforms.iterrows():
        c.drawString(120, height - 170 - (i * 20), f"{row['platform']}: ₹{row['revenue']:,.2f}")

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()

def plot_seasonality_factors():
    months = list(range(1, 13))
    factors = [seasonal_factor(m) for m in months]
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig = px.bar(
        x=month_names,
        y=factors,
        title="Seasonality Factors by Month",
        labels={'x': 'Month', 'y': 'Seasonality Factor'},
        color=factors,
        color_continuous_scale='RdYlGn'
    )
    fig.update_layout(template="plotly_white")
    return fig

def plot_new_vs_returning(tracking_df):
    if tracking_df.empty:
        return None
        
    analysis = tracking_df.groupby('new_customer').agg(
        revenue=('revenue', 'sum'),
        conversions=('conversion_id', 'count'),
        avg_order_value=('revenue', 'mean')
    ).reset_index()
    
    analysis['customer_type'] = analysis['new_customer'].map({True: 'New', False: 'Returning'})
    
    fig = make_subplots(
        rows=1, 
        cols=3,
        specs=[[{"type": "bar"}, {"type": "domain"}, {"type": "xy"}]],
        subplot_titles=(
        "Revenue Distribution", "Conversion Share", "Avg Order Value"
    ))
    
    fig.add_trace(go.Bar(
        x=analysis['customer_type'],
        y=analysis['revenue'],
        name='Revenue',
        marker_color=['#636EFA', '#EF553B']
    ), row=1, col=1)
    
    fig.add_trace(go.Pie(
        labels=analysis['customer_type'],
        values=analysis['conversions'],
        name='Conversions',
        hole=0.5,
        marker_colors=['#636EFA', '#EF553B']
    ), row=1, col=2)
    
    fig.add_trace(go.Bar(
        x=analysis['customer_type'],
        y=analysis['avg_order_value'],
        name='Avg Order Value',
        marker_color=['#636EFA', '#EF553B']
    ), row=1, col=3)
    
    fig.update_layout(
        title="New vs Returning Customer Performance",
        showlegend=False,
        template="plotly_white"
    )
    return fig

def plot_gst_impact(payouts_df):
    if payouts_df.empty:
        return None
        
    total_payout = payouts_df['total_payout'].sum()
    gst_amount = total_payout - (total_payout / (1 + gst_rate))
    net_amount = total_payout - gst_amount
    
    fig = px.pie(
        values=[net_amount, gst_amount],
        names=['Net Payout', 'GST Amount'],
        title="Payout Composition: Net vs GST",
        hole=0.4,
        color_discrete_sequence=['#636EFA', '#EF553B']
    )
    fig.update_traces(textinfo='percent+value')
    fig.update_layout(template="plotly_white")
    return fig

def plot_video_engagement_ratios(posts_df):
    video_df = posts_df[posts_df['video_views'] > 0]
    if video_df.empty:
        return None
        
    video_df['likes/view'] = video_df['likes'] / video_df['video_views']
    video_df['comments/view'] = video_df['comments'] / video_df['video_views']
    
    fig = px.scatter(
        video_df,
        x='likes/view',
        y='comments/view',
        color='platform',
        size='video_views',
        title="Video Engagement Ratios: Likes/View vs Comments/View",
        hover_data=['influencer_id', 'date'],
        template="plotly_white"
    )
    fig.update_layout(
        xaxis_title="Likes per View",
        yaxis_title="Comments per View"
    )
    return fig


def plot_hashtag_analysis(posts_df):
    """Visualize hashtag performance"""
    if posts_df.empty:
        return None
        
    
    all_hashtags = []
    for hashtags in posts_df['hashtags']:
        all_hashtags.extend(hashtags)
        
    hashtag_counts = pd.Series(all_hashtags).value_counts().reset_index()
    hashtag_counts.columns = ['hashtag', 'count']
    
    
    hashtag_perf = []
    for hashtag in hashtag_counts['hashtag'].head(15):
        hashtag_posts = posts_df[posts_df['hashtags'].apply(lambda x: hashtag in x)]
        avg_engagement = hashtag_posts['likes'].mean() if not hashtag_posts.empty else 0
        hashtag_perf.append({
            'hashtag': hashtag,
            'count': hashtag_counts[hashtag_counts['hashtag'] == hashtag]['count'].values[0],
            'avg_engagement': avg_engagement
        })
    
    hashtag_perf = pd.DataFrame(hashtag_perf)
    
    fig = px.bar(
        hashtag_perf,
        x='hashtag',
        y=['count', 'avg_engagement'],
        barmode='group',
        title="Top Hashtags: Usage Count vs Engagement",
        labels={'value': 'Metric', 'variable': 'Metric Type'},
        template="plotly_white"
    )
    fig.update_layout(xaxis_title="Hashtag", yaxis_title="Value")
    return fig

def plot_collab_analysis(posts_df):
    """Visualize collaboration type performance"""
    if posts_df.empty:
        return None
        
    collab_perf = posts_df.groupby('collabs').agg(
        avg_reach=('reach', 'mean'),
        avg_likes=('likes', 'mean'),
        avg_comments=('comments', 'mean')
    ).reset_index()
    
    fig = px.bar(
        collab_perf,
        x='collabs',
        y=['avg_reach', 'avg_likes', 'avg_comments'],
        barmode='group',
        title="Performance by Collaboration Type",
        labels={'value': 'Average', 'variable': 'Metric'},
        template="plotly_white"
    )
    fig.update_layout(xaxis_title="Collaboration Type")
    return fig

def plot_funnel_analysis(posts_df, tracking_df):
    """Visualize conversion funnel"""
    if posts_df.empty or tracking_df.empty:
        return None
        
    
    merged = posts_df.merge(
        tracking_df.groupby('post_id')['conversion_id'].count().reset_index(),
        on='post_id',
        how='left'
    )
    merged['conversions'] = merged['conversion_id'].fillna(0)
    
    
    funnel = {
        'Stage': ['Reach', 'Engagement', 'Conversions'],
        'Count': [
            merged['reach'].sum(),
            merged['likes'].sum() + merged['comments'].sum(),
            merged['conversions'].sum()
        ]
    }
    
    fig = px.funnel(
        funnel, 
        x='Count', 
        y='Stage',
        title="Marketing Funnel: Reach → Engagement → Conversions",
        template="plotly_white"
    )
    fig.update_layout(yaxis_title="Funnel Stage")
    return fig

def plot_audience_heatmap(inf_df):
    """Visualize audience demographics heatmap"""
    if inf_df.empty:
        return None
    inf_df['min_age'] = inf_df['audience_age'].str.split('-').str[0].astype(int)
    inf_df['max_age'] = inf_df['audience_age'].str.split('-').str[1].astype(int)
    inf_df['age_group'] = inf_df.apply(
        lambda x: f"{x['min_age']}-{x['max_age']}", axis=1
    )
    heatmap_data = inf_df.groupby(
        ['audience_gender', 'age_group']
    )['followers'].sum().unstack().fillna(0)
    
    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Age Group", y="Gender Dominance", color="Followers"),
        title="Audience Demographics Heatmap",
        aspect="auto",
        color_continuous_scale='Viridis',
        template="plotly_white"
    )
    return fig

def main():
    st.set_page_config(layout="wide", page_title="HealthKart Influencer Marketing Dashboard", page_icon="📈")
    
    gemini_api_key = st.secrets.get('google', {}).get('api_key', None)
    if gemini_api_key:
        try:
            genai.configure(api_key=gemini_api_key)
        except Exception:
            pass
    else:
        st.warning("Gemini API key not found in .streamlit/secrets.toml under [google] section. AI Advisor will not function.")

    data = generate_data()
    inf_df = data['influencers']
    posts_df = data['posts']
    tracking_df = data['tracking']
    payouts_df = data['payouts']
    comp_df = data['competitors']

    min_date = posts_df['date'].min().date()
    max_date = posts_df['date'].max().date()

    st.sidebar.header("Filters")
    st.sidebar.caption(f"Data available from {min_date} to {max_date}")
    
    start_date = st.sidebar.date_input(
        "Start Date",
        value=min_date,
        min_value=min_date,
        max_value=max_date
    )
    end_date = st.sidebar.date_input(
        "End Date",
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )
    
    if start_date > end_date:
        st.sidebar.error("Error: End date must be after start date.")
        start_date, end_date = min_date, max_date

    selected_platforms = st.sidebar.multiselect("Platforms", platforms, default=platforms)
    selected_categories = st.sidebar.multiselect("Categories", sorted(inf_df['category'].unique()), default=sorted(inf_df['category'].unique()))
    selected_tiers = st.sidebar.multiselect("Influencer Tiers", sorted(inf_df['tier'].unique()), default=sorted(inf_df['tier'].unique()))
    selected_brands = st.sidebar.multiselect("Brands", sorted(tracking_df['brand'].unique()), default=sorted(tracking_df['brand'].unique()))
    selected_products = st.sidebar.multiselect("Products", sorted(tracking_df['product'].unique()), default=sorted(tracking_df['product'].unique()))

    @st.cache_data(ttl=600)
    def get_filtered_inf(inf_df, selected_platforms, selected_categories, selected_tiers):
        return inf_df[(inf_df['platform'].isin(selected_platforms)) & (inf_df['category'].isin(selected_categories)) & (inf_df['tier'].isin(selected_tiers))]
    
    filtered_inf = get_filtered_inf(inf_df, selected_platforms, selected_categories, selected_tiers)

    @st.cache_data(ttl=600)
    def get_filtered_tracking(tracking_df, start_date, end_date, selected_brands, selected_products, posts_df):
        filtered = tracking_df[(tracking_df['date'] >= pd.to_datetime(start_date)) & (tracking_df['date'] <= pd.to_datetime(end_date)) & (tracking_df['brand'].isin(selected_brands)) & (tracking_df['product'].isin(selected_products))]
        filtered = pd.merge(filtered, posts_df[['post_id', 'influencer_id', 'platform']], on=['influencer_id', 'post_id'], how='left')
        return filtered
    
    filtered_tracking = get_filtered_tracking(tracking_df, start_date, end_date, selected_brands, selected_products, posts_df)
    
    posts_for_merge = posts_df[posts_df['post_id'].isin(filtered_tracking['post_id'].unique())][['post_id', 'content_type']].drop_duplicates()
    merged_df = pd.merge(filtered_tracking, posts_for_merge, on='post_id', how='left')

    if filtered_tracking.empty:
        st.warning("No data matches your filters. Showing full dataset.")
        filtered_tracking = pd.merge(tracking_df, posts_df[['post_id', 'influencer_id', 'platform']], on=['influencer_id', 'post_id'], how='left')
        posts_for_merge_full = posts_df[['post_id', 'content_type']].drop_duplicates()
        merged_df = pd.merge(filtered_tracking, posts_for_merge_full, on='post_id', how='left')

    st.title("📊 HealthKart Influencer Campaign Dashboard")
    st.markdown("Comprehensive ROI tracking and optimization for influencer marketing campaigns")

    month = pd.to_datetime(start_date).month
    seasonality = seasonal_factor(month)
    total_revenue = filtered_tracking['revenue'].sum() if not filtered_tracking.empty else 0
    total_payout = payouts_df[payouts_df['influencer_id'].isin(filtered_inf['influencer_id'])]['total_payout'].sum() if not filtered_inf.empty else 0
    overall_roas = calculate_roas(total_revenue, total_payout, seasonality) if total_payout > 0 else 0
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Revenue", f"₹{total_revenue:,.0f}")
    m2.metric("Total Payout", f"₹{total_payout:,.0f}")
    m3.metric("Overall ROAS", f"{overall_roas:.2f}")
    m4.metric("Active Influencers", len(filtered_inf))
    
    with st.expander("📊 Interpret Key Metrics"):
        st.write(f"""
        - **Revenue**: Total earnings from campaigns during selected period
        - **Payout**: Total influencer compensation, representing {(total_payout/total_revenue)*100:.1f}% of revenue
        - **ROAS**: For every ₹1 spent, we earned ₹{overall_roas:.2f} in return
        - **Active Influencers**: {len(filtered_inf)} creators driving campaign performance
        """)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📈 Campaign Performance", 
        "👥 Influencer Insights", 
        "💰 Payout Tracking", 
        "🔍 Advanced Analytics", 
        "🤖 AI Advisor", 
        "⚙️ Settings", 
        "📤 Export Data", 
        "🏛️ Boardroom Insights"
    ])

    with tab1:
        st.subheader("Campaign Overview")
        def add_interpretation(title, interpretation):
            with st.expander(f"📊 Interpret {title}"):
                st.write(interpretation)
                
        def add_campaign_interpretations(fig, title, insights):
            st.plotly_chart(fig, use_container_width=True)
            add_interpretation(title, insights)

        st.subheader("Campaign Performance Analysis")

        st.subheader("Budget Simulation")
        budget_increase = st.slider("Simulate Budget Increase (%)", 0, 100, 10)
        selected_platform_sim = st.selectbox("Apply to platform (or all)", ["All"] + platforms)

        if budget_increase > 0:
            platform_filter = None if selected_platform_sim == "All" else selected_platform_sim
            
            simulated_tracking_results_df = simulate_budget_impact(
                posts_df.copy(),
                filtered_tracking.copy(),
                budget_increase,
                platform_filter
            )

            col1, col2 = st.columns(2)
            with col1:
                original_rev = filtered_tracking['revenue'].sum()
                sim_rev = simulated_tracking_results_df['revenue'].sum()
                st.metric("Original Revenue", f"₹{original_rev:,.0f}")
                st.metric(
                    "Simulated Revenue",
                    f"₹{sim_rev:,.0f}",
                    delta=f"{((sim_rev-original_rev)/original_rev)*100:.1f}%" if original_rev > 0 else "N/A"
                )

            with col2:
                original_conv = len(filtered_tracking)
                sim_conv = len(simulated_tracking_results_df)
                st.metric("Original Conversions", original_conv)
                st.metric(
                    "Simulated Conversions",
                    sim_conv,
                    delta=f"{((sim_conv-original_conv)/original_conv)*100:.1f}%" if original_conv > 0 else "N/A"
                )

            original_ts = (
                filtered_tracking
                .set_index('date')
                .resample('D')['revenue']
                .sum()
                .reset_index()
            )
            simulated_ts = (
                simulated_tracking_results_df
                .set_index('date')
                .resample('D')['revenue']
                .sum()
                .reset_index()
            )

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=original_ts['date'],
                y=original_ts['revenue'],
                mode='lines',
                name='Actual Revenue'
            ))
            fig.add_trace(go.Scatter(
                x=simulated_ts['date'],
                y=simulated_ts['revenue'],
                mode='lines',
                name='Simulated Revenue',
                line=dict(dash='dot')
            ))
            fig.update_layout(
                title="Revenue: Actual vs. Simulated Impact",
                xaxis_title="Date",
                yaxis_title="Revenue",
                hovermode="x unified",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        if filtered_tracking.empty:
            st.info("No data available for the selected filters.")
        else:
            sunburst_df = filtered_tracking.copy()
            if 'brand' in sunburst_df.columns and 'product' in sunburst_df.columns:
                fig_sunburst = px.sunburst(
                    sunburst_df,
                    path=['platform', 'brand', 'product'],
                    values='revenue',
                    title="Revenue Distribution: Platform, Brand & Product Hierarchy",
                    color_continuous_scale=px.colors.sequential.Viridis,
                    template="plotly_white"
                )
                add_campaign_interpretations(
                    fig_sunburst,
                    "Revenue Distribution Hierarchy",
                    """
                    - Outer ring shows product-level revenue distribution
                    - Middle ring represents brand performance within platforms
                    - Inner ring displays platform-wise revenue split
                    - Size of segments indicates relative revenue contribution
                    - Color intensity shows revenue concentration areas
                    """
                )

            ts_data = (
                filtered_tracking
                .set_index('date')
                .resample('W')
                .agg(
                    revenue=('revenue', 'sum'),
                    conversions=('conversion_id', 'count')
                )
                .reset_index()
            )

            if not ts_data.empty:
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(
                    go.Scatter(
                        x=ts_data['date'],
                        y=ts_data['revenue'],
                        name='Revenue',
                        mode='lines+markers'
                    ),
                    secondary_y=False
                )
                fig.add_trace(
                    go.Scatter(
                        x=ts_data['date'],
                        y=ts_data['conversions'],
                        name='Conversions',
                        mode='lines+markers'
                    ),
                    secondary_y=True
                )
                fig.update_layout(
                    title="Performance Trend: Revenue vs. Conversions",
                    xaxis_title="Date",
                    yaxis_title="Revenue",
                    yaxis2_title="Conversions",
                    template="plotly_white"
                )
                add_campaign_interpretations(
                    fig,
                    "Performance Trends",
                    """
                    - Blue line shows revenue trend over time
                    - Orange line represents conversion counts
                    - Parallel movement indicates stable average order value
                    - Divergence suggests changes in purchase behavior
                    - Peaks highlight high-performing campaign periods
                    """
                )
            else:
                st.info("Insufficient data to display Performance Trend.")

            st.subheader("Incremental ROAS Analysis")
            if not ts_data.empty and len(ts_data) >= 2:
                available_dates = ts_data['date'].dt.date.unique().tolist()
                intervention_date = st.selectbox(
                    "Select Intervention Date",
                    available_dates,
                    index=len(available_dates) // 2
                )
                intervention_date = pd.to_datetime(intervention_date)

                inc_data = incremental_roas(
                    ts_data,
                    time_col='date',
                    value_col='revenue',
                    intervention_date=intervention_date
                )

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=inc_data['date'],
                    y=inc_data['revenue'],
                    name='Actual Revenue'
                ))
                fig.add_trace(go.Scatter(
                    x=inc_data['date'],
                    y=inc_data['ma_7'],
                    name='7-day Moving Avg'
                ))
                fig.add_trace(go.Scatter(
                    x=inc_data['date'],
                    y=inc_data['incremental'],
                    name='Incremental',
                    line=dict(dash='dot')
                ))
                fig.update_layout(
                    title="Revenue vs Incremental Impact Analysis",
                    xaxis_title="Date",
                    yaxis_title="Revenue",
                    hovermode="x unified",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No time series data available for incremental analysis.")

        st.subheader("Seasonality Analysis")
        st.plotly_chart(plot_seasonality_factors(), use_container_width=True)
        st.caption("Seasonality factors applied to ROAS calculations throughout the year")

        st.subheader("New vs Returning Customers")
        st.plotly_chart(plot_new_vs_returning(filtered_tracking), use_container_width=True)

        st.subheader("Hashtag Performance")
        st.plotly_chart(plot_hashtag_analysis(posts_df), use_container_width=True)

        st.subheader("Conversion Funnel")
        st.plotly_chart(plot_funnel_analysis(posts_df, filtered_tracking), use_container_width=True)

        st.subheader("Collaboration Performance")
        st.plotly_chart(plot_collab_analysis(posts_df), use_container_width=True)
   

    with tab2:
        st.subheader("Influencer Performance Analysis")
        
        if filtered_inf.empty:
            st.info("No influencers match your filters.")
        else:
            influencer_metrics = calculate_influencer_metrics(filtered_tracking, payouts_df, filtered_inf)
            
            if not influencer_metrics.empty:
                st.subheader("🏆 Top Performing Influencers")
                top_influencers = influencer_metrics.nlargest(10, 'roas')
                
                top_5 = top_influencers.head(5)
                
                fig_radar = go.Figure()
                
                for idx, row in top_5.iterrows():
                    fig_radar.add_trace(go.Scatterpolar(
                        r=[row['roas'], row['avg_order_value']/1000, row['conversion_rate']*100, row['revenue']/10000],
                        theta=['ROAS', 'Avg Order (K)', 'Conv Rate (%)', 'Revenue (10K)'],
                        fill='toself',
                        name=row['name'][:15] + "..." if len(row['name']) > 15 else row['name']
                    ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, max(top_5['roas'].max(), top_5['avg_order_value'].max()/1000, 
                                     top_5['conversion_rate'].max()*100, top_5['revenue'].max()/10000)]
                        )),
                    showlegend=True,
                    title="Top 5 Influencers Performance Radar Chart",
                    template="plotly_white"
                )
                add_campaign_interpretations(
                    fig_radar,
                    "Top Influencers Performance",
                    """
                    - Each axis represents a key performance metric (ROAS, Order Value, Conversion Rate, Revenue)
                    - Larger polygons indicate better overall performance
                    - Balanced shapes suggest consistent performance across metrics
                    - Spikes show exceptional performance in specific areas
                    - Compare patterns to identify influencer strengths
                    """
                )
                
                display_table = top_influencers[['name', 'tier', 'category', 'roas', 'avg_order_value', 'conversion_rate', 'revenue']]
                display_table['roas'] = display_table['roas'].round(2)
                display_table['avg_order_value'] = display_table['avg_order_value'].round(0)
                display_table['conversion_rate'] = display_table['conversion_rate'].round(3)
                st.dataframe(display_table, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    # FIXED TREEMAP CODE
                    # Aggregate data at tier level
                    tier_agg = influencer_metrics.groupby('tier').agg(
                        total_revenue=('revenue', 'sum'),
                        avg_roas=('roas', 'mean')
                    ).reset_index()
    
                    # Combine tier aggregates with individual influencer data
                    treemap_data = pd.concat([
                        tier_agg.assign(name="All", influencer_id=""),
                        influencer_metrics.assign(total_revenue=influencer_metrics['revenue'])
                    ])
    
                    fig_roas = px.treemap(
                        treemap_data,
                        path=['tier', 'name', 'influencer_id'],
                        values='total_revenue',
                        color='avg_roas',
                        color_continuous_scale='RdYlGn',
                        title="ROAS vs Revenue by Tier (Treemap)",
                        template="plotly_white",
                        hover_data=['avg_roas', 'total_revenue']
                    )
                    fig_roas.update_traces(
                        textinfo="label+value+percent parent"
                    )
                    add_campaign_interpretations(
                        fig_roas,
                        "ROAS Distribution",
                        """
                        - Box size represents revenue contribution
                        - Color intensity shows ROAS performance (green = higher)
                        - Hierarchy shows tier-wise influencer distribution
                        - Darker green areas indicate optimal performance
                        - Identify top performers within each tier
                        """
                    )
                
                with col2:
                    tier_avg_data = influencer_metrics.groupby('tier')['avg_order_value'].mean().reset_index()
                    fig_avg_order = px.funnel(
                        tier_avg_data,
                        x='avg_order_value',
                        y='tier',
                        title="Average Order Value by Tier (Funnel)",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig_avg_order, use_container_width=True)
            
            inf_performance = filtered_inf.copy()
            inf_performance['total_revenue'] = inf_performance['influencer_id'].map(
                filtered_tracking.groupby('influencer_id')['revenue'].sum()
            ).fillna(0)
            inf_performance['total_conversions'] = inf_performance['influencer_id'].map(
                filtered_tracking.groupby('influencer_id')['conversion_id'].count()
            ).fillna(0)
            
            fig_bubble = px.scatter(
                inf_performance,
                x='followers',
                y='total_revenue',
                size='total_conversions',
                color='tier',
                hover_data=['name', 'category', 'engagement_rate'],
                title="Influencer Performance: Followers vs Revenue",
                color_discrete_sequence=px.colors.qualitative.Set3,
                template="plotly_white"
            )
            fig_bubble.update_layout(
                xaxis_title="Followers",
                yaxis_title="Total Revenue (₹)",
                hovermode="closest"
            )
            st.plotly_chart(fig_bubble, use_container_width=True)
            
            st.subheader("Video Engagement Ratios")
            st.plotly_chart(plot_video_engagement_ratios(posts_df), use_container_width=True)
        
            st.subheader("Audience Demographics Heatmap")
            st.plotly_chart(plot_audience_heatmap(filtered_inf), use_container_width=True)
            
            st.subheader("👥 Audience Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                age_dist = filtered_inf['audience_age'].value_counts()
                fig_age = px.bar(
                    x=age_dist.index,
                    y=age_dist.values,
                    title="Audience Age Distribution",
                    template="plotly_white"
                )
                fig_age.update_layout(xaxis_title="Age Range", yaxis_title="Number of Influencers")
                st.plotly_chart(fig_age, use_container_width=True)
            
            with col2:
                gender_dist = filtered_inf['audience_gender'].value_counts()
                fig_gender = px.pie(
                    values=gender_dist.values,
                    names=gender_dist.index,
                    title="Audience Gender Split",
                    template="plotly_white"
                )
                st.plotly_chart(fig_gender, use_container_width=True)
            
            location_dist = filtered_inf['location'].value_counts().head(10)
            fig_location = px.bar(
                x=location_dist.values,
                y=location_dist.index,
                orientation='h',
                title="Top 10 Influencer Locations",
                template="plotly_white"
            )
            fig_location.update_layout(xaxis_title="Number of Influencers", yaxis_title="Location")
            st.plotly_chart(fig_location, use_container_width=True)
            
            verified_analysis = filtered_inf.groupby('verified').agg({
                'influencer_id': 'count',
                'followers': 'sum',
                'engagement_rate': 'mean'
            }).reset_index()
            
            col1, col2 = st.columns(2)
            with col1:
                fig_verified = px.bar(
                    verified_analysis,
                    x='verified',
                    y='influencer_id',
                    title="Verified vs Non-verified Influencers",
                    template="plotly_white"
                )
                fig_verified.update_layout(xaxis_title="Verification Status", yaxis_title="Number of Influencers")
                st.plotly_chart(fig_verified, use_container_width=True)
            
            with col2:
                engagement_pie_data = verified_analysis.copy()
                engagement_pie_data['engagement_percentage'] = (engagement_pie_data['engagement_rate'] * 100).round(2)
                
                fig_engagement = px.pie(
                    engagement_pie_data,
                    values='engagement_percentage',
                    names='verified',
                    title="Engagement Rate Distribution by Verification Status (%)",
                    template="plotly_white"
                )
                st.plotly_chart(fig_engagement, use_container_width=True)
            
            st.subheader("Audience Segmentation")
            audience_data = filtered_inf.groupby(['audience_gender', 'audience_age']).agg({
                'influencer_id': 'count',
                'followers': 'sum',
                'engagement_rate': 'mean'
            }).reset_index()
            
            fig_audience = px.treemap(
                audience_data,
                path=['audience_gender', 'audience_age'],
                values='followers',
                color='engagement_rate',
                color_continuous_scale=px.colors.sequential.Viridis,
                title="Audience Distribution by Demographics",
                template="plotly_white"
            )
            add_campaign_interpretations(
                fig_audience,
                "Audience Demographics",
                """
                - Box size shows follower count in each segment
                - Color represents engagement rate (darker = higher)
                - Primary split by gender, secondary by age groups
                - Identify high-engagement demographic segments
                - Use insights for targeted campaign planning
                """
            )           
            
            st.subheader("📹 Video Content Performance")
            video_posts = posts_df[posts_df['content_type'].isin(['Video', 'Reel'])]
            
            if not video_posts.empty:
                video_kpis = video_posts.groupby('platform').agg({
                    'video_views': ['sum', 'mean', 'count'],
                    'likes': ['sum', 'mean'],
                    'comments': ['sum', 'mean']
                }).round(0)
                video_kpis.columns = ['Total Views', 'Avg Views', 'Video Count', 'Total Likes', 'Avg Likes', 'Total Comments', 'Avg Comments']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    views_data = video_kpis[['Total Views', 'Avg Views']].reset_index()
                    fig_views = px.bar(
                        views_data,
                        x='platform',
                        y=['Total Views', 'Avg Views'],
                        title="Video Views by Platform",
                        barmode='group',
                        template="plotly_white"
                    )
                    add_campaign_interpretations(
                        fig_views,
                        "Video Views Analysis",
                        """
                        - Bar height shows total and average views per platform
                        - Compare total reach (blue) vs. typical performance (orange)
                        - Platform comparison reveals content consumption patterns
                        - Higher averages indicate better viewer retention
                        - Use to optimize platform-specific video strategy
                        """
                    )
                
                with col2:
                    engagement_data = video_kpis[['Total Likes', 'Total Comments']].reset_index()
                    fig_engagement = px.scatter(
                        engagement_data,
                        x='Total Likes',
                        y='Total Comments',
                        size='Total Likes',
                        color='platform',
                        title="Video Engagement: Likes vs Comments",
                        template="plotly_white"
                    )
                    add_campaign_interpretations(
                        fig_engagement,
                        "Video Engagement Patterns",
                        """
                        - Position shows relationship between likes and comments
                        - Bubble size indicates overall engagement volume
                        - Colors differentiate performance by platform
                        - Diagonal trend suggests consistent engagement
                        - Outliers highlight exceptional content performance
                        """
                    )
                
                st.dataframe(video_kpis, use_container_width=True)
                
                video_performance = video_posts.groupby('influencer_id').agg({
                    'video_views': 'sum',
                    'likes': 'sum',
                    'comments': 'sum'
                }).reset_index()
                
                fig_video = px.scatter_3d(
                    video_performance,
                    x='video_views',
                    y='likes',
                    z='comments',
                    title="Video Performance: Views vs Engagement",
                    template="plotly_white"
                )
                add_campaign_interpretations(
                    fig_video,
                    "3D Video Performance Analysis",
                    """
                    - X-axis: Total video views by influencer
                    - Y-axis: Like count showing passive engagement
                    - Z-axis: Comment count indicating active engagement
                    - Points in upper-right-back show best performers
                    - Clusters reveal common engagement patterns
                    - Outliers identify unique content success
                    """
                )
            else:
                st.info("No video content data available.")

    with tab3:
        st.subheader("Payout Analysis & Risk Assessment")
        
        filtered_payouts = payouts_df[payouts_df['influencer_id'].isin(filtered_inf['influencer_id'])]
        
        if filtered_payouts.empty:
            st.info("No payout data available for selected filters.")
        else:
            total_payout = filtered_payouts['total_payout'].sum()
            risk_score = monte_carlo_risk(filtered_payouts)
            
            col1, col2 = st.columns([2,1])
            with col1:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=risk_score * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Risk Score (%)"},
                    delta={'reference': 20},
                    gauge={
                        'axis': {'range': [None, 50]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 15], 'color': "lightgreen"},
                            {'range': [15, 30], 'color': "yellow"},
                            {'range': [30, 50], 'color': "red"}
                        ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 30
                            }
                        }
                    ))
                fig_gauge.update_layout(height=400,width=None,margin=dict(l=20, r=20, t=80, b=20),template="plotly_white",autosize=True)

                add_campaign_interpretations(
                    fig_gauge,
                    "Risk Assessment Gauge",
                    """
                    - Gauge shows overall payout risk level
                    - Green zone (0-15%): Low risk, stable payouts
                    - Yellow zone (15-30%): Moderate risk, needs monitoring
                    - Red zone (30%+): High risk, requires immediate attention
                    - Delta shows change from baseline risk of 20%
                    """
                )
            
            with col2:
                st.metric("Total Payout", f"₹{total_payout:,.0f}")
                avg_payout = filtered_payouts['total_payout'].mean()
                st.metric("Average Payout", f"₹{avg_payout:,.0f}")
            
            tier_efficiency = filtered_payouts.merge(filtered_inf[['influencer_id', 'tier', 'followers']], on='influencer_id')
            tier_efficiency['efficiency'] = tier_efficiency['total_payout'] / tier_efficiency['followers']
            
            fig_tier = px.scatter(
                tier_efficiency,
                x='followers',
                y='total_payout',
                color='tier',
                size='efficiency',
                hover_data=['influencer_id'],
                title="Payout Efficiency by Tier",
                color_discrete_sequence=px.colors.qualitative.Set3,
                template="plotly_white"
            )
            fig_tier.update_layout(
                xaxis_title="Followers",
                yaxis_title="Total Payout (₹)"
            )
            st.plotly_chart(fig_tier, use_container_width=True)
           
            status_dist = filtered_payouts['payment_status'].value_counts()
            fig_status = px.pie(
                values=status_dist.values,
                names=status_dist.index,
                title="Payment Status Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3,
                template="plotly_white"
            )
            #st.plotly_chart(fig_status, use_container_width=True)
            add_campaign_interpretations(
                fig_status,
                "Payment Status Distribution",
                """
                - Pie chart shows distribution of payment statuses
                - Helps identify bottlenecks in payment processing
                - Use to monitor and improve payment workflows
                """
            )
            st.subheader("GST Impact Analysis")
            #st.plotly_chart(plot_gst_impact(filtered_payouts), use_container_width=True)
            add_campaign_interpretations(
                plot_gst_impact(filtered_payouts),
                "GST Impact on Payouts",
                """
                - Bar chart shows GST impact on total payouts
                - Helps assess tax implications on influencer compensation
                - Use to optimize payout structures and tax planning
                """
            )
            st.subheader("💰 Payout Distribution Analysis")
            
            payout_analysis = filtered_payouts.merge(
                filtered_inf[['influencer_id', 'platform', 'category']], 
                on='influencer_id', how='left'
            )
            
            col1, col2 = st.columns(2)
            with col1:
                platform_payout = payout_analysis.groupby('platform')['total_payout'].sum().reset_index()
                fig_platform = px.pie(
                    platform_payout,
                    values='total_payout',
                    names='platform',
                    title="Total Payout by Platform",
                    template="plotly_white"
                )
                st.plotly_chart(fig_platform, use_container_width=True)
            
            with col2:
                category_payout = payout_analysis.groupby('category')['total_payout'].sum().reset_index()
                fig_category = px.funnel(
                    category_payout,
                    x='total_payout',
                    y='category',
                    title="Total Payout by Category (Funnel)",
                    template="plotly_white"
                )
                st.plotly_chart(fig_category, use_container_width=True)
            
            status_comparison = payout_analysis.groupby('payment_status').agg({
                'total_payout': ['sum', 'count']
            }).reset_index()
            status_comparison.columns = ['payment_status', 'total_amount', 'count']
            
            if status_comparison['count'].sum() > 0:
                fig_status_comp = px.scatter(
                    status_comparison,
                    x='total_amount',
                    y='count',
                    size='total_amount',
                    color='payment_status',
                    title="Payout Status: Amount vs Count (Scatter)",
                    template="plotly_white"
                )
                st.plotly_chart(fig_status_comp, use_container_width=True)
            else:
                st.warning("No payout data available for status comparison.")
            
            st.subheader("📅 Payout Trends Over Time")
            payout_trends = filtered_payouts.groupby('last_updated').agg({
                'total_payout': 'sum',
                'influencer_id': 'count'
            }).reset_index().rename(columns={'last_updated': 'date'})
            payout_trends['date'] = pd.to_datetime(payout_trends['date'])
            fig_payout_trends = px.line(
                payout_trends,
                x='date',
                y='total_payout',
                title="Total Payout Over Time",
                labels={'total_payout': 'Total Payout (₹)', 'date': 'Date'},
                template="plotly_white"
            )
            fig_payout_trends.update_layout(
                xaxis_title="Date",
                yaxis_title="Total Payout (₹)",
                hovermode="x unified"
            )
            add_campaign_interpretations(
                fig_payout_trends,
                "Payout Trends",
                """
                - Line shows total payout trend over time
                - Peaks indicate high payout periods
                - Use to identify seasonal payout patterns
                - Compare with campaign performance for insights
                - Helps in budget planning and forecasting
                """
            )

    with tab4:
        
        st.subheader("Attribution Analysis")
        markov_results = markov_attribution(filtered_tracking)

        
        fig = px.bar(
            markov_results,
            x='influencer_id',
            y='markov_attribution',
            title="Revenue Attribution by Influencer - Markov Model",
            labels={'markov_attribution': 'Attribution Share', 'influencer_id': 'Influencer'},
            color='markov_attribution',
            color_continuous_scale=px.colors.sequential.Plasma,
            template="plotly_white"
        )
        fig.update_layout(
            xaxis_title="Influencer ID",
            yaxis_title="Attribution Share (%)",
            hovermode="x unified"
        )
        add_campaign_interpretations(
            fig,
            "Markov Attribution Analysis",
            """
            - Bar height shows each influencer's contribution to conversions
            - Color intensity indicates attribution strength
            - Higher bars represent more influential touchpoints
            - Compare relative impact across influencers
            - Use for optimizing influencer allocation
            """
        )

        st.caption(
            "Markov attribution models the customer journey as a sequence of touchpoints, "
            "calculating each influencer's contribution to conversions."
        )
        
        st.subheader("Anomaly Detection")
        anomalies = detect_anomalies(filtered_tracking, 'revenue')
            
        if not anomalies.empty:
            fig_anomaly = px.scatter(
                anomalies,
                x='date',
                y='revenue',
                color='anomaly',
                title="Revenue Anomalies",
                color_discrete_map={0: 'blue', 1: 'red'},
                template="plotly_white"
            )
            add_campaign_interpretations(
                fig_anomaly,
                "Revenue Anomalies",
                """
                - Blue dots represent normal revenue points
                - Red dots indicate anomalous behavior
                - Clustered anomalies suggest systematic issues
                - Isolated anomalies may be campaign spikes
                - Use to identify unusual performance patterns
                """
            )
            
            st.subheader("Cohort Analysis")
            try:
                cohort_matrix = cohort_analysis(filtered_tracking)
                
                if not cohort_matrix.empty and cohort_matrix.shape[0] > 0 and cohort_matrix.shape[1] > 0:
                    fig_cohort = px.imshow(
                        cohort_matrix,
                        title="Customer Retention by Cohort",
                        color_continuous_scale=px.colors.sequential.Viridis,
                        template="plotly_white"
                    )
                    st.plotly_chart(fig_cohort, use_container_width=True)
                else:
                    st.info("Insufficient data for cohort analysis.")
            except Exception as e:
                st.warning(f"Cohort analysis could not be performed: {str(e)}")
                st.info("This may be due to insufficient data or data format issues.")
            
            st.subheader("Enhanced Cohort Analysis")
            try:
                retention_matrix, revenue_matrix = enhanced_cohort_analysis(filtered_tracking)
                
                col1, col2 = st.columns(2)
                with col1:
                    if not retention_matrix.empty:
                        fig_cohort = px.imshow(
                            retention_matrix,
                            title="Customer Retention by Cohort",
                            color_continuous_scale=px.colors.sequential.Viridis,
                            template="plotly_white"
                        )
                        st.plotly_chart(fig_cohort, use_container_width=True)
                    else:
                        st.info("Insufficient data for retention cohort analysis.")
                
                with col2:
                    if not revenue_matrix.empty:
                        fig_revenue_cohort = px.imshow(
                            revenue_matrix,
                            title="Average Order Value by Cohort",
                            color_continuous_scale=px.colors.sequential.Plasma,
                            template="plotly_white"
                        )
                        st.plotly_chart(fig_revenue_cohort, use_container_width=True)
                    else:
                        st.info("Insufficient data for revenue cohort analysis.")
            except Exception as e:
                st.warning(f"Enhanced cohort analysis could not be performed: {str(e)}")
            
            st.subheader("Position-Based Attribution Analysis")
            position_results = improved_attribution(filtered_tracking)
            if not position_results.empty:
                fig_position = px.bar(
                    position_results,
                    x='influencer_id',
                    y='attribution_percentage',
                    title="Position-Based Attribution (30%/40%/30%)",
                    color_discrete_sequence=px.colors.sequential.Reds,
                    template="plotly_white"
                )
                st.plotly_chart(fig_position, use_container_width=True)
            else:
                st.info("Insufficient data for position-based attribution.")
            
            st.subheader("🏆 Competitor Analysis")
            try:
                comp_metrics = competitor_analysis(comp_df, filtered_tracking)
                
                if not comp_metrics.empty:
                    col1, col2 = st.columns(2)
                    with col1:
                        fig_comp_cpa = px.bar(
                            comp_metrics,
                            x='competitor',
                            y='cpa',
                            color='platform',
                            title="Competitor CPA Comparison",
                            template="plotly_white"
                        )
                        st.plotly_chart(fig_comp_cpa, use_container_width=True)
                    
                    with col2:
                        fig_comp_spend = px.bar(
                            comp_metrics,
                            x='competitor',
                            y='spend',
                            color='platform',
                            title="Competitor Spend by Platform",
                            template="plotly_white"
                        )
                        st.plotly_chart(fig_comp_spend, use_container_width=True)
                else:
                    st.info("No competitor data available for analysis.")
            except Exception as e:
                st.warning(f"Competitor analysis could not be performed: {str(e)}")
            
            st.subheader("💳 Payment Method Analysis")
            try:
                payment_analysis = filtered_tracking.groupby('payment_method').agg({
                    'revenue': 'sum',
                    'conversion_id': 'count',
                    'user_id': 'nunique'
                }).reset_index()
                payment_analysis['avg_order_value'] = payment_analysis['revenue'] / payment_analysis['conversion_id']
                
                col1, col2 = st.columns(2)
                with col1:
                    fig_payment_revenue = px.pie(
                        payment_analysis,
                        values='revenue',
                        names='payment_method',
                        title="Revenue by Payment Method",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig_payment_revenue, use_container_width=True)
                
                with col2:
                    fig_payment_avg = px.bar(
                        payment_analysis,
                        x='payment_method',
                        y='avg_order_value',
                        title="Average Order Value by Payment Method",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig_payment_avg, use_container_width=True)
            except Exception as e:
                st.warning(f"Payment analysis could not be performed: {str(e)}")
            
    
            st.subheader("Customer Lifetime Value Analysis")
            try:
                kmf = ltv_survival_analysis(filtered_tracking)
                
                if kmf is not None and hasattr(kmf, 'timeline') and hasattr(kmf, 'survival_function_') and not kmf.survival_function_.empty:
                    fig_survival = go.Figure()
                    fig_survival.add_trace(go.Scatter(
                        x=kmf.timeline,
                        y=kmf.survival_function_.iloc[:, 0],
                        mode='lines',
                        name='Survival Function',
                        line=dict(color='blue', width=3)
                    ))
                    fig_survival.update_layout(
                        title="Customer Survival Function (Probability of Customer Retention)",
                        xaxis_title="Days Since First Purchase",
                        yaxis_title="Survival Probability",
                        template="plotly_white",
                        showlegend=True
                    )
                    st.plotly_chart(fig_survival, use_container_width=True)
                    
                    st.info("📊 **Interpretation**: This shows the probability of customers remaining active over time. Higher values indicate better customer retention.")
                else:
                    st.info("Insufficient data for survival analysis. Need at least 10 customers with meaningful purchase history.")
            except Exception as e:
                st.warning(f"Survival analysis could not be performed: {str(e)}")
                st.info("This may be due to insufficient data or data format issues.")
            

            st.subheader("Revenue Forecasting")
            try:
                forecast_model, forecast_data = forecast_revenue(filtered_tracking)
                
                if forecast_model and forecast_data is not None and not forecast_data.empty:
                    
                    historical_data = forecast_data[forecast_data['ds'] <= forecast_data['ds'].max() - pd.Timedelta(days=30)]
                    forecast_period = forecast_data[forecast_data['ds'] > forecast_data['ds'].max() - pd.Timedelta(days=30)]
                    
                    fig_forecast = go.Figure()
                    
                    fig_forecast.add_trace(go.Scatter(
                        x=historical_data['ds'],
                        y=historical_data['yhat'],
                        mode='lines',
                        name='Historical Trend',
                        line=dict(color='blue', width=2)
                    ))
                    
                    fig_forecast.add_trace(go.Scatter(
                        x=forecast_period['ds'],
                        y=forecast_period['yhat'],
                        mode='lines',
                        name='Forecast',
                        line=dict(color='red', width=3)
                    ))
                    
                    fig_forecast.add_trace(go.Scatter(
                        x=forecast_period['ds'],
                        y=forecast_period['yhat_upper'],
                        mode='lines',
                        name='Upper Bound',
                        line=dict(dash='dash', color='lightgray'),
                        showlegend=False
                    ))
                    fig_forecast.add_trace(go.Scatter(
                        x=forecast_period['ds'],
                        y=forecast_period['yhat_lower'],
                        mode='lines',
                        name='Lower Bound',
                        line=dict(dash='dash', color='lightgray'),
                        fill='tonexty',
                        fillcolor='rgba(255,0,0,0.1)',
                        showlegend=False
                    ))
                    
                    fig_forecast.update_layout(
                        title="Revenue Forecast (Next 30 Days)",
                        xaxis_title="Date",
                        yaxis_title="Revenue (₹)",
                        template="plotly_white",
                        showlegend=True
                    )
                    st.plotly_chart(fig_forecast, use_container_width=True)
                    
                    last_forecast = forecast_period['yhat'].iloc[-1]
                    avg_forecast = forecast_period['yhat'].mean()
                    st.info(f"📈 Forecast Summary: Average daily revenue expected: ₹{avg_forecast:,.0f}, Final day forecast: ₹{last_forecast:,.0f}")
            except Exception as e:
                st.error(f"Forecasting failed: {str(e)}")
            except Exception as e:
                st.warning(f"Revenue forecasting could not be performed: {str(e)}")
                st.info("This may be due to insufficient data or Prophet model requirements.")
        st.subheader("Forecast Components")
        if forecast_model and forecast_data is not None:
            fig_components = forecast_model.plot_components(forecast_data)
            st.pyplot(fig_components)
            st.caption("Detailed breakdown of forecast components: trend, weekly, and yearly seasonality")
    with tab5:
        st.subheader("🤖 AI-Powered Insights & Recommendations")
        
        
        gemini_api_key = st.secrets.get('google', {}).get('api_key', None)
        if not gemini_api_key:
            st.error("❌ Gemini API key not configured. Please add it to .streamlit/secrets.toml")
            st.info("AI features will not work without the API key.")
            return
        
        st.success("✅ Gemini API is configured and ready!")
        
        if filtered_tracking.empty:
            st.info("No data available for AI analysis.")
        else:
        
            context = st.text_area(
                "What specific insights are you looking for? (e.g., 'Which platforms are performing best?', 'How can we optimize our influencer strategy?')",
                placeholder="Describe your analysis needs...",
                height=100
            )
            
            if st.button("Generate AI Insights", type="primary"):
                with st.spinner("🤖 Analyzing data with AI..."):
                    insights = generate_ai_insights(filtered_tracking, context)
                    st.markdown("### AI-Generated Insights")
                    st.markdown(insights)
            
            st.subheader("Quick Analysis")
            quick_analysis_options = [
                "Platform Performance Analysis",
                "Brand Performance Analysis",
                "Product Performance Analysis",
                "Payment Method Analysis",
                "Revenue Optimization Recommendations"
            ]
            
            selected_quick_analysis = st.selectbox("Choose analysis type:", quick_analysis_options)
            
            if st.button("Run Quick Analysis"):
                with st.spinner("Running quick analysis..."):
                    quick_insights = generate_ai_insights(filtered_tracking, selected_quick_analysis)
                    st.markdown("### Quick Analysis Results")
                    st.markdown(quick_insights)

    with tab6:
        st.subheader("⚙️ Dashboard Settings & Configuration")
        
        st.subheader("Data Management")
        if st.button("🔄 Regenerate Data", type="secondary"):
            st.cache_data.clear()
            st.rerun()
        
        st.subheader("Visual Theme")
        theme_options = ["plotly_white", "plotly", "ggplot2", "seaborn", "simple_white"]
        selected_theme = st.selectbox("Select Chart Theme:", theme_options, index=0)
        
        if st.button("Apply Theme"):
            st.success(f"Theme changed to {selected_theme}")
        
        st.subheader("API Configuration")
        
        gemini_api_key = st.secrets.get('google', {}).get('api_key', None)
        
        if gemini_api_key:
            st.success("✅ Gemini API key is configured and ready to use!")
            st.info("API key is securely stored in .streamlit/secrets.toml")
        else:
            st.error("❌ Gemini API key not found in .streamlit/secrets.toml")
            st.info("Please add your API key to .streamlit/secrets.toml under [google] section:")
            st.code("""
[google]
api_key = "your_gemini_api_key_here"
            """)
        
        st.subheader("Export Settings")
        export_format = st.selectbox("Default Export Format:", ["PDF", "CSV", "Excel"])
        include_forecasts = st.checkbox("Include Revenue Forecasts in Exports", value=True)
        include_ai_insights = st.checkbox("Include AI Insights in Exports", value=True)

    with tab7:
        st.subheader("📤 Export & Report Generation")
        
        if filtered_tracking.empty:
            st.info("No data available for export.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Data Export")
                
                if st.button("📊 Export to CSV"):
                    csv_data = filtered_tracking.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name=f"healthkart_campaign_data_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                
                if st.button("📈 Export to Excel"):
                    buffer = BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        filtered_tracking.to_excel(writer, sheet_name='Campaign Data', index=False)
                        filtered_inf.to_excel(writer, sheet_name='Influencer Data', index=False)
                        filtered_payouts.to_excel(writer, sheet_name='Payout Data', index=False)
                    buffer.seek(0)
                    st.download_button(
                        label="Download Excel",
                        data=buffer.getvalue(),
                        file_name=f"healthkart_complete_data_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            
            with col2:
                st.subheader("Report Generation")
                if st.button("📄 Generate PDF Report"):
                    try:
                        pdf_data = generate_pdf_report(filtered_tracking)
                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_data,
                            file_name=f"healthkart_campaign_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf"
                        )
                    except ImportError:
                        st.error("PDF generation requires 'reportlab' package. Please install it using: pip install reportlab")
                    except Exception as e:
                        st.error(f"PDF generation failed: {str(e)}")
                
                st.subheader("Custom Report Builder")
                available_columns = list(filtered_tracking.columns)
                selected_columns = st.multiselect(
                    "Select Data Columns:",
                    available_columns,
                    default=['date', 'platform', 'brand', 'product', 'revenue', 'payment_method']
                )
                
                report_sections = st.multiselect(
                    "Select Report Sections:",
                    ["Executive Summary", "Performance Metrics", "Influencer Analysis", "Financial Analysis", "Forecasts", "AI Insights"],
                    default=["Executive Summary", "Performance Metrics"]
                )
                
                report_start = st.date_input("Report Start Date", value=start_date)
                report_end = st.date_input("Report End Date", value=end_date)
                
                if st.button("🔧 Build Custom Report"):
                    if selected_columns and report_sections:
                        report_data = filtered_tracking[
                            (filtered_tracking['date'] >= pd.to_datetime(report_start)) &
                            (filtered_tracking['date'] <= pd.to_datetime(report_end))
                        ][selected_columns]
                        
                        st.subheader("📊 Custom Report Generated")
                        st.write(f"**Report Period**: {report_start} to {report_end}")
                        st.write(f"**Data Points**: {len(report_data)}")
                    
                        st.dataframe(report_data, use_container_width=True)
                        csv_data = report_data.to_csv(index=False)
                        st.download_button(
                            label="Download Custom Report",
                            data=csv_data,
                            file_name=f"custom_report_{report_start}_{report_end}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("Please select at least one column and one report section.")

    with tab8:
        st.subheader("📊 Strategic Boardroom Insights")
        st.markdown("C-suite level analysis for investment decisions")
        
        selected_model = st.selectbox(
            "AI Model", 
            ["Gemini 1.5 Pro (Comprehensive)", "Gemini 1.5 Flash (Fast)"],
            index=0,
            help="Pro for in-depth analysis, Flash for quick insights"
        )
        
        with st.expander("Strategy Configuration"):
            col1, col2 = st.columns(2)
            with col1:
                time_horizon = st.select_slider(
                    "Time Horizon",
                    options=["0-3 months", "3-12 months", "1-3 years", "3-5 years"],
                    value="1-3 years"
                )
                risk_appetite = st.selectbox(
                    "Risk Appetite",
                    ["Conservative", "Balanced", "Aggressive"],
                    index=1
                )
            
            with col2:
                strategic_goal = st.selectbox(
                    "Primary Goal",
                    ["Revenue Growth", "Market Share", "Brand Equity", "Customer Retention"],
                    index=0
                )
                benchmark = st.selectbox(
                    "Benchmark Against",
                    ["Industry Average", "Top Competitor", "Previous Period"],
                    index=0
                )
        
        with st.expander("Financial Configuration"):
            col1, col2 = st.columns(2)
            with col1:
                wacc = st.number_input("Cost of Capital (WACC %)", 
                                     min_value=0.0, 
                                     max_value=25.0, 
                                     value=10.5, 
                                     step=0.5)
                tax_rate = st.number_input("Corporate Tax Rate (%)",
                                         min_value=0.0,
                                         max_value=50.0,
                                         value=25.0,
                                         step=1.0)
            
            with col2:
                growth_target = st.number_input("Revenue Growth Target (%)",
                                             min_value=-20.0,
                                             max_value=200.0,
                                             value=30.0,
                                             step=5.0)
                inflation = st.number_input("Expected Inflation (%)",
                                          min_value=0.0,
                                          max_value=15.0,
                                          value=3.5,
                                          step=0.5)
        
        if st.button("Generate Boardroom Analysis", type="primary"):
            with st.spinner("Generating executive insights..."):
                try:
                    model_name = "models/gemini-1.5-pro-002" if selected_model.startswith("Gemini 1.5 Pro") else "models/gemini-1.5-flash-002"
                    pro_api_key = st.secrets.get('pro_api_key', None)
                    if pro_api_key:
                        genai.configure(api_key=pro_api_key)
                    financial_context = {
                        "total_revenue": total_revenue,
                        "total_spend": total_payout,
                        "roas": overall_roas,
                        "customer_acquisition_cost": total_payout / len(filtered_tracking) if len(filtered_tracking) > 0 else 0,
                        "customer_lifetime_value": calculate_ltv(filtered_tracking),
                        "financial_parameters": {
                            "wacc": wacc,
                            "tax_rate": tax_rate,
                            "growth_target": growth_target,
                            "inflation": inflation
                        }
                    }
                    report = generate_executive_report(
                        model_name=model_name,
                        financial_context=financial_context,
                        strategy={
                            "time_horizon": time_horizon,
                            "risk_appetite": risk_appetite,
                            "strategic_goal": strategic_goal,
                            "benchmark": benchmark
                        },
                        data_sources={
                            "influencer_performance": get_top_influencers(filtered_tracking, filtered_inf),
                            "platform_performance": get_platform_performance(filtered_tracking),
                            "competitive_landscape": get_competitive_analysis(comp_df)
                        }
                    )
            
                    display_executive_report(report)
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")

    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.8em;'>
        📊 HealthKart Influencer Marketing Dashboard  |  
        Built with Streamlit & Plotly  |  
        Data refreshed every hour
        </div>
        """,
        unsafe_allow_html=True
    )

def calculate_ltv(tracking_df):
    if tracking_df.empty:
        return 0
    user_stats = tracking_df.groupby('user_id').agg({'revenue': 'sum', 'orders': 'sum'})
    avg_revenue = user_stats['revenue'].mean()
    avg_orders = user_stats['orders'].mean()
    return round(avg_revenue * avg_orders, 2)

def get_top_influencers(tracking_df, inf_df, n=5):
    if tracking_df.empty or inf_df.empty:
        return []
    merged = tracking_df.merge(inf_df, on='influencer_id', how='left')
    top = merged.groupby(['influencer_id', 'name']).agg({'revenue': 'sum', 'conversion_id': 'count'}).reset_index()
    top = top.sort_values('revenue', ascending=False).head(n)
    return top.to_dict(orient='records')

def get_platform_performance(tracking_df):
    if tracking_df.empty:
        return []
    perf = tracking_df.groupby('platform').agg({'revenue': 'sum', 'conversion_id': 'count'}).reset_index()
    return perf.to_dict(orient='records')

def get_competitive_analysis(comp_df):
    if comp_df.empty:
        return []


def parse_executive_response(text):
    sections = {'summary': '', 'financial_analysis': '', 'recommendations': '', 'risk_analysis': '', 'metrics': {}, 'risks': []}
    current = None
    for line in text.splitlines():
        if line.strip().lower().startswith('#') or line.strip().lower().startswith('##'):
            if 'executive summary' in line.lower():
                current = 'summary'
            elif 'financial analysis' in line.lower():
                current = 'financial_analysis'
            elif 'recommendation' in line.lower():
                current = 'recommendations'
            elif 'risk' in line.lower():
                current = 'risk_analysis'
            else:
                current = None
        elif current:
            sections[current] += line + '\n'
    return sections

def display_executive_report(report):
    """Display the formatted report"""
    with st.expander("Executive Summary", expanded=True):
        st.markdown(report.get('summary', 'No summary generated'))
    with st.expander("Financial Impact Analysis"):
        st.markdown(report.get('financial_analysis', 'No financial analysis'))
        metrics = {
            'ROAS': report.get('metrics', {}).get('roas', 0),
            'CAC': report.get('metrics', {}).get('cac', 0),
            'LTV': report.get('metrics', {}).get('ltv', 0)
        }
        fig = px.bar(x=list(metrics.keys()), y=list(metrics.values()), title="Key Financial Metrics")
        st.plotly_chart(fig)
    with st.expander("Strategic Recommendations"):
        st.markdown(report.get('recommendations', 'No recommendations'))
    with st.expander("Risk Assessment"):
        st.markdown(report.get('risk_analysis', 'No risk analysis'))
        risks = report.get('risks', [])
        if risks:
            df = pd.DataFrame(risks)
            fig = px.scatter(df, x='probability', y='impact', size='severity', color='category', title="Risk Assessment Matrix")
            st.plotly_chart(fig)
    st.download_button(
        "Download Executive Summary (PDF)",
        generate_pdf(report),
        file_name="boardroom_report.pdf"
    )

def generate_pdf(report):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = A4
    c.drawString(100, height - 50, "Boardroom Executive Summary")
    y = height - 80
    for section, content in report.items():
        if isinstance(content, str) and content.strip():
            c.drawString(100, y, section.replace('_', ' ').title() + ':')
            y -= 20
            for line in content.split('\n'):
                c.drawString(120, y, line[:90])
                y -= 15
                if y < 100:
                    c.showPage()
                    y = height - 80
    c.save()
    buffer.seek(0)
    return buffer.getvalue()

if __name__ == "__main__":
    main()

