import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import zscore
import statsmodels.api as sm
from io import StringIO
import warnings
warnings.filterwarnings('ignore')


# --- CONFIGURATION ---
st.set_page_config(
    page_title="Bengaluru Housing Market Analysis",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- THEME/STYLE CUSTOMIZATION (Inspired by a "Glassmorphic" look) ---
def set_styles():
    st.markdown("""
        <style>
        .css-1d391kg, .eczokfa1 {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 5rem;
            padding-right: 5rem;
        }
        
        /* Main title styling */
        h1 {
            color: #4B0082; /* Indigo */
            font-weight: 800;
            text-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        }
        
        /* KPI card styling */
        div[data-testid="stMetric"] {
            background-color: #f0f2f6; 
            border: 1px solid #e0e0e0;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 4px 4px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        /* Tab styling for a modern look */
        .stTabs [data-baseweb="tab-list"] {
            gap: 15px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: nowrap;
            border-radius: 4px 4px 0 0;
            background-color: #e6e6fa; /* Lavender */
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
            padding-left: 20px;
            padding-right: 20px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #4B0082; /* Indigo */
            color: white;
            font-weight: bold;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #f0f8ff; /* AliceBlue */
            padding-top: 40px;
        }
        
        /* Section headers */
        h2 {
            color: #008080; /* Teal */
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 5px;
            margin-top: 30px;
        }
        
        /* Glassmorphism for plots (subtle) */
        .block-container {
            backdrop-filter: blur(5px);
            background-color: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
            padding: 20px !important;
        }

        </style>
    """, unsafe_allow_html=True)

# --- DATA LOADING & CACHING ---
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        # Ensure correct data types (Crucial after cleaning)
        df['BHK'] = df['BHK'].astype(int)
        df['SqftClean'] = df['SqftClean'].astype(float)
        df['price'] = df['price'].astype(float)
        df['price_per_sqft_new'] = df['price_per_sqft_new'].astype(float)
        # Convert date columns back to datetime for easy sorting/plotting
        df.loc[df['Availability_Year'] == 9999, 'Availability_Year'] = np.nan
        df['Availability_Month'] = df['Availability_Month'].apply(lambda x: x if x != 99 else np.nan)
        df['Date_Sort'] = pd.to_datetime(df['Availability_Year'].astype(str) + '-' + df['Availability_Month'].astype(str) + '-01', errors='coerce')
        return df
    except FileNotFoundError:
        st.error(f"Error: Dataset file '{file_path}' not found. Please ensure it is in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred during data loading or preprocessing: {e}")
        st.stop()


# --- DATA ANALYSIS FUNCTIONS ---

@st.cache_data
def generate_insights(df):
    """Analyzes patterns and generates key bullet-point insights."""
    insights = []
    
    # --- Price Level Analysis ---
    avg_price = df['price'].mean()
    insights.append(f"💰 The **Average Property Price** in the filtered market is **₹{avg_price:,.2f} Lakhs**.")
    
    # --- PPS Analysis ---
    city_avg_pps = df['price_per_sqft_new'].mean()
    
    # Top location by PPS (requires minimum listings for reliability)
    location_pps = df.groupby('location').agg(
        avg_pps=('price_per_sqft_new', 'mean'),
        count=('location', 'size')
    ).reset_index()
    reliable_locations = location_pps[location_pps['count'] >= 20]
    top_pps_loc = reliable_locations.sort_values('avg_pps', ascending=False).iloc[0]
    
    pps_diff = ((top_pps_loc['avg_pps'] / city_avg_pps) - 1) * 100
    
    insights.append(f"📈 **{top_pps_loc['location']}** is the most premium area, with an average PPS **{pps_diff:,.1f}% higher** than the overall market average.")

    # --- BHK Stability Analysis (Using IQR or Std Dev) ---
    bhk_std = df.groupby('BHK')['price'].std().sort_values(ascending=True)
    most_stable_bhk = bhk_std.index[0]
    insights.append(f"🛋️ **BHK {most_stable_bhk}** category shows the **most stable pricing**, indicating lower price volatility and possibly a more saturated market segment.")

    # --- Area Type Performance ---
    area_type_pps = df.groupby('area_type')['price_per_sqft_new'].mean().sort_values(ascending=False)
    best_area_type = area_type_pps.index[0]
    insights.append(f"🏡 The **{best_area_type}** category offers the highest value for money, with the highest average Price Per Square Foot.")
    
    # --- Predictive Hinting ---
    best_value_loc = reliable_locations.sort_values('avg_pps', ascending=True).iloc[0]
    insights.append(f"💡 **Investment Hint:** For the best price-to-sqft value, consider areas like **{best_value_loc['location']}**.")

    return insights

@st.cache_data
def get_clean_bhk(df):
    """Ensures BHK grouping is manageable for plotting by grouping large values."""
    df_plot = df.copy()
    bhk_counts = df_plot['BHK'].value_counts()
    rare_bhks = bhk_counts[bhk_counts < 10].index
    df_plot.loc[df_plot['BHK'].isin(rare_bhks), 'BHK_Grouped'] = '>10 BHK'
    df_plot.loc[df_plot['BHK'] <= 10, 'BHK_Grouped'] = df_plot['BHK'].astype(str)
    return df_plot


# --- CHARTING FUNCTIONS (Plotly & Plotly Express) ---

def plot_kpis(df):
    """Calculates and displays KPI cards."""
    col1, col2, col3, col4 = st.columns(4)
    
    total_properties = len(df)
    avg_price = df['price'].mean()
    avg_pps = df['price_per_sqft_new'].mean()
    total_locations = df['location'].nunique()

    # Apply colored metric styling for KPI Cards
    def metric_card(col, value, label, delta_value, icon):
        col.markdown(f"""
            <div style="background-color: #f0f2f6; border: 1px solid #e0e0e0; padding: 15px; border-radius: 10px; box-shadow: 4px 4px 10px rgba(0,0,0,0.1); text-align: center;">
                <p style="font-size: 1.5rem; color: #4B0082; font-weight: bold;">{value}</p>
                <p style="font-size: 0.9rem; color: #555;">{icon} {label}</p>
            </div>
        """, unsafe_allow_html=True)
    
    # KPI 1: Total Properties
    metric_card(col1, f"{total_properties:,}", "Total Properties", "", "🏡")
    # KPI 2: Average Price (Lakhs)
    metric_card(col2, f"₹{avg_price:,.2f} L", "Avg. Price", "", "💸")
    # KPI 3: Average PPS
    metric_card(col3, f"₹{avg_pps:,.0f}/sqft", "Avg. Price Per Sqft", "", "📈")
    # KPI 4: Total Locations
    metric_card(col4, f"{total_locations:,}", "Total Locations", "", "📍")
    st.markdown("---")


def plot_price_trend(df):
    """1. Price Trend Over Availability Month/Year (Line Chart)"""
    df_trend = df.dropna(subset=['Date_Sort'])
    df_trend = df_trend[df_trend['Date_Sort'].dt.year <= 2025] # Filter out 9999 year
    
    trend_data = df_trend.groupby(df_trend['Date_Sort'].dt.to_period('M'))['price'].mean().reset_index()
    trend_data['Date_Sort'] = trend_data['Date_Sort'].astype(str)
    
    fig = px.line(
        trend_data, 
        x='Date_Sort', 
        y='price', 
        title='Price Trend Over Availability Date (Avg Price in Lakhs)',
        markers=True,
        color_discrete_sequence=['#FF4B4B'] # Streamlit Red
    )
    fig.update_layout(xaxis_title="Availability Month", yaxis_title="Average Price (₹ Lakhs)", hovermode="x unified")
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)


def plot_price_vs_sqft(df):
    """2. Scatter Plot -> Price vs SqftClean (Colored by BHK)"""
    df_plot = get_clean_bhk(df)
    
    fig = px.scatter(
        df_plot,
        x='SqftClean',
        y='price',
        color='BHK_Grouped',
        hover_data=['location', 'price_per_sqft_new'],
        title='Price vs. Area (Sqft) - Colored by BHK',
        opacity=0.6
    )
    fig.update_layout(xaxis_title="Total Square Feet", yaxis_title="Price (₹ Lakhs)")
    st.plotly_chart(fig, use_container_width=True)


def plot_price_by_bhk(df):
    """3. Bar Chart -> Average Price by BHK & 8. Boxplot -> Price by BHK"""
    df_plot = get_clean_bhk(df)
    
    col1, col2 = st.columns(2)
    
    # Bar Chart
    avg_price_bhk = df_plot.groupby('BHK_Grouped')['price'].mean().reset_index()
    avg_price_bhk = avg_price_bhk.sort_values('price', ascending=False)
    
    fig_bar = px.bar(
        avg_price_bhk,
        x='BHK_Grouped',
        y='price',
        title='Average Price by BHK Category',
        color='price',
        color_continuous_scale=px.colors.sequential.Plasma_r
    )
    fig_bar.update_layout(xaxis_title="BHK Category", yaxis_title="Average Price (₹ Lakhs)")
    col1.plotly_chart(fig_bar, use_container_width=True)
    
    # Box Plot
    fig_box = px.box(
        df_plot,
        x='BHK_Grouped',
        y='price',
        title='Price Distribution and Outliers by BHK',
        color='BHK_Grouped'
    )
    fig_box.update_layout(xaxis_title="BHK Category", yaxis_title="Price (₹ Lakhs)")
    col2.plotly_chart(fig_box, use_container_width=True)


def plot_pps_by_area_type(df):
    """4. Bar Chart -> Average Price per Sqft by Area Type & 9. Boxplot -> Price by Area Type"""
    col1, col2 = st.columns(2)
    
    # Bar Chart (Avg PPS by Area Type)
    avg_pps_area = df.groupby('area_type')['price_per_sqft_new'].mean().reset_index()
    avg_pps_area = avg_pps_area.sort_values('price_per_sqft_new', ascending=False)
    
    fig_bar = px.bar(
        avg_pps_area,
        x='area_type',
        y='price_per_sqft_new',
        title='Avg. Price per Sqft by Area Type',
        color='price_per_sqft_new',
        color_continuous_scale=px.colors.sequential.Teal
    )
    fig_bar.update_layout(xaxis_title="Area Type", yaxis_title="Average PPS (₹/Sqft)")
    col1.plotly_chart(fig_bar, use_container_width=True)

    # Box Plot (Price by Area Type)
    fig_box = px.box(
        df,
        x='area_type',
        y='price',
        title='Price Distribution and Outliers by Area Type',
        color='area_type'
    )
    fig_box.update_layout(xaxis_title="Area Type", yaxis_title="Price (₹ Lakhs)")
    col2.plotly_chart(fig_box, use_container_width=True)


def plot_location_prices(df):
    """5. Bar Chart -> Average Price by Location (Top 10 expensive + Top 10 affordable)"""
    location_avg = df.groupby('location')['price'].mean().reset_index()
    
    # Filter for reliable locations (e.g., min 5 listings)
    location_counts = df['location'].value_counts()
    reliable_locations = location_counts[location_counts >= 5].index
    location_avg = location_avg[location_avg['location'].isin(reliable_locations)]
    
    top_10 = location_avg.sort_values('price', ascending=False).head(10)
    bottom_10 = location_avg.sort_values('price', ascending=True).head(10)
    
    # Combine and prepare for plot
    combined_df = pd.concat([top_10, bottom_10])
    combined_df['Category'] = np.where(combined_df['location'].isin(top_10['location']), 'Top 10 Expensive', 'Top 10 Affordable')
    combined_df = combined_df.sort_values(['Category', 'price'], ascending=[False, True])
    
    fig = px.bar(
        combined_df,
        x='location',
        y='price',
        color='Category',
        title='Top 10 Expensive and Affordable Locations (Avg Price in Lakhs)',
        color_discrete_map={'Top 10 Expensive': '#8A2BE2', 'Top 10 Affordable': '#3CB371'},
        hover_data={'price': ':.2f'}
    )
    fig.update_layout(xaxis_title="Location", yaxis_title="Average Price (₹ Lakhs)")
    st.plotly_chart(fig, use_container_width=True)


def plot_histograms(df):
    """6. Histogram -> Price Distribution & 7. Histogram -> Price Per Sqft Distribution"""
    col1, col2 = st.columns(2)
    
    # Price Distribution
    fig_price = px.histogram(
        df,
        x='price',
        nbins=50,
        title='Distribution of Property Price (₹ Lakhs)',
        color_discrete_sequence=['#4682B4'] # Steel Blue
    )
    fig_price.update_layout(xaxis_title="Price (₹ Lakhs)", yaxis_title="Count")
    col1.plotly_chart(fig_price, use_container_width=True)
    
    # Price Per Sqft Distribution
    fig_pps = px.histogram(
        df,
        x='price_per_sqft_new',
        nbins=50,
        title='Distribution of Price Per Sqft',
        color_discrete_sequence=['#DAA520'] # Goldenrod
    )
    fig_pps.update_layout(xaxis_title="Price Per Sqft (₹)", yaxis_title="Count")
    col2.plotly_chart(fig_pps, use_container_width=True)


def plot_correlation_heatmap(df):
    """10. Correlation Heatmap of numeric fields"""
    # Select key numeric columns
    numeric_cols = ['price', 'SqftClean', 'BHK', 'bath', 'balcony', 'price_per_sqft_new', 'Availability_Year']
    df_corr = df[numeric_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=df_corr.values,
        x=df_corr.columns,
        y=df_corr.columns,
        colorscale='Viridis',
        zmin=-1, zmax=1,
        text=df_corr.round(2).values,
        hoverongaps=False
    ))
    fig.update_layout(
        title='Correlation Heatmap of Key Variables',
        xaxis={'tickangle': 45},
        yaxis={'tickangle': -45}
    )
    st.plotly_chart(fig, use_container_width=True)

# Note on Map Plot (11): Since geographic coordinates are not available, a scatter map based on location names is not feasible without geocoding (which is external). We will use a clustered heatmap approximation of location prices in the Location Analysis tab as a substitute for visual geographic density.

def plot_pairplot_analysis(df):
    """12. Pairplot-like analysis for deeper relationships"""
    # Select only the most critical continuous variables for a clear pairplot
    cols = ['price', 'SqftClean', 'price_per_sqft_new']
    df_sample = df[cols].sample(n=min(len(df), 2000), random_state=42) # Sample for performance
    
    fig = px.scatter_matrix(
        df_sample,
        dimensions=cols,
        color='price',
        title='Pairwise Analysis of Key Continuous Variables (Sampled Data)',
        height=900
    )
    fig.update_traces(diagonal_visible=False)
    st.plotly_chart(fig, use_container_width=True)


# --- APPLICATION LAYOUT ---

def main():
    set_styles()
    
    # --- HEADER ---
    st.title("🏠 Bengaluru Housing Price Analyst 📊")
    st.subheader("A World-Class, Interactive Dashboard for Price and Property Analysis")
    st.markdown("---")

    # --- DATA LOADING ---
    df_raw = load_data('Bengaluru_House_Data_Transformed_For_PowerBI.csv')
    df = df_raw.copy()

    # --- SIDEBAR FILTERS ---
    st.sidebar.header("⚙️ Market Filters")
    
    # Filter 1: Location
    location_list = ['All'] + sorted(df['location'].unique().tolist())
    selected_locations = st.sidebar.multiselect("Select Location(s)", location_list, default='All')
    
    if 'All' not in selected_locations:
        df = df[df['location'].isin(selected_locations)]

    # Filter 2: BHK
    bhk_list = sorted(df['BHK'].unique().tolist())
    selected_bhk = st.sidebar.multiselect("Select BHK Categories", bhk_list, default=bhk_list)
    df = df[df['BHK'].isin(selected_bhk)]
    
    # Filter 3: Area Type
    area_type_list = st.sidebar.multiselect("Select Area Type", df['area_type'].unique().tolist(), default=df['area_type'].unique().tolist())
    df = df[df['area_type'].isin(area_type_list)]

    # Filter 4: Price Range
    min_price, max_price = st.sidebar.slider(
        "Price Range (₹ Lakhs)", 
        float(df['price'].min()), 
        float(df['price'].max()), 
        (float(df['price'].min()), float(df['price'].max()))
    )
    df = df[(df['price'] >= min_price) & (df['price'] <= max_price)]

    # Filter 5: Availability Year
    year_list = sorted(df['Availability_Year'].dropna().unique().astype(int).tolist())
    selected_years = st.sidebar.multiselect("Availability Year", year_list, default=year_list)
    df = df[df['Availability_Year'].isin(selected_years)]
    
    
    # --- CHECK FOR EMPTY DATA ---
    if df.empty:
        st.warning("⚠️ No data available based on the selected filters. Please adjust the filters.")
        st.stop()


    # --- MAIN DASHBOARD LAYOUT (TABS) ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🚀 Market Overview", 
        "🏠 BHK & Area Type Insights", 
        "📍 Location & Pricing", 
        "📈 Statistical Deep Dive", 
        "🔍 Outlier Analysis"
    ])

    # --- TAB 1: Market Overview ---
    with tab1:
        st.header("1️⃣ Market Summary & Price Trends 📊")
        plot_kpis(df)
        st.subheader("Price vs. Size Relationship 📏")
        plot_price_vs_sqft(df)

    # --- TAB 2: BHK & Area Type Insights ---
    with tab2:
        st.header("2️⃣ Property Characteristics Insights 🛋️")
        
        st.subheader("Price by BHK and Distribution")
        plot_price_by_bhk(df)
        
        st.subheader("Price per Sqft by Area Type")
        plot_pps_by_area_type(df)
        
        st.subheader("Core Market Distributions")
        plot_histograms(df)

    # --- TAB 3: Location & Pricing ---
    with tab3:
        st.header("3️⃣ Location-Based Analysis 🌍")
        
        st.subheader("Top/Bottom 10 Locations by Average Price")
        plot_location_prices(df)
        
        # 11. Map Plot (Substitution: Clustered Heatmap of Location Prices)
        st.subheader("Location Price Density (Clustered Map Approximation)")
        # Calculate mean PPS for reliable locations
        location_pps = df.groupby('location')['price_per_sqft_new'].mean().reset_index()
        location_pps = location_pps[location_pps['location'].isin(df['location'].value_counts()[df['location'].value_counts() >= 5].index)]
        location_pps = location_pps.sort_values('price_per_sqft_new', ascending=False).head(50) # Top 50 unique locations for visualization
        
        # Create a simple clustered bar chart for density visualization
        fig_map = px.bar(
            location_pps,
            x='location',
            y='price_per_sqft_new',
            title='Top Locations by Avg PPS (Density Approximation)',
            color='price_per_sqft_new',
            color_continuous_scale=px.colors.sequential.Sunset
        )
        fig_map.update_layout(xaxis_title="Location", yaxis_title="Average PPS (₹/Sqft)")
        st.plotly_chart(fig_map, use_container_width=True)

    # --- TAB 4: Statistical Deep Dive ---
    with tab4:
        st.header("4️⃣ Advanced Statistical Analysis 🔬")

        st.subheader("Correlation Analysis: Which Factors Drive Price? 🔑")
        plot_correlation_heatmap(df)
        
        st.subheader("Pairwise Relationship between Price, Sqft, and PPS 🔗")
        plot_pairplot_analysis(df)
        
        st.subheader("Data Scientist Insights & Predictive Hinting 💡")
        insights = generate_insights(df)
        st.markdown(f"**Based on the current filtered data:**")
        
        st.markdown("""
            <ul style="list-style-type: none; padding-left: 0;">
                <li style="margin-bottom: 10px; padding: 10px; border-left: 5px solid #008080; background-color: #e0f0f0;">{}</li>
                <li style="margin-bottom: 10px; padding: 10px; border-left: 5px solid #008080; background-color: #e0f0f0;">{}</li>
                <li style="margin-bottom: 10px; padding: 10px; border-left: 5px solid #008080; background-color: #e0f0f0;">{}</li>
                <li style="margin-bottom: 10px; padding: 10px; border-left: 5px solid #008080; background-color: #e0f0f0;">{}</li>
                <li style="margin-bottom: 10px; padding: 10px; border-left: 5px solid #008080; background-color: #e0f0f0;">{}</li>
            </ul>
        """.format(*insights), unsafe_allow_html=True)


    # --- TAB 5: Outlier Analysis ---
    with tab5:
        st.header("5️⃣ Outlier and Data Quality Analysis 🚨")
        
        st.info("The underlying data was cleaned using threshold filtering. This section analyzes remaining potential outliers based on statistical methods.")

        # Outlier Detection using Z-Score on PPS
        pps_zscore = zscore(df['price_per_sqft_new'])
        df['PPS_ZScore'] = np.abs(pps_zscore)
        
        # Count outliers (Z-Score > 3)
        outlier_count = len(df[df['PPS_ZScore'] > 3])
        total_count = len(df)
        
        col_out1, col_out2, col_out3 = st.columns(3)
        col_out1.metric("Total Properties Analyzed", f"{total_count:,}", delta=None)
        col_out2.metric("Z-Score Outliers (Z > 3)", f"{outlier_count:,}", delta_color="inverse")
        col_out3.metric("Outlier Fraction", f"{(outlier_count / total_count * 100):.2f}%", delta=None)
        
        # Outlier Visualization
        st.subheader("Price Per Sqft Distribution (Highlighting Z-Score Outliers)")
        fig_outlier = px.scatter(
            df,
            x='SqftClean',
            y='price_per_sqft_new',
            color=(df['PPS_ZScore'] > 3).map({True: 'Z-Score Outlier', False: 'Normal'}),
            hover_data=['location', 'price'],
            title='Z-Score Outlier Identification (PPS)',
            color_discrete_map={'Normal': '#1f77b4', 'Z-Score Outlier': '#ff7f0e'}
        )
        st.plotly_chart(fig_outlier, use_container_width=True)

# --- RUN APPLICATION ---
if __name__ == "__main__":
    main()