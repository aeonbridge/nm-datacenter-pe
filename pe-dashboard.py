import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import altair as alt

# Page configuration
st.set_page_config(
    page_title="Pernambuco Data Centers Strategic Dashboard",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


# Load and prepare data
@st.cache_data
def load_data():
    data = {
        'Region': ['Recife Metropolitan', 'Agreste', 'Agreste', 'Agreste',
                   'Zona da Mata Norte', 'Zona da Mata Norte', 'Sertão', 'Sertão', 'Porto de Suape'],
        'Meso_Region': ['Recife', 'Caruaru', 'Garanhuns', 'São Caetano',
                        'Goiana', 'Nazaré da Mata', 'Petrolina', 'Arcoverde', 'Suape'],
        'City': ['Recife (Parqtel, S. Zone)', 'Caruaru', 'Garanhuns', 'São Caetano',
                 'Goiana', 'Nazaré da Mata', 'Petrolina', 'Arcoverde', 'Suape Port Area'],
        'Main_Reasons': ['Advanced grid, fiber, tech hub, submarine cable',
                         'Urban, solar, effluent reuse', 'Urban, solar, effluent reuse',
                         'Solar, effluent reuse, pilot DC', 'Industrial base, grid, some water reuse',
                         'Industrial, grid, water stress', 'Solar/wind, green hydrogen potential',
                         'Solar/wind, green hydrogen potential', 'Energy hub, logistics, grid'],
        'Pros': ['Robust infra, skilled labor, renewables, connectivity',
                 'Lower land cost, renewables, water reuse, expansion',
                 'Lower land cost, renewables, water reuse, expansion',
                 'Renewables, water reuse, expansion',
                 'Industrial infra, near Recife, workforce',
                 'Industrial infra, near Recife, workforce',
                 'Land, renewables, hydrogen power',
                 'Land, renewables, hydrogen power',
                 'Power, logistics, renewables, expansion'],
        'Cons': ['High land cost, urban competition, flood risk, water stress',
                 'Water scarcity, grid upgrades, moderate connectivity',
                 'Water scarcity, grid upgrades, moderate connectivity',
                 'Water scarcity, grid upgrades, moderate connectivity',
                 'Hydro stress, water allocation limits',
                 'Hydro stress, water allocation limits',
                 'Extreme water scarcity, saline groundwater, infra needed',
                 'Extreme water scarcity, saline groundwater, infra needed',
                 'Land competition, water stress, environmental constraints']
    }

    df = pd.DataFrame(data)

    # Add scoring system for analysis
    infrastructure_scores = [9, 6, 6, 7, 7, 6, 4, 5, 8]
    renewable_scores = [7, 8, 8, 9, 5, 5, 9, 9, 7]
    water_availability = [4, 3, 3, 3, 2, 2, 1, 1, 3]
    connectivity_scores = [9, 5, 5, 6, 6, 5, 4, 4, 6]
    land_cost_scores = [3, 7, 7, 7, 6, 6, 9, 9, 5]  # Higher = lower cost

    df['Infrastructure_Score'] = infrastructure_scores
    df['Renewable_Score'] = renewable_scores
    df['Water_Score'] = water_availability
    df['Connectivity_Score'] = connectivity_scores
    df['Land_Cost_Score'] = land_cost_scores
    df['Overall_Score'] = (np.array(infrastructure_scores) +
                           np.array(renewable_scores) +
                           np.array(water_availability) +
                           np.array(connectivity_scores) +
                           np.array(land_cost_scores)) / 5

    return df


# Main dashboard
def main():
    st.markdown('<h1 class="main-header">🏢 Pernambuco Data Centers Strategic Dashboard</h1>',
                unsafe_allow_html=True)
    st.markdown("### Strategic Implementation Analysis for Large-Scale Data Centers")

    # Load data
    df = load_data()

    # Sidebar filters
    st.sidebar.header("🔍 Filters & Options")

    selected_regions = st.sidebar.multiselect(
        "Select Regions",
        options=df['Region'].unique(),
        default=df['Region'].unique()
    )

    score_threshold = st.sidebar.slider(
        "Minimum Overall Score",
        min_value=float(df['Overall_Score'].min()),
        max_value=float(df['Overall_Score'].max()),
        value=float(df['Overall_Score'].min()),
        step=0.1
    )

    show_scores = st.sidebar.checkbox("Show Detailed Scores", value=True)

    # Filter data
    filtered_df = df[
        (df['Region'].isin(selected_regions)) &
        (df['Overall_Score'] >= score_threshold)
        ]

    # Main dashboard layout
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Locations", len(filtered_df))
    with col2:
        st.metric("Avg Overall Score", f"{filtered_df['Overall_Score'].mean():.1f}")
    with col3:
        st.metric("Top Renewable Score", f"{filtered_df['Renewable_Score'].max():.0f}")
    with col4:
        st.metric("Regions Analyzed", len(filtered_df['Region'].unique()))

    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "🗺️ Regional Analysis", "⚡ Infrastructure", "💧 Sustainability", "📈 Scoring Matrix"
    ])

    with tab1:
        st.markdown('<div class="section-header">Location Overview</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])

        with col1:
            # Overall scores radar chart
            fig_radar = create_radar_chart(filtered_df)
            st.plotly_chart(fig_radar, use_container_width=True)

        with col2:
            # Region distribution
            region_counts = filtered_df['Region'].value_counts()
            fig_pie = px.pie(
                values=region_counts.values,
                names=region_counts.index,
                title="Distribution by Region",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)

        # Detailed table
        st.markdown("#### 📋 Detailed Location Analysis")
        display_df = filtered_df[['City', 'Region', 'Main_Reasons', 'Pros', 'Cons']]
        if show_scores:
            display_df = pd.concat([
                display_df,
                filtered_df[['Infrastructure_Score', 'Renewable_Score', 'Water_Score',
                             'Connectivity_Score', 'Overall_Score']].round(1)
            ], axis=1)

        st.dataframe(display_df, use_container_width=True, height=400)

    with tab2:
        st.markdown('<div class="section-header">Regional Comparison</div>', unsafe_allow_html=True)

        # Regional scores comparison
        col1, col2 = st.columns(2)

        with col1:
            fig_bar = px.bar(
                filtered_df,
                x='City',
                y='Overall_Score',
                color='Region',
                title="Overall Suitability Score by Location",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_bar.update_layout(xaxis_tickangle=-45, height=500)
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            # Scatter plot: Infrastructure vs Renewable
            fig_scatter = px.scatter(
                filtered_df,
                x='Infrastructure_Score',
                y='Renewable_Score',
                size='Overall_Score',
                color='Region',
                hover_name='City',
                title="Infrastructure vs Renewable Energy Potential",
                size_max=20
            )
            fig_scatter.update_layout(height=500)
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Regional summary statistics
        st.markdown("#### 📊 Regional Statistics")
        regional_stats = filtered_df.groupby('Region').agg({
            'Overall_Score': ['mean', 'max', 'min'],
            'Infrastructure_Score': 'mean',
            'Renewable_Score': 'mean',
            'Water_Score': 'mean'
        }).round(2)

        regional_stats.columns = ['Avg Score', 'Max Score', 'Min Score', 'Avg Infrastructure', 'Avg Renewable',
                                  'Avg Water']
        st.dataframe(regional_stats, use_container_width=True)

    with tab3:
        st.markdown('<div class="section-header">Infrastructure Analysis</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            # Infrastructure vs Connectivity heatmap
            fig_heatmap = create_infrastructure_heatmap(filtered_df)
            st.plotly_chart(fig_heatmap, use_container_width=True)

        with col2:
            # Infrastructure ranking
            infra_ranking = filtered_df.nlargest(len(filtered_df), 'Infrastructure_Score')[
                ['City', 'Infrastructure_Score', 'Connectivity_Score']
            ]

            fig_infra = go.Figure()
            fig_infra.add_trace(go.Bar(
                name='Infrastructure',
                x=infra_ranking['City'],
                y=infra_ranking['Infrastructure_Score'],
                marker_color='lightblue'
            ))
            fig_infra.add_trace(go.Bar(
                name='Connectivity',
                x=infra_ranking['City'],
                y=infra_ranking['Connectivity_Score'],
                marker_color='orange'
            ))

            fig_infra.update_layout(
                title="Infrastructure & Connectivity Scores",
                xaxis_tickangle=-45,
                barmode='group',
                height=400
            )
            st.plotly_chart(fig_infra, use_container_width=True)

        # Infrastructure insights
        st.markdown("#### 🔍 Key Infrastructure Insights")

        top_infra = filtered_df.loc[filtered_df['Infrastructure_Score'].idxmax()]
        top_connectivity = filtered_df.loc[filtered_df['Connectivity_Score'].idxmax()]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Best Infrastructure:** {top_infra['City']} (Score: {top_infra['Infrastructure_Score']})")
        with col2:
            st.info(
                f"**Best Connectivity:** {top_connectivity['City']} (Score: {top_connectivity['Connectivity_Score']})")
        with col3:
            avg_infra = filtered_df['Infrastructure_Score'].mean()
            st.info(f"**Average Infrastructure Score:** {avg_infra:.1f}")

    with tab4:
        st.markdown('<div class="section-header">Sustainability & Water Analysis</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            # Water vs Renewable scatter
            fig_sustain = px.scatter(
                filtered_df,
                x='Water_Score',
                y='Renewable_Score',
                size='Overall_Score',
                color='Region',
                hover_name='City',
                title="Water Availability vs Renewable Energy",
                labels={'Water_Score': 'Water Availability Score', 'Renewable_Score': 'Renewable Energy Score'}
            )
            st.plotly_chart(fig_sustain, use_container_width=True)

        with col2:
            # Sustainability ranking
            sustainability_score = (filtered_df['Renewable_Score'] + filtered_df['Water_Score']) / 2
            sustain_df = pd.DataFrame({
                'City': filtered_df['City'],
                'Sustainability_Score': sustainability_score,
                'Water_Score': filtered_df['Water_Score'],
                'Renewable_Score': filtered_df['Renewable_Score']
            }).sort_values('Sustainability_Score', ascending=False)

            fig_sustain_bar = px.bar(
                sustain_df,
                x='City',
                y='Sustainability_Score',
                title="Sustainability Ranking",
                color='Sustainability_Score',
                color_continuous_scale='Greens'
            )
            fig_sustain_bar.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_sustain_bar, use_container_width=True)

        # Water challenge analysis
        st.markdown("#### 💧 Water Challenge Assessment")

        water_challenge_df = filtered_df[['City', 'Region', 'Water_Score', 'Cons']].copy()
        water_challenge_df['Challenge_Level'] = pd.cut(
            water_challenge_df['Water_Score'],
            bins=[0, 2, 4, 6, 10],
            labels=['Extreme', 'High', 'Moderate', 'Low']
        )

        challenge_counts = water_challenge_df['Challenge_Level'].value_counts()

        col1, col2 = st.columns([1, 2])
        with col1:
            for level, count in challenge_counts.items():
                color = {'Extreme': '🔴', 'High': '🟠', 'Moderate': '🟡', 'Low': '🟢'}[level]
                st.write(f"{color} **{level} Challenge:** {count} locations")

        with col2:
            st.dataframe(
                water_challenge_df[['City', 'Region', 'Challenge_Level', 'Cons']],
                use_container_width=True
            )

    with tab5:
        st.markdown('<div class="section-header">Comprehensive Scoring Matrix</div>', unsafe_allow_html=True)

        # Scoring matrix heatmap
        score_columns = ['Infrastructure_Score', 'Renewable_Score', 'Water_Score', 'Connectivity_Score',
                         'Land_Cost_Score']
        score_matrix = filtered_df[['City'] + score_columns].set_index('City')

        fig_matrix = px.imshow(
            score_matrix.T,
            labels=dict(x="Location", y="Criteria", color="Score"),
            x=score_matrix.index,
            y=score_columns,
            color_continuous_scale='RdYlGn',
            title="Comprehensive Scoring Matrix"
        )
        fig_matrix.update_layout(height=500)
        st.plotly_chart(fig_matrix, use_container_width=True)

        # Top recommendations
        st.markdown("#### 🏆 Top Recommendations")

        top_3 = filtered_df.nlargest(3, 'Overall_Score')

        for i, (_, location) in enumerate(top_3.iterrows(), 1):
            with st.expander(f"#{i} {location['City']} (Score: {location['Overall_Score']:.1f})"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Region:** {location['Region']}")
                    st.write(f"**Main Advantages:** {location['Pros']}")
                    st.write(f"**Key Reasons:** {location['Main_Reasons']}")
                with col2:
                    st.write(f"**Challenges:** {location['Cons']}")
                    st.write("**Scores:**")
                    st.write(f"- Infrastructure: {location['Infrastructure_Score']}/10")
                    st.write(f"- Renewable Energy: {location['Renewable_Score']}/10")
                    st.write(f"- Water Availability: {location['Water_Score']}/10")
                    st.write(f"- Connectivity: {location['Connectivity_Score']}/10")


def create_radar_chart(df):
    """Create a radar chart showing average scores by region"""
    avg_scores = df.groupby('Region')[
        ['Infrastructure_Score', 'Renewable_Score', 'Water_Score', 'Connectivity_Score']].mean()

    fig = go.Figure()

    categories = ['Infrastructure', 'Renewable Energy', 'Water Availability', 'Connectivity']

    for region in avg_scores.index:
        fig.add_trace(go.Scatterpolar(
            r=avg_scores.loc[region].values,
            theta=categories,
            fill='toself',
            name=region
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )),
        showlegend=True,
        title="Average Scores by Region",
        height=500
    )

    return fig


def create_infrastructure_heatmap(df):
    """Create infrastructure vs connectivity heatmap"""
    fig = px.scatter(
        df,
        x='Infrastructure_Score',
        y='Connectivity_Score',
        size='Overall_Score',
        color='Region',
        hover_name='City',
        title="Infrastructure vs Connectivity Matrix",
        labels={'Infrastructure_Score': 'Infrastructure Score', 'Connectivity_Score': 'Connectivity Score'}
    )

    fig.update_layout(height=500)
    return fig


if __name__ == "__main__":
    main()