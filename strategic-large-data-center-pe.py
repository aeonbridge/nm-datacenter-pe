import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import altair as alt
import anthropic
import json
import time
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Pernambuco Data Centers Strategic Dashboard",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for chat
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'anthropic_client' not in st.session_state:
    st.session_state.anthropic_client = None

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


def initialize_anthropic_client():
    """Initialize Anthropic client with API key"""
    api_key = None

    # Try multiple sources for API key
    try:
        # Method 1: Try Streamlit secrets
        api_key = st.secrets.get("ANTHROPIC_API_KEY")
    except (FileNotFoundError, KeyError, Exception):
        pass

    # Method 2: Try environment variable
    if not api_key:
        import os
        api_key = os.getenv("ANTHROPIC_API_KEY")

    # Method 3: Try session state
    if not api_key:
        api_key = st.session_state.get("anthropic_api_key")

    if not api_key:
        st.sidebar.warning("🔑 Anthropic API Key Required")
        st.sidebar.markdown("""
        **Setup Options:**
        1. **Environment Variable**: `export ANTHROPIC_API_KEY="your-key"`
        2. **Secrets File**: Create `.streamlit/secrets.toml`
        3. **Manual Entry**: Use the input below
        """)

        api_key_input = st.sidebar.text_input(
            "Enter your Anthropic API Key:",
            type="password",
            key="api_key_input",
            help="Get your API key from https://console.anthropic.com/"
        )

        if st.sidebar.button("💾 Save API Key"):
            if api_key_input.strip():
                st.session_state.anthropic_api_key = api_key_input.strip()
                st.sidebar.success("✅ API Key saved successfully!")
                st.rerun()
            else:
                st.sidebar.error("❌ Please enter a valid API key")

        if not api_key_input:
            return None
        else:
            api_key = api_key_input.strip()

    # Validate and create client
    if not api_key or not api_key.startswith('sk-ant-'):
        st.sidebar.error("❌ Invalid API key format. Should start with 'sk-ant-'")
        return None

    try:
        client = anthropic.Anthropic(api_key=api_key)
        # Test the client with a simple request
        return client
    except Exception as e:
        st.sidebar.error(f"❌ Error initializing Anthropic client: {str(e)}")
        return None


def get_context_from_data(df, user_query):
    """Extract relevant context from the dataframe based on user query"""
    query_lower = user_query.lower()

    # Simple keyword matching for context retrieval
    context_data = []

    # Check for specific cities/regions mentioned
    for _, row in df.iterrows():
        city_mentioned = any(word in query_lower for word in row['City'].lower().split())
        region_mentioned = row['Region'].lower() in query_lower

        if city_mentioned or region_mentioned:
            context_data.append({
                'location': row['City'],
                'region': row['Region'],
                'scores': {
                    'infrastructure': row['Infrastructure_Score'],
                    'renewable': row['Renewable_Score'],
                    'water': row['Water_Score'],
                    'connectivity': row['Connectivity_Score'],
                    'overall': row['Overall_Score']
                },
                'pros': row['Pros'],
                'cons': row['Cons'],
                'reasons': row['Main_Reasons']
            })

    return context_data


def query_anthropic(client, user_message, context_data, df):
    """Query Anthropic API with context"""

    # Prepare context from data
    context_str = f"""
    Current Data Context:
    {json.dumps(context_data, indent=2) if context_data else "No specific locations mentioned"}

    Available Locations Summary:
    {df[['City', 'Region', 'Overall_Score']].to_string(index=False)}

    Knowledge Base:
    {KNOWLEDGE_BASE}
    """

    system_prompt = """You are an expert consultant specializing in data center infrastructure planning in Pernambuco, Brazil. 

    You have access to comprehensive data about potential data center locations including:
    - Infrastructure capabilities and power grid information
    - Renewable energy potential and sustainability factors
    - Water availability and environmental constraints
    - Economic considerations and regional advantages
    - Detailed scoring analysis for each location

    Provide accurate, helpful responses based on the provided data and knowledge base. 
    When discussing specific locations, reference their scores and characteristics.
    Always consider both opportunities and challenges in your recommendations.
    Be conversational but professional, and cite specific data points when relevant."""

    try:
        message = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            temperature=0.3,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": f"Context: {context_str}\n\nUser Question: {user_message}"
                }
            ]
        )
        return message.content[0].text
    except Exception as e:
        return f"❌ Error querying Anthropic API: {str(e)}"


def chat_interface(df):
    """Create the chat interface"""
    st.markdown('<div class="section-header">💬 Data Center Strategy Assistant</div>', unsafe_allow_html=True)

    # Initialize Anthropic client
    if not st.session_state.anthropic_client:
        st.session_state.anthropic_client = initialize_anthropic_client()

    if not st.session_state.anthropic_client:
        st.warning("🔑 Please configure your Anthropic API key in the sidebar to use the chat feature.")
        return

    # Chat interface
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("**Ask me about:**")
        st.markdown("- Best locations for specific data center requirements")
        st.markdown("- Infrastructure capabilities and limitations")
        st.markdown("- Sustainability and water management strategies")
        st.markdown("- Regional comparisons and recommendations")

    with col2:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_messages = []
            st.rerun()

    # Display chat messages
    chat_container = st.container()

    with chat_container:
        for message in st.session_state.chat_messages:
            if message["role"] == "user":
                st.markdown(f"**🙋 You:** {message['content']}")
            else:
                st.markdown(f"**🤖 Assistant:** {message['content']}")
            st.markdown("---")

    # Initialize selected question in session state
    if 'selected_question' not in st.session_state:
        st.session_state.selected_question = ""

    # Example questions section
    col1, col2 = st.columns([1, 1])
    with col2:
        if st.button("💡 Show Example Questions"):
            st.session_state.show_examples = not st.session_state.get("show_examples", False)

    if st.session_state.get("show_examples", False):
        st.markdown("**Example Questions:**")
        example_questions = [
            "What's the best location for a large-scale data center with renewable energy requirements?",
            "Compare infrastructure capabilities between Recife and São Caetano",
            "What are the main water challenges for data centers in the Sertão region?",
            "Which locations offer the best connectivity for international data centers?",
            "What sustainability measures should be considered for data centers in Pernambuco?",
            "How does the scoring system work and what do the numbers mean?",
            "What are the green hydrogen opportunities in the Sertão region?",
            "Which region offers the best cost-benefit ratio for data center development?"
        ]

        selected_example = st.selectbox(
            "Select an example question:",
            [""] + example_questions,
            key="example_selector"
        )

        if selected_example and st.button("Use This Question"):
            st.session_state.selected_question = selected_example
            st.session_state.show_examples = False
            st.rerun()

    # Chat input - use selected question if available
    default_value = st.session_state.selected_question if st.session_state.selected_question else ""
    user_input = st.text_input(
        "Ask a question about data center locations in Pernambuco:",
        value=default_value,
        placeholder="e.g., Which location is best for a renewable energy-focused data center?",
        key="chat_input"
    )

    # Clear the selected question after using it
    if st.session_state.selected_question and user_input == st.session_state.selected_question:
        st.session_state.selected_question = ""

    col1, col2, col3 = st.columns([1, 1, 4])

    with col1:
        send_button = st.button("📤 Send", type="primary")

    # Process user input
    if send_button and user_input.strip():
        # Add user message to chat
        st.session_state.chat_messages.append({"role": "user", "content": user_input})

        # Get context from data
        context_data = get_context_from_data(df, user_input)

        # Show thinking indicator
        with st.spinner("🤔 Analyzing your question..."):
            # Query Anthropic
            response = query_anthropic(st.session_state.anthropic_client, user_input, context_data, df)

        # Add assistant response to chat
        st.session_state.chat_messages.append({"role": "assistant", "content": response})

        # Clear the selected question and rerun
        st.session_state.selected_question = ""
        st.rerun()


def export_chat_history():
    """Export chat history as downloadable file"""
    if st.session_state.chat_messages:
        chat_export = {
            "export_date": datetime.now().isoformat(),
            "chat_history": st.session_state.chat_messages
        }

        return json.dumps(chat_export, indent=2, ensure_ascii=False)
    return None


# Main dashboard
def main():
    st.markdown('<h1 class="main-header">🏢 Pernambuco Data Centers Strategic Dashboard</h1>',
                unsafe_allow_html=True)
    st.markdown("### Strategic Implementation Analysis for Large-Scale Data Centers")

    # Load data
    df = load_data()

    # Sidebar filters
    st.sidebar.header("🔍 Filters & Options")

    # API Configuration section
    with st.sidebar.expander("🔧 API Configuration", expanded=not st.session_state.get("anthropic_api_key")):
        st.markdown("**Anthropic API Setup**")

        # Check current status
        if st.session_state.get("anthropic_api_key"):
            masked_key = st.session_state.anthropic_api_key[:8] + "..." + st.session_state.anthropic_api_key[-4:]
            st.success(f"✅ API Key configured: {masked_key}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Update Key"):
                    del st.session_state.anthropic_api_key
                    if 'anthropic_client' in st.session_state:
                        del st.session_state.anthropic_client
                    st.rerun()
            with col2:
                if st.button("🧪 Test API"):
                    if st.session_state.anthropic_client:
                        st.success("🟢 API is working!")
                    else:
                        st.error("🔴 API test failed")
        else:
            st.info("💡 Enter your Anthropic API key to enable chat features")
            st.markdown("""
            **Get your API key:**
            1. Visit [Anthropic Console](https://console.anthropic.com/)
            2. Create an account or sign in
            3. Generate an API key
            4. Enter it below
            """)

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
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", "🗺️ Regional Analysis", "⚡ Infrastructure", "💧 Sustainability", "📈 Scoring Matrix",
        "💬 AI Assistant"
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

    with tab6:
        # Chat interface tab
        chat_interface(filtered_df)

        # Export chat functionality
        if st.session_state.chat_messages:
            st.markdown("---")
            col1, col2 = st.columns([1, 1])
            with col1:
                chat_export = export_chat_history()
                if chat_export:
                    st.download_button(
                        label="📥 Download Chat History",
                        data=chat_export,
                        file_name=f"pernambuco_dc_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            with col2:
                st.info(f"💬 Chat messages: {len(st.session_state.chat_messages)}")


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


# Knowledge base content for RAG
KNOWLEDGE_BASE = """
# Pernambuco Data Centers Knowledge Base

## Regional Data
- Recife Metropolitan: Advanced power grid, fiber infrastructure, tech hub with Porto Digital, submarine cable plans
- Agreste (Caruaru, Garanhuns, São Caetano): Urban centers, high solar potential, effluent reuse projects, lower land costs
- Zona da Mata Norte (Goiana, Nazaré da Mata): Industrial base, existing grid, high hydrological stress
- Sertão (Petrolina, Arcoverde): Extreme solar/wind potential, green hydrogen opportunities, severe water scarcity
- Porto de Suape: Energy transition hub, strong logistics, renewable projects

## Infrastructure Details
- Via Mangue Substation (Recife): 52 MVA capacity, serves 260,000 people, most advanced in Brazil
- Neoenergia Investment: R$5.1 billion through 2028 for grid modernization
- Renewable Capacity: Wind farms like Fontes dos Ventos II (99 MW), Solar expansion 51% in 2024
- Fiber Networks: 22,000+ km in Recife area, submarine cable initiatives

## Sustainability Factors
- Water Scarcity: Chronic in Sertão and Agreste, 85% municipalities with rationing
- Renewable Integration: Strong solar/wind potential, green hydrogen projects
- Cooling Solutions: Air-to-liquid cooling, wastewater reuse, closed-loop systems
- Environmental Constraints: Flood risks in coastal areas, saline groundwater in interior

## Economic Considerations
- Job Creation: 26,000 employed in solar sector, tech hub with 475 companies in Porto Digital
- Land Costs: Higher in Recife Metropolitan, lower in interior regions
- Investment Incentives: BNDES financing, renewable energy zones, tax breaks
- Community Impact: Need for benefit agreements, vocational training, infrastructure sharing

## Key Challenges and Solutions
- Water Management: Effluent reuse, desalination, water-efficient cooling
- Grid Reliability: Modernization projects, renewable integration, demand response
- Skilled Labor: Porto Digital ecosystem, training programs, workforce development
- Environmental Impact: Sustainability standards, community engagement, green technology adoption

# Strategic Implementation of Large Data Centers in Pernambuco: A Policy Framework for Sustainable Growth  

## Executive Summary  
Pernambuco possesses unique advantages for data center development, including a growing renewable energy infrastructure, strategic geographic positioning, and targeted investments in digital connectivity. However, challenges related to water scarcity, grid reliability, and regional disparities in infrastructure require careful mitigation. This report synthesizes Scopus-indexed research and regional data to evaluate optimal locations, weighing economic benefits against environmental and social impacts. Key recommendations include prioritizing zones with renewable energy clusters, accelerating submarine cable projects in Recife, and implementing water-efficient cooling technologies to align with Brazil’s sustainability goals.  

---

## Energy Infrastructure and Renewable Integration  

### Current Energy Landscape  
Pernambuco’s electricity grid is undergoing significant modernization, with **R$5.1 billion** allocated by Neoenergia Pernambuco for substation expansions and high-voltage line installations through 2028[4][12]. The state’s renewable energy capacity has grown rapidly, including:  
- **Wind Energy**: The 99 MW Fontes dos Ventos II wind farm, operational since 2022, supplies corporate clients like Lojas Renner and avoids 6,200 tonnes of CO₂ annually[9].  
- **Solar Energy**: Residential solar capacity expanded by 51% in 2024, reaching 856.7 MW, supported by financing programs[10]. The 114 MWp Luiz Gonzaga solar plant in Terra Nova, funded by SPIC Brasil and Recurrent Energy, exemplifies large-scale projects attracting private investment[13].  

#### Pros for Data Centers  
- **Grid Modernization**: Neoenergia’s Via Mangue substation in Recife adds 52 MVA of power, sufficient for 260,000 residents, demonstrating scalability for high-density data centers[12].  
- **Renewable Clusters**: Regions like Terra Nova and Cabo de Santo Agostinho offer co-location opportunities with solar/wind farms, reducing reliance on fossil fuels[9][13].  

#### Cons and Mitigation Strategies  
- **Intermittency Risks**: Solar and wind generation variability necessitates hybrid systems with battery storage or grid redundancy.  
- **Transmission Bottlenecks**: Rural areas lack the 500 kV infrastructure seen in Rio Grande do Norte’s Serra do Tigre Wind Complex[17]. Public-private partnerships could prioritize transmission upgrades near renewable zones.  

---

## Water Resources and Cooling Demands  

### Water Availability Challenges  
Urban Pernambuco faces intermittent water supply, with 85% of municipalities implementing rationing during droughts[8]. Data centers using evaporative cooling could exacerbate stress in regions like the Recife Metropolitan Area (RMA), where demand already outpaces supply.  

#### Sustainable Cooling Solutions  
- **Air-to-Liquid Cooling**: Deploy closed-loop systems, as used in Nordic data centers, to reduce water dependency[18].  
- **Wastewater Reuse**: Partner with COMPESA to integrate treated wastewater into cooling systems, leveraging Recife’s condominial sewerage innovations[8].  

---

## Geographic and Connectivity Advantages  

### Recife’s Digital Ecosystem  
Recife’s Parque Tecnológico hosts Um Telecom’s Tier III data center, which benefits from:  
- **Low Latency**: 13 ms latency to 90% of Northeast Brazil’s population[6].  
- **Fiber Optic Networks**: Proximity to BR-101, BR-232, and BR-408 highways ensures redundant fiber links[6].  
- **Submarine Cable Potential**: The city’s push for an international cable landing could decentralize Brazil’s connectivity from Fortaleza, enhancing redundancy[7].  

#### Risks in Coastal Zones  
- **Flood Vulnerability**: Coastal sites require elevated infrastructure, increasing capital costs.  
- **Land Competition**: Tourism and urban development in Boa Viagem may limit expansion.  

### Interior Regions: Terra Nova and Petrolina  
- **Pros**: Lower land costs, abundant solar resources, and reduced water competition.  
- **Cons**: Limited fiber connectivity and skilled labor pools. The state could incentivize edge data centers here to support agritech and rural digitalization.  

---

## Economic and Social Impacts  

### Positive Outcomes  
- **Job Creation**: The solar sector already employs 26,000 in Pernambuco; data centers could add technical roles and stimulate ancillary services[10].  
- **Tech Hub Development**: Recife’s Porto Digital (475 tech firms) would benefit from low-latency infrastructure for AI and IoT applications[7].  

### Negative Externalities  
- **Gentrification**: Rising property prices near data centers could displace communities in RMA.  
- **Energy Price Volatility**: Neoenergia’s 2025 tariff adjustments (-7.1% for high-voltage users, +3% for low-voltage) highlight grid strain risks[5]. Policymakers must ensure data centers contribute to grid stability via demand-response agreements.  

---

## Policy Recommendations  

1. **Zoning and Incentives**  
   - Designate **Renewable Energy Zones** in Terra Nova and Cabo de Santo Agostinho with tax breaks for data centers using ≥80% renewables.  
   - Require water recycling systems as a condition for permits in water-stressed municipalities.  

2. **Infrastructure Investments**  
   - Fast-track the Recife submarine cable project to attract hyperscalers[7].  
   - Expand 500 kV transmission lines to interior solar/wind hubs, mirroring Rio Grande do Norte’s model[17].  

3. **Community Engagement**  
   - Mandate community benefit agreements (CBAs) requiring data centers to fund vocational training and local water infrastructure.  
   - Establish a regional task force to monitor resource use and equitable growth.  

---

## Conclusion  
Pernambuco’s path to becoming a data center hub hinges on balancing its renewable energy potential with environmental stewardship. Recife’s existing infrastructure and submarine cable ambitions position it as the immediate priority, while interior regions offer long-term scalability. By adopting tiered incentives, enforcing sustainability standards, and fostering cross-sector collaboration, policymakers can ensure data centers drive inclusive growth without compromising ecological resilience.  

[Cited Sources: 4, 5, 6, 7, 8, 9, 10, 12, 13, 17, 18]

Sources
[1] 10 Key Factors to Consider When Siting a Data Center - Transect https://www.transect.com/blog/10-key-factors-to-consider-when-siting-a-data-center
[2] Data Center Location Factors - The Rise of New Hubs - Ecoblox https://ecoblox.com/blog/2025/03/07/data-center-locations-rise-of-new-hubs/
[3] Why is site selection so important for the data center industry? - DCD https://www.datacenterdynamics.com/en/opinions/why-is-site-selection-so-important-for-the-data-center-industry/
[4] Neoenergia Pernambuco announces record investment of R$ 5.1 ... https://www.neoenergia.com/en/w/neoenergia-investimentos-pernambuco
[5] Aneel aprova aumento de 0,61% para Neoenergia PE - CanalEnergia https://www.canalenergia.com.br/noticias/53309869/aneel-aprova-aumento-de-061-para-neoenergia-pe
[6] Um Telecom expande infraestrutura digital do Nordeste com 1º Data ... https://inforchannel.com.br/2024/07/16/um-telecom-expande-infraestrutura-digital-do-nordeste-com-1o-data-center-tier-iii-de-pernambuco/
[7] Prefeitura busca atrair cabo submarino para o Recife https://www.diariodepernambuco.com.br/colunas/diarioeconomico/2025/03/todos-querem-o-cabo-submarino-e-recife-entrou-na-briga.html
[8] Water supply and sanitation in Pernambuco - Wikipedia https://en.wikipedia.org/wiki/Water_supply_and_sanitation_in_Pernambuco
[9] Enel Green Power begins operation at wind farm in Pernambuco ... https://valorinternational.globo.com/business/news/2022/04/25/enel-green-power-begins-operation-at-wind-farm-in-pernambuco.ghtml
[10] Pernambuco expande energia solar residencial em 51% e ocupa 3ª ... https://www.diariodepernambuco.com.br/noticia/economia/2024/06/pernambuco-expande-energia-solar-residencial-em-51-e-ocupa-3-posicao.html
[11] 5 Considerations for Choosing Data Center Locations https://blog.equinix.com/blog/2024/08/06/5-considerations-for-choosing-data-center-locations/
[12] Neoenergia inaugurates more modern substation to reinforce ... https://www.neoenergia.com/en/w/neoenergia-subestacao-via-mangue
[13] Planta solar em Pernambuco terá investimentos de R$ 400 milhões https://movimentoeconomico.com.br/economia/energia/2024/11/05/planta-solar-em-pernambuco-tera-investimentos-de-r-400-milhoes/
[14] 7 considerations for data center site selection - TechTarget https://www.techtarget.com/searchdatacenter/tip/Considerations-for-data-center-site-selection
[15] Electricity sector in Brazil - Wikipedia https://en.wikipedia.org/wiki/Electricity_sector_in_Brazil
[16] Essential considerations for effective data center site selection https://www.flexential.com/resources/blog/essential-considerations-effective-data-center-site-selection
[17] GE Vernova's Grid Solutions to supply air-insulated substations to ... https://www.gevernova.com/news/press-releases/ge-vernova-grid-solutions-to-supply-air-insulated-substations-to-casa-dos-ventos-serra-do-tigre-wind-complex-brazil
[18] The 8Cs: What Goes into Choosing Data Centre Locations? https://dataxconnect.com/insights-choosing-data-centre-locations/
[19] Neoenergia Pernambuco - CB Insights https://www.cbinsights.com/company/neoenergia-pernambuco
[20] Executive Roundtable: Data Center Site Selection Implications https://www.datacenterfrontier.com/executive-roundtable/article/55248872/executive-roundtable-data-center-site-selection-implications
[21] Does It Matter Where My Data Center Is Located? - Datacenters.com https://www.datacenters.com/news/does-it-matter-where-my-data-center-is-located
[22] Data Centers: Site Selection 101 https://siteselection.com/data-centers-site-selection-101/
[23] Wind energy assessment and wind farm simulation in Triunfo https://www.sciencedirect.com/science/article/abs/pii/S0960148110001862
[24] #pernambuco #energydiversification #brazil #portfolio | Shon R. Hiatt https://www.linkedin.com/posts/shon-r-hiatt-a874327_pernambuco-energydiversification-brazil-activity-7261769712114556928-5H3j
[25] Recife - REN21 https://www.ren21.net/cities-2021/cities/recife/recife/
[26] UM Telecom acelera transformação digital de Pernambuco https://www.telesintese.com.br/um-telecom-acelera-transformacao-digital-de-pernambuco/
[27] Selecione a localidade para ver as estatísticas de tráfego - IX.br https://ix.br/trafego/agregado/pe
[28] Agregado Topologia Participantes Adesão PIX Adesão CIX Recife/PE https://ix.br/particip/pe
[29] Brazil - Pernambuco Sustainable Water Project : environmental and ... https://documents.worldbank.org/pt/publication/documents-reports/documentdetail/503941468238482238/executive-summary
[30] [PDF] Water stress in the watersheds of the state of Pernambuco https://swat.tamu.edu/media/h2ddrogv/danielatavares_presentation-water-stress.pdf
[31] Aneel define índice de reajuste das tarifas de energia elétrica em ... https://www.neoenergia.com/web/pernambuco/w/aneel-define-indice-de-reajuste-das-tarifas-de-energia-eletrica-em-pernambuco
[32] ANEEL apresenta proposta de revisão tarifária da Neoenergia ... https://fiepe.org.br/aneel-apresenta-proposta-de-revisao-tarifaria-da-neoenergia-pernambuco/
[33] Brazilian islands to have 85% solar and battery storage by 2027 https://www.pv-magazine.com/2024/11/05/brazilian-islands-to-have-85-solar-and-battery-storage-by-2027/
[34] Neoenergia and Pernambuco State Government enter into a ... https://www.neoenergia.com/en/w/neoenergia-e-governo-de-pernambuco-assinam-memorando-de-entendimento-para-producao-de-hidrogenio-verde-1
[35] [PDF] Annual Wind–Energy Report 2022 https://abeeolica.org.br/wp-content/uploads/2023/08/WIND-ENERGY-REPORT-2022-1.pdf
[36] 810 MW Solar Complex Planned For Brazil's Pernambuco https://taiyangnews.info/markets/810-mw-solar-complex-planned-for-brazils-pernambuco
[37] Brazil Climate - Weather conditions in Recife - Aventura do Brasil https://www.aventuradobrasil.com/info/brazil-climate/recife/
[38] Power plant profile: Pernambuco Wind Farm, Brazil https://www.power-technology.com/data-insights/power-plant-profile-pernambuco-wind-farm-brazil/
[39] Climate and Average Weather Year Round in Recife Pernambuco ... https://weatherspark.com/y/31432/Average-Weather-in-Recife-Brazil
[40] Recife - IX.br https://ix.br/adesao/pe/
[41] IX.br https://ix.br
[42] IX.br (PTT.br) Recife - PeeringDB https://www.peeringdb.com/ix/705
[43] Recife Internet Exchanges - Data Center Map https://www.datacentermap.com/ixp/g/recife/
[44] Logo do IX.br https://ix.br/sobre
[45] Cabos de internet no mar? Entenda como a internet chega até você ... https://g1.globo.com/tecnologia/noticia/2023/10/04/cabos-de-internet-no-mar-entenda-como-a-internet-chega-ate-voce-e-se-ha-risco-de-uma-acao-derrubar-a-conexao-no-brasil.ghtml
[46] Pernambuco Water and Sanitation Efficiency and Expansion Project https://www.ndb.int/project/pernambuco-water-and-sanitation-efficiency-and-expansion-project/
[47] [PDF] The Current Situation of Water Supply and Business Opportunties in ... https://itpo-tokyo.unido.org/files/086f46edc032dff46c4e0e0c233f70dd.pdf
[48] Co‐Producing Interdisciplinary Knowledge and Action for ... https://pmc.ncbi.nlm.nih.gov/articles/PMC6450448/
[49] História e Perfil - COMPESA https://servicos.compesa.com.br/historia-e-perfil/
[50] Determination of hydrological stress in a river basin in northeastern ... https://www.scielo.br/j/rbrh/a/H5n6Shby9FM4FsySTFNqQvf/
[51] BRAZIL: Published Rural Water Supply and Sanitary Sewage ... https://www.macsonline.de/brazil/brazil-published-rural-water-supply-and-sanitary-sewage-system-study-in-pernambuco
[52] Abastecimento de Água - COMPESA https://servicos.compesa.com.br/abastecimento-de-agua/
[53] Pernambuco (Nordeste) (Brazil) - Areas - Countries - Online access https://www.thewindpower.net/zones_en_26_125.php
[54] Enel Green Power begins operation at wind farm in Pernambuco ... https://murray.adv.br/en/enel-green-power-begins-operation-at-wind-farm-in-pernambuco/
[55] Ventos do Araripe III (Pernambuco) (Brazil) - The Wind Power https://www.thewindpower.net/windfarm_en_24029_ventos-do-araripe-iii-(pernambuco).php
[56] Top five onshore wind power plants in development in Brazil https://www.power-technology.com/data-insights/top-5-onshore-wind-power-plants-in-development-in-brazil/
[57] Brasil alcança marco histórico em energia solar e Pernambuco ... https://www.absolar.org.br/noticia/https-www-ne9-com-br-brasil-alcanca-marco-historico-em-energia-solar-e-pernambuco-amplia-participacao/

## Regions in Pernambuco with Strong Power Infrastructure for Large Data Centers

Based on current and planned investments, several regions in Pernambuco are particularly well-suited for supporting large-scale data centers due to their robust and expanding power infrastructure:

### 1. **Recife (Especially the South Zone: Pina and Boa Viagem)**
- **Via Mangue Substation:** Recently inaugurated, this substation increases energy supply by 30% for the neighborhoods of Pina and Boa Viagem, adding 52 MVA of installed power—enough to serve approximately 260,000 people. It is one of the most technologically advanced in Brazil, with fully sheltered equipment, underground circuits, and advanced automation for reliability and rapid recovery from outages[2][3][8].
- **Planned Expansion:** The Via Mangue is the first of 13 new substations planned by Neoenergia Pernambuco through 2028, which will collectively boost the state’s distribution grid capacity by 10%[2][3][4][8].
- **Data Center Investment:** The first large-scale Tier 3 data center in Pernambuco is being built in Recife’s Parqtel, leveraging this modernized grid and supported by significant financing for further infrastructure (generators, substations, etc.)[1][6].

### 2. **Interior Regions with Renewable Energy Integration**
- **Arcoverde and Surroundings:** The Arcoverde Project has added a new substation (Arcoverde II) and expanded two others (Garanhuns II and Caetés II), along with 139 km of new transmission lines. This area now has 400 MVA of additional capacity, specifically designed to facilitate the flow of renewable (especially wind) energy and support growing power demands in the interior[5][7].
- **São Caetano (Agreste):** Home to a solar plant already supplying part of a data center’s demand, this region benefits from proximity to renewable generation and recent grid investments[1].
- **Petrolina, Garanhuns, Paulista:** These cities have substations under construction or recently completed, further strengthening the interior grid and making them potential candidates for future data center projects[3][4].

### 3. **Strategic Considerations**
- **Proximity to Renewable Energy:** Areas near wind and solar farms, such as those in the Agreste and Sertão, are attractive for sustainable data center operations, aligning with environmental and operational cost goals[1][5].
- **Urban Infrastructure:** Recife offers the best combination of robust power, fiber connectivity, and access to skilled labor, making it the immediate priority for large data centers[1][6][8].
- **Planned Submarine Cable:** Recife is also being considered for a new submarine cable landing, which would further enhance its attractiveness for hyperscale data centers[1].

---

## Summary Table: Key Regions and Power Infrastructure

| Region                | Power Infrastructure Highlights                                  | Suitability for Large Data Centers     |
|-----------------------|-----------------------------------------------------------------|----------------------------------------|
| Recife (Pina, Boa Viagem) | Via Mangue substation (52 MVA), 13 substations planned, Tier 3 DC | Excellent (urban, reliable, scalable)  |
| Arcoverde & Agreste   | Arcoverde II substation, 400 MVA, renewable integration         | Very Good (renewable, growing grid)    |
| São Caetano           | Solar plant, grid upgrades                                      | Good (renewable, less urbanized)       |
| Petrolina, Garanhuns, Paulista | New substations under construction                      | Good (expanding capacity, regional)    |

---

## Conclusion

**Recife**—particularly its South Zone (Pina and Boa Viagem)—currently offers the most robust and scalable power infrastructure for large data centers in Pernambuco, thanks to major investments like the Via Mangue substation and ongoing grid modernization. **Interior regions** such as Arcoverde, São Caetano, Petrolina, and Garanhuns are also emerging as strong candidates, especially for data centers prioritizing renewable energy integration and long-term expansion[1][2][3][4][5][6][7][8].

Sources
[1] Pernambuco Entra No Promissor Mercado Dos Data Centers https://algomais.com/pernambuco-entra-no-promissor-mercado-dos-data-centers/
[2] Neoenergia inaugurates more modern substation to reinforce ... https://www.neoenergia.com/en/w/neoenergia-subestacao-via-mangue
[3] Neoenergia inaugura subestação mais moderna para reforçar ... https://www.neoenergia.com/w/neoenergia-subestacao-via-mangue
[4] Power Substations - Neoenergia https://www.neoenergia.com/en/w/power-substations
[5] Sterlite Power Advances Its Business Strategy in Brazil with ... https://energetica-india.net/news/sterlite-power-advances-its-business-strategy-in-brazil-with-successful-divestment
[6] Com R$ 41 milhões do BNDES, Pernambuco terá primeiro data ... https://agenciagov.ebc.com.br/noticias/202406/com-r-41-milhoes-do-bndes-pernambuco-tera-primeiro-data-center-tier-3
[7] Sterlite Power delivers its first project in Brazil 28 months ahead of ... https://www.sterlitepower.com/press-release/sterlite-power-delivers-its-first-project-brazil-28-months-ahead-schedule
[8] Neoenergia Inaugura Subestação Via Mangue Para Reforçar ... https://cenarioenergia.com.br/2024/05/27/neoenergia-inaugura-subestacao-via-mangue-para-reforcar-fornecimento-de-energia-em-pernambuco/
[9] Brazil's Um Telecom secures $7.6 million from BNDES for ... https://www.datacenterdynamics.com/en/news/brazils-um-telecom-secures-76-million-from-bndes-for-pernambuco-data-center/
[10] With R$41 million from BNDES, Pernambuco will have its first Tier 3 ... https://www.bnamericas.com/en/news/with-r41-million-from-bndes-pernambuco-will-have-its-first-tier-3-datacenter
[11] Buy the power: Data center deals on the rise in the US | M&A Explorer https://mergers.whitecase.com/highlights/buy-the-power-data-center-deals-on-the-rise-in-the-us
[12] Neoenergia inaugura subestação em Boa Viagem com capacidade ... https://www.cbnrecife.com/artigo/neoenergia-inaugura-subestacao-em-boa-viagem-com-capacidade-para-atender-260-mil-pessoas
[13] Neoenergia opens three substations to expand energy supply in the ... https://www.neoenergia.com/en/w/neoenergia-subestacoes-bahia-pernambuco
[14] Neoenergia inaugura subestação em Boa Viagem que beneficia 16 ... https://www.folhape.com.br/economia/neoenergia-inaugura-subestacao-em-boa-viagem-que-beneficia-16-mil/338014/
[15] [PDF] SCENARIOS FOR OFFSHORE WIND DEVELOPMENT IN BRAZIL https://documents1.worldbank.org/curated/en/099071824152541731/pdf/P1790301a6fb9702119dfe173210dbb9b56.pdf
[16] [PDF] Definitional Mission to Evaluate ICT Projects in Brazil: Volume 4 https://www.jhellerstein.com/Pernambuco.pdf
[17] Brazil's power regulator approves 359MW of new grid capacity for ... https://www.datacenterdynamics.com/en/news/brazils-power-regulator-approves-359mw-of-new-grid-capacity-for-data-centers-in-s%C3%A3o-paulo/
[18] White Paper: The Power Strain: Can the Grid Manage the Data ... https://www.wmeng.com/news-events/the-power-strain-can-the-u-s-grid-handle-the-ai-and-data-center-boom/


"""


if __name__ == "__main__":
    main()