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
if 'anthropic_api_key' not in st.session_state:
    st.session_state.anthropic_api_key = None
if 'api_key_verified' not in st.session_state:
    st.session_state.api_key_verified = False

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

    # Add coordinates for mapping
    coordinates = [
        (-8.0522, -34.9286),  # Recife
        (-8.2834, -35.9761),  # Caruaru
        (-8.8905, -36.4919),  # Garanhuns
        (-8.2745, -35.8714),  # São Caetano
        (-7.5597, -35.0044),  # Goiana
        (-7.7481, -35.2318),  # Nazaré da Mata
        (-9.3891, -40.5006),  # Petrolina
        (-8.4194, -36.7611),  # Arcoverde
        (-8.3590, -34.9544)  # Suape
    ]

    df['Latitude'] = [coord[0] for coord in coordinates]
    df['Longitude'] = [coord[1] for coord in coordinates]

    return df


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
"""


def initialize_anthropic_client():
    """Initialize Anthropic client with API key"""
    import os

    api_key = None

    # Method 1: Try session state first (highest priority)
    if st.session_state.get("anthropic_api_key"):
        api_key = st.session_state.anthropic_api_key
        st.sidebar.info(f"🔑 Using session API key: {api_key[:8]}...{api_key[-4:]}")

    # Method 2: Try environment variable
    elif os.getenv("ANTHROPIC_API_KEY"):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        st.sidebar.info(f"🌍 Using environment API key: {api_key[:8]}...{api_key[-4:]}")

    # Method 3: Try Streamlit secrets (with better error handling)
    else:
        try:
            if hasattr(st, 'secrets') and "ANTHROPIC_API_KEY" in st.secrets:
                api_key = st.secrets["ANTHROPIC_API_KEY"]
                st.sidebar.info(f"📁 Using secrets API key: {api_key[:8]}...{api_key[-4:]}")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Secrets access failed: {str(e)}")

    if not api_key:
        st.sidebar.error("❌ No API key found in any source")
        return None

    # Validate API key format
    if not api_key.startswith('sk-ant-'):
        st.sidebar.error("❌ Invalid API key format. Must start with 'sk-ant-'")
        return None

    # Create and test client
    try:
        client = anthropic.Anthropic(api_key=api_key)
        st.session_state.api_key_verified = True
        st.sidebar.success("✅ Anthropic client initialized successfully!")
        return client
    except Exception as e:
        st.sidebar.error(f"❌ Failed to create Anthropic client: {str(e)}")
        st.session_state.api_key_verified = False
        return None


def save_api_key_to_session(api_key):
    """Save API key to session state and reset client"""
    if api_key and api_key.strip():
        st.session_state.anthropic_api_key = api_key.strip()
        st.session_state.anthropic_client = None  # Reset client to force reinit
        st.session_state.api_key_verified = False
        return True
    return False


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
            model="claude-3-5-sonnet-20241022",  # Updated to newer model
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


def create_executive_summary(df, selected_regions, score_threshold):
    """Create a dynamic executive summary based on current filters"""
    st.markdown('<div class="section-header">📋 Executive Summary</div>', unsafe_allow_html=True)

    # Header with key insights
    st.markdown("### Strategic Data Center Investment Analysis - Pernambuco, Brazil")
    st.markdown("*Dynamic analysis based on current filter selections*")

    # Key metrics bar
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Analyzed Locations", len(df), delta=f"{len(df) - 9} from baseline")
    with col2:
        avg_score = df['Overall_Score'].mean()
        st.metric("Average Viability", f"{avg_score:.1f}/10", delta=f"{avg_score - 5.8:.1f}")
    with col3:
        top_score = df['Overall_Score'].max()
        best_location = df.loc[df['Overall_Score'].idxmax(), 'City']
        st.metric("Top Location", best_location, delta=f"Score: {top_score:.1f}")
    with col4:
        viable_count = len(df[df['Overall_Score'] >= 7])
        st.metric("High-Viability Sites", viable_count, delta=f"{(viable_count / len(df) * 100):.0f}% of total")

    st.markdown("---")

    # Executive recommendations
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 🎯 Strategic Recommendations")

        # Dynamic recommendations based on data
        top_3 = df.nlargest(3, 'Overall_Score')

        # Primary recommendation
        primary = top_3.iloc[0]
        st.markdown(f"""
        **🥇 PRIMARY RECOMMENDATION: {primary['City']}**

        With an overall score of **{primary['Overall_Score']:.1f}/10**, {primary['City']} emerges as the optimal location for data center investment. 
        Key advantages include {primary['Pros'].lower()}, making it ideal for immediate deployment.

        *Investment Priority: HIGH*
        """)

        # Secondary recommendations
        if len(top_3) > 1:
            secondary = top_3.iloc[1]
            st.markdown(f"""
            **🥈 SECONDARY OPTION: {secondary['City']}**

            Scoring **{secondary['Overall_Score']:.1f}/10**, {secondary['City']} offers strong potential as an alternative or expansion site. 
            Consider for {secondary['Main_Reasons'].lower()} requirements.

            *Investment Priority: MEDIUM*
            """)

    with col2:
        st.markdown("### 📊 Market Analysis")

        # Regional distribution analysis
        region_analysis = df.groupby('Region').agg({
            'Overall_Score': 'mean',
            'Infrastructure_Score': 'mean',
            'Renewable_Score': 'mean',
            'Water_Score': 'mean'
        }).round(1)

        # Best performing region
        best_region = region_analysis['Overall_Score'].idxmax()
        best_region_score = region_analysis['Overall_Score'].max()

        st.markdown(f"""
        **Regional Leadership:**

        **{best_region}** leads regional performance with average score of **{best_region_score:.1f}/10**.
        """)


def create_main_chat_interface(df):
    """Create the main chat interface for the right column"""
    st.markdown("### 🤖 AI Data Center Assistant")

    # Check API key status
    has_api_key = bool(st.session_state.get("anthropic_api_key"))
    has_client = bool(st.session_state.get("anthropic_client"))

    if not has_api_key:
        st.warning("🔑 API Key Required")

        # API setup form
        with st.form("main_chat_api_setup"):
            st.markdown("**Enter your Anthropic API key to start chatting:**")
            api_key = st.text_input(
                "API Key:",
                type="password",
                placeholder="sk-ant-api03-...",
                help="Get your key from https://console.anthropic.com/"
            )
            setup_btn = st.form_submit_button("🚀 Start AI Assistant", type="primary")

            if setup_btn and api_key:
                if api_key.startswith('sk-ant-'):
                    if save_api_key_to_session(api_key):
                        st.session_state.anthropic_client = initialize_anthropic_client()
                        if st.session_state.anthropic_client:
                            st.success("✅ AI Assistant is ready!")
                            st.balloons()
                            st.rerun()
                        else:
                            st.error("❌ Failed to initialize AI. Please check your key.")
                    else:
                        st.error("❌ Failed to save API key")
                else:
                    st.error("❌ Invalid API key format")
        return

    # Initialize client if needed
    if not has_client:
        with st.spinner("🔄 Initializing AI Assistant..."):
            st.session_state.anthropic_client = initialize_anthropic_client()

    if not st.session_state.anthropic_client:
        st.error("❌ AI Assistant unavailable")
        return

    # AI is ready - show quick questions
    st.success("🟢 AI Assistant Active")

    # Quick action buttons
    st.markdown("#### 💡 Quick Questions")
    quick_questions = [
        ("🏆", "Best overall location?"),
        ("⚖️", "Compare Recife vs São Caetano"),
        ("💧", "Water management challenges?"),
        ("🌱", "Renewable energy opportunities?")
    ]

    # Display quick questions in a grid
    cols = st.columns(2)
    for i, (emoji, question) in enumerate(quick_questions):
        col_idx = i % 2
        with cols[col_idx]:
            if st.button(f"{emoji} {question}", key=f"main_quick_{i}", use_container_width=True):
                # Add to chat and get response
                st.session_state.chat_messages.append({"role": "user", "content": question})

                with st.spinner("🤔 Analyzing..."):
                    context_data = get_context_from_data(df, question)
                    response = query_anthropic(st.session_state.anthropic_client, question, context_data, df)
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})

                st.rerun()

    # Chat input
    st.markdown("#### ✍️ Ask Your Question")
    with st.form("main_chat_form"):
        user_question = st.text_area(
            "Type your question about data center locations in Pernambuco:",
            placeholder="e.g., What factors should I consider for a sustainable data center?",
            height=100,
            key="main_chat_input"
        )

        send_btn = st.form_submit_button("📤 Send Message", type="primary", use_container_width=True)

        if send_btn and user_question.strip():
            # Add user message
            st.session_state.chat_messages.append({"role": "user", "content": user_question.strip()})

            # Get AI response
            with st.spinner("🤔 Analyzing your question..."):
                try:
                    context_data = get_context_from_data(df, user_question)
                    response = query_anthropic(st.session_state.anthropic_client, user_question, context_data, df)
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    # Chat history
    if st.session_state.chat_messages:
        st.markdown("#### 💬 Conversation History")

        for message in st.session_state.chat_messages[-6:]:  # Show last 6 messages
            if message["role"] == "user":
                st.markdown(f"""
                <div style="background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin: 5px 0;">
                <strong>🙋‍♂️ You:</strong><br>{message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: #f1f8e9; padding: 10px; border-radius: 10px; margin: 5px 0;">
                <strong>🤖 AI:</strong><br>{message['content']}
                </div>
                """, unsafe_allow_html=True)


def create_sidebar_chat(df):
    """Create compact sidebar chat for quick access"""
    st.markdown("### 🚀 Quick AI Access")

    # Status indicator
    if st.session_state.get("anthropic_client"):
        st.success("🟢 AI Ready")

        # Quick question in sidebar
        with st.form("sidebar_quick_form"):
            quick_q = st.text_input(
                "Quick question:",
                placeholder="e.g., Best location?",
                key="sidebar_quick_input"
            )
            if st.form_submit_button("Ask"):
                if quick_q.strip():
                    st.session_state.chat_messages.append({"role": "user", "content": quick_q.strip()})
                    with st.spinner("🤔"):
                        context_data = get_context_from_data(df, quick_q)
                        response = query_anthropic(st.session_state.anthropic_client, quick_q, context_data, df)
                        st.session_state.chat_messages.append({"role": "assistant", "content": response})
                    st.rerun()

        # Recent activity
        if st.session_state.chat_messages:
            st.markdown("**Latest Response:**")
            last_response = st.session_state.chat_messages[-1]
            if last_response["role"] == "assistant":
                truncated = last_response["content"][:100] + "..." if len(last_response["content"]) > 100 else \
                last_response["content"]
                st.caption(truncated)
    else:
        st.warning("🔴 AI Not Ready")
        st.caption("Configure API key in main chat →")


def create_map_view(df):
    """Create interactive map view of data center locations"""
    st.markdown('<div class="section-header">🌍 Interactive Map View</div>', unsafe_allow_html=True)

    # Map controls
    col1, col2, col3 = st.columns(3)

    with col1:
        color_by = st.selectbox(
            "Color locations by:",
            ["Overall_Score", "Infrastructure_Score", "Renewable_Score", "Water_Score", "Connectivity_Score"],
            key="map_color_selector"
        )

    with col2:
        size_by = st.selectbox(
            "Size bubbles by:",
            ["Overall_Score", "Infrastructure_Score", "Renewable_Score", "Water_Score", "Connectivity_Score"],
            key="map_size_selector"
        )

    with col3:
        map_style = st.selectbox(
            "Map style:",
            ["open-street-map", "carto-positron", "carto-darkmatter"],
            key="map_style_selector"
        )

    # Create the map
    fig_map = px.scatter_map(
        df,
        lat="Latitude",
        lon="Longitude",
        color=color_by,
        size=size_by,
        hover_name="City",
        hover_data={
            "Region": True,
            "Overall_Score": ":.1f",
            "Infrastructure_Score": ":.1f",
            "Renewable_Score": ":.1f",
            "Water_Score": ":.1f",
            "Connectivity_Score": ":.1f",
            "Latitude": False,
            "Longitude": False
        },
        color_continuous_scale="Viridis",
        size_max=25,
        zoom=6,
        title=f"Pernambuco Data Center Locations - Colored by {color_by.replace('_', ' ')}",
        height=600
    )

    # Center map on Pernambuco
    fig_map.update_layout(
        margin=dict(l=0, r=0, t=50, b=0)
    )

    st.plotly_chart(fig_map, use_container_width=True)


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
    with st.sidebar.expander("🔧 API Configuration", expanded=not st.session_state.get("api_key_verified", False)):
        st.markdown("**Anthropic API Setup**")

        # Show current status first
        if st.session_state.get("anthropic_api_key"):
            masked_key = st.session_state.anthropic_api_key[:8] + "..." + st.session_state.anthropic_api_key[-4:]
            st.success(f"✅ API Key configured: {masked_key}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🧪 Test Key", key="sidebar_test_btn"):
                    test_client = initialize_anthropic_client()
                    if test_client:
                        st.success("🟢 API Key working!")
                        st.session_state.anthropic_client = test_client
                    else:
                        st.error("🔴 API Key failed")
            with col2:
                if st.button("🗑️ Remove Key", key="sidebar_clear_btn"):
                    st.session_state.anthropic_api_key = None
                    st.session_state.anthropic_client = None
                    st.session_state.api_key_verified = False
                    st.rerun()
        else:
            # Input field for API key
            st.markdown("**Enter your API key:**")

            # Create a form to handle the input properly
            with st.form("sidebar_api_form"):
                sidebar_api_key = st.text_input(
                    "Anthropic API Key",
                    type="password",
                    placeholder="sk-ant-...",
                    help="Get your key from https://console.anthropic.com/",
                    key="sidebar_api_input"
                )

                submitted = st.form_submit_button("💾 Save API Key")

                if submitted and sidebar_api_key:
                    if sidebar_api_key.startswith('sk-ant-'):
                        if save_api_key_to_session(sidebar_api_key):
                            st.session_state.anthropic_client = initialize_anthropic_client()
                            if st.session_state.anthropic_client:
                                st.success("✅ API Key saved successfully!")
                                st.rerun()
                            else:
                                st.error("❌ Key saved but client failed")
                        else:
                            st.error("❌ Failed to save key")
                    else:
                        st.error("❌ Invalid key format")

            # Instructions
            st.markdown("**Alternative methods:**")
            st.code("export ANTHROPIC_API_KEY='your-key'", language="bash")

        # Debug info at the bottom (no nested expander)
        st.markdown("---")
        st.markdown("**Debug Info:**")
        st.caption(f"Session key: {bool(st.session_state.get('anthropic_api_key'))}")
        st.caption(f"Client exists: {bool(st.session_state.get('anthropic_client'))}")
        st.caption(f"Verified: {st.session_state.get('api_key_verified', False)}")

    st.sidebar.markdown("---")

    # Additional Features Section
    st.sidebar.markdown("### 🚀 Additional Features")

    # Knowledge Base button
    if st.sidebar.button("📚 Knowledge Base", key="knowledge_base_btn",
                         help="Access comprehensive data center knowledge base"):
        st.sidebar.info(
            "🔄 Knowledge Base feature coming soon! This will provide access to:\n\n• Technical documentation\n• Best practices\n• Regulatory guidelines\n• Industry standards\n• Case studies")

    # Create Work Group button
    if st.sidebar.button("👥 Create Work Group", key="create_workgroup_btn",
                         help="Organize collaborative teams for projects"):
        st.sidebar.info(
            "🔄 Work Group feature coming soon! This will enable:\n\n• Team collaboration\n• Project management\n• Document sharing\n• Task assignment\n• Progress tracking")

    st.sidebar.markdown("---")

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

    st.markdown("---")

    # Split screen layout: Dashboard (2/3) + Chat (1/3)
    dashboard_col, chat_col = st.columns([2, 1])

    with dashboard_col:
        st.markdown("### 📊 Dashboard Analysis")

        # Tabs for dashboard content
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📋 Executive Summary", "📊 Overview", "🗺️ Regional", "⚡ Infrastructure", "💧 Sustainability", "📈 Scoring",
            "🌍 Map"
        ])

        with tab1:
            # Executive Summary Tab
            create_executive_summary(filtered_df, selected_regions, score_threshold)

        with tab2:
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

        with tab3:
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

        with tab4:
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

        with tab5:
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

        with tab6:
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

        with tab7:
            # Map View Tab
            create_map_view(filtered_df)

    # Right column: Dedicated Chat Interface
    with chat_col:
        st.markdown("### 🤖 AI Assistant")
        create_main_chat_interface(filtered_df)

    # Sidebar AI Assistant (Always Visible)
    with st.sidebar:
        st.markdown("---")
        create_sidebar_chat(filtered_df)


if __name__ == "__main__":
    main()