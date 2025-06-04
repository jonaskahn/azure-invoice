import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from typing import Optional, Dict, Any
from pathlib import Path
import base64


# Configuration Constants
class Config:
    # Label and Display
    MAX_LABEL_LENGTH = 40
    TOP_ITEMS_COUNT = 10

    # Chart Dimensions
    CHART_WIDTH = 800
    CHART_HEIGHT = 500
    LARGE_CHART_WIDTH = 1000
    SMALL_CHART_HEIGHT = 400

    # Text and Rotation
    LABEL_FONT_SIZE = 12
    ROTATION_ANGLE = 45

    # Chart Styling
    SPINE_COLOR = '#CCCCCC'
    SPINE_LINEWIDTH = 0.8
    GRID_ALPHA = 0.3
    GRID_LINESTYLE = '-'
    GRID_LINEWIDTH = 0.5

    # Streamlit specific
    SIDEBAR_WIDTH = 300
    CHART_THEME = "streamlit"  # or "plotly", "plotly_white"


class AzureInvoiceData:
    """Encapsulates Azure invoice data operations."""

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        self._clean_data()

    def _clean_data(self) -> None:
        """Clean and prepare data for analysis."""
        if self.df is None:
            return

        # Ensure numeric conversion for Cost and Quantity
        self.df['Cost'] = pd.to_numeric(self.df.get('Cost', 0), errors='coerce').fillna(0)
        self.df['Quantity'] = pd.to_numeric(self.df.get('Quantity', 0), errors='coerce').fillna(0)

    def get_cost_by_resource_group(self) -> pd.Series:
        """Calculate total cost grouped by ResourceGroup."""
        if self.df is None or 'ResourceGroup' not in self.df.columns:
            return pd.Series(dtype=float)
        return self.df.groupby('ResourceGroup')['Cost'].sum().sort_values(ascending=False)

    def get_cost_by_machine(self) -> pd.Series:
        """Calculate total cost grouped by ResourceName (machine)."""
        if self.df is None or 'ResourceName' not in self.df.columns:
            return pd.Series(dtype=float)
        return self.df.groupby('ResourceName')['Cost'].sum().sort_values(ascending=False)

    def get_usage_by_resource_group(self) -> pd.Series:
        """Calculate total usage grouped by ResourceGroup."""
        if self.df is None or 'ResourceGroup' not in self.df.columns:
            return pd.Series(dtype=float)
        return self.df.groupby('ResourceGroup')['Quantity'].sum().sort_values(ascending=False)

    def get_usage_by_machine(self) -> pd.Series:
        """Calculate total usage grouped by ResourceName (machine)."""
        if self.df is None or 'ResourceName' not in self.df.columns:
            return pd.Series(dtype=float)
        return self.df.groupby('ResourceName')['Quantity'].sum().sort_values(ascending=False)

    def get_cost_by_resource_group_and_machine(self) -> pd.DataFrame:
        """Get cost breakdown by resource group and machine."""
        if (self.df is None or
                'ResourceGroup' not in self.df.columns or
                'ResourceName' not in self.df.columns):
            return pd.DataFrame()

        return (self.df.groupby(['ResourceGroup', 'ResourceName'])['Cost']
                .sum()
                .reset_index()
                .sort_values(['ResourceGroup', 'Cost'], ascending=[True, False]))

    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the dashboard."""
        if self.df is None or self.df.empty:
            return {}

        return {
            'total_cost': self.df['Cost'].sum(),
            'total_quantity': self.df['Quantity'].sum(),
            'total_records': len(self.df),
            'date_range': {
                'start': self.df['Date'].min() if 'Date' in self.df.columns else None,
                'end': self.df['Date'].max() if 'Date' in self.df.columns else None
            },
            'unique_resource_groups': self.df['ResourceGroup'].nunique() if 'ResourceGroup' in self.df.columns else 0,
            'unique_machines': self.df['ResourceName'].nunique() if 'ResourceName' in self.df.columns else 0
        }


class StreamlitChartCreator:
    """Creates Plotly charts optimized for Streamlit display."""

    def __init__(self):
        self.theme = Config.CHART_THEME

    def format_label(self, label: str, max_length: int = Config.MAX_LABEL_LENGTH) -> str:
        """Format label to specified length, padding with spaces if needed."""
        if len(label) <= max_length:
            return label.ljust(max_length)
        return label[:max_length - 3] + "..."

    def create_cost_by_resource_group_chart(self, cost_data: pd.Series) -> go.Figure:
        """Create interactive bar chart for total cost by resource group."""
        if cost_data.empty:
            return go.Figure()

        # Format labels
        formatted_labels = [self.format_label(str(label)) for label in cost_data.index]

        fig = go.Figure(data=[
            go.Bar(
                x=formatted_labels,
                y=cost_data.values,
                text=[f'${value:,.2f}' for value in cost_data.values],
                textposition='outside',
                marker_color='lightblue',
                hovertemplate='<b>%{x}</b><br>Cost: $%{y:,.2f}<extra></extra>'
            )
        ])

        fig.update_layout(
            title={
                'text': 'Total Cost by Resource Group',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            xaxis_title='Resource Group',
            yaxis_title='Cost (USD)',
            height=int(Config.CHART_HEIGHT * 1.1),
            showlegend=False,
            xaxis={'tickangle': Config.ROTATION_ANGLE},
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        # Remove top and right spines (clean chart appearance)
        fig.update_xaxes(showline=True, linewidth=1, linecolor=Config.SPINE_COLOR, mirror=False)
        fig.update_yaxes(showline=True, linewidth=1, linecolor=Config.SPINE_COLOR, mirror=False)
        fig.update_layout(
            xaxis=dict(showgrid=True, gridcolor='lightgray', gridwidth=0.5),
            yaxis=dict(showgrid=True, gridcolor='lightgray', gridwidth=0.5)
        )

        return fig

    def create_top_machines_chart(self, cost_data: pd.Series) -> go.Figure:
        """Create interactive bar chart for top machines by cost."""
        if cost_data.empty:
            return go.Figure()

        top_machines = cost_data.head(Config.TOP_ITEMS_COUNT)
        formatted_labels = [self.format_label(str(label)) for label in top_machines.index]

        fig = go.Figure(data=[
            go.Bar(
                x=formatted_labels,
                y=top_machines.values,
                text=[f'${value:,.2f}' for value in top_machines.values],
                textposition='outside',
                marker_color='lightcoral',
                hovertemplate='<b>%{x}</b><br>Cost: $%{y:,.2f}<extra></extra>'
            )
        ])

        fig.update_layout(
            title={
                'text': f'Top {Config.TOP_ITEMS_COUNT} Machines by Total Cost',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            xaxis_title='Machine (ResourceName)',
            yaxis_title='Cost (USD)',
            height=int(Config.CHART_HEIGHT * 1.1),
            showlegend=False,
            xaxis={'tickangle': Config.ROTATION_ANGLE},
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        # Clean chart styling
        fig.update_xaxes(showline=True, linewidth=1, linecolor=Config.SPINE_COLOR, mirror=False)
        fig.update_yaxes(showline=True, linewidth=1, linecolor=Config.SPINE_COLOR, mirror=False)
        fig.update_layout(
            xaxis=dict(showgrid=True, gridcolor='lightgray', gridwidth=0.5),
            yaxis=dict(showgrid=True, gridcolor='lightgray', gridwidth=0.5)
        )

        return fig

    def create_resource_group_breakdown_chart(self, cost_data: pd.DataFrame, resource_group: str) -> go.Figure:
        """Create chart for a specific resource group."""
        sub_df = cost_data[cost_data['ResourceGroup'] == resource_group].head(Config.TOP_ITEMS_COUNT)
        if sub_df.empty:
            return go.Figure()

        formatted_labels = [self.format_label(name) for name in sub_df['ResourceName']]

        fig = go.Figure(data=[
            go.Bar(
                x=formatted_labels,
                y=sub_df['Cost'].values,
                text=[f'${value:,.2f}' for value in sub_df['Cost'].values],
                textposition='outside',
                marker_color='lightgreen',
                hovertemplate='<b>%{x}</b><br>Cost: $%{y:,.2f}<extra></extra>'
            )
        ])

        fig.update_layout(
            title={
                'text': f'Top {Config.TOP_ITEMS_COUNT} Machines in: {resource_group}',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            xaxis_title='Machine (ResourceName)',
            yaxis_title='Cost (USD)',
            height=int(Config.SMALL_CHART_HEIGHT * 1.1),
            showlegend=False,
            xaxis={'tickangle': Config.ROTATION_ANGLE},
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        # Clean chart styling
        fig.update_xaxes(showline=True, linewidth=1, linecolor=Config.SPINE_COLOR, mirror=False)
        fig.update_yaxes(showline=True, linewidth=1, linecolor=Config.SPINE_COLOR, mirror=False)
        fig.update_layout(
            xaxis=dict(showgrid=True, gridcolor='lightgray', gridwidth=0.5),
            yaxis=dict(showgrid=True, gridcolor='lightgray', gridwidth=0.5)
        )

        return fig

    def create_cost_usage_comparison_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create alternative dual-axis bar chart for cost vs usage comparison."""
        if df is None or df.empty:
            return go.Figure()

        # Aggregate by resource group
        agg_data = df.groupby('ResourceGroup').agg({
            'Cost': 'sum',
            'Quantity': 'sum'
        }).reset_index()

        if agg_data.empty:
            return go.Figure()

        # Sort by cost descending
        agg_data = agg_data.sort_values('Cost', ascending=False)

        # Format labels
        formatted_labels = [self.format_label(str(label)) for label in agg_data['ResourceGroup']]

        # Create subplot with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add cost bars
        fig.add_trace(
            go.Bar(
                x=formatted_labels,
                y=agg_data['Cost'],
                name='Cost (USD)',
                marker_color='lightblue',
                text=[f'${value:,.0f}' for value in agg_data['Cost']],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Cost: $%{y:,.2f}<extra></extra>'
            ),
            secondary_y=False,
        )

        # Add usage line
        fig.add_trace(
            go.Scatter(
                x=formatted_labels,
                y=agg_data['Quantity'],
                mode='lines+markers',
                name='Usage (Hours)',
                line=dict(color='red', width=3),
                marker=dict(size=8, color='red'),
                text=[f'{value:,.0f}h' for value in agg_data['Quantity']],
                textposition='top center',
                hovertemplate='<b>%{x}</b><br>Usage: %{y:,.2f} hours<extra></extra>'
            ),
            secondary_y=True,
        )

        # Update layout
        fig.update_xaxes(title_text="Resource Group", tickangle=Config.ROTATION_ANGLE)
        fig.update_yaxes(title_text="Cost (USD)", secondary_y=False)
        fig.update_yaxes(title_text="Usage (Hours)", secondary_y=True)

        fig.update_layout(
            title={
                'text': 'Cost and Usage Comparison by Resource Group',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            height=int(Config.CHART_HEIGHT * 1.1),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        return fig


class StreamlitDashboard:
    """Main Streamlit dashboard orchestrator."""

    def __init__(self):
        self.chart_creator = StreamlitChartCreator()
        self.setup_page_config()

    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Azure Invoice Analyzer",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Add CSS for print/PDF export optimization
        st.markdown("""
        <style>
        /* Print styles for PDF export */
        @media print {
            /* Show main content */
            body, html, .stApp {
                visibility: visible !important;
                background-color: white !important;
                color: black !important;
                font-size: 12px !important;
            }

            /* Hide Streamlit UI elements */
            header[data-testid="stHeader"] {
                display: none !important;
            }

            div[data-testid="stSidebar"] {
                display: none !important;
            }

            div[data-testid="stToolbar"] {
                display: none !important;
            }

            footer {
                display: none !important;
            }

            /* Show main content area */
            .main, .block-container {
                display: block !important;
                visibility: visible !important;
                max-width: 100% !important;
                padding: 1rem !important;
                margin: 0 !important;
            }

            /* Ensure charts are visible */
            .js-plotly-plot, .plotly {
                display: block !important;
                visibility: visible !important;
                page-break-inside: avoid !important;
                margin-bottom: 1rem !important;
                background-color: white !important;
            }

            /* Style headers */
            h1, h2, h3 {
                color: black !important;
                page-break-after: avoid !important;
                margin-top: 1rem !important;
                margin-bottom: 0.5rem !important;
            }

            /* Style metrics */
            div[data-testid="metric-container"] {
                display: inline-block !important;
                margin: 0.5rem !important;
                padding: 0.5rem !important;
                border: 1px solid #ddd !important;
                background-color: #f9f9f9 !important;
            }

            /* Style tables */
            .dataframe, table {
                font-size: 10px !important;
                border-collapse: collapse !important;
                width: 100% !important;
                margin-bottom: 1rem !important;
            }

            .dataframe th, .dataframe td, table th, table td {
                border: 1px solid #ddd !important;
                padding: 4px !important;
                text-align: left !important;
            }

            /* Ensure text is visible */
            p, div, span {
                color: black !important;
                visibility: visible !important;
            }

            /* Page breaks */
            .stSubheader {
                page-break-before: auto !important;
                margin-top: 1.5rem !important;
            }

            /* Force content visibility */
            * {
                -webkit-print-color-adjust: exact !important;
                color-adjust: exact !important;
            }
        }

        /* Regular screen styles */
        .print-button {
            background-color: #ff6b6b;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            border: none;
            cursor: pointer;
            width: 100%;
            margin-bottom: 1rem;
        }

        .print-button:hover {
            background-color: #ff5252;
        }

        /* Print header only visible when printing */
        .print-header {
            display: none;
        }

        @media print {
            .print-header {
                display: block !important;
                text-align: center;
                border-bottom: 2px solid #ccc;
                padding-bottom: 10px;
                margin-bottom: 20px;
                page-break-after: avoid;
            }
        }
        </style>
        """, unsafe_allow_html=True)

    def display_header(self):
        """Display application header."""
        st.title("🔍 Azure Invoice Analyzer")
        st.markdown("""
        **Comprehensive analysis of Azure billing data with interactive visualizations**

        Upload your Azure invoice CSV file to get detailed cost breakdowns, usage analytics, 
        and resource group insights.
        """)
        st.divider()

    def display_file_uploader(self) -> Optional[pd.DataFrame]:
        """Handle file upload and return processed DataFrame."""
        uploaded_file = st.file_uploader(
            "Choose your Azure Invoice CSV file",
            type=['csv'],
            help="Upload your Azure invoice CSV file (supports up to 200MB)"
        )

        if uploaded_file is not None:
            try:
                with st.spinner("Loading and processing your data..."):
                    # Read the CSV with date parsing
                    df = pd.read_csv(uploaded_file, parse_dates=['Date'], low_memory=False)

                    # Display basic file info
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Records", len(df))
                    with col2:
                        st.metric("Columns", len(df.columns))
                    with col3:
                        file_size_mb = uploaded_file.size / (1024 * 1024)
                        st.metric("File Size", f"{file_size_mb:.2f} MB")

                    return df

            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                st.info(
                    "Please ensure your CSV file has the correct format with columns: Date, Cost, Quantity, ResourceGroup, ResourceName")
                return None

        return None

    def display_data_summary(self, data: AzureInvoiceData):
        """Display data summary metrics."""
        summary = data.get_data_summary()

        if not summary:
            return

        st.subheader("📈 Executive Summary")

        # Main metrics row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Cost",
                f"${summary['total_cost']:,.2f}",
                help="Total cost across all resources"
            )

        with col2:
            st.metric(
                "Total Usage",
                f"{summary['total_quantity']:,.2f} hrs",
                help="Total usage hours across all resources"
            )

        with col3:
            st.metric(
                "Resource Groups",
                summary['unique_resource_groups'],
                help="Number of unique resource groups"
            )

        with col4:
            st.metric(
                "Machines",
                summary['unique_machines'],
                help="Number of unique machines/resources"
            )

        # Date range and key insights
        col1, col2 = st.columns(2)

        with col1:
            if summary['date_range']['start'] and summary['date_range']['end']:
                st.info(
                    f"📅 **Data Period:** {summary['date_range']['start'].strftime('%Y-%m-%d')} to {summary['date_range']['end'].strftime('%Y-%m-%d')}")

        with col2:
            # Calculate average cost per machine for insights
            avg_cost_per_machine = summary['total_cost'] / summary['unique_machines'] if summary[
                                                                                             'unique_machines'] > 0 else 0
            st.info(f"💡 **Average Cost per Machine:** ${avg_cost_per_machine:,.2f}")

        # Add print-only summary table with proper string formatting
        total_cost = summary['total_cost']
        total_quantity = summary['total_quantity']
        unique_resource_groups = summary['unique_resource_groups']
        unique_machines = summary['unique_machines']

        st.markdown(f"""
        <div class="print-summary" style="display: none;">
        <style>
        @media print {{
            .print-summary {{
                display: block !important;
                margin: 20px 0;
                padding: 15px;
                border: 1px solid #ddd;
                background-color: #f9f9f9;
            }}
            .print-summary table {{
                width: 100%;
                border-collapse: collapse;
            }}
            .print-summary th, .print-summary td {{
                padding: 8px;
                text-align: left;
                border: 1px solid #ddd;
            }}
            .print-summary th {{
                background-color: #f2f2f2;
                font-weight: bold;
            }}
        }}
        </style>
        <h3>Summary Statistics</h3>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Cost</td><td>${total_cost:,.2f}</td></tr>
            <tr><td>Total Usage Hours</td><td>{total_quantity:,.2f}</td></tr>
            <tr><td>Resource Groups</td><td>{unique_resource_groups}</td></tr>
            <tr><td>Total Machines</td><td>{unique_machines}</td></tr>
            <tr><td>Average Cost per Machine</td><td>${avg_cost_per_machine:,.2f}</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

    def display_cost_analysis(self, data: AzureInvoiceData):
        """Display cost analysis charts."""
        st.subheader("💰 Cost Analysis")

        # Get data
        cost_by_rg = data.get_cost_by_resource_group()
        cost_by_machine = data.get_cost_by_machine()

        if cost_by_rg.empty and cost_by_machine.empty:
            st.warning("No cost data available for analysis.")
            return

        # Display charts vertically for better visibility
        if not cost_by_rg.empty:
            fig1 = self.chart_creator.create_cost_by_resource_group_chart(cost_by_rg)
            # Enhanced config for better PDF export
            fig1.update_layout(
                font=dict(size=12),
                title_font_size=16,
                margin=dict(l=50, r=50, t=80, b=50)
            )
            st.plotly_chart(fig1, use_container_width=True, config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToAdd': ['downloadSVG'],
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'cost_by_resource_group',
                    'height': int(Config.CHART_HEIGHT * 1.1),
                    'width': 1200,
                    'scale': 2
                }
            })

        if not cost_by_machine.empty:
            fig2 = self.chart_creator.create_top_machines_chart(cost_by_machine)
            # Enhanced config for better PDF export
            fig2.update_layout(
                font=dict(size=12),
                title_font_size=16,
                margin=dict(l=50, r=50, t=80, b=50)
            )
            st.plotly_chart(fig2, use_container_width=True, config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToAdd': ['downloadSVG'],
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': f'top_{Config.TOP_ITEMS_COUNT}_machines',
                    'height': int(Config.CHART_HEIGHT * 1.1),
                    'width': 1200,
                    'scale': 2
                }
            })

        # Cost vs Usage analysis - dual-axis comparison
        if not data.df.empty:
            st.subheader("📊 Cost vs Usage Analysis")

            fig3 = self.chart_creator.create_cost_usage_comparison_chart(data.df)
            # Enhanced config for better PDF export
            fig3.update_layout(
                font=dict(size=12),
                title_font_size=16,
                margin=dict(l=50, r=50, t=80, b=50)
            )
            st.plotly_chart(fig3, use_container_width=True, config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToAdd': ['downloadSVG'],
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'cost_vs_usage_comparison',
                    'height': int(Config.CHART_HEIGHT * 1.1),
                    'width': 1200,
                    'scale': 2
                }
            })
            st.info(
                "💡 **Chart Guide:** Blue bars show cost (left axis), red line shows usage (right axis). This makes it easy to spot cost-inefficient resource groups.")

    def display_resource_group_breakdown(self, data: AzureInvoiceData):
        """Display detailed breakdown by resource group."""
        st.subheader("🏗️ Resource Group Breakdown")

        cost_by_rg_machine = data.get_cost_by_resource_group_and_machine()

        if cost_by_rg_machine.empty:
            st.warning("No resource group data available.")
            return

        # Resource group selector
        resource_groups = cost_by_rg_machine['ResourceGroup'].unique()

        if len(resource_groups) == 0:
            st.warning("No resource groups found in the data.")
            return

        # Create tabs for each resource group
        tabs = st.tabs([f"📁 {rg[:20]}..." if len(rg) > 20 else f"📁 {rg}" for rg in resource_groups[:5]])

        for i, resource_group in enumerate(resource_groups[:5]):  # Limit to first 5 tabs
            with tabs[i]:
                fig = self.chart_creator.create_resource_group_breakdown_chart(cost_by_rg_machine, resource_group)
                if fig.data:
                    st.plotly_chart(fig, use_container_width=True)

                # Show data table for this resource group - now uses dynamic TOP_ITEMS_COUNT
                rg_data = cost_by_rg_machine[cost_by_rg_machine['ResourceGroup'] == resource_group].head(
                    Config.TOP_ITEMS_COUNT)
                if not rg_data.empty:
                    st.dataframe(
                        rg_data[['ResourceName', 'Cost']].round(2),
                        use_container_width=True,
                        hide_index=True
                    )

    def display_data_tables(self, data: AzureInvoiceData):
        """Display detailed data tables."""
        st.subheader("📊 Detailed Data Tables")

        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(
            ["💰 Cost by Resource Group", "🖥️ Cost by Machine", "⏱️ Usage by Resource Group", "⚙️ Usage by Machine"])

        with tab1:
            cost_by_rg = data.get_cost_by_resource_group()
            if not cost_by_rg.empty:
                df_display = pd.DataFrame({
                    'Resource Group': cost_by_rg.index,
                    'Total Cost ($)': cost_by_rg.values.round(2)
                })
                st.dataframe(df_display, use_container_width=True, hide_index=True)
            else:
                st.info("No cost data by resource group available.")

        with tab2:
            cost_by_machine = data.get_cost_by_machine()
            if not cost_by_machine.empty:
                df_display = pd.DataFrame({
                    'Machine': cost_by_machine.index,
                    'Total Cost ($)': cost_by_machine.values.round(2)
                })
                st.dataframe(df_display, use_container_width=True, hide_index=True)
            else:
                st.info("No cost data by machine available.")

        with tab3:
            usage_by_rg = data.get_usage_by_resource_group()
            if not usage_by_rg.empty:
                df_display = pd.DataFrame({
                    'Resource Group': usage_by_rg.index,
                    'Total Usage (hrs)': usage_by_rg.values.round(2)
                })
                st.dataframe(df_display, use_container_width=True, hide_index=True)
            else:
                st.info("No usage data by resource group available.")

        with tab4:
            usage_by_machine = data.get_usage_by_machine()
            if not usage_by_machine.empty:
                df_display = pd.DataFrame({
                    'Machine': usage_by_machine.index,
                    'Total Usage (hrs)': usage_by_machine.values.round(2)
                })
                st.dataframe(df_display, use_container_width=True, hide_index=True)
            else:
                st.info("No usage data by machine available.")

    def display_sidebar_controls(self, data: Optional[AzureInvoiceData]):
        """Display sidebar with controls and options."""
        with st.sidebar:
            st.header("⚙️ Analysis Options")

            if data is not None:
                st.subheader("📋 Export Options")

                # Print/PDF Export button
                if st.button("🖨️ Print / Export PDF", use_container_width=True):
                    # Enhanced print functionality with better content visibility
                    st.markdown("""
                    <script>
                    // Wait for content to load then print
                    setTimeout(function() {
                        // Ensure all elements are visible for print
                        document.body.style.visibility = 'visible';
                        window.print();
                    }, 500);
                    </script>
                    """, unsafe_allow_html=True)
                    st.success("📄 Print dialog opening... Choose 'Save as PDF' and enable 'Background graphics'!")
                    st.info(
                        "💡 **Print Tips:** Enable 'Background graphics' and use Portrait orientation for best results.")

                # Individual chart downloads
                if st.button("📊 Download All Charts", use_container_width=True):
                    st.info("💡 **Tip**: Right-click on any chart → 'Download plot as a png' for individual charts")

                st.subheader("🎨 Chart Options")

                # Chart customization options
                chart_height = st.slider(
                    "Chart Height",
                    300,
                    800,
                    Config.CHART_HEIGHT,
                    help="Base chart height in pixels (actual height will be +10% for better visibility)"
                )
                Config.CHART_HEIGHT = chart_height

                # Top items count control
                top_items = st.slider(
                    "Show Top N Machines",
                    min_value=5,
                    max_value=100,
                    value=Config.TOP_ITEMS_COUNT,
                    step=1,
                    help="Number of top machines to display in charts"
                )
                Config.TOP_ITEMS_COUNT = top_items

            else:
                st.info("Upload a CSV file to access analysis options.")

            st.subheader("ℹ️ About")
            st.markdown("""
            **Azure Invoice Analyzer**

            This tool helps you analyze Azure billing data with:
            - Interactive cost visualizations
            - Resource group breakdowns  
            - Usage analytics
            - Cost vs usage comparisons
            - PDF export capabilities

            **Supported CSV Format:**
            - Date, Cost, Quantity, ResourceGroup, ResourceName columns
            """)

    def run(self):
        """Main application entry point."""
        # Display header
        self.display_header()

        # File upload
        df = self.display_file_uploader()

        if df is not None:
            # Create data object
            data = AzureInvoiceData(df)

            # Display sidebar controls
            self.display_sidebar_controls(data)

            # Display main analysis
            with st.container():
                # Data summary
                self.display_data_summary(data)
                st.divider()

                # Cost analysis
                self.display_cost_analysis(data)
                st.divider()

                # Resource group breakdown
                self.display_resource_group_breakdown(data)
                st.divider()

                # Data tables
                self.display_data_tables(data)

                # Success message
                st.success("✅ Analysis complete! Use the sidebar options to customize views and export data.")

        else:
            # Simple instruction when no file is uploaded
            st.info("📁 Please upload your Azure Invoice CSV file to begin analysis!")

            st.markdown("""
            **Required CSV columns:**
            - Date, Cost, Quantity, ResourceGroup, ResourceName
            """)


# Streamlit app entry point
def main():
    """Main function to run the Streamlit app."""
    dashboard = StreamlitDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()