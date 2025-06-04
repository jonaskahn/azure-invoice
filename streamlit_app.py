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
            height=Config.CHART_HEIGHT,
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
            height=Config.CHART_HEIGHT,
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
            height=Config.SMALL_CHART_HEIGHT,
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
            height=Config.CHART_HEIGHT,
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

    def display_header(self):
        """Display application header."""
        st.title("🔍 Azure Invoice Analyzer")
        st.markdown("""
        **Comprehensive analysis of Azure billing data with interactive visualizations**

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

        st.subheader("📈 Data Summary")

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

        # Date range if available
        if summary['date_range']['start'] and summary['date_range']['end']:
            st.info(
                f"📅 Data Period: {summary['date_range']['start'].strftime('%Y-%m-%d')} to {summary['date_range']['end'].strftime('%Y-%m-%d')}")

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
            st.plotly_chart(fig1, use_container_width=True)

        if not cost_by_machine.empty:
            fig2 = self.chart_creator.create_top_machines_chart(cost_by_machine)
            st.plotly_chart(fig2, use_container_width=True)

        # Cost vs Usage analysis - dual-axis comparison
        if not data.df.empty:
            st.subheader("📊 Cost vs Usage Analysis")

            fig3 = self.chart_creator.create_cost_usage_comparison_chart(data.df)
            st.plotly_chart(fig3, use_container_width=True)
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
                st.subheader("🎨 Chart Options")

                # Chart customization options
                chart_height = st.slider("Chart Height", 300, 800, Config.CHART_HEIGHT)
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