from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from plotly.subplots import make_subplots


class SimpleInvoiceData:
    """
    Simple Azure invoice data operations for basic service usage analysis.
    
    This class provides consistent calculation methods for all simple template analysis.
    All charts and tables use the same underlying calculation formulas to ensure 
    consistent results across different sections.
    
    Calculation Standards:
    - Cost calculations: df.groupby(column)['Cost'].sum().sort_values(ascending=False)
    - Usage calculations: df.groupby(column)['Quantity'].sum().sort_values(ascending=False)
    - Efficiency calculations: Total_Cost / Total_Quantity per group
    - All methods handle missing data and return empty Series/DataFrames gracefully
    """
    
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        self._clean_data()
    
    def _clean_data(self) -> None:
        """Clean and prepare simple data for analysis."""
        if self.df is None:
            return
        
        # Ensure numeric conversion for Cost and Quantity
        self.df['Cost'] = pd.to_numeric(self.df.get('Cost', 0), errors='coerce').fillna(0)
        self.df['Quantity'] = pd.to_numeric(self.df.get('Quantity', 0), errors='coerce').fillna(0)
        
        # Ensure required columns exist
        required_columns = ['ServiceName', 'ServiceType', 'ServiceRegion', 'ServiceResource']
        for col in required_columns:
            if col not in self.df.columns:
                self.df[col] = 'Unknown'
                st.warning(f"Missing column '{col}' - created with default values")
    
    def get_cost_by_service(self) -> pd.Series:
        """Calculate total cost grouped by ServiceName."""
        if self.df is None or 'ServiceName' not in self.df.columns:
            return pd.Series(dtype=float)
        return self.df.groupby('ServiceName')['Cost'].sum().sort_values(ascending=False)
    
    def get_cost_by_service_type(self) -> pd.Series:
        """Calculate total cost grouped by ServiceType."""
        if self.df is None or 'ServiceType' not in self.df.columns:
            return pd.Series(dtype=float)
        return self.df.groupby('ServiceType')['Cost'].sum().sort_values(ascending=False)
    
    def get_cost_by_region(self) -> pd.Series:
        """Calculate total cost grouped by ServiceRegion."""
        if self.df is None or 'ServiceRegion' not in self.df.columns:
            return pd.Series(dtype=float)
        return self.df.groupby('ServiceRegion')['Cost'].sum().sort_values(ascending=False)
    
    def get_cost_by_resource(self) -> pd.Series:
        """Calculate total cost grouped by ServiceResource."""
        if self.df is None or 'ServiceResource' not in self.df.columns:
            return pd.Series(dtype=float)
        return self.df.groupby('ServiceResource')['Cost'].sum().sort_values(ascending=False)
    
    def get_usage_by_service(self) -> pd.Series:
        """Calculate total usage grouped by ServiceName."""
        if self.df is None or 'ServiceName' not in self.df.columns:
            return pd.Series(dtype=float)
        return self.df.groupby('ServiceName')['Quantity'].sum().sort_values(ascending=False)
    
    def get_usage_by_resource(self) -> pd.Series:
        """Calculate total usage grouped by ServiceResource."""
        if self.df is None or 'ServiceResource' not in self.df.columns:
            return pd.Series(dtype=float)
        return self.df.groupby('ServiceResource')['Quantity'].sum().sort_values(ascending=False)
    
    def get_usage_by_service_type(self) -> pd.Series:
        """Calculate total usage grouped by ServiceType."""
        if self.df is None or 'ServiceType' not in self.df.columns:
            return pd.Series(dtype=float)
        return self.df.groupby('ServiceType')['Quantity'].sum().sort_values(ascending=False)
    
    def get_usage_by_region(self) -> pd.Series:
        """Calculate total usage grouped by ServiceRegion."""
        if self.df is None or 'ServiceRegion' not in self.df.columns:
            return pd.Series(dtype=float)
        return self.df.groupby('ServiceRegion')['Quantity'].sum().sort_values(ascending=False)
    
    def get_service_efficiency_metrics(self) -> pd.DataFrame:
        """Calculate efficiency metrics (cost per unit) by service."""
        if self.df is None or self.df.empty:
            return pd.DataFrame()
        
        # Group by ServiceName
        service_metrics = self.df.groupby('ServiceName').agg({
            'Cost': 'sum',
            'Quantity': 'sum'
        }).round(4)
        
        # Filter out services with zero quantity
        service_metrics = service_metrics[service_metrics['Quantity'] > 0]
        
        if service_metrics.empty:
            return pd.DataFrame()
        
        # Calculate efficiency score (cost per unit)
        service_metrics['EfficiencyScore'] = service_metrics['Cost'] / service_metrics['Quantity']
        service_metrics = service_metrics.sort_values('Cost', ascending=False)
        
        return service_metrics.reset_index()
    
    def get_resource_efficiency_metrics(self) -> pd.DataFrame:
        """Calculate efficiency metrics (cost per unit) by resource."""
        if self.df is None or self.df.empty:
            return pd.DataFrame()
        
        # Group by ServiceResource
        resource_metrics = self.df.groupby('ServiceResource').agg({
            'Cost': 'sum',
            'Quantity': 'sum'
        }).round(4)
        
        # Filter out resources with zero quantity
        resource_metrics = resource_metrics[resource_metrics['Quantity'] > 0]
        
        if resource_metrics.empty:
            return pd.DataFrame()
        
        # Calculate efficiency score (cost per unit)
        resource_metrics['EfficiencyScore'] = resource_metrics['Cost'] / resource_metrics['Quantity']
        resource_metrics = resource_metrics.sort_values('Cost', ascending=False)
        
        return resource_metrics.reset_index()
    
    def get_service_type_breakdown(self) -> pd.DataFrame:
        """Get detailed breakdown by service type with regions."""
        if self.df is None or self.df.empty:
            return pd.DataFrame()
        
        breakdown = self.df.groupby(['ServiceName', 'ServiceType', 'ServiceRegion']).agg({
            'Cost': 'sum',
            'Quantity': 'sum'
        }).round(4).reset_index()
        
        # Calculate percentages
        total_cost = breakdown['Cost'].sum()
        breakdown['Cost_Percentage'] = (breakdown['Cost'] / total_cost * 100).round(2) if total_cost > 0 else 0
        
        return breakdown.sort_values('Cost', ascending=False)
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the simple dashboard."""
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
            'unique_services': self.df['ServiceName'].nunique() if 'ServiceName' in self.df.columns else 0,
            'unique_service_types': self.df['ServiceType'].nunique() if 'ServiceType' in self.df.columns else 0,
            'unique_regions': self.df['ServiceRegion'].nunique() if 'ServiceRegion' in self.df.columns else 0,
            'unique_resources': self.df['ServiceResource'].nunique() if 'ServiceResource' in self.df.columns else 0,
            'unique_subscriptions': self.df['SubscriptionName'].nunique() if 'SubscriptionName' in self.df.columns else 0
        }


class SimpleChartCreator:
    """
    Simple chart creator for basic service usage analysis.
    
    This class ensures all charts use consistent data sources and calculation methods.
    
    Consistency Standards:
    - All charts receive pre-calculated data from SimpleInvoiceData methods
    - No direct DataFrame aggregation in chart methods
    - Same category calculations produce identical results across all chart sections
    - All methods use the standardized calculation formulas defined in SimpleInvoiceData
    """
    
    def __init__(self):
        pass
    
    def format_label(self, label: str, max_length: int=40) -> str:
        """Format label to specified length."""
        if len(label) <= max_length:
            return label.ljust(max_length)
        return label[:max_length - 3] + "..."
    
    def create_cost_by_service_chart(self, cost_data: pd.Series, top_items: int=None) -> go.Figure:
        """Create bar chart for cost by service."""
        if cost_data.empty:
            return go.Figure()
        
        # Use configuration value if not specified
        if top_items is None:
            from streamlit_app import Config
            top_items = Config.TOP_ITEMS_COUNT
            
        top_services = cost_data.head(top_items)
        formatted_labels = [self.format_label(str(label)) for label in top_services.index]
        
        fig = go.Figure(data=[
            go.Bar(
                x=formatted_labels,
                y=top_services.values,
                text=[f'${value:,.2f}' for value in top_services.values],
                textposition='outside',
                marker_color='lightblue',
                hovertemplate='<b>%{x}</b><br>Cost: $%{y:,.2f}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title={
                'text': '🔧 Total Cost by Service',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            xaxis_title='Service Name',
            yaxis_title='Cost (USD)',
            height=500,
            showlegend=False,
            xaxis={'tickangle': 45},
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def create_cost_by_region_chart(self, cost_data: pd.Series) -> go.Figure:
        """Create pie chart for cost by region."""
        if cost_data.empty:
            return go.Figure()
        
        fig = go.Figure(data=[go.Pie(
            labels=cost_data.index,
            values=cost_data.values,
            hole=0.4,
            textinfo='label+percent+value',
            texttemplate='<b>%{label}</b><br>%{percent}<br>$%{value:,.2f}',
            hovertemplate='<b>%{label}</b><br>Cost: $%{value:,.2f}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title={
                'text': '🌍 Cost Distribution by Region',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            height=500,
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def create_cost_by_resource_chart(self, cost_data: pd.Series, top_items: int=None) -> go.Figure:
        """Create horizontal bar chart for cost by resource."""
        if cost_data.empty:
            return go.Figure()
        
        # Use configuration value if not specified
        if top_items is None:
            from streamlit_app import Config
            top_items = Config.TOP_ITEMS_COUNT
            
        top_resources = cost_data.head(top_items)
        
        fig = go.Figure(data=[go.Bar(
            y=top_resources.index,
            x=top_resources.values,
            orientation='h',
            text=[f'${value:,.2f}' for value in top_resources.values],
            textposition='outside',
            marker_color='lightcoral',
            hovertemplate='<b>%{y}</b><br>Cost: $%{x:,.2f}<extra></extra>'
        )])
        
        fig.update_layout(
            title={
                'text': '💻 Top Resources by Cost',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            xaxis_title='Cost (USD)',
            yaxis_title='Service Resource',
            height=max(400, len(top_resources) * 40),
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=150, r=50, t=80, b=50)
        )
        
        return fig
    
    def create_usage_vs_cost_chart(self, cost_by_service: pd.Series, usage_by_service: pd.Series) -> go.Figure:
        """Create scatter plot showing usage vs cost by service using consistent calculation methods."""
        if cost_by_service.empty or usage_by_service.empty:
            return go.Figure()
        
        # Use the same data sources as other service charts for consistency
        # Align usage data to match cost data ordering
        aligned_usage = usage_by_service.reindex(cost_by_service.index).fillna(0)
        
        # Create DataFrame for plotting
        service_summary = pd.DataFrame({
            'ServiceName': cost_by_service.index,
            'Cost': cost_by_service.values,
            'Quantity': aligned_usage.values
        })
        
        # Remove services with zero values to avoid clutter
        service_summary = service_summary[(service_summary['Cost'] > 0) & (service_summary['Quantity'] > 0)]
        
        if service_summary.empty:
            return go.Figure()
        
        fig = go.Figure(data=go.Scatter(
            x=service_summary['Quantity'],
            y=service_summary['Cost'],
            mode='markers+text',
            text=service_summary['ServiceName'],
            textposition='top center',
            marker=dict(
                size=12,
                color=service_summary['Cost'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Cost (USD)")
            ),
            hovertemplate='<b>%{text}</b><br>Usage: %{x:,.0f}<br>Cost: $%{y:,.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': '📊 Service Usage vs Cost Analysis',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            xaxis_title='Total Usage (Quantity)',
            yaxis_title='Total Cost (USD)',
            height=500,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def create_service_efficiency_chart(self, efficiency_data: pd.DataFrame) -> go.Figure:
        """Create efficiency chart for services using consistent efficiency calculation."""
        if efficiency_data.empty:
            return go.Figure()
        
        top_services = efficiency_data.head(15)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Total Cost by Service", "Cost per Unit"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Cost bar chart - uses same data as cost_by_service calculations
        fig.add_trace(
            go.Bar(
                x=top_services['ServiceName'],
                y=top_services['Cost'],
                name='Total Cost',
                marker_color='lightblue',
                text=[f'${cost:,.0f}' for cost in top_services['Cost']],
                textposition='outside',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Efficiency line chart - uses same calculation as efficiency metrics
        fig.add_trace(
            go.Scatter(
                x=top_services['ServiceName'],
                y=top_services['EfficiencyScore'],
                mode='lines+markers+text',
                name='Cost per Unit',
                line=dict(color='red', width=3),
                marker=dict(size=8, color='red'),
                text=[f'${score:.3f}' for score in top_services['EfficiencyScore']],
                textposition='top center',
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Service", tickangle=45, row=1, col=1)
        fig.update_xaxes(title_text="Service", tickangle=45, row=1, col=2)
        fig.update_yaxes(title_text="Cost (USD)", row=1, col=1)
        fig.update_yaxes(title_text="Cost per Unit (USD)", row=1, col=2)
        
        fig.update_layout(
            title={
                'text': '⚡ Service Efficiency Analysis',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            height=600,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig 


class SimpleDashboard:
    """Simple Azure invoice dashboard with basic analysis features."""
    
    def __init__(self):
        self.chart_creator = SimpleChartCreator()
        # Configuration Constants - Fixed values for consistent display
        self.MAX_LABEL_LENGTH = 40
        self.TOP_ITEMS_COUNT = 12  # Show top 12 items in charts
        self.CHART_HEIGHT = 500

    def display_simple_summary(self, data: SimpleInvoiceData):
        """Display simple data summary."""
        summary = data.get_data_summary()
        
        if not summary:
            return
        
        st.header("📈 Simple Invoice Summary")
        
        # Main metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Total Cost",
                f"${summary['total_cost']:,.2f}",
                help="Total cost across all services"
            )
        
        with col2:
            st.metric(
                "Total Usage",
                f"{summary['total_quantity']:,.0f} units",
                help="Total usage across all services"
            )
        
        with col3:
            st.metric(
                "Services",
                summary['unique_services'],
                help="Number of unique services"
            )
        
        with col4:
            st.metric(
                "Regions",
                summary['unique_regions'],
                help="Number of unique regions"
            )
        
        with col5:
            st.metric(
                "Resources",
                summary['unique_resources'],
                help="Number of unique resources"
            )
    
    def display_simple_service_analysis(self, data: SimpleInvoiceData):
        """Display simple service analysis charts."""
        st.header("🔧 Service Analysis")
        
        # Get service data
        cost_by_service = data.get_cost_by_service()
        cost_by_region = data.get_cost_by_region()
        
        if cost_by_service.empty and cost_by_region.empty:
            st.warning("No service data available for analysis.")
            return
        
        # Service cost chart
        if not cost_by_service.empty:
            fig1 = self.chart_creator.create_cost_by_service_chart(cost_by_service, self.TOP_ITEMS_COUNT)
            st.plotly_chart(fig1, use_container_width=True)
        
        # Region cost chart
        if not cost_by_region.empty:
            fig2 = self.chart_creator.create_cost_by_region_chart(cost_by_region)
            st.plotly_chart(fig2, use_container_width=True)
    
    def display_simple_resource_analysis(self, data: SimpleInvoiceData):
        """Display simple resource analysis."""
        st.header("💻 Resource Analysis")
        
        cost_by_resource = data.get_cost_by_resource()
        
        if cost_by_resource.empty:
            st.warning("No resource data available for analysis.")
            return
        
        # Resource cost chart
        fig = self.chart_creator.create_cost_by_resource_chart(cost_by_resource, self.TOP_ITEMS_COUNT)
        st.plotly_chart(fig, use_container_width=True)
    
    def display_simple_efficiency_analysis(self, data: SimpleInvoiceData):
        """Display simple efficiency analysis."""
        st.header("⚡ Efficiency Analysis")
        
        # Usage vs Cost scatter plot - using consistent calculation methods
        cost_by_service = data.get_cost_by_service()
        usage_by_service = data.get_usage_by_service()
        fig1 = self.chart_creator.create_usage_vs_cost_chart(cost_by_service, usage_by_service)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Service efficiency metrics
        service_efficiency = data.get_service_efficiency_metrics()
        
        if not service_efficiency.empty:
            fig2 = self.chart_creator.create_service_efficiency_chart(service_efficiency)
            st.plotly_chart(fig2, use_container_width=True)
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            
            total_cost = service_efficiency['Cost'].sum()
            total_usage = service_efficiency['Quantity'].sum()
            avg_efficiency = service_efficiency['EfficiencyScore'].mean()
            
            with col1:
                st.metric("Analyzed Cost", f"${total_cost:,.2f}")
            with col2:
                st.metric("Analyzed Usage", f"{total_usage:,.0f} units")
            with col3:
                st.metric("Avg Cost/Unit", f"${avg_efficiency:.4f}")
    
    def display_simple_detailed_tables(self, data: SimpleInvoiceData):
        """Display simple detailed data tables."""
        st.header("📊 Detailed Data Tables")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔧 Services",
            "🌍 Regions",
            "💻 Resources",
            "📋 Service Breakdown"
        ])
        
        with tab1:
            cost_by_service = data.get_cost_by_service()
            usage_by_service = data.get_usage_by_service()
            
            if not cost_by_service.empty and not usage_by_service.empty:
                service_df = pd.DataFrame({
                    'Service': cost_by_service.index,
                    'Total Cost ($)': cost_by_service.values.round(2),
                    'Total Usage': usage_by_service.reindex(cost_by_service.index).values.round(2)
                })
                st.dataframe(service_df, use_container_width=True, hide_index=True)
            else:
                st.info("No service data available.")
        
        with tab2:
            cost_by_region = data.get_cost_by_region()
            usage_by_region = data.get_usage_by_region()
            
            if not cost_by_region.empty:
                region_df = pd.DataFrame({
                    'Region': cost_by_region.index,
                    'Total Cost ($)': cost_by_region.values.round(2)
                })
                if not usage_by_region.empty:
                    region_df['Total Usage'] = usage_by_region.reindex(cost_by_region.index).values.round(2)
                
                st.dataframe(region_df, use_container_width=True, hide_index=True)
            else:
                st.info("No region data available.")
        
        with tab3:
            cost_by_resource = data.get_cost_by_resource()
            usage_by_resource = data.get_usage_by_resource()
            
            if not cost_by_resource.empty:
                resource_df = pd.DataFrame({
                    'Resource': cost_by_resource.index,
                    'Total Cost ($)': cost_by_resource.values.round(2)
                })
                if not usage_by_resource.empty:
                    resource_df['Total Usage'] = usage_by_resource.reindex(cost_by_resource.index).values.round(2)
                
                st.dataframe(resource_df.head(20), use_container_width=True, hide_index=True)  # Show top 20
            else:
                st.info("No resource data available.")
        
        with tab4:
            service_breakdown = data.get_service_type_breakdown()
            if not service_breakdown.empty:
                display_df = service_breakdown.copy()
                display_df['Cost'] = display_df['Cost'].apply(lambda x: f"${x:,.2f}")
                display_df['Quantity'] = display_df['Quantity'].apply(lambda x: f"{x:,.2f}")
                display_df['Cost_Percentage'] = display_df['Cost_Percentage'].apply(lambda x: f"{x:.1f}%")
                
                display_df.columns = ['Service', 'Type', 'Region', 'Cost', 'Usage', 'Cost %']
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.info("No service breakdown data available.")

    def run_simple_analysis(self, data: SimpleInvoiceData):
        """Run the complete simple analysis dashboard."""
        st.info("📊 **Simple Template Active** - Basic service usage analysis")

        with st.container():
            # Simple summary
            self.display_simple_summary(data)
            st.divider()

            # Service analysis
            self.display_simple_service_analysis(data)
            st.divider()

            # Resource analysis
            self.display_simple_resource_analysis(data)
            st.divider()

            # Efficiency analysis
            self.display_simple_efficiency_analysis(data)
            st.divider()

            # Detailed tables
            self.display_simple_detailed_tables(data)

            st.success("✅ Simple analysis complete! Service costs, regional distribution, and efficiency metrics calculated.")
            
            # Calculation consistency documentation
            with st.expander("📊 **Calculation Consistency Guarantee**", expanded=False):
                st.markdown("""
                **🎯 Consistent Calculations Across All Sections:**
                
                This simple template ensures that **all calculations use identical formulas** across every chart and table section:
                
                **📈 Standard Calculation Methods:**
                - **Cost by Service**: `df.groupby('ServiceName')['Cost'].sum().sort_values(ascending=False)`
                - **Cost by Region**: `df.groupby('ServiceRegion')['Cost'].sum().sort_values(ascending=False)`
                - **Cost by Resource**: `df.groupby('ServiceResource')['Cost'].sum().sort_values(ascending=False)`
                - **Usage by Service**: `df.groupby('ServiceName')['Quantity'].sum().sort_values(ascending=False)`
                - **Usage by Region**: `df.groupby('ServiceRegion')['Quantity'].sum().sort_values(ascending=False)`
                - **Usage by Resource**: `df.groupby('ServiceResource')['Quantity'].sum().sort_values(ascending=False)`
                - **Efficiency Score**: `Total_Cost ÷ Total_Quantity` per category
                
                **🔍 Consistency Features:**
                - ✅ **Same Service** appears with **identical cost values** in all charts and tables
                - ✅ **Same Region** shows **identical cost totals** across all sections
                - ✅ **Same Resource** displays **consistent values** everywhere
                - ✅ **Efficiency calculations** use the exact same cost and usage data
                - ✅ **No discrepancies** between chart data and table data
                
                **📊 How We Ensure Consistency:**
                1. **Centralized Calculation Methods**: All data comes from standardized methods in `SimpleInvoiceData`
                2. **No Duplicate Aggregations**: Charts receive pre-calculated data, never aggregate directly
                3. **Unified Data Sources**: All sections use the same calculation functions
                4. **Consistent Sorting**: All results sorted by cost/usage descending for predictable ordering
                
                **🎯 Benefits:**
                - **Reliable Analysis**: Same category always shows same results
                - **Trustworthy Data**: No calculation inconsistencies to confuse analysis
                - **Clear Comparisons**: Easy to cross-reference values between different views
                - **Audit Trail**: Single source of truth for all calculations
                """) 
 
