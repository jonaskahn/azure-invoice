from typing import Optional, Dict, Any
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


class SimpleInvoiceData:
    """Simple Azure invoice data operations for basic service usage analysis."""
    
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
    """Simple chart creator for basic service usage analysis."""
    
    def __init__(self):
        pass
    
    def format_label(self, label: str, max_length: int=40) -> str:
        """Format label to specified length."""
        if len(label) <= max_length:
            return label.ljust(max_length)
        return label[:max_length - 3] + "..."
    
    def create_cost_by_service_chart(self, cost_data: pd.Series, top_items: int=10) -> go.Figure:
        """Create bar chart for cost by service."""
        if cost_data.empty:
            return go.Figure()
        
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
    
    def create_cost_by_resource_chart(self, cost_data: pd.Series, top_items: int=10) -> go.Figure:
        """Create horizontal bar chart for cost by resource."""
        if cost_data.empty:
            return go.Figure()
        
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
    
    def create_usage_vs_cost_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create scatter plot showing usage vs cost by service."""
        if df is None or df.empty:
            return go.Figure()
        
        # Aggregate by service
        service_summary = df.groupby('ServiceName').agg({
            'Cost': 'sum',
            'Quantity': 'sum'
        }).reset_index()
        
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
        """Create efficiency chart for services."""
        if efficiency_data.empty:
            return go.Figure()
        
        top_services = efficiency_data.head(15)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Total Cost by Service", "Cost per Unit"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Cost bar chart
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
        
        # Efficiency line chart
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
