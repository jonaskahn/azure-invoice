import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import base64
import numpy as np
from datetime import datetime, timedelta


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
    CHART_THEME = "streamlit"

    # Cost category colors
    CATEGORY_COLORS = {
        'Managed Disks': '#FF6B6B',
        'CDN': '#4ECDC4',
        'Network/IP': '#45B7D1',
        'Backup': '#96CEB4',
        'Load Balancer': '#FECA57',
        'VM Compute': '#FF9FF3',
        'Other Storage': '#54A0FF',
        'Bandwidth': '#5F27CD',
        'Key Vault': '#00D2D3',
        'Other': '#C7ECEE'
    }


class CostCategoryAnalyzer:
    """Analyzes and classifies Azure costs into business categories."""

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        self.category_definitions = self._define_cost_categories()

    def _define_cost_categories(self) -> Dict[str, Dict]:
        """Define cost category classification rules based on strategy document."""
        return {
            'Managed Disks': {
                'filter_type': 'meter_subcategory',
                'values': [
                    'Premium SSD Managed Disks',
                    'Standard HDD Managed Disks',
                    'Standard SSD Managed Disks',
                    'Ultra SSD Managed Disks'
                ],
                'description': 'Azure managed disk storage costs'
            },
            'CDN': {
                'filter_type': 'meter_category',
                'values': ['Content Delivery Network'],
                'description': 'Content Delivery Network costs'
            },
            'Network/IP': {
                'filter_type': 'meter_category',
                'values': ['Virtual Network'],
                'description': 'Virtual network and IP address costs'
            },
            'Backup': {
                'filter_type': 'consumed_service',
                'values': ['Microsoft.RecoveryServices'],
                'description': 'Backup and recovery services'
            },
            'Load Balancer': {
                'filter_type': 'meter_category',
                'values': ['Load Balancer'],
                'description': 'Load balancer costs'
            },
            'VM Compute': {
                'filter_type': 'complex',
                'logic': 'microsoft_compute_minus_disks',
                'description': 'Virtual machine compute costs (excluding disks)'
            },
            'Other Storage': {
                'filter_type': 'complex',
                'logic': 'storage_minus_managed_disks',
                'description': 'Storage costs excluding managed disks'
            },
            'Bandwidth': {
                'filter_type': 'meter_category',
                'values': ['Bandwidth'],
                'description': 'Data transfer and bandwidth costs'
            },
            'Key Vault': {
                'filter_type': 'consumed_service',
                'values': ['Microsoft.KeyVault'],
                'description': 'Key Vault service costs'
            }
        }

    def classify_costs(self) -> pd.DataFrame:
        """Classify each row into cost categories."""
        if self.df is None or self.df.empty:
            return pd.DataFrame()

        df_classified = self.df.copy()
        df_classified['CostCategory'] = 'Other'  # Default category

        # Apply classification rules
        for category, rules in self.category_definitions.items():
            if rules['filter_type'] == 'meter_subcategory':
                mask = df_classified['MeterSubcategory'].isin(rules['values'])
            elif rules['filter_type'] == 'meter_category':
                mask = df_classified['MeterCategory'].isin(rules['values'])
            elif rules['filter_type'] == 'consumed_service':
                mask = df_classified['ConsumedService'].isin(rules['values'])
            elif rules['filter_type'] == 'complex':
                mask = self._apply_complex_logic(df_classified, rules['logic'])
            else:
                continue

            df_classified.loc[mask, 'CostCategory'] = category

        return df_classified

    def _apply_complex_logic(self, df: pd.DataFrame, logic: str) -> pd.Series:
        """Apply complex classification logic."""
        if logic == 'microsoft_compute_minus_disks':
            # VM compute costs: Microsoft.Compute service minus managed disk subcategories
            compute_mask = df['ConsumedService'] == 'Microsoft.Compute'
            disk_subcategories = self.category_definitions['Managed Disks']['values']
            not_disk_mask = ~df['MeterSubcategory'].isin(disk_subcategories)
            return compute_mask & not_disk_mask

        elif logic == 'storage_minus_managed_disks':
            # Other storage: Storage category minus managed disks
            storage_mask = df['MeterCategory'] == 'Storage'
            disk_subcategories = self.category_definitions['Managed Disks']['values']
            not_disk_mask = ~df['MeterSubcategory'].isin(disk_subcategories)
            return storage_mask & not_disk_mask

        return pd.Series([False] * len(df))

    def get_category_summary(self) -> pd.DataFrame:
        """Get summary statistics by cost category."""
        df_classified = self.classify_costs()

        summary = df_classified.groupby('CostCategory').agg({
            'Cost': ['sum', 'count', 'mean'],
            'Quantity': 'sum'
        }).round(4)

        # Flatten column names
        summary.columns = ['Total_Cost', 'Record_Count', 'Avg_Cost', 'Total_Quantity']
        summary = summary.reset_index()

        # Calculate percentages
        total_cost = summary['Total_Cost'].sum()
        summary['Cost_Percentage'] = (summary['Total_Cost'] / total_cost * 100).round(2)

        # Sort by cost descending
        summary = summary.sort_values('Total_Cost', ascending=False)

        return summary

    def get_service_provider_analysis(self) -> pd.DataFrame:
        """Analyze costs by Azure service provider."""
        if self.df is None or self.df.empty:
            return pd.DataFrame()

        provider_summary = self.df.groupby('ConsumedService').agg({
            'Cost': ['sum', 'count'],
            'Quantity': 'sum'
        }).round(4)

        provider_summary.columns = ['Total_Cost', 'Record_Count', 'Total_Quantity']
        provider_summary = provider_summary.reset_index()

        # Calculate percentages
        total_cost = provider_summary['Total_Cost'].sum()
        provider_summary['Cost_Percentage'] = (provider_summary['Total_Cost'] / total_cost * 100).round(2)

        return provider_summary.sort_values('Total_Cost', ascending=False)

    def validate_cost_reconciliation(self) -> Dict[str, Any]:
        """Validate that all costs are properly categorized."""
        df_classified = self.classify_costs()

        original_total = self.df['Cost'].sum()
        categorized_total = df_classified['Cost'].sum()
        difference = abs(original_total - categorized_total)

        # Category coverage
        category_totals = df_classified.groupby('CostCategory')['Cost'].sum()
        uncategorized_cost = category_totals.get('Other', 0)
        categorized_cost = categorized_total - uncategorized_cost

        return {
            'original_total': original_total,
            'categorized_total': categorized_total,
            'difference': difference,
            'reconciliation_success': difference < 0.01,
            'categorized_cost': categorized_cost,
            'uncategorized_cost': uncategorized_cost,
            'categorization_coverage': (categorized_cost / original_total * 100) if original_total > 0 else 0,
            'category_breakdown': category_totals.to_dict()
        }


class AzureInvoiceData:
    """Enhanced Azure invoice data operations with cost category analysis."""

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        self._clean_data()
        self.cost_analyzer = CostCategoryAnalyzer(self.df) if self.df is not None else None

    def _clean_data(self) -> None:
        """Clean and prepare data for analysis."""
        if self.df is None:
            return

        # Ensure numeric conversion for Cost and Quantity
        self.df['Cost'] = pd.to_numeric(self.df.get('Cost', 0), errors='coerce').fillna(0)
        self.df['Quantity'] = pd.to_numeric(self.df.get('Quantity', 0), errors='coerce').fillna(0)

        # Ensure required columns exist
        required_columns = ['ConsumedService', 'MeterCategory', 'MeterSubcategory']
        for col in required_columns:
            if col not in self.df.columns:
                self.df[col] = 'Unknown'
                st.warning(f"Missing column '{col}' - created with default values")

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

    def get_efficiency_metrics(self) -> pd.DataFrame:
        """Calculate efficiency metrics (cost per unit)."""
        if self.df is None or self.df.empty:
            return pd.DataFrame()

        # Calculate cost per unit for each resource
        efficiency_df = self.df.copy()
        efficiency_df['CostPerUnit'] = efficiency_df['Cost'] / efficiency_df['Quantity'].replace(0, np.nan)

        # Group by resource and calculate efficiency metrics
        efficiency_summary = efficiency_df.groupby('ResourceName').agg({
            'Cost': 'sum',
            'Quantity': 'sum',
            'CostPerUnit': 'mean'
        }).round(4)

        efficiency_summary = efficiency_summary[efficiency_summary['Quantity'] > 0]
        efficiency_summary['EfficiencyScore'] = efficiency_summary['Cost'] / efficiency_summary['Quantity']

        return efficiency_summary.sort_values('Cost', ascending=False)

    def get_data_summary(self) -> Dict[str, Any]:
        """Get enhanced summary statistics for the dashboard."""
        if self.df is None or self.df.empty:
            return {}

        # Basic metrics
        basic_summary = {
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

        # Enhanced metrics
        if self.cost_analyzer:
            validation = self.cost_analyzer.validate_cost_reconciliation()
            category_summary = self.cost_analyzer.get_category_summary()

            basic_summary.update({
                'cost_validation': validation,
                'top_cost_category': category_summary.iloc[0][
                    'CostCategory'] if not category_summary.empty else 'Unknown',
                'top_category_cost': category_summary.iloc[0]['Total_Cost'] if not category_summary.empty else 0,
                'categorization_coverage': validation['categorization_coverage'],
                'unique_services': self.df['ConsumedService'].nunique(),
                'unique_meter_categories': self.df['MeterCategory'].nunique()
            })

        return basic_summary

    def get_machines_by_resource_group(self, resource_group: str) -> pd.DataFrame:
        """Get machines (ResourceName) and their costs for a specific resource group."""
        if self.df is None or self.df.empty:
            return pd.DataFrame()

        if 'ResourceGroup' not in self.df.columns or 'ResourceName' not in self.df.columns:
            return pd.DataFrame()

        # Filter out rows with NaN values in ResourceGroup or ResourceName
        clean_df = self.df.dropna(subset=['ResourceGroup', 'ResourceName'])
        rg_data = clean_df[clean_df['ResourceGroup'] == resource_group]

        if rg_data.empty:
            return pd.DataFrame()

        machine_summary = rg_data.groupby('ResourceName').agg({
            'Cost': 'sum',
            'Quantity': 'sum',
            'ConsumedService': lambda x: ', '.join(x.dropna().astype(str).unique()),
            'MeterCategory': lambda x: ', '.join(x.dropna().astype(str).unique())
        }).round(4).reset_index()

        # Calculate percentage within resource group
        total_rg_cost = machine_summary['Cost'].sum()
        machine_summary['Cost_Percentage'] = (machine_summary['Cost'] / total_rg_cost * 100).round(
            2) if total_rg_cost > 0 else 0

        machine_summary = machine_summary.sort_values('Cost', ascending=False)

        return machine_summary

    def get_machine_cost_breakdown(self, resource_name: str) -> pd.DataFrame:
        """Get cost breakdown by category for a specific machine."""
        if self.df is None or self.df.empty or not self.cost_analyzer:
            return pd.DataFrame()

        if 'ResourceName' not in self.df.columns:
            return pd.DataFrame()

        # Get classified data
        df_classified = self.cost_analyzer.classify_costs()

        # Filter out rows with NaN values in ResourceName
        df_classified = df_classified.dropna(subset=['ResourceName'])

        # Get exact matches first
        machine_data = df_classified[df_classified['ResourceName'] == resource_name]

        # Also look for related resources (disks, network interfaces, etc. that might be associated with this VM)
        # Common patterns: vm-name-disk, vm-name-nic, vm-name_OsDisk, etc.
        related_data = df_classified[
            (df_classified['ResourceName'].str.contains(resource_name, case=False, na=False)) |
            (df_classified['ResourceName'].str.startswith(resource_name + '-', na=False)) |
            (df_classified['ResourceName'].str.startswith(resource_name + '_', na=False))
            ]

        # Combine exact and related matches, removing duplicates
        combined_data = pd.concat([machine_data, related_data]).drop_duplicates()

        if combined_data.empty:
            return pd.DataFrame()

        # Group by cost category
        category_breakdown = combined_data.groupby('CostCategory').agg({
            'Cost': 'sum',
            'Quantity': 'sum',
            'ConsumedService': lambda x: ', '.join(x.dropna().astype(str).unique()),
            'MeterCategory': lambda x: ', '.join(x.dropna().astype(str).unique()),
            'MeterSubcategory': lambda x: ', '.join(x.dropna().astype(str).unique()),
            'ResourceName': lambda x: ', '.join(x.dropna().astype(str).unique())  # Show all related resources
        }).round(4).reset_index()

        # Calculate percentages
        total_machine_cost = category_breakdown['Cost'].sum()
        category_breakdown['Cost_Percentage'] = (category_breakdown['Cost'] / total_machine_cost * 100).round(
            2) if total_machine_cost > 0 else 0

        category_breakdown = category_breakdown.sort_values('Cost', ascending=False)

        return category_breakdown

    def get_machine_related_resources(self, resource_name: str) -> pd.DataFrame:
        """Get all resources related to a specific machine (including disks, NICs, etc.)."""
        if self.df is None or self.df.empty:
            return pd.DataFrame()

        if 'ResourceName' not in self.df.columns:
            return pd.DataFrame()

        # Look for all resources that might be related to this machine
        related_mask = (
                (self.df['ResourceName'] == resource_name) |
                (self.df['ResourceName'].str.contains(resource_name, case=False, na=False)) |
                (self.df['ResourceName'].str.startswith(resource_name + '-', na=False)) |
                (self.df['ResourceName'].str.startswith(resource_name + '_', na=False))
        )

        related_resources = self.df[related_mask]

        if related_resources.empty:
            return pd.DataFrame()

        # Group by actual resource name to show what's included
        resource_summary = related_resources.groupby('ResourceName').agg({
            'Cost': 'sum',
            'Quantity': 'sum',
            'ConsumedService': lambda x: ', '.join(x.dropna().astype(str).unique()),
            'MeterCategory': lambda x: ', '.join(x.dropna().astype(str).unique()),
            'MeterSubcategory': lambda x: ', '.join(x.dropna().astype(str).unique())
        }).round(4).reset_index()

        resource_summary = resource_summary.sort_values('Cost', ascending=False)

        return resource_summary

    def get_all_resource_groups(self) -> list:
        """Get list of all resource groups."""
        if self.df is None or self.df.empty or 'ResourceGroup' not in self.df.columns:
            return []

        # Filter out NaN values and convert to string, then get unique values
        resource_groups = self.df['ResourceGroup'].dropna().astype(str).unique().tolist()

        # Filter out empty strings and 'nan' strings, then sort
        resource_groups = [rg for rg in resource_groups if rg and rg.lower() != 'nan']

        return sorted(resource_groups)


class StreamlitChartCreator:
    """Enhanced chart creator with cost category visualizations."""

    def __init__(self):
        self.theme = Config.CHART_THEME

    def format_label(self, label: str, max_length: int = Config.MAX_LABEL_LENGTH) -> str:
        """Format label to specified length, padding with spaces if needed."""
        if len(label) <= max_length:
            return label.ljust(max_length)
        return label[:max_length - 3] + "..."

    def create_cost_category_pie_chart(self, category_summary: pd.DataFrame) -> go.Figure:
        """Create pie chart showing cost breakdown by category."""
        if category_summary.empty:
            return go.Figure()

        # Get colors for categories
        colors = [Config.CATEGORY_COLORS.get(cat, '#CCCCCC') for cat in category_summary['CostCategory']]

        fig = go.Figure(data=[go.Pie(
            labels=category_summary['CostCategory'],
            values=category_summary['Total_Cost'],
            hole=0.4,
            marker_colors=colors,
            textinfo='label+percent+value',
            texttemplate='<b>%{label}</b><br>%{percent}<br>$%{value:,.2f}',
            hovertemplate='<b>%{label}</b><br>Cost: $%{value:,.2f}<br>Percentage: %{percent}<extra></extra>'
        )])

        fig.update_layout(
            title={
                'text': '💰 Cost Breakdown by Category',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            height=Config.CHART_HEIGHT,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.05
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        return fig

    def create_cost_category_bar_chart(self, category_summary: pd.DataFrame) -> go.Figure:
        """Create horizontal bar chart for cost categories."""
        if category_summary.empty:
            return go.Figure()

        # Get colors for categories
        colors = [Config.CATEGORY_COLORS.get(cat, '#CCCCCC') for cat in category_summary['CostCategory']]

        fig = go.Figure(data=[go.Bar(
            y=category_summary['CostCategory'],
            x=category_summary['Total_Cost'],
            orientation='h',
            marker_color=colors,
            text=[f'${value:,.2f} ({pct:.1f}%)'
                  for value, pct in zip(category_summary['Total_Cost'], category_summary['Cost_Percentage'])],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Cost: $%{x:,.2f}<br>Records: %{customdata}<extra></extra>',
            customdata=category_summary['Record_Count']
        )])

        fig.update_layout(
            title={
                'text': '📊 Cost Categories (Detailed View)',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            xaxis_title='Cost (USD)',
            yaxis_title='Cost Category',
            height=max(400, len(category_summary) * 50),
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=150, r=50, t=80, b=50)
        )

        return fig

    def create_service_provider_chart(self, provider_summary: pd.DataFrame) -> go.Figure:
        """Create chart for service provider analysis."""
        if provider_summary.empty:
            return go.Figure()

        # Take top 10 providers
        top_providers = provider_summary.head(10)

        fig = go.Figure(data=[go.Bar(
            x=top_providers['ConsumedService'],
            y=top_providers['Total_Cost'],
            marker_color='lightblue',
            text=[f'${value:,.2f}' for value in top_providers['Total_Cost']],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Cost: $%{y:,.2f}<br>Records: %{customdata}<extra></extra>',
            customdata=top_providers['Record_Count']
        )])

        fig.update_layout(
            title={
                'text': '🏢 Cost by Azure Service Provider',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            xaxis_title='Service Provider',
            yaxis_title='Cost (USD)',
            height=Config.CHART_HEIGHT,
            showlegend=False,
            xaxis={'tickangle': Config.ROTATION_ANGLE},
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        return fig

    def create_efficiency_metrics_chart(self, efficiency_data: pd.DataFrame) -> go.Figure:
        """Create efficiency metrics visualization."""
        if efficiency_data.empty:
            return go.Figure()

        # Take top 15 resources by cost
        top_resources = efficiency_data.head(15)

        # Create dual-axis chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add cost bars
        fig.add_trace(
            go.Bar(
                x=top_resources.index,
                y=top_resources['Cost'],
                name='Total Cost',
                marker_color='lightcoral',
                yaxis='y',
                offsetgroup=1
            ),
            secondary_y=False,
        )

        # Add efficiency score line
        fig.add_trace(
            go.Scatter(
                x=top_resources.index,
                y=top_resources['EfficiencyScore'],
                mode='lines+markers',
                name='Cost per Unit',
                line=dict(color='blue', width=3),
                marker=dict(size=8)
            ),
            secondary_y=True,
        )

        fig.update_xaxes(title_text="Resource Name", tickangle=45)
        fig.update_yaxes(title_text="Total Cost (USD)", secondary_y=False)
        fig.update_yaxes(title_text="Cost per Unit", secondary_y=True)

        fig.update_layout(
            title={
                'text': '⚡ Resource Efficiency Analysis',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            height=Config.CHART_HEIGHT,
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        return fig

    def create_machine_cost_breakdown_chart(self, machine_breakdown: pd.DataFrame, machine_name: str) -> go.Figure:
        """Create cost breakdown chart for a specific machine showing all categories."""
        if machine_breakdown.empty:
            return go.Figure()

        # Get colors for categories
        colors = [Config.CATEGORY_COLORS.get(cat, '#CCCCCC') for cat in machine_breakdown['CostCategory']]

        # Create combined pie and bar chart
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "domain"}, {"type": "xy"}]],
            subplot_titles=["Cost Distribution", "Category Details"],
            column_widths=[0.4, 0.6]
        )

        # Add pie chart
        fig.add_trace(
            go.Pie(
                labels=machine_breakdown['CostCategory'],
                values=machine_breakdown['Cost'],
                hole=0.4,
                marker_colors=colors,
                textinfo='label+percent',
                textposition='outside',
                hovertemplate='<b>%{label}</b><br>Cost: $%{value:,.2f}<br>Percentage: %{percent}<extra></extra>',
                showlegend=False
            ),
            row=1, col=1
        )

        # Add horizontal bar chart
        fig.add_trace(
            go.Bar(
                y=machine_breakdown['CostCategory'],
                x=machine_breakdown['Cost'],
                orientation='h',
                marker_color=colors,
                text=[f'${value:,.2f}' for value in machine_breakdown['Cost']],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Cost: $%{x:,.2f}<br>Quantity: %{customdata}<extra></extra>',
                customdata=machine_breakdown['Quantity'],
                showlegend=False
            ),
            row=1, col=2
        )

        # Clean machine name for title
        clean_machine_name = machine_name[:50] + "..." if len(machine_name) > 50 else machine_name

        fig.update_layout(
            title={
                'text': f'🖥️ Cost Breakdown for: {clean_machine_name}',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            height=500,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        # Update subplot titles
        fig.update_annotations(font_size=14)

        return fig

    def create_machine_details_chart(self, machine_breakdown: pd.DataFrame, machine_name: str) -> go.Figure:
        """Create detailed breakdown chart showing services and meter categories for a machine."""
        if machine_breakdown.empty:
            return go.Figure()

        # Create stacked bar chart showing different service components
        fig = go.Figure()

        # Add bars for each category
        for i, row in machine_breakdown.iterrows():
            color = Config.CATEGORY_COLORS.get(row['CostCategory'], '#CCCCCC')

            fig.add_trace(go.Bar(
                x=[row['CostCategory']],
                y=[row['Cost']],
                name=row['CostCategory'],
                marker_color=color,
                text=f"${row['Cost']:,.2f}",
                textposition='auto',
                hovertemplate=f"""
                <b>{row['CostCategory']}</b><br>
                Cost: $%{{y:,.2f}}<br>
                Quantity: {row['Quantity']:,.2f}<br>
                Service: {row['ConsumedService']}<br>
                Meter: {row['MeterCategory']}<br>
                Subcategory: {row['MeterSubcategory'][:50]}...
                <extra></extra>"""
            ))

        clean_machine_name = machine_name[:40] + "..." if len(machine_name) > 40 else machine_name

        fig.update_layout(
            title={
                'text': f'🔍 Detailed Service Breakdown: {clean_machine_name}',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            xaxis_title='Cost Category',
            yaxis_title='Cost (USD)',
            height=450,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis={'tickangle': 45}
        )

        return fig

    def create_cost_by_resource_group_chart(self, cost_data: pd.Series) -> go.Figure:
        """Create interactive bar chart for total cost by resource group."""
        if cost_data.empty:
            return go.Figure()

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
                'text': '🏗️ Total Cost by Resource Group',
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
                'text': f'🖥️ Top {Config.TOP_ITEMS_COUNT} Machines by Total Cost',
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

        return fig

    def create_cost_usage_comparison_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create alternative dual-axis bar chart for cost vs usage comparison."""
        if df is None or df.empty:
            return go.Figure()

        agg_data = df.groupby('ResourceGroup').agg({
            'Cost': 'sum',
            'Quantity': 'sum'
        }).reset_index()

        if agg_data.empty:
            return go.Figure()

        agg_data = agg_data.sort_values('Cost', ascending=False)
        formatted_labels = [self.format_label(str(label)) for label in agg_data['ResourceGroup']]

        fig = make_subplots(specs=[[{"secondary_y": True}]])

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

        fig.update_xaxes(title_text="Resource Group", tickangle=Config.ROTATION_ANGLE)
        fig.update_yaxes(title_text="Cost (USD)", secondary_y=False)
        fig.update_yaxes(title_text="Usage (Hours)", secondary_y=True)

        fig.update_layout(
            title={
                'text': '📈 Cost and Usage Comparison by Resource Group',
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
    """Enhanced Streamlit dashboard with cost category analysis."""

    def __init__(self):
        self.chart_creator = StreamlitChartCreator()
        self.setup_page_config()

    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Azure Invoice Analyzer Pro",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Enhanced CSS for better styling
        st.markdown("""
        <style>
        /* Print styles for PDF export */
        @media print {
            body, html, .stApp {
                visibility: visible !important;
                background-color: white !important;
                color: black !important;
                font-size: 12px !important;
            }
            header[data-testid="stHeader"] { display: none !important; }
            div[data-testid="stSidebar"] { display: none !important; }
            div[data-testid="stToolbar"] { display: none !important; }
            footer { display: none !important; }
            .main, .block-container {
                display: block !important;
                visibility: visible !important;
                max-width: 100% !important;
                padding: 1rem !important;
                margin: 0 !important;
            }
            .js-plotly-plot, .plotly {
                display: block !important;
                visibility: visible !important;
                page-break-inside: avoid !important;
                margin-bottom: 1rem !important;
                background-color: white !important;
            }
        }

        /* Enhanced metric styling */
        div[data-testid="metric-container"] {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        /* Enhanced info boxes */
        .stInfo {
            background-color: #e7f3ff;
            border-left: 5px solid #1f77b4;
        }

        .stSuccess {
            background-color: #f0fff4;
            border-left: 5px solid #28a745;
        }

        .stWarning {
            background-color: #fffbf0;
            border-left: 5px solid #ffc107;
        }

        .stError {
            background-color: #fff5f5;
            border-left: 5px solid #dc3545;
        }
        </style>
        """, unsafe_allow_html=True)

    def display_header(self):
        """Display enhanced application header."""
        st.title("🔍 Azure Invoice Analyzer Pro")
        st.markdown("""
        **Advanced Azure billing analysis with cost categorization, efficiency metrics, and validation**

        Upload your Azure invoice CSV file to get detailed cost breakdowns, usage analytics, 
        service provider insights, and resource optimization recommendations.
        """)
        st.divider()

    def display_file_uploader(self) -> Optional[pd.DataFrame]:
        """Handle file upload and return processed DataFrame."""
        uploaded_file = st.file_uploader(
            "Choose your Azure Invoice CSV file",
            type=['csv'],
            help="Upload your Azure invoice CSV file with columns: Date, Cost, Quantity, ResourceGroup, ResourceName, ConsumedService, MeterCategory, MeterSubcategory"
        )

        if uploaded_file is not None:
            try:
                with st.spinner("Loading and processing your data..."):
                    df = pd.read_csv(uploaded_file, parse_dates=['Date'], low_memory=False)

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Records", len(df))
                    with col2:
                        st.metric("Columns", len(df.columns))
                    with col3:
                        file_size_mb = uploaded_file.size / (1024 * 1024)
                        st.metric("File Size", f"{file_size_mb:.2f} MB")
                    with col4:
                        total_cost = pd.to_numeric(df.get('Cost', 0), errors='coerce').sum()
                        st.metric("Total Cost", f"${total_cost:,.2f}")

                    return df

            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                st.info(
                    "Please ensure your CSV file has the correct format with columns: Date, Cost, Quantity, ResourceGroup, ResourceName, ConsumedService, MeterCategory, MeterSubcategory")
                return None

        return None

    def display_enhanced_summary(self, data: AzureInvoiceData):
        """Display enhanced data summary with validation."""
        summary = data.get_data_summary()

        if not summary:
            return

        st.header("📈 Executive Summary & Validation")

        # Detailed validation status
        if 'cost_validation' in summary:
            validation = summary['cost_validation']

            # Cost reconciliation overview
            col1, col2 = st.columns([3, 1])

            with col1:
                if validation['reconciliation_success']:
                    st.success(f"✅ **Cost Reconciliation:** ${validation['original_total']:,.2f} fully reconciled")
                else:
                    st.error(f"❌ **Cost Reconciliation:** Difference of ${validation['difference']:,.2f} found")

            with col2:
                coverage = validation['categorization_coverage']
                if coverage >= 95:
                    st.success(f"📊 **Coverage:** {coverage:.1f}%")
                elif coverage >= 80:
                    st.warning(f"📊 **Coverage:** {coverage:.1f}%")
                else:
                    st.error(f"📊 **Coverage:** {coverage:.1f}%")

            # Detailed reconciliation breakdown
            with st.expander("🔍 **Detailed Cost Reconciliation Breakdown**", expanded=False):
                st.markdown("### 💰 Cost Reconciliation Details")

                # Create reconciliation table
                reconciliation_data = {
                    'Metric': [
                        'Original Invoice Total',
                        'Sum of Categorized Costs',
                        'Difference (Original - Categorized)',
                        'Categorized Costs',
                        'Uncategorized Costs ("Other")',
                        'Reconciliation Status'
                    ],
                    'Amount ($)': [
                        f"${validation['original_total']:,.2f}",
                        f"${validation['categorized_total']:,.2f}",
                        f"${validation['difference']:,.2f}",
                        f"${validation['categorized_cost']:,.2f}",
                        f"${validation['uncategorized_cost']:,.2f}",
                        "✅ Success" if validation['reconciliation_success'] else "❌ Failed"
                    ],
                    'Percentage (%)': [
                        "100.0%",
                        f"{(validation['categorized_total'] / validation['original_total'] * 100):.2f}%" if validation[
                                                                                                                'original_total'] > 0 else "0%",
                        f"{(validation['difference'] / validation['original_total'] * 100):.2f}%" if validation[
                                                                                                         'original_total'] > 0 else "0%",
                        f"{validation['categorization_coverage']:.2f}%",
                        f"{(validation['uncategorized_cost'] / validation['original_total'] * 100):.2f}%" if validation[
                                                                                                                 'original_total'] > 0 else "0%",
                        f"{validation['categorization_coverage']:.1f}% Categorized"
                    ]
                }

                reconciliation_df = pd.DataFrame(reconciliation_data)
                st.dataframe(reconciliation_df, hide_index=True, use_container_width=True)

                # Category-by-category breakdown
                st.markdown("### 📊 Category-by-Category Breakdown")
                category_breakdown = validation['category_breakdown']

                breakdown_data = []
                for category, amount in sorted(category_breakdown.items(), key=lambda x: x[1], reverse=True):
                    percentage = (amount / validation['original_total'] * 100) if validation[
                                                                                      'original_total'] > 0 else 0
                    breakdown_data.append({
                        'Cost Category': category,
                        'Amount ($)': f"${amount:,.2f}",
                        'Percentage (%)': f"{percentage:.2f}%",
                        'Status': "🎯 Categorized" if category != 'Other' else "⚠️ Uncategorized"
                    })

                breakdown_df = pd.DataFrame(breakdown_data)
                st.dataframe(breakdown_df, hide_index=True, use_container_width=True)

                # Reconciliation insights
                st.markdown("### 💡 Reconciliation Insights")

                if validation['reconciliation_success']:
                    st.info("✅ **Perfect Reconciliation**: All invoice costs have been accounted for in the analysis.")
                else:
                    st.warning(
                        f"⚠️ **Reconciliation Gap**: ${validation['difference']:,.2f} difference detected. This may indicate:")
                    st.markdown("""
                    - Data processing errors
                    - Missing or corrupted records
                    - Unexpected data formats
                    - New Azure service types not yet categorized
                    """)

                if validation['uncategorized_cost'] > 0:
                    uncategorized_pct = (validation['uncategorized_cost'] / validation['original_total'] * 100)
                    if uncategorized_pct > 5:
                        st.warning(
                            f"🔍 **High Uncategorized Costs**: {uncategorized_pct:.1f}% (${validation['uncategorized_cost']:,.2f}) of costs are in 'Other' category.")
                        st.markdown("**Recommended Actions:**")
                        st.markdown("""
                        - Review 'Other' category items in detailed tables
                        - Check for new Azure service types
                        - Verify MeterCategory and MeterSubcategory values
                        - Update category classification rules if needed
                        """)
                    else:
                        st.success(
                            f"✅ **Low Uncategorized Costs**: Only {uncategorized_pct:.1f}% in 'Other' category - excellent categorization coverage!")

                # Mathematical verification
                st.markdown("### 🧮 Mathematical Verification")
                calculated_total = sum(category_breakdown.values())
                verification_success = abs(calculated_total - validation['original_total']) < 0.01

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original Total", f"${validation['original_total']:,.2f}")
                with col2:
                    st.metric("Calculated Total", f"${calculated_total:,.2f}")
                with col3:
                    difference = abs(calculated_total - validation['original_total'])
                    st.metric("Difference", f"${difference:,.2f}",
                              delta="✅ Verified" if verification_success else "❌ Error")

        # Main metrics
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "Total Cost",
                f"${summary['total_cost']:,.2f}",
                help="Total cost across all resources"
            )

        with col2:
            st.metric(
                "Total Usage",
                f"{summary['total_quantity']:,.0f} hrs",
                help="Total usage hours across all resources"
            )

        with col3:
            st.metric(
                "Top Category",
                summary.get('top_cost_category', 'Unknown'),
                f"${summary.get('top_category_cost', 0):,.2f}",
                help="Highest cost category"
            )

        with col4:
            st.metric(
                "Resource Groups",
                summary['unique_resource_groups'],
                help="Number of unique resource groups"
            )

        with col5:
            st.metric(
                "Services",
                summary.get('unique_services', 0),
                help="Number of unique Azure services"
            )

    def display_cost_category_analysis(self, data: AzureInvoiceData):
        """Display cost category analysis."""
        st.header("💰 Cost Category Analysis")

        if not data.cost_analyzer:
            st.error("Cost analyzer not available - missing required columns")
            return

        category_summary = data.cost_analyzer.get_category_summary()

        if category_summary.empty:
            st.warning("No cost category data available for analysis.")
            return

        # Display category summary table
        col1, col2 = st.columns([2, 1])

        with col1:
            # Pie chart
            fig_pie = self.chart_creator.create_cost_category_pie_chart(category_summary)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Category summary table
            st.markdown("**📋 Category Summary**")
            display_df = category_summary[['CostCategory', 'Total_Cost', 'Cost_Percentage']].copy()
            display_df.columns = ['Category', 'Cost ($)', 'Percentage (%)']
            display_df['Cost ($)'] = display_df['Cost ($)'].apply(lambda x: f"${x:,.2f}")
            st.dataframe(display_df, hide_index=True, use_container_width=True)

        # Horizontal bar chart for detailed view
        fig_bar = self.chart_creator.create_cost_category_bar_chart(category_summary)
        st.plotly_chart(fig_bar, use_container_width=True)

        # Cost insights
        top_category = category_summary.iloc[0]
        st.info(
            f"💡 **Key Insight:** {top_category['CostCategory']} represents {top_category['Cost_Percentage']:.1f}% of total costs (${top_category['Total_Cost']:,.2f})")

    def display_service_provider_analysis(self, data: AzureInvoiceData):
        """Display service provider analysis."""
        st.header("🏢 Service Provider Analysis")

        if not data.cost_analyzer:
            return

        provider_summary = data.cost_analyzer.get_service_provider_analysis()

        if provider_summary.empty:
            st.warning("No service provider data available.")
            return

        # Service provider chart
        fig = self.chart_creator.create_service_provider_chart(provider_summary)
        st.plotly_chart(fig, use_container_width=True)

        # Provider summary table
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("**📊 Service Provider Breakdown**")
            display_df = provider_summary.head(10)[
                ['ConsumedService', 'Total_Cost', 'Cost_Percentage', 'Record_Count']].copy()
            display_df.columns = ['Service Provider', 'Cost ($)', 'Percentage (%)', 'Records']
            display_df['Cost ($)'] = display_df['Cost ($)'].apply(lambda x: f"${x:,.2f}")
            st.dataframe(display_df, hide_index=True, use_container_width=True)

        with col2:
            # Top provider insights
            top_provider = provider_summary.iloc[0]
            st.info(f"**Top Provider:** {top_provider['ConsumedService']}")
            st.metric("Cost", f"${top_provider['Total_Cost']:,.2f}")
            st.metric("Share", f"{top_provider['Cost_Percentage']:.1f}%")

    def display_efficiency_analysis(self, data: AzureInvoiceData):
        """Display efficiency analysis."""
        st.header("⚡ Resource Efficiency Analysis")

        efficiency_data = data.get_efficiency_metrics()

        if efficiency_data.empty:
            st.warning("No efficiency data available (requires quantity > 0).")
            return

        # Efficiency chart
        fig = self.chart_creator.create_efficiency_metrics_chart(efficiency_data)
        st.plotly_chart(fig, use_container_width=True)

        # Efficiency insights
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🎯 Most Expensive Resources**")
            top_cost = efficiency_data.head(5)[['Cost', 'EfficiencyScore']].copy()
            top_cost['Cost'] = top_cost['Cost'].apply(lambda x: f"${x:,.2f}")
            top_cost['EfficiencyScore'] = top_cost['EfficiencyScore'].apply(lambda x: f"${x:,.4f}/unit")
            top_cost.columns = ['Total Cost', 'Cost per Unit']
            st.dataframe(top_cost, use_container_width=True)

        with col2:
            st.markdown("**💡 Efficiency Insights**")
            avg_efficiency = efficiency_data['EfficiencyScore'].mean()
            high_efficiency_resources = (efficiency_data['EfficiencyScore'] > avg_efficiency * 2).sum()

            st.metric("Average Cost/Unit", f"${avg_efficiency:.4f}")
            st.metric("High-Cost Resources", high_efficiency_resources)

            if high_efficiency_resources > 0:
                st.warning(f"⚠️ {high_efficiency_resources} resources have above-average cost per unit")

    def display_interactive_drill_down(self, data: AzureInvoiceData):
        """Display interactive drill-down: Resource Group -> Machines -> Cost Categories."""
        st.subheader("🔍 Interactive Drill-Down Analysis")
        st.markdown(
            "**Select a resource group to see its machines, then click on any machine to see its cost breakdown by category.**")

        # Get all resource groups with error handling
        try:
            resource_groups = data.get_all_resource_groups()
        except Exception as e:
            st.error(f"Error loading resource groups: {str(e)}")
            st.info("This might be due to missing or invalid data in the ResourceGroup column.")
            return

        if not resource_groups:
            st.warning("No resource groups found in the data.")
            st.info("Please ensure your CSV file has a 'ResourceGroup' column with valid data.")
            return

        # Resource Group Selection
        col1, col2 = st.columns([1, 3])

        with col1:
            selected_rg = st.selectbox(
                "🏗️ **Select Resource Group:**",
                resource_groups,
                help="Choose a resource group to see its machines and costs"
            )

        with col2:
            # Show resource group summary with error handling
            if selected_rg:
                try:
                    rg_data = data.df[data.df[
                                          'ResourceGroup'] == selected_rg] if 'ResourceGroup' in data.df.columns else pd.DataFrame()
                    if not rg_data.empty:
                        rg_cost = rg_data['Cost'].sum()
                        rg_machines = rg_data[
                            'ResourceName'].dropna().nunique() if 'ResourceName' in rg_data.columns else 0
                        rg_quantity = rg_data['Quantity'].sum()

                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Total Cost", f"${rg_cost:,.2f}")
                        with col_b:
                            st.metric("Machines", f"{rg_machines}")
                        with col_c:
                            st.metric("Total Usage", f"{rg_quantity:,.0f} hrs")
                except Exception as e:
                    st.warning(f"Error calculating resource group summary: {str(e)}")

        if not selected_rg:
            st.info("👆 Please select a resource group to begin drill-down analysis.")
            return

        # Get machines for selected resource group with error handling
        try:
            machines_data = data.get_machines_by_resource_group(selected_rg)
        except Exception as e:
            st.error(f"Error loading machines for resource group '{selected_rg}': {str(e)}")
            return

        if machines_data.empty:
            st.warning(f"No machines found in resource group: {selected_rg}")
            return

        # Display machines table with selection capability
        st.markdown(f"### 🖥️ Machines in Resource Group: **{selected_rg}**")

        # Format machines data for display
        try:
            display_machines = machines_data.copy()
            display_machines['Cost'] = display_machines['Cost'].apply(lambda x: f"${x:,.2f}")
            display_machines['Quantity'] = display_machines['Quantity'].apply(lambda x: f"{x:,.2f}")
            display_machines['Cost_Percentage'] = display_machines['Cost_Percentage'].apply(lambda x: f"{x:.1f}%")

            display_machines.columns = ['Machine Name', 'Total Cost', 'Total Usage', 'Services Used',
                                        'Meter Categories', 'Cost %']

            # Create clickable machine selection
            st.markdown("**Click on a machine name below to see its detailed cost breakdown:**")

            # Machine selection using radio buttons for better UX
            machine_options = machines_data['ResourceName'].tolist()

            # Create a more compact selection method
            selected_machine = st.selectbox(
                "🖥️ **Select Machine for Detailed Analysis:**",
                [""] + machine_options,
                format_func=lambda
                    x: "Choose a machine..." if x == "" else f"{x} (${machines_data[machines_data['ResourceName'] == x]['Cost'].iloc[0]:,.2f})" if x else x,
                help="Select a machine to see its cost breakdown by category"
            )

            # Display machines table
            st.dataframe(display_machines, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Error formatting machine data: {str(e)}")
            return

        # Machine cost breakdown analysis
        if selected_machine and selected_machine != "":
            st.markdown("---")

            # Get machine cost breakdown by category
            machine_breakdown = data.get_machine_cost_breakdown(selected_machine)

            if machine_breakdown.empty:
                st.warning(f"No cost breakdown available for machine: {selected_machine}")
                return

            # Display machine details header
            machine_cost = machine_breakdown['Cost'].sum()
            machine_quantity = machine_breakdown['Quantity'].sum()

            st.markdown(f"### 🎯 Detailed Analysis: **{selected_machine}**")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Cost", f"${machine_cost:,.2f}")
            with col2:
                st.metric("Total Usage", f"{machine_quantity:,.2f} hrs")
            with col3:
                st.metric("Categories", len(machine_breakdown))
            with col4:
                avg_cost_per_hour = machine_cost / machine_quantity if machine_quantity > 0 else 0
                st.metric("Cost/Hour", f"${avg_cost_per_hour:.4f}")

            # Create cost breakdown charts
            col1, col2 = st.columns(2)

            with col1:
                # Pie + Bar combination chart
                fig_breakdown = self.chart_creator.create_machine_cost_breakdown_chart(machine_breakdown,
                                                                                       selected_machine)
                st.plotly_chart(fig_breakdown, use_container_width=True)

            with col2:
                # Detailed service breakdown
                fig_details = self.chart_creator.create_machine_details_chart(machine_breakdown, selected_machine)
                st.plotly_chart(fig_details, use_container_width=True)

            # Category breakdown table
            st.markdown("#### 📊 Category Breakdown Table")

            # Format breakdown data for display
            display_breakdown = machine_breakdown.copy()
            display_breakdown['Cost'] = display_breakdown['Cost'].apply(lambda x: f"${x:,.2f}")
            display_breakdown['Quantity'] = display_breakdown['Quantity'].apply(lambda x: f"{x:,.2f}")
            display_breakdown['Cost_Percentage'] = display_breakdown['Cost_Percentage'].apply(lambda x: f"{x:.1f}%")

            display_breakdown = display_breakdown[
                ['CostCategory', 'Cost', 'Cost_Percentage', 'Quantity', 'ConsumedService', 'MeterCategory']]
            display_breakdown.columns = ['Cost Category', 'Cost', '% of Machine', 'Quantity', 'Service Provider',
                                         'Meter Category']

            st.dataframe(display_breakdown, use_container_width=True, hide_index=True)

            # Insights and recommendations
            st.markdown("#### 💡 Machine Cost Insights")

            # Generate insights based on breakdown
            top_category = machine_breakdown.iloc[0]
            insights = []

            # Cost concentration insight
            if top_category['Cost_Percentage'] > 80:
                insights.append(
                    f"🎯 **Highly Concentrated**: {top_category['Cost_Percentage']:.1f}% of costs come from {top_category['CostCategory']}")
            elif top_category['Cost_Percentage'] > 50:
                insights.append(
                    f"📊 **Moderately Concentrated**: {top_category['Cost_Percentage']:.1f}% of costs come from {top_category['CostCategory']}")
            else:
                insights.append(
                    f"📈 **Well Distributed**: Costs are spread across multiple categories, with {top_category['CostCategory']} being the largest at {top_category['Cost_Percentage']:.1f}%")

            # Category-specific insights
            for _, row in machine_breakdown.iterrows():
                category = row['CostCategory']
                cost = row['Cost']
                percentage = row['Cost_Percentage']

                if category == 'Managed Disks' and percentage > 50:
                    insights.append(
                        f"💾 **Storage Heavy**: {percentage:.1f}% of costs are disk storage - consider disk optimization")
                elif category == 'VM Compute' and percentage > 60:
                    insights.append(
                        f"🖥️ **Compute Intensive**: {percentage:.1f}% of costs are VM compute - review instance sizing")
                elif category == 'CDN' and cost > 10:
                    insights.append(
                        f"🌐 **High CDN Usage**: ${cost:.2f} in CDN costs - review content delivery patterns")
                elif category == 'Backup' and percentage > 20:
                    insights.append(
                        f"💽 **Backup Heavy**: {percentage:.1f}% of costs are backup services - review backup policies")

            # Efficiency insight
            if machine_quantity > 0:
                if avg_cost_per_hour > 1:
                    insights.append(
                        f"💰 **High Cost per Hour**: ${avg_cost_per_hour:.4f}/hour - review resource optimization")
                elif avg_cost_per_hour < 0.1:
                    insights.append(f"✅ **Efficient Usage**: ${avg_cost_per_hour:.4f}/hour - good cost efficiency")

            # Display insights
            for insight in insights:
                st.info(insight)

            # Recommendations based on analysis
            st.markdown("#### 🚀 Optimization Recommendations")

            recommendations = []

            # Storage optimization
            storage_categories = ['Managed Disks', 'Other Storage']
            storage_cost = machine_breakdown[machine_breakdown['CostCategory'].isin(storage_categories)]['Cost'].sum()
            storage_pct = (storage_cost / machine_cost * 100) if machine_cost > 0 else 0

            if storage_pct > 70:
                recommendations.append(
                    "💾 **Storage Optimization**: Consider disk tier optimization (Premium SSD → Standard SSD where appropriate)")

            # Compute optimization
            compute_categories = ['VM Compute']
            compute_cost = machine_breakdown[machine_breakdown['CostCategory'].isin(compute_categories)]['Cost'].sum()
            compute_pct = (compute_cost / machine_cost * 100) if machine_cost > 0 else 0

            if compute_pct > 60:
                recommendations.append(
                    "🖥️ **Compute Optimization**: Review VM sizing and consider reserved instances for steady workloads")

            # Network optimization
            network_categories = ['Network/IP', 'CDN', 'Bandwidth']
            network_cost = machine_breakdown[machine_breakdown['CostCategory'].isin(network_categories)]['Cost'].sum()
            network_pct = (network_cost / machine_cost * 100) if machine_cost > 0 else 0

            if network_pct > 30:
                recommendations.append(
                    "🌐 **Network Optimization**: Review data transfer patterns and CDN configuration")

            # General recommendations
            recommendations.append(
                "📊 **Regular Monitoring**: Set up cost alerts for this machine to track spending trends")
            recommendations.append(
                "🔍 **Resource Tagging**: Ensure proper tagging for better cost allocation and governance")

            for recommendation in recommendations:
                st.markdown(f"- {recommendation}")

        else:
            st.info("👆 Please select a machine from the dropdown to see its detailed cost breakdown by category.")

    def display_traditional_analysis(self, data: AzureInvoiceData):
        """Display traditional resource group and machine analysis."""
        st.header("🏗️ Resource Analysis")

        # Get traditional data
        cost_by_rg = data.get_cost_by_resource_group()
        cost_by_machine = data.get_cost_by_machine()

        if cost_by_rg.empty and cost_by_machine.empty:
            st.warning("No resource data available for analysis.")
            return

        # Resource group analysis
        if not cost_by_rg.empty:
            fig1 = self.chart_creator.create_cost_by_resource_group_chart(cost_by_rg)
            st.plotly_chart(fig1, use_container_width=True)

        # Top machines analysis
        if not cost_by_machine.empty:
            fig2 = self.chart_creator.create_top_machines_chart(cost_by_machine)
            st.plotly_chart(fig2, use_container_width=True)

        # Cost vs Usage analysis
        if not data.df.empty:
            fig3 = self.chart_creator.create_cost_usage_comparison_chart(data.df)
            st.plotly_chart(fig3, use_container_width=True)

    def display_uncategorized_analysis(self, data: AzureInvoiceData):
        """Display detailed analysis of uncategorized items."""
        st.header("🔍 Uncategorized Items Analysis")

        if not data.cost_analyzer:
            st.info("Cost analyzer not available for uncategorized analysis.")
            return

        # Get validation data
        validation = data.cost_analyzer.validate_cost_reconciliation()
        df_classified = data.cost_analyzer.classify_costs()
        uncategorized_items = df_classified[df_classified['CostCategory'] == 'Other']

        if uncategorized_items.empty:
            st.success(
                "🎉 **Excellent Categorization!** All costs have been successfully categorized into business categories.")
            st.info(
                "💡 This means your Azure invoice contains only known service types that our classification rules can handle.")
            return

        # Uncategorized summary metrics
        uncategorized_cost = uncategorized_items['Cost'].sum()
        uncategorized_count = len(uncategorized_items)
        uncategorized_percentage = (uncategorized_cost / validation['original_total'] * 100) if validation[
                                                                                                    'original_total'] > 0 else 0

        # Status indicators
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Uncategorized Cost", f"${uncategorized_cost:,.2f}")
        with col2:
            st.metric("Percentage of Total", f"{uncategorized_percentage:.2f}%")
        with col3:
            st.metric("Number of Items", f"{uncategorized_count:,}")
        with col4:
            # Status based on percentage
            if uncategorized_percentage < 1:
                st.success("✅ Excellent")
            elif uncategorized_percentage < 5:
                st.warning("⚠️ Good")
            else:
                st.error("❌ Needs Review")

        # Alert level based on uncategorized percentage
        if uncategorized_percentage > 10:
            st.error(
                f"🚨 **High Uncategorized Costs**: {uncategorized_percentage:.1f}% of costs are uncategorized. This requires immediate attention!")
        elif uncategorized_percentage > 5:
            st.warning(
                f"⚠️ **Moderate Uncategorized Costs**: {uncategorized_percentage:.1f}% of costs are uncategorized. Consider reviewing classification rules.")
        elif uncategorized_percentage > 1:
            st.info(
                f"💡 **Low Uncategorized Costs**: {uncategorized_percentage:.1f}% of costs are uncategorized. This is normal for evolving Azure services.")
        else:
            st.success(
                f"✅ **Minimal Uncategorized Costs**: Only {uncategorized_percentage:.1f}% uncategorized. Excellent categorization coverage!")

        # Show uncategorized items in a simple table
        st.subheader("📋 Uncategorized Items Summary")
        if not uncategorized_items.empty:
            # Group by service characteristics for summary
            service_breakdown = uncategorized_items.groupby(['ConsumedService', 'MeterCategory']).agg({
                'Cost': 'sum',
                'ResourceName': 'count'
            }).round(4).reset_index()

            service_breakdown.columns = ['Service Provider', 'Meter Category', 'Total Cost', 'Item Count']
            service_breakdown['Total Cost'] = service_breakdown['Total Cost'].apply(lambda x: f"${x:,.2f}")
            service_breakdown = service_breakdown.sort_values('Total Cost', ascending=False,
                                                              key=lambda x: x.str.replace('$', '').str.replace(',',
                                                                                                               '').astype(
                                                                  float))

            st.dataframe(service_breakdown, use_container_width=True, hide_index=True)

    def display_detailed_tables(self, data: AzureInvoiceData):
        """Display detailed data tables."""
        st.header("📊 Detailed Data Tables")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "💰 Cost Categories",
            "🏢 Service Providers",
            "🏗️ Resource Groups",
            "🖥️ Top Machines",
            "⚡ Efficiency Metrics"
        ])

        with tab1:
            if data.cost_analyzer:
                category_summary = data.cost_analyzer.get_category_summary()
                if not category_summary.empty:
                    st.dataframe(category_summary, use_container_width=True, hide_index=True)
                else:
                    st.info("No cost category data available.")

        with tab2:
            if data.cost_analyzer:
                provider_summary = data.cost_analyzer.get_service_provider_analysis()
                if not provider_summary.empty:
                    st.dataframe(provider_summary, use_container_width=True, hide_index=True)
                else:
                    st.info("No service provider data available.")

        with tab3:
            cost_by_rg = data.get_cost_by_resource_group()
            if not cost_by_rg.empty:
                df_display = pd.DataFrame({
                    'Resource Group': cost_by_rg.index,
                    'Total Cost ($)': cost_by_rg.values.round(2)
                })
                st.dataframe(df_display, use_container_width=True, hide_index=True)
            else:
                st.info("No resource group data available.")

        with tab4:
            cost_by_machine = data.get_cost_by_machine()
            if not cost_by_machine.empty:
                df_display = pd.DataFrame({
                    'Machine': cost_by_machine.index,
                    'Total Cost ($)': cost_by_machine.values.round(2)
                }).head(20)  # Show top 20
                st.dataframe(df_display, use_container_width=True, hide_index=True)
            else:
                st.info("No machine data available.")

        with tab5:
            efficiency_data = data.get_efficiency_metrics()
            if not efficiency_data.empty:
                st.dataframe(efficiency_data.head(20), use_container_width=True)
            else:
                st.info("No efficiency data available.")

    def display_sidebar_controls(self, data: Optional[AzureInvoiceData]):
        """Display enhanced sidebar with controls and options."""
        with st.sidebar:
            st.subheader("⚙️ Analysis Options")

            if data is not None:
                # Enhanced validation summary in sidebar
                if data.cost_analyzer:
                    validation = data.cost_analyzer.validate_cost_reconciliation()

                    st.markdown("**🔍 Cost Reconciliation**")

                    # Key reconciliation metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Original Total", f"${validation['original_total']:,.0f}")
                        st.metric("Categorized", f"${validation['categorized_cost']:,.0f}")
                    with col2:
                        st.metric("Coverage", f"{validation['categorization_coverage']:.1f}%")
                        st.metric("Uncategorized", f"${validation['uncategorized_cost']:,.0f}")

                    # Reconciliation status
                    if validation['reconciliation_success']:
                        st.success("✅ Reconciliation: PASSED")
                    else:
                        st.error(f"❌ Reconciliation: FAILED")
                        st.error(f"Difference: ${validation['difference']:,.2f}")

                    # Category summary in sidebar
                    st.markdown("**📊 Quick Category Breakdown:**")
                    category_breakdown = validation['category_breakdown']
                    for category, amount in sorted(category_breakdown.items(), key=lambda x: x[1], reverse=True)[:5]:
                        if amount > 0:
                            percentage = (amount / validation['original_total'] * 100) if validation[
                                                                                              'original_total'] > 0 else 0
                            icon = "🎯" if category != 'Other' else "⚠️"
                            st.text(f"{icon} {category}: ${amount:,.0f} ({percentage:.1f}%)")

                    if validation['uncategorized_cost'] > 0:
                        st.warning(f"💡 Review ${validation['uncategorized_cost']:,.0f} in 'Other' category")

                    st.divider()

                st.markdown("**📋 Export Options**")

                if st.button("🖨️ Print / Export PDF", use_container_width=True):
                    st.markdown("""
                    <script>
                    setTimeout(function() {
                        document.body.style.visibility = 'visible';
                        window.print();
                    }, 500);
                    </script>
                    """, unsafe_allow_html=True)
                    st.success("📄 Print dialog opening...")

                st.markdown("**🔍 Interactive Analysis**")

                # Quick stats about drill-down capability with error handling
                try:
                    resource_groups = data.get_all_resource_groups()
                    if resource_groups:
                        st.info(f"📊 **Available for Drill-Down:**")
                        st.text(f"• {len(resource_groups)} Resource Groups")

                        total_machines = 0
                        if 'ResourceName' in data.df.columns:
                            # Count unique machines, excluding NaN values
                            total_machines = data.df['ResourceName'].dropna().nunique()
                        st.text(f"• {total_machines} Unique Machines")

                        if data.cost_analyzer:
                            category_summary = data.cost_analyzer.get_category_summary()
                            active_categories = len(category_summary[category_summary[
                                                                         'Total_Cost'] > 0]) if not category_summary.empty else 0
                            st.text(f"• {active_categories} Active Cost Categories")
                    else:
                        st.warning("No valid resource groups found for drill-down analysis")
                except Exception as e:
                    st.warning(f"Unable to load drill-down stats: {str(e)}")

                st.markdown("**🎨 Chart Options**")

                chart_height = st.slider("Chart Height", 300, 800, Config.CHART_HEIGHT)
                Config.CHART_HEIGHT = chart_height

                top_items = st.slider("Show Top N Items", 5, 50, Config.TOP_ITEMS_COUNT)
                Config.TOP_ITEMS_COUNT = top_items

            else:
                st.info("Upload a CSV file to access analysis options.")

            st.markdown("**ℹ️ About**")
            st.markdown("""
            **Azure Invoice Analyzer Pro**

            Advanced features:
            - 🎯 9 cost categories classification
            - 🔍 Cost reconciliation & validation  
            - 🏢 Service provider analysis
            - ⚡ Resource efficiency metrics
            - 🔍 **Interactive drill-down**: Resource Group → Machines → Cost Categories
            - 📊 Interactive visualizations with drill-down charts
            - 📄 PDF export capabilities

            **Interactive Drill-Down Flow:**
            1. Select a Resource Group
            2. View all machines in that group
            3. Click on any machine to see its cost breakdown across all 9 categories
            4. Get optimization recommendations

            **Required CSV Columns:**
            Date, Cost, Quantity, ResourceGroup, ResourceName, ConsumedService, MeterCategory, MeterSubcategory
            """)

    def run(self):
        """Enhanced main application entry point."""
        self.display_header()

        df = self.display_file_uploader()

        if df is not None:
            data = AzureInvoiceData(df)
            self.display_sidebar_controls(data)

            with st.container():
                # Enhanced summary with validation
                self.display_enhanced_summary(data)
                st.divider()

                # Cost category analysis (new primary feature)
                self.display_cost_category_analysis(data)
                st.divider()

                # Service provider analysis (new feature)
                self.display_service_provider_analysis(data)
                st.divider()

                # Efficiency analysis (new feature)
                self.display_efficiency_analysis(data)
                st.divider()

                # Interactive drill-down analysis (new feature)
                self.display_interactive_drill_down(data)
                st.divider()

                # Traditional resource analysis
                self.display_traditional_analysis(data)
                st.divider()

                # Uncategorized items analysis (new prominent section)
                self.display_uncategorized_analysis(data)
                st.divider()

                # Enhanced detailed tables
                self.display_detailed_tables(data)

                st.success(
                    "✅ Enhanced analysis complete! Cost categories classified, validation performed, and efficiency metrics calculated.")

        else:
            st.info("📁 Please upload your Azure Invoice CSV file to begin enhanced analysis!")

            with st.expander("📋 Required CSV Format"):
                st.markdown("""
                Your CSV file should contain these columns:
                - **Date**: Invoice date
                - **Cost**: Cost amount (numeric)
                - **Quantity**: Usage quantity (numeric)
                - **ResourceGroup**: Azure resource group
                - **ResourceName**: Individual resource name
                - **ConsumedService**: Azure service provider (e.g., Microsoft.Compute)
                - **MeterCategory**: Service category (e.g., Storage, Virtual Network)
                - **MeterSubcategory**: Detailed service type (e.g., Premium SSD Managed Disks)
                """)


def main():
    """Main function to run the enhanced Streamlit app."""
    dashboard = StreamlitDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()