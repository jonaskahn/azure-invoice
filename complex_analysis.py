from typing import Optional, Dict, Any
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


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

    def get_cost_by_resource_group(self, use_classified: bool=True) -> pd.Series:
        """Calculate total cost grouped by ResourceGroup."""
        if self.df is None or 'ResourceGroup' not in self.df.columns:
            return pd.Series(dtype=float)

        # Use classified data if available and requested
        if self.cost_analyzer and use_classified:
            df_to_use = self.cost_analyzer.classify_costs()
        else:
            df_to_use = self.df.copy()

        return df_to_use.groupby('ResourceGroup')['Cost'].sum().sort_values(ascending=False)

    def _get_unified_machine_costs(self, include_related: bool=True) -> Dict[str, float]:
        """Unified method to calculate machine costs with consistent logic across all sections."""
        if self.df is None or self.df.empty:
            return {}

        if 'ResourceName' not in self.df.columns:
            return {}

        # Use classified data if available, otherwise use raw data
        if self.cost_analyzer and include_related:
            df_to_use = self.cost_analyzer.classify_costs()
        else:
            df_to_use = self.df.copy()

        # Filter out rows with NaN values in ResourceName
        clean_df = df_to_use.dropna(subset=['ResourceName'])

        if clean_df.empty:
            return {}

        # Get all unique machine names
        machine_names = clean_df['ResourceName'].unique()
        machine_costs = {}

        for machine_name in machine_names:
            # Skip resources that appear to be related resources themselves (contain typical suffixes)
            if include_related and any(suffix in machine_name.lower() for suffix in
                                       ['-disk', '_osdisk', '-nic', '-ip', '-nsg', 'disk-', 'snapshot']):
                continue

            if include_related:
                # Get exact matches first
                machine_data = clean_df[clean_df['ResourceName'] == machine_name]

                # Also look for related resources (disks, network interfaces, etc.)
                related_data = clean_df[
                    (clean_df['ResourceName'].str.contains(machine_name, case=False, na=False)) | 
                    (clean_df['ResourceName'].str.startswith(machine_name + '-', na=False)) | 
                    (clean_df['ResourceName'].str.startswith(machine_name + '_', na=False))
                    ]

                # Combine exact and related matches, removing duplicates
                combined_data = pd.concat([machine_data, related_data]).drop_duplicates()

                if not combined_data.empty:
                    machine_costs[machine_name] = combined_data['Cost'].sum()
            else:
                # Simple exact match only
                machine_data = clean_df[clean_df['ResourceName'] == machine_name]
                if not machine_data.empty:
                    machine_costs[machine_name] = machine_data['Cost'].sum()

        return machine_costs

    def get_cost_by_machine(self, include_related: bool=True) -> pd.Series:
        """Calculate total cost grouped by ResourceName (machine)."""
        machine_costs = self._get_unified_machine_costs(include_related=include_related)
        if not machine_costs:
            return pd.Series(dtype=float)

        return pd.Series(machine_costs).sort_values(ascending=False)

    def get_efficiency_metrics(self, include_related: bool=True) -> pd.DataFrame:
        """Calculate efficiency metrics (cost per unit) using unified machine calculation logic."""
        if self.df is None or self.df.empty:
            return pd.DataFrame()

        # Get unified machine costs
        machine_costs = self._get_unified_machine_costs(include_related=include_related)

        if not machine_costs:
            return pd.DataFrame()

        # Use the same data source as unified calculation
        if self.cost_analyzer and include_related:
            df_to_use = self.cost_analyzer.classify_costs()
        else:
            df_to_use = self.df.copy()

        # Calculate quantities for each machine using the same logic
        machine_quantities = {}
        clean_df = df_to_use.dropna(subset=['ResourceName'])

        for machine_name in machine_costs.keys():
            if include_related:
                # Get exact matches first
                machine_data = clean_df[clean_df['ResourceName'] == machine_name]

                # Also look for related resources
                related_data = clean_df[
                    (clean_df['ResourceName'].str.contains(machine_name, case=False, na=False)) | 
                    (clean_df['ResourceName'].str.startswith(machine_name + '-', na=False)) | 
                    (clean_df['ResourceName'].str.startswith(machine_name + '_', na=False))
                    ]

                # Combine exact and related matches, removing duplicates
                combined_data = pd.concat([machine_data, related_data]).drop_duplicates()
                machine_quantities[machine_name] = combined_data['Quantity'].sum()
            else:
                # Simple exact match only
                machine_data = clean_df[clean_df['ResourceName'] == machine_name]
                machine_quantities[machine_name] = machine_data['Quantity'].sum()

        # Build efficiency dataframe
        efficiency_data = []
        for machine_name in machine_costs.keys():
            cost = machine_costs[machine_name]
            quantity = machine_quantities.get(machine_name, 0)

            if quantity > 0:
                efficiency_data.append({
                    'ResourceName': machine_name,
                    'Cost': cost,
                    'Quantity': quantity,
                    'CostPerUnit': cost / quantity,
                    'EfficiencyScore': cost / quantity
                })

        if not efficiency_data:
            return pd.DataFrame()

        efficiency_summary = pd.DataFrame(efficiency_data)
        efficiency_summary = efficiency_summary.round(4)

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

        # Use the same data source as breakdown - get classified data if available
        if self.cost_analyzer:
            df_to_use = self.cost_analyzer.classify_costs()
        else:
            df_to_use = self.df.copy()

        # Filter out rows with NaN values in ResourceGroup or ResourceName
        clean_df = df_to_use.dropna(subset=['ResourceGroup', 'ResourceName'])
        rg_data = clean_df[clean_df['ResourceGroup'] == resource_group]

        if rg_data.empty:
            return pd.DataFrame()

        # Get all unified machine costs for this resource group
        all_machine_costs = self._get_unified_machine_costs(include_related=True)

        # Filter to only machines that have resources in this resource group
        rg_machine_names = set()
        for machine_name in rg_data['ResourceName'].unique():
            # Skip infrastructure resources
            if not any(suffix in machine_name.lower() for suffix in
                       ['-disk', '_osdisk', '-nic', '-ip', '-nsg', 'disk-', 'snapshot']):
                # Check if this machine or its related resources are in this resource group
                machine_related = rg_data[
                    (rg_data['ResourceName'] == machine_name) | 
                    (rg_data['ResourceName'].str.contains(machine_name, case=False, na=False)) | 
                    (rg_data['ResourceName'].str.startswith(machine_name + '-', na=False)) | 
                    (rg_data['ResourceName'].str.startswith(machine_name + '_', na=False))
                    ]
                if not machine_related.empty:
                    rg_machine_names.add(machine_name)

        # Build machine summary for this resource group
        machine_totals = []
        for machine_name in rg_machine_names:
            if machine_name in all_machine_costs:
                # Get detailed info for this machine in this resource group
                machine_related = rg_data[
                    (rg_data['ResourceName'] == machine_name) | 
                    (rg_data['ResourceName'].str.contains(machine_name, case=False, na=False)) | 
                    (rg_data['ResourceName'].str.startswith(machine_name + '-', na=False)) | 
                    (rg_data['ResourceName'].str.startswith(machine_name + '_', na=False))
                    ]

                total_quantity = machine_related['Quantity'].sum()
                services = ', '.join(machine_related['ConsumedService'].dropna().astype(str).unique())
                meter_categories = ', '.join(machine_related['MeterCategory'].dropna().astype(str).unique())

                machine_totals.append({
                    'ResourceName': machine_name,
                    'Cost': all_machine_costs[machine_name],
                    'Quantity': total_quantity,
                    'ConsumedService': services,
                    'MeterCategory': meter_categories
                })

        if not machine_totals:
            return pd.DataFrame()

        machine_summary = pd.DataFrame(machine_totals)

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
        pass

    def format_label(self, label: str, max_length: int=40) -> str:
        """Format label to specified length, padding with spaces if needed."""
        if len(label) <= max_length:
            return label.ljust(max_length)
        return label[:max_length - 3] + "..."

    def create_cost_category_pie_chart(self, category_summary: pd.DataFrame) -> go.Figure:
        """Create pie chart showing cost breakdown by category."""
        if category_summary.empty:
            return go.Figure()

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

        # Get colors for categories
        colors = [CATEGORY_COLORS.get(cat, '#CCCCCC') for cat in category_summary['CostCategory']]

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
            height=500,
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

        # Get colors for categories
        colors = [CATEGORY_COLORS.get(cat, '#CCCCCC') for cat in category_summary['CostCategory']]

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
            height=500,
            showlegend=False,
            xaxis={'tickangle': 45},
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        return fig

    def create_efficiency_metrics_chart(self, efficiency_data: pd.DataFrame) -> go.Figure:
        """Create comprehensive efficiency metrics visualization with enhanced clarity."""
        if efficiency_data.empty:
            return go.Figure()

        # Take top 15 resources by cost
        top_resources = efficiency_data.head(15).copy()

        # Calculate efficiency categories for color coding
        efficiency_median = efficiency_data['EfficiencyScore'].median()
        efficiency_mean = efficiency_data['EfficiencyScore'].mean()

        # Define efficiency categories
        def get_efficiency_category(score):
            if score > efficiency_mean * 1.5:
                return 'High Cost/Unit'
            elif score > efficiency_median:
                return 'Above Average'
            else:
                return 'Efficient'

        top_resources['EfficiencyCategory'] = top_resources['EfficiencyScore'].apply(get_efficiency_category)

        # Color mapping for efficiency categories
        color_map = {
            'High Cost/Unit': '#FF6B6B',  # Red - needs attention
            'Above Average': '#FFD93D',  # Yellow - monitor
            'Efficient': '#6BCF7F'  # Green - good
        }

        colors = [color_map[cat] for cat in top_resources['EfficiencyCategory']]

        # Create subplot with secondary y-axis
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Total Cost by Resource',
                'Usage Hours by Resource',
                'Cost per Unit ($/hour)',
                'Efficiency Distribution'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"type": "pie"}]
            ],
            vertical_spacing=0.25,
            horizontal_spacing=0.15
        )

        # 1. Total Cost Bar Chart (top left)
        fig.add_trace(
            go.Bar(
                x=top_resources.index,
                y=top_resources['Cost'],
                name='Total Cost',
                marker_color=colors,
                text=[f'${cost:,.0f}' for cost in top_resources['Cost']],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Cost: $%{y:,.2f}<br>Category: %{customdata}<extra></extra>',
                customdata=top_resources['EfficiencyCategory'],
                showlegend=False
            ),
            row=1, col=1
        )

        # 2. Usage Hours Bar Chart (top right)
        fig.add_trace(
            go.Bar(
                x=top_resources.index,
                y=top_resources['Quantity'],
                name='Usage Hours',
                marker_color='lightblue',
                text=[f'{qty:,.0f}h' for qty in top_resources['Quantity']],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Hours: %{y:,.0f}<extra></extra>',
                showlegend=False
            ),
            row=1, col=2
        )

        # 3. Cost per Unit Line Chart (bottom left)
        fig.add_trace(
            go.Scatter(
                x=top_resources.index,
                y=top_resources['EfficiencyScore'],
                mode='lines+markers+text',
                name='Cost per Unit',
                line=dict(color='navy', width=3),
                marker=dict(size=10, color=colors, line=dict(width=2, color='white')),
                text=[f'${score:.3f}' for score in top_resources['EfficiencyScore']],
                textposition='top center',
                hovertemplate='<b>%{x}</b><br>Cost/Unit: $%{y:.4f}<br>Category: %{customdata}<extra></extra>',
                customdata=top_resources['EfficiencyCategory'],
                showlegend=False
            ),
            row=2, col=1
        )

        # 4. Efficiency Category Pie Chart (bottom right)
        category_counts = top_resources['EfficiencyCategory'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=category_counts.index,
                values=category_counts.values,
                marker_colors=[color_map[cat] for cat in category_counts.index],
                textinfo='label+percent+value',
                texttemplate='<b>%{label}</b><br>%{percent}<br>(%{value} resources)',
                hovertemplate='<b>%{label}</b><br>Resources: %{value}<br>Percentage: %{percent}<extra></extra>',
                showlegend=False
            ),
            row=2, col=2
        )

        # Update axes labels
        fig.update_xaxes(title_text="Resources", tickangle=45, row=1, col=1)
        fig.update_xaxes(title_text="Resources", tickangle=45, row=1, col=2)
        fig.update_xaxes(title_text="Resources", tickangle=45, row=2, col=1)

        fig.update_yaxes(title_text="Cost (USD)", row=1, col=1)
        fig.update_yaxes(title_text="Hours", row=1, col=2)
        fig.update_yaxes(title_text="Cost per Hour (USD)", row=2, col=1)

        fig.update_layout(
            title={
                'text': '⚡ Comprehensive Resource Efficiency Analysis',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': 'darkblue'}
            },
            height=800,
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(t=100, b=50, l=50, r=50)
        )

        return fig

    def create_machine_cost_breakdown_chart(self, machine_breakdown: pd.DataFrame, machine_name: str) -> go.Figure:
        """Create cost breakdown chart for a specific machine showing all categories."""
        if machine_breakdown.empty:
            return go.Figure()

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

        # Get colors for categories
        colors = [CATEGORY_COLORS.get(cat, '#CCCCCC') for cat in machine_breakdown['CostCategory']]

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

        # Create stacked bar chart showing different service components
        fig = go.Figure()

        # Add bars for each category
        for i, row in machine_breakdown.iterrows():
            color = CATEGORY_COLORS.get(row['CostCategory'], '#CCCCCC')

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
            height=550,
            showlegend=False,
            xaxis={'tickangle': 45},
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        return fig

    def create_top_machines_chart(self, cost_data: pd.Series, top_items: int=10) -> go.Figure:
        """Create interactive bar chart for top machines by cost."""
        if cost_data.empty:
            return go.Figure()

        top_machines = cost_data.head(top_items)
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
                'text': f'🖥️ Top {top_items} Machines by Total Cost',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            xaxis_title='Machine (ResourceName)',
            yaxis_title='Cost (USD)',
            height=550,
            showlegend=False,
            xaxis={'tickangle': 45},
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        return fig

    def create_resource_group_breakdown_chart(self, cost_data: pd.DataFrame, resource_group: str, top_items: int=10) -> go.Figure:
        """Create chart for a specific resource group."""
        sub_df = cost_data[cost_data['ResourceGroup'] == resource_group].head(top_items)
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
                'text': f'Top {top_items} Machines in: {resource_group}',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            xaxis_title='Machine (ResourceName)',
            yaxis_title='Cost (USD)',
            height=440,
            showlegend=False,
            xaxis={'tickangle': 45},
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

        fig.update_xaxes(title_text="Resource Group", tickangle=45)
        fig.update_yaxes(title_text="Cost (USD)", secondary_y=False)
        fig.update_yaxes(title_text="Usage (Hours)", secondary_y=True)

        fig.update_layout(
            title={
                'text': '📈 Cost and Usage Comparison by Resource Group',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            height=550,
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
