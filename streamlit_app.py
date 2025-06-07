from typing import Optional, Dict, Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Import separated analysis modules
from complex_analysis import AzureInvoiceData, CostCategoryAnalyzer, StreamlitChartCreator
from simple_analysis import SimpleInvoiceData, SimpleChartCreator


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

# ===================================================================
# COMPLEX INVOICE DATA CLASSES (OLD LOGIC - ISOLATED)
# ===================================================================


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
        """Calculate total cost grouped by ResourceGroup.
        
        Args:
            use_classified: If True, uses classified data when available (default: True)
        """
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
        """Calculate total cost grouped by ResourceName (machine).
        
        Args:
            include_related: If True, includes related resources (disks, NICs, etc.) in machine costs
        """
        machine_costs = self._get_unified_machine_costs(include_related=include_related)
        if not machine_costs:
            return pd.Series(dtype=float)

        return pd.Series(machine_costs).sort_values(ascending=False)

    def get_usage_by_resource_group(self, use_classified: bool=True) -> pd.Series:
        """Calculate total usage grouped by ResourceGroup.
        
        Args:
            use_classified: If True, uses classified data when available (default: True)
        """
        if self.df is None or 'ResourceGroup' not in self.df.columns:
            return pd.Series(dtype=float)

        # Use classified data if available and requested
        if self.cost_analyzer and use_classified:
            df_to_use = self.cost_analyzer.classify_costs()
        else:
            df_to_use = self.df.copy()

        return df_to_use.groupby('ResourceGroup')['Quantity'].sum().sort_values(ascending=False)

    def get_usage_by_machine(self, include_related: bool=True, use_classified: bool=True) -> pd.Series:
        """Calculate total usage using unified machine logic.
        
        Args:
            include_related: If True, includes related resources in machine calculations
            use_classified: If True, uses classified data when available (default: True)
        """
        if self.df is None or 'ResourceName' not in self.df.columns:
            return pd.Series(dtype=float)

        # Use the same data source as unified calculation
        if self.cost_analyzer and use_classified:
            df_to_use = self.cost_analyzer.classify_costs()
        else:
            df_to_use = self.df.copy()

        # Calculate quantities for each machine using the same logic as unified costs
        machine_quantities = {}
        clean_df = df_to_use.dropna(subset=['ResourceName'])

        if clean_df.empty:
            return pd.Series(dtype=float)

        # Get machine names the same way as unified costs
        machine_names = clean_df['ResourceName'].unique()

        for machine_name in machine_names:
            # Skip infrastructure resources if including related
            if include_related and any(suffix in machine_name.lower() for suffix in
                                       ['-disk', '_osdisk', '-nic', '-ip', '-nsg', 'disk-', 'snapshot']):
                continue

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

        return pd.Series(machine_quantities).sort_values(ascending=False)

    def get_cost_by_resource_group_and_machine(self, use_classified: bool=True) -> pd.DataFrame:
        """Get cost breakdown by resource group and machine.
        
        Args:
            use_classified: If True, uses classified data when available (default: True)
        """
        if (self.df is None or
                'ResourceGroup' not in self.df.columns or
                'ResourceName' not in self.df.columns):
            return pd.DataFrame()

        # Use classified data if available and requested
        if self.cost_analyzer and use_classified:
            df_to_use = self.cost_analyzer.classify_costs()
        else:
            df_to_use = self.df.copy()

        return (df_to_use.groupby(['ResourceGroup', 'ResourceName'])['Cost']
                .sum()
                .reset_index()
                .sort_values(['ResourceGroup', 'Cost'], ascending=[True, False]))

    def get_efficiency_metrics(self, include_related: bool=True) -> pd.DataFrame:
        """Calculate efficiency metrics (cost per unit) using unified machine calculation logic.
        
        Args:
            include_related: If True, includes related resources in machine costs
        """
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

    def debug_machine_calculation(self, resource_group: str, machine_name: str) -> Dict[str, Any]:
        """Debug function to compare machine calculations across all methods."""
        if self.df is None or self.df.empty:
            return {}

        debug_info = {
            'machine_name': machine_name,
            'resource_group': resource_group
        }

        # 1. Get data from machines_by_resource_group method (unified logic)
        machines_data = self.get_machines_by_resource_group(resource_group)
        machine_row = machines_data[machines_data['ResourceName'] == machine_name]
        if not machine_row.empty:
            debug_info['table_total'] = machine_row['Cost'].iloc[0]
        else:
            debug_info['table_total'] = 0

        # 2. Get data from machine_cost_breakdown method (unified logic)
        breakdown_data = self.get_machine_cost_breakdown(machine_name)
        if not breakdown_data.empty:
            debug_info['breakdown_total'] = breakdown_data['Cost'].sum()
            debug_info['breakdown_by_category'] = breakdown_data[['CostCategory', 'Cost']].to_dict('records')
        else:
            debug_info['breakdown_total'] = 0
            debug_info['breakdown_by_category'] = []

        # 3. Get data from cost_by_machine method (unified logic with related resources)
        cost_by_machine_series = self.get_cost_by_machine(include_related=True)
        debug_info['cost_by_machine_total'] = cost_by_machine_series.get(machine_name, 0)

        # 4. Get data from cost_by_machine method (simple exact match only)
        cost_by_machine_simple = self.get_cost_by_machine(include_related=False)
        debug_info['cost_by_machine_simple'] = cost_by_machine_simple.get(machine_name, 0)

        # 5. Get efficiency metrics
        efficiency_data = self.get_efficiency_metrics(include_related=True)
        efficiency_row = efficiency_data[efficiency_data['ResourceName'] == machine_name]
        if not efficiency_row.empty:
            debug_info['efficiency_total'] = efficiency_row['Cost'].iloc[0]
        else:
            debug_info['efficiency_total'] = 0

        # 6. Get related resources to see what's included
        related_resources = self.get_machine_related_resources(machine_name)
        if not related_resources.empty:
            debug_info['related_resources'] = related_resources[['ResourceName', 'Cost']].to_dict('records')
            debug_info['related_resources_total'] = related_resources['Cost'].sum()
        else:
            debug_info['related_resources'] = []
            debug_info['related_resources_total'] = 0

        # Calculate differences and check consistency
        totals = [
            debug_info.get('table_total', 0),
            debug_info.get('breakdown_total', 0),
            debug_info.get('cost_by_machine_total', 0),
            debug_info.get('efficiency_total', 0)
        ]

        debug_info['max_difference'] = max(totals) - min(totals) if totals else 0
        debug_info['all_match'] = debug_info['max_difference'] < 0.01
        debug_info['totals_summary'] = {
            'table': debug_info.get('table_total', 0),
            'breakdown': debug_info.get('breakdown_total', 0),
            'cost_by_machine': debug_info.get('cost_by_machine_total', 0),
            'cost_by_machine_simple': debug_info.get('cost_by_machine_simple', 0),
            'efficiency': debug_info.get('efficiency_total', 0)
        }

        return debug_info

    def get_efficiency_resource_breakdown(self, include_related: bool=True) -> pd.DataFrame:
        """Get detailed breakdown of efficiency metrics by resource group and machine."""
        if self.df is None or self.df.empty:
            return pd.DataFrame()

        # Get efficiency data
        efficiency_data = self.get_efficiency_metrics(include_related=include_related)

        if efficiency_data.empty:
            return pd.DataFrame()

        # Use the same data source as efficiency metrics
        if self.cost_analyzer and include_related:
            df_to_use = self.cost_analyzer.classify_costs()
        else:
            df_to_use = self.df.copy()

        if 'ResourceGroup' not in df_to_use.columns:
            return pd.DataFrame()

        clean_df = df_to_use.dropna(subset=['ResourceName', 'ResourceGroup'])

        # Build detailed breakdown
        breakdown_data = []

        for resource_name in efficiency_data.index:
            # Ensure resource_name is a string to avoid regex errors
            resource_name_str = str(resource_name)

            # Find which resource group this machine belongs to
            if include_related:
                # Get exact matches first
                machine_data = clean_df[clean_df['ResourceName'] == resource_name]

                # Also look for related resources - use string version for pattern matching
                try:
                    related_data = clean_df[
                        (clean_df['ResourceName'].str.contains(resource_name_str, case=False, na=False)) | 
                        (clean_df['ResourceName'].str.startswith(resource_name_str + '-', na=False)) | 
                        (clean_df['ResourceName'].str.startswith(resource_name_str + '_', na=False))
                        ]
                except (TypeError, ValueError):
                    # If string operations fail, just use exact matches
                    related_data = pd.DataFrame()

                # Combine exact and related matches, removing duplicates
                if not related_data.empty:
                    combined_data = pd.concat([machine_data, related_data]).drop_duplicates()
                else:
                    combined_data = machine_data
            else:
                combined_data = clean_df[clean_df['ResourceName'] == resource_name]

            if not combined_data.empty:
                # Get the primary resource group (most common one for this resource)
                resource_groups = combined_data['ResourceGroup'].value_counts()
                primary_rg = resource_groups.index[0] if len(resource_groups) > 0 else 'Unknown'

                # Get efficiency metrics for this resource
                efficiency_row = efficiency_data.loc[resource_name]

                # Get additional details
                services = ', '.join(combined_data['ConsumedService'].dropna().unique()[:3])  # Top 3 services
                if len(combined_data['ConsumedService'].dropna().unique()) > 3:
                    services += "..."

                breakdown_data.append({
                    'ResourceName': resource_name,
                    'ResourceGroup': primary_rg,
                    'Cost': efficiency_row['Cost'],
                    'Quantity': efficiency_row['Quantity'],
                    'EfficiencyScore': efficiency_row['EfficiencyScore'],
                    'CostPerUnit': efficiency_row['CostPerUnit'],
                    'RelatedResources': len(combined_data),
                    'PrimaryServices': services
                })

        if not breakdown_data:
            return pd.DataFrame()

        breakdown_df = pd.DataFrame(breakdown_data)
        breakdown_df = breakdown_df.sort_values('Cost', ascending=False)

        return breakdown_df

    def get_all_resource_groups(self) -> list:
        """Get list of all resource groups."""
        if self.df is None or self.df.empty or 'ResourceGroup' not in self.df.columns:
            return []

        # Filter out NaN values and convert to string, then get unique values
        resource_groups = self.df['ResourceGroup'].dropna().astype(str).unique().tolist()

        # Filter out empty strings and 'nan' strings, then sort
        resource_groups = [rg for rg in resource_groups if rg and rg.lower() != 'nan']

        return sorted(resource_groups)

# ===================================================================
# SIMPLE CHART CREATOR (NEW LOGIC - ISOLATED) 
# Note: SimpleChartCreator class is now imported from simple_analysis.py
# ===================================================================

# ===================================================================
# COMPLEX CHART CREATOR (OLD LOGIC - ISOLATED)
# ===================================================================


class StreamlitChartCreator:
    """Enhanced chart creator with cost category visualizations."""

    def __init__(self):
        self.theme = Config.CHART_THEME

    def format_label(self, label: str, max_length: int=Config.MAX_LABEL_LENGTH) -> str:
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
        self.chart_creator = StreamlitChartCreator()  # Complex charts
        self.simple_chart_creator = SimpleChartCreator()  # Simple charts
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

    def display_file_uploader(self) -> tuple[Optional[pd.DataFrame], str]:
        """Handle file upload and template selection, return processed DataFrame and template type."""
        
        # Template selection
        st.subheader("📋 Select Analysis Template")
        template_type = st.radio(
            "Choose the analysis template that matches your CSV format:",
            ["Complex (Advanced Azure Invoice)", "Simple (Basic Service Usage)"],
            help="Select the template that matches your CSV file structure"
        )
        
        # Convert to simple string for easier handling
        template = "complex" if "Complex" in template_type else "simple"
        
        # Display expected format based on template
        if template == "complex":
            st.info("""
            **Complex Template** expects these columns:
            `Date, Cost, Quantity, ResourceGroup, ResourceName, ConsumedService, MeterCategory, MeterSubcategory`
            
            This provides advanced cost categorization, efficiency analysis, and drill-down capabilities.
            """)
        else:
            st.info("""
            **Simple Template** expects these columns:
            `SubscriptionName, SubscriptionGuid, Date, ResourceGuid, ServiceName, ServiceType, ServiceRegion, ServiceResource, Quantity, Cost`
            
            This provides straightforward service and resource cost analysis.
            """)
        
        uploaded_file = st.file_uploader(
            f"Choose your Azure Invoice CSV file ({template_type})",
            type=['csv'],
            help=f"Upload your CSV file matching the {template} template format"
        )

        if uploaded_file is not None:
            try:
                with st.spinner("Loading and processing your data..."):
                    df = pd.read_csv(uploaded_file, parse_dates=['Date'], low_memory=False)
                    
                    # Validate columns based on template
                    if template == "complex":
                        required_cols = ['Date', 'Cost', 'Quantity', 'ResourceGroup', 'ResourceName', 'ConsumedService', 'MeterCategory', 'MeterSubcategory']
                        missing_cols = [col for col in required_cols if col not in df.columns]
                        if missing_cols:
                            st.error(f"Complex template missing required columns: {missing_cols}")
                            st.info("Please ensure your CSV has all required columns for the Complex template.")
                            return None, template
                    else:
                        required_cols = ['SubscriptionName', 'Date', 'ServiceName', 'ServiceType', 'ServiceResource', 'Quantity', 'Cost']
                        missing_cols = [col for col in required_cols if col not in df.columns]
                        if missing_cols:
                            st.error(f"Simple template missing required columns: {missing_cols}")
                            st.info("Please ensure your CSV has all required columns for the Simple template.")
                            return None, template

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

                    return df, template

            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                st.info(f"Please ensure your CSV file has the correct format for the {template} template.")
                return None, template

        return None, template

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

        # Chart calculation explanations
        with st.expander("📊 **How These Charts Are Calculated**", expanded=False):
            st.markdown("""
            **Cost Category Analysis Breakdown:**
            
            **📈 Pie Chart Calculations:**
            - **Total Cost**: Sum of all 'Cost' values for each category: `SUM(Cost) GROUP BY CostCategory`
            - **Percentage**: Each category's cost divided by total cost: `(Category Cost / Total Cost) × 100`
            - **Values Shown**: Dollar amount and percentage for each slice
            
            **📊 Bar Chart Calculations:**
            - **X-axis (Cost)**: Same total cost per category as pie chart
            - **Text Labels**: Shows both dollar amount and percentage: `$X,XXX.XX (XX.X%)`
            - **Hover Info**: Includes record count (number of line items in each category)
            
            **🏷️ Category Classification Rules:**
            - **Managed Disks**: Premium/Standard SSD/HDD Managed Disks subcategories
            - **VM Compute**: Microsoft.Compute service MINUS disk costs
            - **CDN**: Content Delivery Network meter category
            - **Network/IP**: Virtual Network meter category
            - **Backup**: Microsoft.RecoveryServices consumed service
            - **Load Balancer**: Load Balancer meter category
            - **Other Storage**: Storage category MINUS managed disks
            - **Bandwidth**: Bandwidth meter category
            - **Key Vault**: Microsoft.KeyVault consumed service
            - **Other**: All remaining costs not classified above
            
            **📊 Data Processing Steps:**
            1. Group invoice data by cost category
            2. Calculate total cost: `df.groupby('CostCategory')['Cost'].sum()`
            3. Calculate record count: `df.groupby('CostCategory')['Cost'].count()`
            4. Calculate percentages: `(category_total / grand_total) * 100`
            5. Sort by total cost descending for display
            """)

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

        # Chart calculation explanations
        with st.expander("📊 **How Service Provider Chart Is Calculated**", expanded=False):
            st.markdown("""
            **Service Provider Analysis Breakdown:**
            
            **📊 Bar Chart Calculations:**
            - **Y-axis (Cost)**: Total cost per Azure service: `SUM(Cost) GROUP BY ConsumedService`
            - **X-axis**: Azure service provider names (top 10 by cost)
            - **Text Labels**: Dollar amounts displayed above each bar: `$X,XXX.XX`
            - **Hover Info**: Service name, total cost, and number of records
            
            **🔍 Data Source Fields:**
            - **ConsumedService**: Azure service that generated the cost (e.g., Microsoft.Compute, Microsoft.Storage)
            - **Cost**: Dollar amount charged for each service usage
            - **Record_Count**: Number of individual billing line items per service
            
            **📈 Processing Logic:**
            1. Group all invoice data by 'ConsumedService' field
            2. Sum costs: `df.groupby('ConsumedService')['Cost'].sum()`
            3. Count records: `df.groupby('ConsumedService')['Cost'].count()`
            4. Calculate percentages: `(service_total / grand_total) * 100`
            5. Sort by total cost descending
            6. Display top 10 services by cost
            
            **💡 Understanding the Data:**
            - Higher bars = More expensive Azure services
            - Record count = How many billing entries (frequency of usage)
            - Percentage = What portion of total bill each service represents
            """)

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
        """Display comprehensive efficiency analysis with enhanced insights."""
        st.header("⚡ Resource Efficiency Analysis")

        with st.spinner('⚡ Calculating resource efficiency metrics...'):
            efficiency_data = data.get_efficiency_metrics(include_related=True)

        if efficiency_data.empty:
            st.warning("No efficiency data available (requires quantity > 0).")
            st.info(
                "💡 Efficiency analysis requires resources with usage data (Quantity > 0). Check if your data includes usage metrics.")
            return

        # Summary metrics at the top
        col1, col2, col3, col4 = st.columns(4)

        total_cost = efficiency_data['Cost'].sum()
        total_hours = efficiency_data['Quantity'].sum()
        avg_efficiency = efficiency_data['EfficiencyScore'].mean()
        efficiency_median = efficiency_data['EfficiencyScore'].median()

        with col1:
            st.metric("Total Cost", f"${total_cost:,.2f}")
        with col2:
            st.metric("Total Usage", f"{total_hours:,.0f} hours")
        with col3:
            st.metric("Avg Cost/Hour", f"${avg_efficiency:.4f}")
        with col4:
            st.metric("Median Cost/Hour", f"${efficiency_median:.4f}")

        # Enhanced efficiency chart
        fig = self.chart_creator.create_efficiency_metrics_chart(efficiency_data)
        st.plotly_chart(fig, use_container_width=True)

        # Detailed insights and recommendations
        st.markdown("### 📊 Efficiency Insights & Recommendations")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🎯 Cost Optimization Targets")

            # High cost per unit resources
            high_cost_threshold = avg_efficiency * 1.5
            high_cost_resources = efficiency_data[efficiency_data['EfficiencyScore'] > high_cost_threshold]

            if not high_cost_resources.empty:
                st.warning(
                    f"⚠️ **{len(high_cost_resources)} resources** have high cost per hour (>${high_cost_threshold:.4f}+)")

                display_high_cost = high_cost_resources.head(5)[['Cost', 'Quantity', 'EfficiencyScore']].copy()
                display_high_cost['Cost'] = display_high_cost['Cost'].apply(lambda x: f"${x:,.2f}")
                display_high_cost['Quantity'] = display_high_cost['Quantity'].apply(lambda x: f"{x:,.0f}")
                display_high_cost['EfficiencyScore'] = display_high_cost['EfficiencyScore'].apply(lambda x: f"${x:.4f}")
                display_high_cost.columns = ['Total Cost', 'Hours', 'Cost/Hour']

                st.dataframe(display_high_cost, use_container_width=True)

                # Calculate potential savings
                potential_savings = high_cost_resources['Cost'].sum() * 0.2  # Assume 20% savings potential
                st.info(f"💰 **Optimization Potential**: ~${potential_savings:,.2f} (20% reduction target)")
            else:
                st.success("✅ All resources have reasonable cost efficiency!")

        with col2:
            st.markdown("#### 💡 Efficiency Categories")

            # Categorize resources
            efficient_resources = efficiency_data[efficiency_data['EfficiencyScore'] <= efficiency_median]
            above_avg_resources = efficiency_data[
                (efficiency_data['EfficiencyScore'] > efficiency_median) & 
                (efficiency_data['EfficiencyScore'] <= avg_efficiency * 1.5)
                ]
            high_cost_resources = efficiency_data[efficiency_data['EfficiencyScore'] > avg_efficiency * 1.5]

            # Display categories with metrics
            st.metric(
                "🟢 Efficient Resources",
                f"{len(efficient_resources)}",
                delta=f"${efficient_resources['Cost'].sum():,.0f} total cost"
            )

            st.metric(
                "🟡 Above Average",
                f"{len(above_avg_resources)}",
                delta=f"${above_avg_resources['Cost'].sum():,.0f} total cost"
            )

            st.metric(
                "🔴 High Cost/Hour",
                f"{len(high_cost_resources)}",
                delta=f"${high_cost_resources['Cost'].sum():,.0f} total cost"
            )

        # Resource Group and Machine Analysis
        st.markdown("### 🏗️ Resource Group & Machine Breakdown")

        # Get detailed breakdown with resource groups - show loading indicator
        with st.spinner('🔄 Calculating resource group and machine breakdown...'):
            resource_breakdown = data.get_efficiency_resource_breakdown(include_related=True)

        if not resource_breakdown.empty:
            # Resource Group Summary - First Row
            st.markdown("#### 📊 Cost by Resource Group")

            with st.spinner('📊 Processing resource group summaries...'):
                rg_summary = resource_breakdown.groupby('ResourceGroup').agg({
                    'Cost': 'sum',
                    'Quantity': 'sum',
                    'ResourceName': 'count'
                }).round(2)
                rg_summary['AvgEfficiency'] = rg_summary['Cost'] / rg_summary['Quantity']
                rg_summary = rg_summary.sort_values('Cost', ascending=False)

            # Format for display
            rg_display = rg_summary.copy()
            rg_display['Cost'] = rg_display['Cost'].apply(lambda x: f"${x:,.2f}")
            rg_display['Quantity'] = rg_display['Quantity'].apply(lambda x: f"{x:,.0f}")
            rg_display['AvgEfficiency'] = rg_display['AvgEfficiency'].apply(lambda x: f"${x:.4f}")
            rg_display.columns = ['Total Cost', 'Total Hours', 'Resources', 'Avg Cost/Hour']

            st.dataframe(rg_display, use_container_width=True)

            # Add space between rows
            st.markdown("")

            # Top Cost Resources by Group - Second Row
            st.markdown("#### 🎯 Top Cost Resources by Group")

            # Show top 2 most expensive resources per top 3 resource groups
            top_rgs = rg_summary.head(3).index.tolist()

            for rg in top_rgs:
                rg_resources = resource_breakdown[resource_breakdown['ResourceGroup'] == rg].head(2)

                st.markdown(f"**{rg}:**")
                for _, resource in rg_resources.iterrows():
                    cost = resource['Cost']
                    efficiency = resource['EfficiencyScore']
                    st.write(f"• `{resource['ResourceName']}`: ${cost:,.2f} (${efficiency:.4f}/hr)")
                st.write("")

        # Detailed efficiency breakdown table
        st.markdown("#### 📋 Complete Resource & Group Breakdown")

        if not resource_breakdown.empty:
            with st.spinner('🔄 Formatting detailed breakdown table...'):

                # Add efficiency categories to the display
                def categorize_efficiency(score):
                    if score > avg_efficiency * 1.5:
                        return "🔴 High Cost"
                    elif score > efficiency_median:
                        return "🟡 Above Average"
                    else:
                        return "🟢 Efficient"

                display_data = resource_breakdown.copy()
                display_data['Category'] = display_data['EfficiencyScore'].apply(categorize_efficiency)
                display_data['Cost'] = display_data['Cost'].apply(lambda x: f"${x:,.2f}")
                display_data['Quantity'] = display_data['Quantity'].apply(lambda x: f"{x:,.0f}")
                display_data['CostPerUnit'] = display_data['CostPerUnit'].apply(lambda x: f"${x:.4f}")
                display_data['EfficiencyScore'] = display_data['EfficiencyScore'].apply(lambda x: f"${x:.4f}")

                # Reorder and rename columns
                display_data = display_data[
                    ['Category', 'ResourceGroup', 'ResourceName', 'Cost', 'Quantity', 'CostPerUnit', 'RelatedResources',
                     'PrimaryServices']]
                display_data.columns = ['Efficiency', 'Resource Group', 'Machine/Resource', 'Total Cost', 'Hours',
                                        'Cost/Hour', 'Related', 'Primary Services']

            st.dataframe(display_data, use_container_width=True, hide_index=True)

            # Cost distribution insights
            st.markdown("#### 💡 Resource Group Insights")

            with st.spinner('💡 Calculating resource group insights...'):
                # Calculate resource group statistics
                total_cost_all = resource_breakdown['Cost'].sum()
                top_rg_cost = rg_summary.iloc[0]['Cost'] if not rg_summary.empty else 0
                top_rg_name = rg_summary.index[0] if not rg_summary.empty else 'Unknown'
                top_rg_percentage = (top_rg_cost / total_cost_all * 100) if total_cost_all > 0 else 0

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Most Expensive Group",
                    top_rg_name,
                    delta=f"${top_rg_cost:,.2f} ({top_rg_percentage:.1f}%)"
                )

            with col2:
                # Find most efficient resource group
                if not rg_summary.empty:
                    most_efficient_rg = rg_summary.loc[rg_summary['AvgEfficiency'].idxmin()]
                    st.metric(
                        "Most Efficient Group",
                        most_efficient_rg.name,
                        delta=f"${most_efficient_rg['AvgEfficiency']:.4f}/hour avg"
                    )

            with col3:
                # Resource group count
                unique_rgs = resource_breakdown['ResourceGroup'].nunique()
                st.metric(
                    "Resource Groups",
                    unique_rgs,
                    delta=f"{len(resource_breakdown)} total resources"
                )
        else:
            st.warning("Unable to generate resource group breakdown - ResourceGroup data may be missing.")

        # Chart calculation explanations
        with st.expander("📊 **How Enhanced Efficiency Analysis Works**", expanded=False):
            st.markdown(f"""
            **Comprehensive Resource Efficiency Analysis:**
            
            **📊 Four-Panel Dashboard Breakdown:**
            
            **1. Total Cost by Resource (Top Left)**
            - **Purpose**: Shows which resources consume the most budget
            - **Color Coding**: 🟢 Efficient | 🟡 Above Average | 🔴 High Cost/Hour
            - **Calculation**: `SUM(Cost) GROUP BY ResourceName`
            
            **2. Usage Hours by Resource (Top Right)**
            - **Purpose**: Shows actual resource utilization
            - **Helps Identify**: Under-utilized expensive resources
            - **Calculation**: `SUM(Quantity) GROUP BY ResourceName`
            
            **3. Cost per Unit Analysis (Bottom Left)**
            - **Purpose**: Shows efficiency trends across resources
            - **Formula**: `Total_Cost ÷ Total_Quantity` per resource
            - **Color Coded**: Same as cost chart for consistency
            
            **4. Efficiency Distribution (Bottom Right)**
            - **Purpose**: Shows proportion of resources in each category
            - **Categories**: 
              - 🟢 **Efficient**: ≤ ${efficiency_median:.4f}/hour (median)
              - 🟡 **Above Average**: ${efficiency_median:.4f} - ${avg_efficiency * 1.5:.4f}/hour
              - 🔴 **High Cost**: > ${avg_efficiency * 1.5:.4f}/hour (1.5x mean)
            
            **🧮 Efficiency Score Calculation:**
            ```
            EfficiencyScore = Total_Resource_Cost ÷ Total_Resource_Hours
            
            Includes:
            - VM compute costs + related infrastructure (disks, networking)
            - All usage hours for that logical resource
            - Unified calculation logic across all sections
            ```
            
            **🎯 Optimization Insights:**
            - **High Cost + High Usage**: Consider reserved instances or rightsizing
            - **High Cost + Low Usage**: Review if resource is needed
            - **Low Cost + High Usage**: Good efficiency, potential to scale
            - **Low Cost + Low Usage**: Consider consolidation
            
            **📈 Data Processing:**
            1. Uses unified machine calculation logic (includes related resources)
            2. Filters resources with `Quantity > 0`
            3. Calculates cost per hour for each logical resource
            4. Categorizes by efficiency relative to statistical benchmarks
            5. Shows top 15 by total cost for detailed analysis
            
            **🔍 What This Tells You:**
            - Which resources offer the best value for money
            - Where to focus cost optimization efforts
            - Resource utilization patterns and efficiency trends
            - Budget distribution across your infrastructure
            """)

        # Action recommendations
        st.markdown("### 🚀 Recommended Actions")

        if not high_cost_resources.empty:
            st.markdown("#### 🔴 High Priority (High Cost/Hour Resources)")
            st.markdown(f"""
            - **Review {len(high_cost_resources)} high-cost resources** - these may benefit from:
              - 💻 **VM Rightsizing**: Check if VMs are over-provisioned
              - 💾 **Storage Optimization**: Review disk tiers (Premium → Standard where appropriate)
              - 📅 **Reserved Instances**: Consider 1-3 year commitments for steady workloads
              - ⏰ **Scheduling**: Implement auto-shutdown for development/test resources
            """)

        if not above_avg_resources.empty:
            st.markdown("#### 🟡 Medium Priority (Above Average Cost/Hour)")
            st.markdown(f"""
            - **Monitor {len(above_avg_resources)} above-average resources** for optimization opportunities
            - Set up cost alerts to track spending trends
            - Review usage patterns for potential rightsizing
            """)

        st.markdown("#### ✅ General Best Practices")
        st.markdown("""
        - **Regular Review**: Schedule monthly efficiency reviews
        - **Cost Alerts**: Set up budget alerts for top resources
        - **Tagging Strategy**: Ensure proper resource tagging for better cost allocation
        - **Automation**: Implement auto-scaling and scheduled shutdowns where appropriate
        """)

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
            with st.spinner(f'🔄 Loading machines for resource group: {selected_rg}...'):
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

            # Display machines table
            st.dataframe(display_machines, use_container_width=True, hide_index=True)

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

        except Exception as e:
            st.error(f"Error formatting machine data: {str(e)}")
            return

        # Machine cost breakdown analysis
        if selected_machine and selected_machine != "":
            st.markdown("---")

            # Get machine cost breakdown by category
            with st.spinner(f'🔍 Analyzing cost breakdown for: {selected_machine}...'):
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

            # Chart calculation explanations for machine breakdown
            with st.expander("📊 **How Machine Cost Breakdown Charts Are Calculated**", expanded=False):
                st.markdown("""
                **Machine Cost Analysis Breakdown:**
                
                **🔍 Machine Total Cost Calculation:**
                - **Includes**: Main VM resource + All related resources (disks, NICs, public IPs, etc.)
                - **Pattern Matching**: Resources with names containing, starting with `machine-name-` or `machine-name_`
                - **Example**: For VM `myserver`, includes `myserver`, `myserver-disk1`, `myserver_OsDisk`, `myserver-nic`, etc.
                - **Important**: Both the table view and breakdown now use the same calculation method for consistency
                
                **🥧 Pie Chart (Left) Calculations:**
                - **Slices**: Each cost category for this specific machine and its related resources
                - **Values**: Total cost per category: `SUM(Cost) WHERE ResourceName matches machine pattern GROUP BY CostCategory`
                - **Percentages**: Each category's cost divided by machine total: `(Category Cost / Machine Total) × 100`
                - **Data Source**: Filters all invoice data to this machine and related resources
                
                **📊 Bar Chart (Right) Calculations:**
                - **Y-axis**: Same cost categories as pie chart
                - **X-axis**: Cost amounts in dollars
                - **Sorting**: Categories ordered by cost (highest first)
                - **Additional Info**: Shows service provider and quantity details
                
                **🔍 Machine Data Processing Steps:**
                1. Find machine resources: `df[df['ResourceName'] matches machine patterns]`
                2. Classify costs using same category rules as main analysis
                3. Group by cost category: `GROUP BY CostCategory`
                4. Calculate totals: `SUM(Cost)`, `SUM(Quantity)`
                5. Calculate percentages: `(category_cost / machine_total) × 100`
                
                **📈 What Each Number Means:**
                - **Total Cost**: All charges for this machine and its infrastructure across all Azure services
                - **Category Cost**: How much this machine spent on each type of service
                - **Percentage**: What portion of this machine's costs each category represents
                - **Quantity**: Usage amount (hours, GB, operations) for each category
                
                **💡 Understanding Machine Categories:**
                - **VM Compute**: The virtual machine instance costs (CPU, RAM)
                - **Managed Disks**: Storage attached to this machine
                - **Network/IP**: Public IP addresses and network resources
                - **Backup**: Backup services for this machine
                - **Other**: Any additional services associated with this machine
                
                **🚀 Optimization Opportunities:**
                - High **VM Compute** costs → Consider resizing or reserved instances
                - High **Managed Disks** costs → Review disk tiers (Premium vs Standard)
                - High **Network/IP** costs → Review public IP usage and data transfer
                - High **Backup** costs → Review backup policies and retention
                
                **🔧 Recent Fix:**
                - Fixed inconsistency where machine table showed different totals than breakdown
                - Now both views include related infrastructure costs for accurate totals
                """)

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

            # Debug section to help troubleshoot calculation differences
            with st.expander("🔧 **Debug: Calculation Comparison Across All Sections**", expanded=False):
                try:
                    debug_info = data.debug_machine_calculation(selected_rg, selected_machine)

                    if debug_info:
                        # Overall consistency check
                        max_diff = debug_info.get('max_difference', 0)
                        all_match = debug_info.get('all_match', False)

                        if all_match:
                            st.success(f"✅ **All calculations match!** Maximum difference: ${max_diff:,.2f}")
                        else:
                            st.error(f"❌ **Calculations inconsistent!** Maximum difference: ${max_diff:,.2f}")

                        # Show all calculation methods
                        st.markdown("**Comparison Across All Calculation Methods:**")
                        totals_summary = debug_info.get('totals_summary', {})

                        col1, col2, col3, col4, col5 = st.columns(5)

                        with col1:
                            st.metric("RG Table", f"${totals_summary.get('table', 0):,.2f}")
                            st.caption("machines_by_resource_group()")

                        with col2:
                            st.metric("Breakdown", f"${totals_summary.get('breakdown', 0):,.2f}")
                            st.caption("machine_cost_breakdown()")

                        with col3:
                            st.metric("Cost by Machine", f"${totals_summary.get('cost_by_machine', 0):,.2f}")
                            st.caption("get_cost_by_machine(related=True)")

                        with col4:
                            st.metric("Simple Cost", f"${totals_summary.get('cost_by_machine_simple', 0):,.2f}")
                            st.caption("get_cost_by_machine(related=False)")

                        with col5:
                            st.metric("Efficiency", f"${totals_summary.get('efficiency', 0):,.2f}")
                            st.caption("get_efficiency_metrics()")

                        st.markdown("**Related Resources Included:**")
                        if debug_info.get('related_resources'):
                            related_df = pd.DataFrame(debug_info['related_resources'])
                            related_df['Cost'] = related_df['Cost'].apply(lambda x: f"${x:,.2f}")
                            st.dataframe(related_df, use_container_width=True, hide_index=True)
                            st.write(
                                f"**Total from related resources:** ${debug_info.get('related_resources_total', 0):,.2f}")
                        else:
                            st.write("No related resources found")

                        st.markdown("**Breakdown by Category:**")
                        if debug_info.get('breakdown_by_category'):
                            breakdown_debug_df = pd.DataFrame(debug_info['breakdown_by_category'])
                            breakdown_debug_df['Cost'] = breakdown_debug_df['Cost'].apply(lambda x: f"${x:,.2f}")
                            st.dataframe(breakdown_debug_df, use_container_width=True, hide_index=True)
                        else:
                            st.write("No breakdown data available")

                except Exception as e:
                    st.error(f"Debug error: {str(e)}")

                st.markdown("""
                **How to interpret this debug info:**
                
                **✅ What Should Match (using unified logic):**
                - **RG Table**: Cost in the resource group machines table
                - **Breakdown**: Sum of categories in detailed breakdown  
                - **Cost by Machine**: Cost from get_cost_by_machine(related=True)
                - **Efficiency**: Cost from efficiency metrics
                
                **📊 What Will Be Different:**
                - **Simple Cost**: Only exact ResourceName matches (no related resources)
                
                **🔧 Recent Fix:**
                All major calculation methods now use unified logic with consistent:
                - Data source (classified data when available)
                - Related resource inclusion patterns
                - Infrastructure resource filtering
                
                **🎯 Expected Results:**
                After the fix, RG Table, Breakdown, Cost by Machine, and Efficiency should all show the same total.
                Simple Cost will be lower as it excludes related resources like disks and network components.
                """)

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
        cost_by_rg = data.get_cost_by_resource_group(use_classified=True)
        cost_by_machine = data.get_cost_by_machine(include_related=True)

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
            # Use classified data for consistency if available
            if data.cost_analyzer:
                df_for_chart = data.cost_analyzer.classify_costs()
            else:
                df_for_chart = data.df
            fig3 = self.chart_creator.create_cost_usage_comparison_chart(df_for_chart)
            st.plotly_chart(fig3, use_container_width=True)

        # Chart calculation explanations
        with st.expander("📊 **How Resource Analysis Charts Are Calculated**", expanded=False):
            st.markdown("""
            **Resource Analysis Breakdown:**
            
            **🏗️ Cost by Resource Group Chart:**
            - **Y-axis**: Total cost per resource group: `SUM(Cost) GROUP BY ResourceGroup`
            - **X-axis**: Resource group names (ordered by total cost)
            - **Data Source**: Groups all resources by their ResourceGroup field
            - **Calculation**: `df.groupby('ResourceGroup')['Cost'].sum()`
            
            **🖥️ Top Machines Chart:**
            - **Y-axis**: Total cost per machine/resource: `SUM(Cost) GROUP BY ResourceName`
            - **X-axis**: Resource/machine names (top machines by cost)
            - **Sorting**: Descending by total cost (most expensive first)
            - **Calculation**: `df.groupby('ResourceName')['Cost'].sum().sort_values(ascending=False)`
            
            **📊 Cost vs Usage Comparison Chart:**
            - **Left Y-axis (Bars)**: Total cost per resource group
            - **Right Y-axis (Line)**: Total usage hours per resource group
            - **Dual Purpose**: Shows relationship between spending and actual usage
            - **Calculations**:
              - Cost: `df.groupby('ResourceGroup')['Cost'].sum()`
              - Usage: `df.groupby('ResourceGroup')['Quantity'].sum()`
            
            **🔍 What These Charts Tell You:**
            - **Resource Group Chart**: Which departments/projects are spending the most
            - **Top Machines Chart**: Which individual resources are your biggest cost drivers
            - **Cost vs Usage Chart**: Whether high costs correlate with high usage (efficiency check)
            
            **💡 Optimization Insights:**
            - Look for resource groups with high costs but low usage
            - Identify top machines that might be over-provisioned
            - Find resource groups where cost doesn't match usage patterns
            
            **📈 Data Processing Notes:**
            - Charts only include resources with valid ResourceGroup and ResourceName values
            - Costs are aggregated from all service categories for each resource
            - Usage is measured in the original quantity units (typically hours)
            """)

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
            cost_by_rg = data.get_cost_by_resource_group(use_classified=True)
            if not cost_by_rg.empty:
                df_display = pd.DataFrame({
                    'Resource Group': cost_by_rg.index,
                    'Total Cost ($)': cost_by_rg.values.round(2)
                })
                st.dataframe(df_display, use_container_width=True, hide_index=True)
            else:
                st.info("No resource group data available.")

        with tab4:
            cost_by_machine = data.get_cost_by_machine(include_related=True)
            if not cost_by_machine.empty:
                df_display = pd.DataFrame({
                    'Machine': cost_by_machine.index,
                    'Total Cost ($)': cost_by_machine.values.round(2)
                }).head(20)  # Show top 20
                st.dataframe(df_display, use_container_width=True, hide_index=True)
            else:
                st.info("No machine data available.")

        with tab5:
            efficiency_data = data.get_efficiency_metrics(include_related=True)
            if not efficiency_data.empty:
                st.dataframe(efficiency_data.head(20), use_container_width=True)
            else:
                st.info("No efficiency data available.")

    # ===================================================================
    # SIMPLE DASHBOARD METHODS (NEW LOGIC - ISOLATED)
    # ===================================================================
    
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
            fig1 = self.simple_chart_creator.create_cost_by_service_chart(cost_by_service)
            st.plotly_chart(fig1, use_container_width=True)
        
        # Region cost chart
        if not cost_by_region.empty:
            fig2 = self.simple_chart_creator.create_cost_by_region_chart(cost_by_region)
            st.plotly_chart(fig2, use_container_width=True)
    
    def display_simple_resource_analysis(self, data: SimpleInvoiceData):
        """Display simple resource analysis."""
        st.header("💻 Resource Analysis")
        
        cost_by_resource = data.get_cost_by_resource()
        
        if cost_by_resource.empty:
            st.warning("No resource data available for analysis.")
            return
        
        # Resource cost chart
        fig = self.simple_chart_creator.create_cost_by_resource_chart(cost_by_resource)
        st.plotly_chart(fig, use_container_width=True)
    
    def display_simple_efficiency_analysis(self, data: SimpleInvoiceData):
        """Display simple efficiency analysis."""
        st.header("⚡ Efficiency Analysis")
        
        # Usage vs Cost scatter plot - using consistent calculation methods
        cost_by_service = data.get_cost_by_service()
        usage_by_service = data.get_usage_by_service()
        fig1 = self.simple_chart_creator.create_usage_vs_cost_chart(cost_by_service, usage_by_service)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Service efficiency metrics
        service_efficiency = data.get_service_efficiency_metrics()
        
        if not service_efficiency.empty:
            fig2 = self.simple_chart_creator.create_service_efficiency_chart(service_efficiency)
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

    def display_simple_sidebar_controls(self, data: Optional[SimpleInvoiceData]):
        """Display simple sidebar with controls and options."""
        with st.sidebar:
            st.subheader("⚙️ Simple Analysis Options")
            
            if data is not None:
                summary = data.get_data_summary()
                
                st.markdown("**📊 Quick Stats:**")
                st.text(f"• Services: {summary.get('unique_services', 0)}")
                st.text(f"• Service Types: {summary.get('unique_service_types', 0)}")
                st.text(f"• Regions: {summary.get('unique_regions', 0)}")
                st.text(f"• Resources: {summary.get('unique_resources', 0)}")
                st.text(f"• Subscriptions: {summary.get('unique_subscriptions', 0)}")
                
                st.divider()
                
                st.markdown("**📋 Export Options**")
                if st.button("🖨️ Print / Export PDF", use_container_width=True):
                    st.success("📄 Print dialog opening...")
                
                st.markdown("**🎨 Chart Options**")
                chart_height = st.slider("Chart Height", 300, 800, Config.CHART_HEIGHT)
                Config.CHART_HEIGHT = chart_height
                
                top_items = st.slider("Show Top N Items", 5, 50, Config.TOP_ITEMS_COUNT)
                Config.TOP_ITEMS_COUNT = top_items
            
            else:
                st.info("Upload a CSV file to access analysis options.")
            
            st.markdown("**ℹ️ About Simple Template**")
            st.markdown("""
            **Simple Azure Invoice Analyzer**
            
            Features:
            - 🔧 Service cost analysis
            - 🌍 Regional cost distribution  
            - 💻 Resource cost breakdown
            - ⚡ Basic efficiency metrics
            - 📊 Interactive visualizations
            
            **Required CSV Columns:**
            SubscriptionName, Date, ServiceName, ServiceType, ServiceRegion, ServiceResource, Quantity, Cost
            """)

    # ===================================================================
    # COMPLEX DASHBOARD METHODS (OLD LOGIC - ISOLATED) 
    # ===================================================================

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
        """Enhanced main application entry point with template support."""
        self.display_header()

        df, template = self.display_file_uploader()

        if df is not None:
            if template == "complex":
                # ===================================================================
                # COMPLEX TEMPLATE LOGIC (OLD LOGIC - ISOLATED)
                # ===================================================================
                st.info("🔧 **Complex Template Active** - Advanced Azure invoice analysis with cost categorization")
                
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
                # ===================================================================
                # SIMPLE TEMPLATE LOGIC (NEW LOGIC - ISOLATED)
                # ===================================================================
                st.info("📊 **Simple Template Active** - Basic service usage analysis")
                
                data = SimpleInvoiceData(df)
                self.display_simple_sidebar_controls(data)

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

        else:
            st.info("📁 Please select a template and upload your Azure Invoice CSV file to begin analysis!")

            with st.expander("📋 Template Formats"):
                st.markdown("""
                **Complex Template (Advanced Analysis):**
                - **Date**: Invoice date
                - **Cost**: Cost amount (numeric)
                - **Quantity**: Usage quantity (numeric)
                - **ResourceGroup**: Azure resource group
                - **ResourceName**: Individual resource name
                - **ConsumedService**: Azure service provider (e.g., Microsoft.Compute)
                - **MeterCategory**: Service category (e.g., Storage, Virtual Network)
                - **MeterSubcategory**: Detailed service type (e.g., Premium SSD Managed Disks)
                
                **Simple Template (Basic Analysis):**
                - **SubscriptionName**: Azure subscription name
                - **SubscriptionGuid**: Subscription ID
                - **Date**: Invoice date
                - **ResourceGuid**: Resource ID
                - **ServiceName**: Service name (e.g., Virtual Machines)
                - **ServiceType**: Service type (e.g., Dsv4 Series)
                - **ServiceRegion**: Azure region (e.g., US West 2)
                - **ServiceResource**: Resource type (e.g., D2s v4)
                - **Quantity**: Usage quantity (numeric)
                - **Cost**: Cost amount (numeric)
                """)


def main():
    """Main function to run the enhanced Streamlit app."""
    dashboard = StreamlitDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
