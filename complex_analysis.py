from typing import Dict, Any

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
        # Add validation on initialization
        self._validate_data_integrity()

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

    def _validate_data_integrity(self) -> None:
        """Validate data integrity and log any anomalies."""
        if self.df is None or self.df.empty:
            return

        # Check for duplicate rows
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            st.warning(f"⚠️ Found {duplicates} duplicate rows in the data")

        # Check for negative costs
        negative_costs = (self.df['Cost'] < 0).sum()
        if negative_costs > 0:
            st.warning(f"⚠️ Found {negative_costs} rows with negative costs")

        # Check for extremely high individual costs
        total_cost = self.df['Cost'].sum()
        max_individual_cost = self.df['Cost'].max()
        if max_individual_cost > total_cost * 0.5:  # If any single row is more than 50% of total
            st.warning(
                f"⚠️ Found unusually high individual cost: ${max_individual_cost:,.2f} (Total: ${total_cost:,.2f})")

    def get_cost_by_resource_group(self, use_classified: bool=True) -> pd.Series:
        """Calculate total cost grouped by ResourceGroup with comprehensive validation."""
        if self.df is None or 'ResourceGroup' not in self.df.columns:
            return pd.Series(dtype=float)

        # Use classified data if available and requested
        if self.cost_analyzer and use_classified:
            df_to_use = self.cost_analyzer.classify_costs()
        else:
            df_to_use = self.df.copy()

        # Validate source data
        original_total = self.df['Cost'].sum()
        print(f"DEBUG: Original invoice total: ${original_total:,.2f}")

        # Clean data before grouping
        clean_df = df_to_use.copy()

        # Remove any rows with NaN costs or resource groups
        initial_rows = len(clean_df)
        clean_df = clean_df.dropna(subset=['Cost', 'ResourceGroup'])
        removed_rows = initial_rows - len(clean_df)

        if removed_rows > 0:
            print(f"DEBUG: Removed {removed_rows} rows with NaN values")

        # Ensure Cost is numeric
        clean_df['Cost'] = pd.to_numeric(clean_df['Cost'], errors='coerce')
        clean_df = clean_df.dropna(subset=['Cost'])

        # Group by ResourceGroup and sum costs
        cost_by_rg = clean_df.groupby('ResourceGroup')['Cost'].sum().sort_values(ascending=False)

        # Comprehensive validation
        grouped_total = cost_by_rg.sum()
        max_single_rg = cost_by_rg.max() if not cost_by_rg.empty else 0

        print(f"DEBUG: After grouping total: ${grouped_total:,.2f}")
        print(f"DEBUG: Max single RG cost: ${max_single_rg:,.2f}")

        # Check for impossible values
        if max_single_rg > original_total * 1.1:  # Allow 10% tolerance for rounding
            print(f"ERROR: Resource group cost ${max_single_rg:,.2f} exceeds total ${original_total:,.2f}!")
            print("ERROR: This indicates a serious calculation error!")

            # Find the problematic resource group
            problematic_rg = cost_by_rg.idxmax()
            print(f"ERROR: Problematic RG: {problematic_rg}")

            # Debug the problematic resource group
            problematic_data = clean_df[clean_df['ResourceGroup'] == problematic_rg]
            print(f"ERROR: RG has {len(problematic_data)} records")
            print(f"ERROR: Individual costs: {problematic_data['Cost'].tolist()[:10]}")
            print(f"ERROR: Sum of individual costs: ${problematic_data['Cost'].sum():,.2f}")

            # Return empty series to prevent showing incorrect data
            return pd.Series(dtype=float)

        # Check for reasonable total reconciliation
        difference = abs(grouped_total - original_total)
        tolerance = original_total * 0.01  # 1% tolerance

        if difference > tolerance:
            print(f"WARNING: Reconciliation difference: ${difference:,.2f} (tolerance: ${tolerance:,.2f})")

            # Check for NaN resource groups
            nan_cost = self.df[self.df['ResourceGroup'].isna()]['Cost'].sum()
            if nan_cost > 0:
                print(f"INFO: ${nan_cost:,.2f} in costs have NaN resource groups")

        # Final validation - log top resource groups
        print("DEBUG: Top 5 resource groups:")
        for rg, cost in cost_by_rg.head(5).items():
            percentage = (cost / original_total * 100) if original_total > 0 else 0
            print(f"  {rg}: ${cost:,.2f} ({percentage:.1f}%)")

        return cost_by_rg

    def _get_related_resources_for_machine(self, machine_name: str, df_to_use: pd.DataFrame) -> list:
        """Standardized method to find all related resources for a machine.
        
        This ensures ALL machine calculation methods use identical logic.
        """
        machine_resources = [machine_name]  # Start with the exact machine name
        
        # Get all unique resource names, sorted by length (shorter names first to avoid substring issues)
        all_resources = sorted(df_to_use['ResourceName'].dropna().unique(), key=len)
        
        # Look for related resources using precise matching
        for resource in all_resources:
            if resource == machine_name:
                continue  # Already included
                
            # Use standardized matching logic
            is_related = False
            
            # Pattern 1: resource starts with machine name + separator
            if (resource.startswith(machine_name + '-') or
                    resource.startswith(machine_name + '_')):
                is_related = True
                
            # Pattern 2: resource contains machine name but ensure it's a word boundary
            elif machine_name in resource:
                import re
                # Create pattern that matches machine_name as a complete word/segment
                pattern = r'\b' + re.escape(machine_name) + r'[\-_]'
                if re.search(pattern, resource, re.IGNORECASE):
                    is_related = True
                    
            if is_related:
                machine_resources.append(resource)
                
        return machine_resources

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

        if not include_related:
            # Simple exact match only - just group by ResourceName
            return clean_df.groupby('ResourceName')['Cost'].sum().to_dict()

        # For related resources logic, we need to be more careful to avoid double-counting
        machine_costs = {}
        processed_resources = set()  # Track which resources we've already assigned to a machine

        # Get all unique resource names, sorted by length (shorter names first to avoid substring issues)
        all_resources = sorted(clean_df['ResourceName'].unique(), key=len)

        # Identify potential main machines (resources that don't look like related resources)
        main_machines = []
        for resource in all_resources:
            # Skip if this looks like a related resource (contains typical suffixes)
            if any(suffix in resource.lower() for suffix in
                   ['-disk', '_osdisk', '-nic', '-ip', '-nsg', 'disk-', 'snapshot-']):
                continue
            main_machines.append(resource)

        # Process each main machine and find its related resources using STANDARDIZED method
        for machine_name in main_machines:
            if machine_name in processed_resources:
                continue

            # Use the standardized method to find related resources
            machine_resources = self._get_related_resources_for_machine(machine_name, clean_df)
            
            # Filter out already processed resources
            machine_resources = [r for r in machine_resources if r not in processed_resources]
            
            if machine_resources:
                # Calculate total cost for this machine and its related resources
                machine_data = clean_df[clean_df['ResourceName'].isin(machine_resources)]
                if not machine_data.empty:
                    machine_costs[machine_name] = machine_data['Cost'].sum()
                        
                    # Mark these resources as processed
                    for resource in machine_resources:
                        processed_resources.add(resource)

        # Handle any remaining unprocessed resources as standalone machines
        for resource in all_resources:
            if resource not in processed_resources:
                resource_data = clean_df[clean_df['ResourceName'] == resource]
                if not resource_data.empty:
                    machine_costs[resource] = resource_data['Cost'].sum()

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

        # Calculate quantities for each machine using the SAME STANDARDIZED logic as cost calculation
        machine_quantities = {}
        clean_df = df_to_use.dropna(subset=['ResourceName'])

        if not include_related:
            # Simple exact match only - just group by ResourceName
            machine_quantities = clean_df.groupby('ResourceName')['Quantity'].sum().to_dict()
        else:
            # Use the SAME STANDARDIZED logic as the unified cost calculation
            processed_resources = set()
            all_resources = sorted(clean_df['ResourceName'].unique(), key=len)

            # Identify potential main machines (same logic as cost calculation)
            main_machines = []
            for resource in all_resources:
                if any(suffix in resource.lower() for suffix in
                       ['-disk', '_osdisk', '-nic', '-ip', '-nsg', 'disk-', 'snapshot-']):
                    continue
                main_machines.append(resource)

            # Process each main machine using STANDARDIZED method
            for machine_name in main_machines:
                if machine_name in processed_resources:
                    continue

                # Use the STANDARDIZED method to find related resources
                machine_resources = self._get_related_resources_for_machine(machine_name, clean_df)
                
                # Filter out already processed resources
                machine_resources = [r for r in machine_resources if r not in processed_resources]
                
                if machine_resources:
                    # Calculate total quantity for this machine and its related resources
                    machine_data = clean_df[clean_df['ResourceName'].isin(machine_resources)]
                    if not machine_data.empty:
                        machine_quantities[machine_name] = machine_data['Quantity'].sum()
                            
                        # Mark these resources as processed
                        for resource in machine_resources:
                            processed_resources.add(resource)

            # Handle any remaining unprocessed resources as standalone machines
            for resource in all_resources:
                if resource not in processed_resources:
                    resource_data = clean_df[clean_df['ResourceName'] == resource]
                    if not resource_data.empty:
                        machine_quantities[resource] = resource_data['Quantity'].sum()

        # Build efficiency dataframe using only machines that have both cost and quantity data
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
            'unique_records': len(self.df.drop_duplicates()),  # Add unique record count
            'duplicate_records': len(self.df) - len(self.df.drop_duplicates()),  # Add duplicate count
            'date_range': {
                'start': self.df['Date'].min() if 'Date' in self.df.columns else None,
                'end': self.df['Date'].max() if 'Date' in self.df.columns else None
            },
            'unique_resource_groups': self.df['ResourceGroup'].nunique() if 'ResourceGroup' in self.df.columns else 0,
            'unique_machines': self.df['ResourceName'].nunique() if 'ResourceName' in self.df.columns else 0,
            'negative_costs': (self.df['Cost'] < 0).sum(),  # Add negative cost count
            'max_single_cost': self.df['Cost'].max(),  # Add max single cost
            'rows_with_nan_rg': self.df['ResourceGroup'].isna().sum() if 'ResourceGroup' in self.df.columns else 0
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
        """Get machines (ResourceName) and their costs for a specific resource group.
        
        Now uses the STANDARDIZED method to ensure consistency with all other calculations.
        """
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

        # STEP 1: Get ALL machines that have resources in this resource group using STANDARDIZED logic
        # Get all machines from the unified calculation
        all_machine_costs = self._get_unified_machine_costs(include_related=True)

        # Filter to only machines that have resources in this resource group
        # BUT show the FULL machine cost (like detailed analysis), not just the portion in this RG
        machines_in_rg = {}
        
        for machine_name, total_machine_cost in all_machine_costs.items():
            # Use the STANDARDIZED method to find all related resources for this machine
            machine_resources = self._get_related_resources_for_machine(machine_name, clean_df)
            
            # Check if ANY of this machine's resources are in the target resource group
            machine_resources_in_rg = []
            has_resources_in_rg = False
            
            # Find resources in this RG for metadata (services, categories)
            rg_services = set()
            rg_categories = set()
            rg_quantity = 0
            
            for resource_name in machine_resources:
                resource_data_in_rg = rg_data[rg_data['ResourceName'] == resource_name]
                if not resource_data_in_rg.empty:
                    machine_resources_in_rg.append(resource_name)
                    has_resources_in_rg = True
                    rg_quantity += resource_data_in_rg['Quantity'].sum()
                    rg_services.update(resource_data_in_rg['ConsumedService'].dropna().astype(str).unique())
                    rg_categories.update(resource_data_in_rg['MeterCategory'].dropna().astype(str).unique())
            
            # If this machine has ANY resources in this RG, include it with FULL cost
            if has_resources_in_rg:
                # Get full machine metadata from ALL its resources (not just RG-specific)
                all_machine_data = clean_df[clean_df['ResourceName'].isin(machine_resources)]
                full_services = set()
                full_categories = set()
                full_quantity = 0
                
                if not all_machine_data.empty:
                    full_services.update(all_machine_data['ConsumedService'].dropna().astype(str).unique())
                    full_categories.update(all_machine_data['MeterCategory'].dropna().astype(str).unique())
                    full_quantity = all_machine_data['Quantity'].sum()
                
                machines_in_rg[machine_name] = {
                    'cost': total_machine_cost,  # FULL machine cost, not just RG portion
                    'quantity': full_quantity,  # FULL machine quantity
                    'services': ', '.join(sorted(full_services)),
                    'meter_categories': ', '.join(sorted(full_categories)),
                    'resources_in_rg': machine_resources_in_rg
                }

        # STEP 2: Convert to the expected DataFrame format
        if not machines_in_rg:
            return pd.DataFrame()

        machine_totals = []
        for machine_name, machine_info in machines_in_rg.items():
            machine_totals.append({
                'ResourceName': machine_name,
                'Cost': machine_info['cost'],
                'Quantity': machine_info['quantity'],
                'ConsumedService': machine_info['services'],
                'MeterCategory': machine_info['meter_categories']
            })

        machine_summary = pd.DataFrame(machine_totals)

        # Calculate percentage within resource group
        total_rg_cost = machine_summary['Cost'].sum()
        machine_summary['Cost_Percentage'] = (machine_summary['Cost'] / total_rg_cost * 100).round(
            2) if total_rg_cost > 0 else 0

        machine_summary = machine_summary.sort_values('Cost', ascending=False)

        return machine_summary

    def get_machines_by_resource_group_rg_only(self, resource_group: str) -> pd.DataFrame:
        """Get machines costs ONLY from resources within the specified resource group.
        
        This method ensures Sum of Machines = Resource Group Total by only counting
        resources that physically exist within the target resource group.
        """
        if self.df is None or self.df.empty:
            return pd.DataFrame()

        if 'ResourceGroup' not in self.df.columns or 'ResourceName' not in self.df.columns:
            return pd.DataFrame()

        # Use the same data source as other methods
        if self.cost_analyzer:
            df_to_use = self.cost_analyzer.classify_costs()
        else:
            df_to_use = self.df.copy()

        # Filter out rows with NaN values and get ONLY resources in this RG
        clean_df = df_to_use.dropna(subset=['ResourceGroup', 'ResourceName'])
        rg_data = clean_df[clean_df['ResourceGroup'] == resource_group]

        if rg_data.empty:
            return pd.DataFrame()

        # Group by ResourceName to get individual resource costs WITHIN this RG only
        machine_costs_in_rg = rg_data.groupby('ResourceName').agg({
            'Cost': 'sum',
            'Quantity': 'sum',
            'ConsumedService': lambda x: ', '.join(x.unique()),
            'MeterCategory': lambda x: ', '.join(x.unique())
        }).round(4).reset_index()

        # Calculate percentage within resource group
        total_rg_cost = machine_costs_in_rg['Cost'].sum()
        machine_costs_in_rg['Cost_Percentage'] = (machine_costs_in_rg['Cost'] / total_rg_cost * 100).round(2) if total_rg_cost > 0 else 0

        machine_costs_in_rg = machine_costs_in_rg.sort_values('Cost', ascending=False)

        return machine_costs_in_rg

    def _classify_resource_type(self, resource_name: str, machine_name: str) -> str:
        """Helper method to classify resource types for difference breakdown."""
        if resource_name == machine_name:
            return "🖥️ Main Machine"
        elif any(suffix in resource_name.lower() for suffix in ['_osdisk', 'osdisk']):
            return "💾 OS Disk"
        elif any(suffix in resource_name.lower() for suffix in ['_lun_', 'lun']):
            return "💿 Data Disk"
        elif any(suffix in resource_name.lower() for suffix in ['-nic', 'nic']):
            return "🌐 Network Interface"
        elif any(suffix in resource_name.lower() for suffix in ['-ip', 'publicip']):
            return "📡 Public IP"
        elif any(suffix in resource_name.lower() for suffix in ['-nsg', 'nsg']):
            return "🛡️ Network Security"
        else:
            return "🔧 Other Resource"

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

        # Use the STANDARDIZED method to find related resources - ensures consistency
        machine_resources = self._get_related_resources_for_machine(resource_name, df_classified)

        # Get data for this machine and its related resources
        combined_data = df_classified[df_classified['ResourceName'].isin(machine_resources)]

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

        for _, row in efficiency_data.iterrows():
            resource_name = row['ResourceName']
            # Ensure resource_name is a string to avoid regex errors
            resource_name_str = str(resource_name)

            # Find which resource group this machine belongs to
            if include_related:
                # Use the STANDARDIZED method to find related resources
                machine_resources = self._get_related_resources_for_machine(resource_name, clean_df)
                combined_data = clean_df[clean_df['ResourceName'].isin(machine_resources)]
            else:
                combined_data = clean_df[clean_df['ResourceName'] == resource_name]

            if not combined_data.empty:
                # Get the primary resource group (most common one for this resource)
                resource_groups = combined_data['ResourceGroup'].value_counts()
                primary_rg = resource_groups.index[0] if len(resource_groups) > 0 else 'Unknown'

                # Get additional details
                services = ', '.join(combined_data['ConsumedService'].dropna().unique()[:3])  # Top 3 services
                if len(combined_data['ConsumedService'].dropna().unique()) > 3:
                    services += "..."

                breakdown_data.append({
                    'ResourceName': resource_name,
                    'ResourceGroup': primary_rg,
                    'Cost': row['Cost'],
                    'Quantity': row['Quantity'],
                    'EfficiencyScore': row['EfficiencyScore'],
                    'CostPerUnit': row['CostPerUnit'],
                    'RelatedResources': len(combined_data),
                    'PrimaryServices': services
                })

        if not breakdown_data:
            return pd.DataFrame()

        breakdown_df = pd.DataFrame(breakdown_data)
        breakdown_df = breakdown_df.sort_values('Cost', ascending=False)

        return breakdown_df

    def manual_calculate_resource_group(self, resource_group: str) -> dict:
        """Manually calculate resource group costs step by step to identify issues."""
        if self.df is None or self.df.empty:
            return {}

        # Get classified data
        if self.cost_analyzer:
            df_classified = self.cost_analyzer.classify_costs()
        else:
            df_classified = self.df.copy()

        # STEP 1: Get ALL resources in this resource group (this is the TRUTH)
        clean_df = df_classified.dropna(subset=['Cost', 'ResourceGroup', 'ResourceName'])
        clean_df['Cost'] = pd.to_numeric(clean_df['Cost'], errors='coerce')
        clean_df = clean_df.dropna(subset=['Cost'])

        rg_data = clean_df[clean_df['ResourceGroup'] == resource_group]

        # TRUTH: Traditional method - simple sum
        truth_total = rg_data['Cost'].sum()

        # STEP 2: Get all unique resources with their costs
        all_resources = []
        resource_costs = {}
        for resource_name in rg_data['ResourceName'].unique():
            if pd.notna(resource_name):
                resource_data = rg_data[rg_data['ResourceName'] == resource_name]
                resource_cost = resource_data['Cost'].sum()
                resource_costs[resource_name] = resource_cost
                all_resources.append({
                    'resource': resource_name,
                    'cost': resource_cost,
                    'records': len(resource_data),
                    'is_infrastructure': any(suffix in resource_name.lower() for suffix in
                                             ['-disk', '_osdisk', '-nic', '-ip', '-nsg', 'disk-', 'snapshot'])
                })

        # STEP 3: Verify total
        manual_total = sum([r['cost'] for r in all_resources])

        # STEP 4: Create SIMPLE machine assignment (NO overlapping patterns)
        machines = {}
        assigned_resources = set()

        # First pass: Assign exact matches for main machines (non-infrastructure)
        for resource in all_resources:
            if not resource['is_infrastructure'] and resource['resource'] not in assigned_resources:
                machine_name = resource['resource']
                machines[machine_name] = {
                    'cost': resource['cost'],
                    'resources': [resource['resource']],
                    'resource_count': 1
                }
                assigned_resources.add(resource['resource'])

        # Second pass: Assign infrastructure resources to their parent machines
        for resource in all_resources:
            if resource['is_infrastructure'] and resource['resource'] not in assigned_resources:
                resource_name = resource['resource']
                assigned = False

                # Try to find a parent machine for this infrastructure resource
                for machine_name in machines.keys():
                    # Simple pattern: infrastructure resource starts with machine name
                    if (resource_name.startswith(machine_name + '-') or
                            resource_name.startswith(machine_name + '_')):
                        machines[machine_name]['cost'] += resource['cost']
                        machines[machine_name]['resources'].append(resource_name)
                        machines[machine_name]['resource_count'] += 1
                        assigned_resources.add(resource_name)
                        assigned = True
                        break

                # If not assigned, create standalone entry
                if not assigned:
                    machines[resource_name] = {
                        'cost': resource['cost'],
                        'resources': [resource_name],
                        'resource_count': 1
                    }
                    assigned_resources.add(resource_name)

        # Calculate machine total
        machine_total = sum([m['cost'] for m in machines.values()])

        # STEP 5: Identify any unassigned resources
        unassigned_resources = []
        for resource in all_resources:
            if resource['resource'] not in assigned_resources:
                unassigned_resources.append(resource)

        unassigned_total = sum([r['cost'] for r in unassigned_resources])

        return {
            'resource_group': resource_group,
            'truth_total': truth_total,
            'manual_total': manual_total,
            'machine_total': machine_total,
            'unassigned_total': unassigned_total,
            'total_resources': len(all_resources),
            'assigned_resources': len(assigned_resources),
            'unassigned_resources': len(unassigned_resources),
            'machines': machines,
            'all_resources': sorted(all_resources, key=lambda x: x['cost'], reverse=True),
            'unassigned': unassigned_resources,
            'validation': {
                'truth_vs_manual': abs(truth_total - manual_total) < 0.01,
                'truth_vs_machine': abs(truth_total - machine_total) < 0.01,
                'all_assigned': len(unassigned_resources) == 0
            }
        }

    def debug_resource_group_calculation(self, resource_group: str) -> dict:
        """Debug and manually calculate resource group costs using both methods."""
        if self.df is None or self.df.empty:
            return {}

        # Get classified data
        if self.cost_analyzer:
            df_classified = self.cost_analyzer.classify_costs()
        else:
            df_classified = self.df.copy()

        # METHOD 1: Traditional - simple groupby (what get_cost_by_resource_group does)
        traditional_df = df_classified.dropna(subset=['Cost', 'ResourceGroup'])
        traditional_df['Cost'] = pd.to_numeric(traditional_df['Cost'], errors='coerce')
        traditional_df = traditional_df.dropna(subset=['Cost'])

        # Filter for this resource group
        rg_traditional = traditional_df[traditional_df['ResourceGroup'] == resource_group]
        traditional_total = rg_traditional['Cost'].sum()
        traditional_count = len(rg_traditional)

        # METHOD 2: Drill-down - NEW SIMPLIFIED method (what get_machines_by_resource_group does)
        rg_data = df_classified.dropna(subset=['ResourceGroup', 'ResourceName'])
        rg_data = rg_data[rg_data['ResourceGroup'] == resource_group]

        drill_down_total = 0
        drill_down_machines = []

        # Get all unique ResourceNames in this RG (excluding infrastructure)
        main_machines = set()
        for resource_name in rg_data['ResourceName'].unique():
            if pd.notna(resource_name):
                # Skip infrastructure resources
                if not any(suffix in resource_name.lower() for suffix in
                           ['-disk', '_osdisk', '-nic', '-ip', '-nsg', 'disk-', 'snapshot']):
                    main_machines.add(resource_name)

        # For each machine, calculate cost using ONLY resources in THIS resource group
        for machine_name in main_machines:
            # Find all resources in THIS RG that are related to this machine
            machine_related = rg_data[
                (rg_data['ResourceName'] == machine_name) | 
                (rg_data['ResourceName'].str.contains(machine_name, case=False, na=False)) | 
                (rg_data['ResourceName'].str.startswith(machine_name + '-', na=False)) | 
                (rg_data['ResourceName'].str.startswith(machine_name + '_', na=False))
                ]

            machine_cost_in_rg = machine_related['Cost'].sum()
            if machine_cost_in_rg > 0:
                drill_down_total += machine_cost_in_rg
                related_resources = machine_related['ResourceName'].unique()
                drill_down_machines.append({
                    'machine': machine_name,
                    'cost_in_rg': machine_cost_in_rg,
                    'resources_in_rg': len(machine_related),
                    'related_resources': ', '.join(related_resources[:3]) + (
                        f' (+{len(related_resources) - 3} more)' if len(related_resources) > 3 else '')
                })

        # Manual verification - get all unique resources in this RG
        all_resources_in_rg = rg_traditional['ResourceName'].unique()
        resource_details = []
        manual_total = 0

        for resource in all_resources_in_rg:
            if pd.notna(resource):  # Skip NaN resources
                resource_data = rg_traditional[rg_traditional['ResourceName'] == resource]
                resource_cost = resource_data['Cost'].sum()
                manual_total += resource_cost
                resource_details.append({
                    'resource': resource,
                    'cost': resource_cost,
                    'records': len(resource_data)
                })

        return {
            'resource_group': resource_group,
            'traditional_method': {
                'total_cost': traditional_total,
                'record_count': traditional_count,
                'unique_resources': len(all_resources_in_rg)
            },
            'drill_down_method': {
                'total_cost': drill_down_total,
                'machine_count': len(drill_down_machines),
                'machines': drill_down_machines
            },
            'manual_verification': {
                'total_cost': manual_total,
                'resource_count': len(resource_details),
                'resources': sorted(resource_details, key=lambda x: x['cost'], reverse=True)[:10]  # Top 10
            },
            'difference': abs(traditional_total - drill_down_total),
            'match': abs(traditional_total - drill_down_total) < 0.01
        }

    def debug_machine_calculation(self, resource_group: str, machine_name: str) -> dict:
        """Debug method to compare different calculation methods for a specific machine."""
        try:
            debug_info = {}

            # 1. Get from resource group table calculation
            machines_by_rg = self.get_machines_by_resource_group(resource_group)
            table_cost = 0
            if not machines_by_rg.empty and machine_name in machines_by_rg['ResourceName'].values:
                table_cost = machines_by_rg[machines_by_rg['ResourceName'] == machine_name]['Cost'].iloc[0]

            # 2. Get from breakdown calculation
            breakdown = self.get_machine_cost_breakdown(machine_name)
            breakdown_cost = breakdown['Cost'].sum() if not breakdown.empty else 0

            # 3. Get from cost by machine (with related)
            cost_by_machine = self.get_cost_by_machine(include_related=True)
            cost_by_machine_value = cost_by_machine.get(machine_name, 0)

            # 4. Get from cost by machine (simple/exact match only)
            cost_by_machine_simple = self.get_cost_by_machine(include_related=False)
            cost_by_machine_simple_value = cost_by_machine_simple.get(machine_name, 0)

            # 5. Get from efficiency metrics
            efficiency_data = self.get_efficiency_metrics(include_related=True)
            efficiency_cost = 0
            if not efficiency_data.empty and machine_name in efficiency_data.index:
                efficiency_cost = efficiency_data.loc[machine_name, 'Cost']

            # Collect all totals
            totals = {
                'table': table_cost,
                'breakdown': breakdown_cost,
                'cost_by_machine': cost_by_machine_value,
                'cost_by_machine_simple': cost_by_machine_simple_value,
                'efficiency': efficiency_cost
            }

            # Check if all match (within small tolerance)
            all_values = [v for v in totals.values() if v > 0]
            max_diff = max(all_values) - min(all_values) if len(all_values) > 1 else 0
            all_match = max_diff < 0.01

            debug_info.update({
                'totals_summary': totals,
                'max_difference': max_diff,
                'all_match': all_match
            })

            # Get breakdown details
            if not breakdown.empty:
                breakdown_details = []
                for _, row in breakdown.iterrows():
                    breakdown_details.append({
                        'Category': row['CostCategory'],
                        'Cost': row['Cost'],
                        'Quantity': row['Quantity'],
                        'Service': row['ConsumedService']
                    })
                debug_info['breakdown_by_category'] = breakdown_details

            # Get related resources details
            if self.cost_analyzer:
                df_to_use = self.cost_analyzer.classify_costs()
            else:
                df_to_use = self.df.copy()

            # Use the STANDARDIZED method to find related resources
            machine_resources = self._get_related_resources_for_machine(machine_name, df_to_use.dropna(subset=['ResourceName']))
            related_data = df_to_use[df_to_use['ResourceName'].isin(machine_resources)]

            if not related_data.empty:
                related_details = []
                for _, row in related_data.iterrows():
                    related_details.append({
                        'ResourceName': row['ResourceName'],
                        'Cost': row['Cost'],
                        'Quantity': row['Quantity'],
                        'Service': row.get('ConsumedService', 'Unknown')
                    })
                debug_info['related_resources'] = related_details
                debug_info['related_resources_total'] = related_data['Cost'].sum()

            return debug_info

        except Exception as e:
            return {'error': str(e), 'totals_summary': {}, 'max_difference': 0, 'all_match': False}


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

    def create_resource_group_breakdown_chart(self, cost_data: pd.DataFrame, resource_group: str,
                                              top_items: int=10) -> go.Figure:
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


class ComplexDashboard:
    """Complex Azure invoice dashboard with advanced analysis features."""

    def __init__(self):
        self.chart_creator = StreamlitChartCreator()
        # Configuration Constants - Fixed values for consistent display
        self.MAX_LABEL_LENGTH = 40
        self.TOP_ITEMS_COUNT = 15  # Show top 15 items in charts
        self.CHART_HEIGHT = 500
        self.CATEGORY_COLORS = {
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

    def display_enhanced_summary(self, data: AzureInvoiceData):
        """Display enhanced data summary with validation."""
        summary = data.get_data_summary()

        if not summary:
            return

        st.header("📈 Executive Summary & Validation")

        # Add data integrity check section
        if summary.get('duplicate_records', 0) > 0 or summary.get('negative_costs', 0) > 0 or summary.get(
                'rows_with_nan_rg', 0) > 0:
            with st.expander("⚠️ **Data Integrity Warnings**", expanded=False):
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    duplicates = summary.get('duplicate_records', 0)
                    if duplicates > 0:
                        st.error(f"**Duplicate Rows:** {duplicates}")
                        st.caption("May cause inflated costs")
                    else:
                        st.success("**No Duplicates** ✓")

                with col2:
                    negative = summary.get('negative_costs', 0)
                    if negative > 0:
                        st.warning(f"**Negative Costs:** {negative}")
                        st.caption("Check for credits/refunds")
                    else:
                        st.success("**No Negative Costs** ✓")

                with col3:
                    nan_rg = summary.get('rows_with_nan_rg', 0)
                    if nan_rg > 0:
                        st.info(f"**Missing Resource Groups:** {nan_rg}")
                        st.caption("Not included in RG analysis")
                    else:
                        st.success("**All RGs Present** ✓")

                with col4:
                    max_cost = summary.get('max_single_cost', 0)
                    total_cost = summary.get('total_cost', 1)
                    if max_cost > total_cost * 0.3:
                        st.warning(f"**Max Single Cost:** ${max_cost:,.2f}")
                        st.caption(f"{(max_cost / total_cost * 100):.1f}% of total")
                    else:
                        st.success("**Normal Cost Distribution** ✓")

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

        # Enhanced efficiency chart (custom layout)
        # Get the four traces from the chart creator
        fig_full = self.chart_creator.create_efficiency_metrics_chart(efficiency_data)
        traces = fig_full.data
        # traces: [0]=cost bar, [1]=usage bar, [2]=cost/unit line, [3]=pie
        # Layout: top row 50:50, bottom row 60:40
        col_top1, col_top2 = st.columns(2)
        col_bot1, col_bot2 = st.columns([3, 2])
        # Top row
        with col_top1:
            fig_cost = go.Figure(traces[0])
            fig_cost.update_layout(title='Total Cost by Resource', height=350, margin=dict(t=40, b=20))
            st.plotly_chart(fig_cost, use_container_width=True)
        with col_top2:
            fig_usage = go.Figure(traces[1])
            fig_usage.update_layout(title='Usage Hours by Resource', height=350, margin=dict(t=40, b=20))
            st.plotly_chart(fig_usage, use_container_width=True)
        # Bottom row (now 50:50)
        col_bot1, col_bot2 = st.columns(2)
        with col_bot1:
            fig_cpu = go.Figure(traces[2])
            fig_cpu.update_layout(
                title='Cost per Unit',
                height=400,
                margin=dict(t=50, b=40),
                font=dict(size=16),
                xaxis_title='Resource',
                yaxis_title='Cost per Unit ($/hr)'
            )
            st.plotly_chart(fig_cpu, use_container_width=True)
        with col_bot2:
            fig_pie = go.Figure(traces[3])
            fig_pie.update_layout(
                title='Efficiency Distribution',
                height=400,
                margin=dict(t=50, b=40),
                font=dict(size=16),
                legend=dict(orientation='h', y=-0.2, x=0.5, xanchor='center'),
            )
            fig_pie.update_traces(textfont_size=16, pull=[0.05, 0, 0], textinfo='label+percent+value', showlegend=True)
            st.plotly_chart(fig_pie, use_container_width=True)

        # Detailed insights and recommendations
        st.markdown("### 📊 Efficiency Insights & Recommendations")

        col1, col2 = st.columns([6,4])

        with col1:
            st.markdown("#### 🎯 Cost Optimization Targets")

            # High cost per unit resources
            high_cost_threshold = avg_efficiency * 1.5
            high_cost_resources = efficiency_data[efficiency_data['EfficiencyScore'] > high_cost_threshold]

            if not high_cost_resources.empty:
                st.warning(
                    f"⚠️ **{len(high_cost_resources)} resources** have high cost per hour (>${high_cost_threshold:.4f}+)")

                # Include Resource Name and Resource Group for better clarity
                # Map each resource to its primary resource group
                resource_group_map = data.df.groupby('ResourceName')['ResourceGroup'].agg(lambda x: x.mode()[0] if not x.mode().empty else 'Unknown')
                # Prepare display with ResourceName, ResourceGroup, Cost, Quantity, EfficiencyScore
                display_high_cost = high_cost_resources.head(5)[['ResourceName', 'Cost', 'Quantity', 'EfficiencyScore']].copy()
                display_high_cost['Resource Group'] = display_high_cost['ResourceName'].map(resource_group_map)
                display_high_cost = display_high_cost[['ResourceName', 'Resource Group', 'Cost', 'Quantity', 'EfficiencyScore']]
                # Format values
                display_high_cost['Cost'] = display_high_cost['Cost'].apply(lambda x: f"${x:,.2f}")
                display_high_cost['Quantity'] = display_high_cost['Quantity'].apply(lambda x: f"{x:,.0f}")
                display_high_cost['EfficiencyScore'] = display_high_cost['EfficiencyScore'].apply(lambda x: f"${x:.4f}")
                display_high_cost.columns = ['Resource Name', 'Resource Group', 'Total Cost', 'Hours', 'Cost/Hour']
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
                # Use the CORRECTED resource group calculation method instead of grouping breakdown data
                rg_costs = data.get_cost_by_resource_group(use_classified=True)

                # Calculate additional metrics from efficiency data
                rg_metrics = resource_breakdown.groupby('ResourceGroup').agg({
                    'Quantity': 'sum',
                    'ResourceName': 'count'
                }).round(2)

                # Combine the corrected costs with metrics
                rg_summary = pd.DataFrame(index=rg_costs.index)
                rg_summary['Cost'] = rg_costs.values
                rg_summary['Quantity'] = rg_metrics['Quantity'].reindex(rg_summary.index, fill_value=0)
                rg_summary['ResourceName'] = rg_metrics['ResourceName'].reindex(rg_summary.index, fill_value=0)
                rg_summary['AvgEfficiency'] = rg_summary['Cost'] / rg_summary['Quantity'].replace(0,
                                                                                                  1)  # Avoid division by zero
                rg_summary = rg_summary.sort_values('Cost', ascending=False)

            # Format for display
            rg_display = rg_summary.copy()
            rg_display['Cost'] = rg_display['Cost'].apply(lambda x: f"${x:,.2f}")
            rg_display['Quantity'] = rg_display['Quantity'].apply(lambda x: f"{x:,.0f}")
            rg_display['AvgEfficiency'] = rg_display['AvgEfficiency'].apply(lambda x: f"${x:.4f}")
            rg_display.columns = ['Total Cost', 'Total Hours', 'Resources', 'Avg Cost/Hour']

            # Create two columns with 6:4 ratio for Resource Group table and Top Cost Resources
            col1, col2 = st.columns([6, 4])
            
            with col1:
                st.dataframe(rg_display, use_container_width=True)
            
            with col2:
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

    def display_data_duplication_analysis(self, data: AzureInvoiceData):
        """Display comprehensive data duplication analysis."""
        st.header("🔍 Data Quality & Duplication Analysis")
        st.markdown("**Analyzing case sensitivity issues and data duplicates that may affect calculations**")
        
        # Quick duplication check
        df = data.df
        
        # Resource Group Case Sensitivity Check
        st.subheader("🏗️ Resource Group Case Sensitivity Analysis")
        
        if 'ResourceGroup' in df.columns:
            # Get unique resource groups
            rg_original = df['ResourceGroup'].dropna().unique()
            rg_lower_map = {}
            case_issues = []
            
            for rg in rg_original:
                rg_lower = str(rg).lower()
                if rg_lower in rg_lower_map:
                    existing = rg_lower_map[rg_lower]
                    case_issues.append({
                        'normalized': rg_lower,
                        'variant_1': existing,
                        'variant_2': rg,
                        'cost_1': df[df['ResourceGroup'] == existing]['Cost'].sum(),
                        'cost_2': df[df['ResourceGroup'] == rg]['Cost'].sum(),
                        'records_1': len(df[df['ResourceGroup'] == existing]),
                        'records_2': len(df[df['ResourceGroup'] == rg])
                    })
                else:
                    rg_lower_map[rg_lower] = rg
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original RG Count", len(rg_original))
            with col2:
                st.metric("Unique (case-insensitive)", len(rg_lower_map))
            with col3:
                st.metric("Case Issues Found", len(case_issues))
            
            if case_issues:
                st.error(f"🚨 **CRITICAL: Found {len(case_issues)} resource group case sensitivity issues!**")
                
                # Check specifically for PLG-PRD-GWC-RG01
                plg_issue = None
                for issue in case_issues:
                    if 'plg-prd-gwc-rg01' in issue['normalized']:
                        plg_issue = issue
                        break
                
                if plg_issue:
                    st.error("🎯 **PLG-PRD-GWC-RG01 Case Issue Confirmed!**")
                    st.markdown(f"""
                    **The issue you identified is confirmed:**
                    - **Variant 1**: `{plg_issue['variant_1']}` - {plg_issue['records_1']} records, ${plg_issue['cost_1']:,.2f}
                    - **Variant 2**: `{plg_issue['variant_2']}` - {plg_issue['records_2']} records, ${plg_issue['cost_2']:,.2f}
                    - **Combined Cost**: ${plg_issue['cost_1'] + plg_issue['cost_2']:,.2f}
                    - **This explains why you see two different groups!**
                    """)
                
                # Show all case issues
                with st.expander("🔍 **View All Case Sensitivity Issues**", expanded=False):
                    case_df_data = []
                    for issue in case_issues:
                        case_df_data.append({
                            'Normalized Name': issue['normalized'],
                            'Variant 1': issue['variant_1'],
                            'Variant 2': issue['variant_2'],
                            'V1 Records': issue['records_1'],
                            'V2 Records': issue['records_2'],
                            'V1 Cost': f"${issue['cost_1']:,.2f}",
                            'V2 Cost': f"${issue['cost_2']:,.2f}",
                            'Combined Cost': f"${issue['cost_1'] + issue['cost_2']:,.2f}"
                        })
                    
                    case_df = pd.DataFrame(case_df_data)
                    st.dataframe(case_df, use_container_width=True, hide_index=True)
                    
                    total_affected_cost = sum(issue['cost_1'] + issue['cost_2'] for issue in case_issues)
                    st.warning(f"💰 **Total affected cost**: ${total_affected_cost:,.2f}")
            else:
                st.success("✅ No resource group case sensitivity issues found!")
        
        # Exact Duplicate Analysis
        st.subheader("🔄 Exact Duplicate Records Analysis")
        
        # Check for exact duplicates
        duplicate_mask = df.duplicated(keep=False)
        exact_duplicates = df[duplicate_mask]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Exact Duplicates", len(exact_duplicates))
        with col3:
            if len(exact_duplicates) > 0:
                excess_cost = exact_duplicates['Cost'].sum() - exact_duplicates.drop_duplicates()['Cost'].sum()
                st.metric("Excess Cost", f"${excess_cost:,.2f}")
            else:
                st.metric("Excess Cost", "$0.00")
        
        if len(exact_duplicates) > 0:
            st.error(f"❌ **Found {len(exact_duplicates)} exact duplicate records!**")
            
            with st.expander("🔍 **View Sample Duplicates**", expanded=False):
                # Show sample duplicates grouped
                duplicate_groups = exact_duplicates.groupby(list(exact_duplicates.columns)).size().reset_index(name='count')
                duplicate_groups = duplicate_groups[duplicate_groups['count'] > 1].sort_values('count', ascending=False)
                
                if not duplicate_groups.empty:
                    st.dataframe(duplicate_groups.head(10), use_container_width=True, hide_index=True)
                    st.info(f"Showing top 10 of {len(duplicate_groups)} duplicate patterns")
        else:
            st.success("✅ No exact duplicate records found!")
        
        # Key Field Duplicates (same resource, date, service but different cost)
        st.subheader("🔑 Business Logic Duplicates")
        
        key_fields = ['ResourceGroup', 'ResourceName', 'Date', 'ConsumedService']
        available_keys = [field for field in key_fields if field in df.columns]
        
        if len(available_keys) >= 3:
            business_dupes = df[df.duplicated(subset=available_keys, keep=False)]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Business Duplicates", len(business_dupes))
            with col2:
                st.info(f"Key fields: {', '.join(available_keys)}")
            
            if len(business_dupes) > 0:
                # Check for cost variance in business duplicates
                variance_issues = []
                for keys, group in business_dupes.groupby(available_keys):
                    if len(group) > 1 and group['Cost'].std() > 0.01:  # Different costs for same business key
                        variance_issues.append({
                            'keys': dict(zip(available_keys, keys)),
                            'record_count': len(group),
                            'cost_variance': group['Cost'].std(),
                            'cost_values': group['Cost'].tolist()
                        })
                
                if variance_issues:
                    st.warning(f"⚠️ **Found {len(variance_issues)} patterns with cost variance!**")
                    
                    with st.expander("🔍 **View Cost Variance Issues**", expanded=False):
                        for i, issue in enumerate(variance_issues[:5]):
                            st.markdown(f"**Issue {i+1}:**")
                            for key, value in issue['keys'].items():
                                st.write(f"- {key}: {value}")
                            st.write(f"- Record Count: {issue['record_count']}")
                            st.write(f"- Cost Variance: ${issue['cost_variance']:.2f}")
                            st.write(f"- Cost Values: {issue['cost_values']}")
                            st.markdown("---")
                else:
                    st.success("✅ No cost variance issues in business duplicates!")
            else:
                st.success("✅ No business duplicate records found!")
        
        # Summary and Recommendations
        st.subheader("💡 Data Quality Recommendations")
        
        recommendations = []
        
        if case_issues:
            recommendations.append(f"🏗️ **Fix Case Sensitivity**: Standardize resource group names (affects ${sum(issue['cost_1'] + issue['cost_2'] for issue in case_issues):,.2f})")
        
        if len(exact_duplicates) > 0:
            recommendations.append(f"🔄 **Remove Duplicates**: Delete {len(exact_duplicates)} exact duplicate records")
        
        if 'variance_issues' in locals() and variance_issues:
            recommendations.append(f"🔑 **Investigate Cost Variance**: Review {len(variance_issues)} patterns with same business keys but different costs")
        
        if recommendations:
            for rec in recommendations:
                st.warning(rec)
            
            st.error("**⚠️ Data quality issues found! These issues may cause:**")
            st.markdown("""
            - Double-counting of costs in resource group analysis
            - Incorrect machine groupings and related resource detection
            - Inflated totals in dashboards
            - Inconsistent calculations between sections
            """)
        else:
            st.success("✅ **Data quality looks good!** No major duplication issues detected.")

    def run_complex_analysis(self, data: AzureInvoiceData):
        """Run the complete complex analysis dashboard."""
        st.info("🔧 **Complex Template Active** - Advanced Azure invoice analysis with cost categorization")

        # Add data duplication analysis at the top
        self.display_data_duplication_analysis(data)
        st.divider()

        # Add overall consistency validation at the top
        if data.cost_analyzer:
            with st.expander("🔍 **Overall Calculation Consistency Check**", expanded=False):
                # Perform comprehensive validation across all sections
                validation = data.cost_analyzer.validate_cost_reconciliation()
                raw_total = data.df['Cost'].sum()
                classified_total = validation['original_total']

                st.markdown("### 📊 Data Source Consistency")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Raw Data Total", f"${raw_total:,.2f}")
                with col2:
                    st.metric("Classified Data Total", f"${classified_total:,.2f}")
                with col3:
                    diff = abs(raw_total - classified_total)
                    st.metric("Difference", f"${diff:,.2f}",
                              delta="✅ Consistent" if diff < 0.01 else "❌ Error")

                if diff < 0.01:
                    st.success("✅ **Data consistency verified** - all sections use the same data source")
                else:
                    st.error("❌ **Data inconsistency detected** - sections may show different values")

                st.markdown("### 🎯 Section Data Sources")
                st.markdown("""
                **All sections now use CLASSIFIED data for consistency:**
                - ✅ **Executive Summary**: Uses classified data for validation
                - ✅ **Cost Category Analysis**: Uses classified data (primary source)
                - ✅ **Service Provider Analysis**: Uses classified data
                - ✅ **Efficiency Analysis**: Uses classified data
                - ✅ **Interactive Drill-Down**: Uses classified data (FIXED)
                - ✅ **Resource Analysis**: Uses classified data
                - ✅ **Detailed Tables**: Uses classified data
                
                **This ensures that:**
                - Resource Group costs are identical across all sections
                - Machine costs are consistent in all views
                - Category breakdowns match everywhere
                - No cross-session calculation differences
                """)

                # Quick validation of key metrics
                st.markdown("### 🧮 Quick Validation")
                rg_costs = data.get_cost_by_resource_group(use_classified=True)
                machine_costs = data.get_cost_by_machine(include_related=True)

                col1, col2, col3 = st.columns(3)
                with col1:
                    rg_total = rg_costs.sum() if not rg_costs.empty else 0
                    st.metric("Resource Groups Total", f"${rg_total:,.2f}")
                with col2:
                    machine_total = machine_costs.sum() if not machine_costs.empty else 0
                    st.metric("Machines Total", f"${machine_total:,.2f}")
                with col3:
                    rg_machine_diff = abs(rg_total - machine_total)
                    st.metric("RG vs Machine Diff", f"${rg_machine_diff:,.2f}",
                              delta="✅ Match" if rg_machine_diff < classified_total * 0.01 else "⚠️ Different scope")

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

    # Continue with more complex methods...

    def display_interactive_drill_down(self, data: AzureInvoiceData):
        """Display interactive drill-down: Resource Group -> Machines -> Cost Categories."""
        st.subheader("🔍 Interactive Drill-Down Analysis")
        st.markdown(
            "**Select a resource group to see its machines, then click on any machine to see its cost breakdown by category.**")

        # Add consistency validation info
        st.info(
            "📊 **Data Source**: Using **classified data** (same as Resource Analysis and Cost Categories for consistency)")

        with st.expander("🔍 **Cross-Section Consistency Validation**", expanded=False):
            if data.cost_analyzer:
                # Get all resource group costs from traditional analysis
                traditional_rg_costs = data.get_cost_by_resource_group(use_classified=True)
                classified_df = data.cost_analyzer.classify_costs()

                st.markdown("**Resource Group Cost Comparison:**")
                st.markdown("Comparing costs between this section and Traditional Resource Analysis...")

                # Show top 5 resource groups comparison
                top_rgs = traditional_rg_costs.head(5)
                comparison_data = []

                for rg, traditional_cost in top_rgs.items():
                    # Calculate the same RG cost using the same method as this section
                    drill_down_rg_data = classified_df[classified_df['ResourceGroup'] == rg]
                    drill_down_cost = drill_down_rg_data['Cost'].sum() if not drill_down_rg_data.empty else 0

                    diff = abs(traditional_cost - drill_down_cost)
                    status = "✅ Match" if diff < 0.01 else f"❌ Diff: ${diff:,.2f}"

                    comparison_data.append({
                        'Resource Group': rg,
                        'Traditional Analysis': f"${traditional_cost:,.2f}",
                        'Drill-Down Analysis': f"${drill_down_cost:,.2f}",
                        'Status': status
                    })

                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)

                # Overall status
                all_match = all('✅' in row['Status'] for row in comparison_data)
                if all_match:
                    st.success("✅ **Perfect Consistency**: All resource group costs match between sections!")
                else:
                    st.error("❌ **Inconsistency Detected**: Resource group costs differ between sections!")
            else:
                st.warning("⚠️ Cost analyzer not available - using raw data")

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
            # Show resource group summary with error handling - USE SAME DATA SOURCE AS TRADITIONAL ANALYSIS
            if selected_rg:
                try:
                    # Use the SAME data source as traditional analysis for consistency
                    if data.cost_analyzer:
                        df_to_use = data.cost_analyzer.classify_costs()
                    else:
                        df_to_use = data.df.copy()

                    rg_data = df_to_use[df_to_use[
                                            'ResourceGroup'] == selected_rg] if 'ResourceGroup' in df_to_use.columns else pd.DataFrame()
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

                        # Add validation indicator to show data source consistency
                        # Get the resource group cost from traditional analysis method for comparison
                        traditional_rg_costs = data.get_cost_by_resource_group(use_classified=True)
                        if selected_rg in traditional_rg_costs:
                            traditional_cost = traditional_rg_costs[selected_rg]
                            if abs(rg_cost - traditional_cost) < 0.01:
                                st.success("✅ Consistent with Traditional Analysis")
                            else:
                                st.error(f"❌ Inconsistent: Traditional shows ${traditional_cost:,.2f}")

                        # Show detailed breakdown comparison
                        with st.expander("🔧 **Debug: Calculation Method Comparison**", expanded=False):
                            st.markdown(f"**Resource Group: {selected_rg}**")

                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.markdown("**Traditional Method:**")
                                st.markdown("- Groups by ResourceGroup")
                                st.markdown("- Sums ALL costs in RG")
                                st.markdown("- Simple aggregation")
                                st.metric("Traditional Total", f"${traditional_cost:,.2f}")

                            with col_b:
                                st.markdown("**Drill-Down Method:**")
                                st.markdown("- Calculates per-machine costs")
                                st.markdown("- Only costs within THIS RG")
                                st.markdown("- Machine-by-machine sum")
                                st.metric("Drill-Down Total", f"${rg_cost:,.2f}")

                            diff = abs(rg_cost - traditional_cost)
                            if diff < 0.01:
                                st.success(f"✅ **FIXED**: Methods now match! Difference: ${diff:.2f}")
                            else:
                                st.error(f"❌ **Still inconsistent**: Difference: ${diff:,.2f}")
                                st.markdown("**Possible causes:**")
                                st.markdown("- Related resources spanning multiple RGs")
                                st.markdown("- Machine grouping logic differences")
                                st.markdown("- Data processing inconsistencies")

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

        calc_method = st.radio(
            "Please select calculation method:",
            ["Show Full Machine Costs (includes related resources from other RGs)",
             "Show RG-Only Costs (only resources within this RG)"],
            horizontal=True
        )

        # Recalculate machines data based on selected method
        if calc_method.startswith("Show RG-Only"):
            st.info("🔍 **RG-Only Mode**: Showing only the portion of each machine's cost that exists within this specific resource group.")
            # Use the RG-only calculation method
            machines_data_display = data.get_machines_by_resource_group_rg_only(selected_rg)
        else:
            st.info("🌐 **Full Machine Mode**: Showing complete machine costs (includes related resources from all resource groups).")
            machines_data_display = machines_data

        # Format machines data for display
        try:
            display_machines = machines_data_display.copy()
            display_machines['Cost'] = display_machines['Cost'].apply(lambda x: f"${x:,.2f}")
            display_machines['Quantity'] = display_machines['Quantity'].apply(lambda x: f"{x:,.2f}")
            display_machines['Cost_Percentage'] = display_machines['Cost_Percentage'].apply(lambda x: f"{x:.1f}%")

            display_machines.columns = ['Machine Name', 'Total Cost', 'Total Usage', 'Services Used',
                                        'Meter Categories', 'Cost %']

            # Create clickable machine selection
            st.markdown("**Click on a machine name below to see its detailed cost breakdown:**")

            # Display machines table
            st.dataframe(display_machines, use_container_width=True, hide_index=True)

            # Add enhanced validation section with clear explanations
            if not machines_data_display.empty:
                machine_total = machines_data_display['Cost'].sum()

                # Get traditional RG cost for comparison
                traditional_rg_costs = data.get_cost_by_resource_group(use_classified=True)
                traditional_cost = traditional_rg_costs.get(selected_rg, 0) if selected_rg in traditional_rg_costs else 0

                st.markdown("#### 🔍 **Cost Calculation Validation**")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Sum of Machines", f"${machine_total:,.2f}")
                    st.caption("Machine costs using selected method")
                with col2:
                    st.metric("Resource Group Total", f"${traditional_cost:,.2f}")
                    st.caption("Traditional RG aggregation")
                with col3:
                    diff = abs(machine_total - traditional_cost)
                    status = "✅ Match" if diff < 0.01 else f"❌ Diff: ${diff:,.2f}"
                    st.metric("Validation", status)

                # Enhanced explanation based on calculation method
                if diff >= 0.01:
                    if calc_method.startswith("Show Full Machine"):
                        st.warning(f"⚠️ **Expected Difference**: ${diff:,.2f}")
                        st.markdown(f"""
                        **Why there's a difference in Full Machine Mode:**
                        - **Sum of Machines (${machine_total:,.2f})**: Includes ALL related resources for each machine, even if they're in other resource groups
                        - **Resource Group Total (${traditional_cost:,.2f})**: Only includes resources physically located in **{selected_rg}**
                        - **Difference (${diff:,.2f})**: Represents machine-related resources located in other resource groups
                        
                        💡 **This is normal behavior** - machines often have disks, NICs, and other components in different resource groups.
                        """)
                        
                        # Show detailed breakdown of what's included in the difference
                        with st.expander("🔍 **View Detailed Breakdown: What's in the Difference?**", expanded=False):
                            try:
                                st.markdown("**📋 Resources Contributing to the Difference:**")
                                
                                # Get classified data
                                if data.cost_analyzer:
                                    df_classified = data.cost_analyzer.classify_costs()
                                else:
                                    df_classified = data.df.copy()
                                
                                df_classified = df_classified.dropna(subset=['ResourceGroup', 'ResourceName'])
                                
                                # For each machine in the current RG, find ALL its related resources
                                difference_breakdown = []
                                rg_data = df_classified[df_classified['ResourceGroup'] == selected_rg]
                                
                                for machine in machines_data_display['ResourceName'].unique():
                                    # Get ALL related resources for this machine (across all RGs)
                                    machine_resources = data._get_related_resources_for_machine(machine, df_classified)
                                    all_machine_data = df_classified[df_classified['ResourceName'].isin(machine_resources)]
                                    
                                    # Separate resources IN current RG vs OTHER RGs
                                    in_current_rg = all_machine_data[all_machine_data['ResourceGroup'] == selected_rg]
                                    in_other_rgs = all_machine_data[all_machine_data['ResourceGroup'] != selected_rg]
                                    
                                    if not in_other_rgs.empty:
                                        # Group by ResourceGroup to show where the "extra" resources are
                                        for other_rg, other_rg_data in in_other_rgs.groupby('ResourceGroup'):
                                            for _, resource in other_rg_data.iterrows():
                                                difference_breakdown.append({
                                                    'Machine': machine,
                                                    'Additional Resource': resource['ResourceName'],
                                                    'Located in RG': other_rg,
                                                    'Cost': resource['Cost'],
                                                    'Resource Type': data._classify_resource_type(resource['ResourceName'], machine),
                                                    'Service': resource['ConsumedService'],
                                                    'Category': resource.get('CostCategory', 'Unknown')
                                                })
                                
                                if difference_breakdown:
                                    diff_df = pd.DataFrame(difference_breakdown)
                                    
                                    # Sort by cost descending
                                    diff_df = diff_df.sort_values('Cost', ascending=False)
                                    
                                    # Format cost for display
                                    diff_df_display = diff_df.copy()
                                    diff_df_display['Cost'] = diff_df_display['Cost'].apply(lambda x: f"${x:,.2f}")
                                    
                                    st.dataframe(diff_df_display, use_container_width=True, hide_index=True)
                                    
                                    # Summary by resource group
                                    st.markdown("**📊 Summary by Resource Group:**")
                                    rg_summary = diff_df.groupby('Located in RG')['Cost'].sum().sort_values(ascending=False)
                                    
                                    rg_summary_display = []
                                    for rg, cost in rg_summary.items():
                                        count = len(diff_df[diff_df['Located in RG'] == rg])
                                        rg_summary_display.append({
                                            'Resource Group': rg,
                                            'Additional Cost': f"${cost:,.2f}",
                                            'Resources': count
                                        })
                                    
                                    rg_summary_df = pd.DataFrame(rg_summary_display)
                                    st.dataframe(rg_summary_df, use_container_width=True, hide_index=True)
                                    
                                    # Verification
                                    total_additional = diff_df['Cost'].sum()
                                    st.info(f"💡 **Total Additional Cost**: ${total_additional:,.2f} (Expected difference: ${diff:,.2f})")
                                    
                                    if abs(total_additional - diff) < 0.01:
                                        st.success("✅ **Perfect Match**: Additional resources account for the full difference!")
                                    else:
                                        st.warning(f"⚠️ **Partial Match**: Additional resources (${total_additional:,.2f}) don't fully explain difference (${diff:,.2f})")
                                
                                else:
                                    st.info("No additional resources found in other resource groups.")
                                    
                            except Exception as e:
                                st.error(f"Error generating difference breakdown: {str(e)}")
                    else:
                        st.error(f"❌ **Unexpected Difference**: ${diff:,.2f}")
                        st.markdown("""
                        **In RG-Only Mode, these should match exactly.**
                        This indicates a calculation issue that needs investigation.
                        """)
                else:
                    st.success("✅ **Perfect Match**: Machine calculations are consistent with resource group totals.")

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

            # Defensive: Ensure '% of Machine' column exists
            if '% of Machine' not in machine_breakdown.columns:
                # Try to create it from Cost
                try:
                    if machine_cost > 0:
                        machine_breakdown['% of Machine'] = (machine_breakdown['Cost'] / machine_cost * 100).round(2)
                    else:
                        machine_breakdown['% of Machine'] = 0.0
                except Exception as e:
                    st.error(f"Could not calculate '% of Machine' column: {str(e)}")
                    return

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
                ['CostCategory', 'Cost', '% of Machine', 'Quantity', 'ConsumedService', 'MeterCategory']]
            display_breakdown.columns = ['Cost Category', 'Cost', '% of Machine', 'Quantity', 'Service Provider',
                                         'Meter Category']

            st.dataframe(display_breakdown, use_container_width=True, hide_index=True)

            # NEW: Resource-by-Resource Breakdown Table
            st.markdown("#### 🔍 Resource-by-Resource Breakdown")
            st.markdown("**All individual resources included in this machine's cost calculation:**")

            # Get the detailed resource breakdown
            try:
                # Use the same data source and method as the breakdown calculation
                if data.cost_analyzer:
                    df_classified = data.cost_analyzer.classify_costs()
                else:
                    df_classified = data.df.copy()

                df_classified = df_classified.dropna(subset=['ResourceName'])
                
                # Use the STANDARDIZED method to find related resources
                machine_resources = data._get_related_resources_for_machine(selected_machine, df_classified)
                
                # Get data for all related resources
                combined_data = df_classified[df_classified['ResourceName'].isin(machine_resources)]
                
                if not combined_data.empty:
                    # Show INDIVIDUAL records (not grouped) so each cost category is independent
                    resource_records = combined_data.copy()
                    
                    # Calculate percentage of each record's contribution to total machine cost
                    total_machine_cost_for_percentage = resource_records['Cost'].sum()
                    resource_records['Cost_Percentage'] = (resource_records['Cost'] / total_machine_cost_for_percentage * 100).round(2) if total_machine_cost_for_percentage > 0 else 0
                    
                    # Sort by cost descending
                    resource_records = resource_records.sort_values('Cost', ascending=False)
                    
                    # Add resource type classification
                    def classify_resource_type(resource_name):
                        if resource_name == selected_machine:
                            return "🖥️ Main Machine"
                        elif any(suffix in resource_name.lower() for suffix in ['_osdisk', 'osdisk']):
                            return "💾 OS Disk"
                        elif any(suffix in resource_name.lower() for suffix in ['_lun_', 'lun']):
                            return "💿 Data Disk"
                        elif any(suffix in resource_name.lower() for suffix in ['-nic', 'nic']):
                            return "🌐 Network Interface"
                        elif any(suffix in resource_name.lower() for suffix in ['-ip', 'publicip']):
                            return "📡 Public IP"
                        elif any(suffix in resource_name.lower() for suffix in ['-nsg', 'nsg']):
                            return "🛡️ Network Security"
                        else:
                            return "🔧 Other Resource"
                    
                    resource_records['Resource_Type'] = resource_records['ResourceName'].apply(classify_resource_type)
                    
                    # Format for display
                    display_resources = resource_records.copy()
                    display_resources['Cost'] = display_resources['Cost'].apply(lambda x: f"${x:,.2f}")
                    display_resources['Quantity'] = display_resources['Quantity'].apply(lambda x: f"{x:,.2f}")
                    display_resources['Cost_Percentage'] = display_resources['Cost_Percentage'].apply(lambda x: f"{x:.1f}%")
                    
                    # Truncate long text fields for better display
                    display_resources['MeterSubcategory'] = display_resources['MeterSubcategory'].apply(
                        lambda x: str(x)[:50] + "..." if len(str(x)) > 50 else str(x)
                    )
                    
                    # Add billing period info if available
                    if 'Date' in display_resources.columns:
                        display_resources['Date'] = pd.to_datetime(display_resources['Date']).dt.strftime('%Y-%m-%d')
                    
                    # Select and reorder columns for display
                    available_columns = ['Resource_Type', 'ResourceName', 'Cost', 'Cost_Percentage', 'Quantity',
                                       'CostCategory', 'ConsumedService', 'MeterCategory', 'MeterSubcategory']
                    
                    # Add Date if available
                    if 'Date' in display_resources.columns:
                        available_columns.insert(2, 'Date')
                    
                    # Only include columns that exist in the DataFrame
                    display_columns = [col for col in available_columns if col in display_resources.columns]
                    display_resources = display_resources[display_columns]
                    
                    # Set column names
                    column_names = {
                        'Resource_Type': 'Type',
                        'ResourceName': 'Resource Name',
                        'Cost': 'Cost',
                        'Cost_Percentage': '% of Total',
                        'Quantity': 'Quantity',
                        'CostCategory': 'Cost Category',
                        'ConsumedService': 'Service Provider',
                        'MeterCategory': 'Meter Category',
                        'MeterSubcategory': 'Meter Subcategory',
                        'Date': 'Date'
                    }
                    
                    display_resources.columns = [column_names.get(col, col) for col in display_resources.columns]
                    
                    # Show records with no artificial limit - user can see all records
                    max_records = st.slider("Number of records to display", min_value=10, max_value=len(resource_records), value=len(resource_records), step=5)
                    
                    st.dataframe(display_resources.head(max_records), use_container_width=True, hide_index=True)
                    
                    if len(resource_records) > max_records:
                        st.info(f"💡 Showing top {max_records} of {len(resource_records)} total records. Adjust the slider above to see more.")
                    
                    # Summary stats - calculate from unique resources for summary
                    unique_resources = resource_records['ResourceName'].nunique()
                    main_machine_cost = resource_records[resource_records['ResourceName'] == selected_machine]['Cost'].sum()
                    related_cost = total_machine_cost_for_percentage - main_machine_cost
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Records", len(resource_records))
                    with col2:
                        st.metric("Unique Resources", unique_resources)
                    with col3:
                        st.metric("Main Machine Cost", f"${main_machine_cost:,.2f}")
                    with col4:
                        st.metric("Total Machine Cost", f"${total_machine_cost_for_percentage:,.2f}")
                    
                    # Category breakdown summary
                    category_summary = resource_records.groupby('CostCategory')['Cost'].sum().sort_values(ascending=False)
                    st.markdown("#### 📊 Category Summary from Individual Records")
                    
                    category_display = []
                    for category, cost in category_summary.items():
                        record_count = len(resource_records[resource_records['CostCategory'] == category])
                        percentage = (cost / total_machine_cost_for_percentage * 100) if total_machine_cost_for_percentage > 0 else 0
                        category_display.append({
                            'Cost Category': category,
                            'Total Cost': f"${cost:,.2f}",
                            'Records': record_count,
                            'Percentage': f"{percentage:.1f}%"
                        })
                    
                    category_df = pd.DataFrame(category_display)
                    st.dataframe(category_df, use_container_width=True, hide_index=True)
                    
                else:
                    st.warning("No resource data found for this machine.")
                    
            except Exception as e:
                st.error(f"Error generating resource breakdown: {str(e)}")
                st.info("This might be due to data processing issues. Please check the debug section below for more details.")

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

                        st.markdown("**🔍 Detailed Machine Analysis (New Individual Records Logic):**")
                        
                        # Get the actual data using the SAME method as the new Resource-by-Resource breakdown
                        try:
                            if data.cost_analyzer:
                                df_for_analysis = data.cost_analyzer.classify_costs()
                            else:
                                df_for_analysis = data.df.copy()

                            df_for_analysis = df_for_analysis.dropna(subset=['ResourceName'])
                            
                            # Use the STANDARDIZED method to find related resources (same as new breakdown)
                            found_resources = data._get_related_resources_for_machine(selected_machine, df_for_analysis)
                            combined_records = df_for_analysis[df_for_analysis['ResourceName'].isin(found_resources)]
                            
                            st.write(f"**Machine name searched:** `{selected_machine}`")
                            st.write(f"**Resources found by standardized method:** {len(found_resources)}")
                            st.write(f"**Total individual records:** {len(combined_records)}")
                            
                            if not combined_records.empty:
                                # Show individual record breakdown (first 10 records)
                                st.markdown("**📋 Sample Individual Records (New Logic):**")
                                sample_records = combined_records.sort_values('Cost', ascending=False).head(10)
                                sample_display = []
                                
                                for _, record in sample_records.iterrows():
                                    sample_display.append({
                                        'Resource Name': record['ResourceName'],
                                        'Cost': f"${record['Cost']:,.2f}",
                                        'Category': record['CostCategory'],
                                        'Service': record['ConsumedService'],
                                        'Meter Category': record['MeterCategory'],
                                        'Quantity': f"{record['Quantity']:,.2f}"
                                    })
                                
                                sample_df = pd.DataFrame(sample_display)
                                st.dataframe(sample_df, use_container_width=True, hide_index=True)
                                
                                # Category summary from individual records
                                st.markdown("**📊 Category Summary (From Individual Records):**")
                                category_totals = combined_records.groupby('CostCategory').agg({
                                    'Cost': 'sum',
                                    'ResourceName': 'count'  # Count of individual records
                                }).round(2)
                                category_totals.columns = ['Total Cost', 'Record Count']
                                category_totals = category_totals.sort_values('Total Cost', ascending=False)
                                
                                category_display = []
                                total_all_categories = category_totals['Total Cost'].sum()
                                
                                for category, row in category_totals.iterrows():
                                    percentage = (row['Total Cost'] / total_all_categories * 100) if total_all_categories > 0 else 0
                                    category_display.append({
                                        'Cost Category': category,
                                        'Total Cost': f"${row['Total Cost']:,.2f}",
                                        'Records': int(row['Record Count']),
                                        'Percentage': f"{percentage:.1f}%"
                                    })
                                
                                category_summary_df = pd.DataFrame(category_display)
                                st.dataframe(category_summary_df, use_container_width=True, hide_index=True)
                                
                                # Validation against old method
                                total_from_individual_records = combined_records['Cost'].sum()
                                st.success(f"**✅ NEW METHOD TOTAL:** ${total_from_individual_records:,.2f}")
                                
                                # Compare with breakdown total
                                if debug_info.get('breakdown_by_category'):
                                    old_breakdown_total = sum([cat['Cost'] for cat in debug_info['breakdown_by_category']])
                                    diff = abs(total_from_individual_records - old_breakdown_total)
                                    if diff < 0.01:
                                        st.success(f"✅ **PERFECT CONSISTENCY**: Individual records total matches breakdown method (${old_breakdown_total:,.2f})")
                                    else:
                                        st.warning(f"⚠️ **DIFFERENCE**: Individual records (${total_from_individual_records:,.2f}) vs breakdown (${old_breakdown_total:,.2f}) = ${diff:,.2f}")
                                
                                # Resource-level summary
                                st.markdown("**🖥️ Resource-Level Summary:**")
                                resource_summary = combined_records.groupby('ResourceName').agg({
                                    'Cost': 'sum',
                                    'CostCategory': lambda x: len(x.unique()),  # Number of different categories per resource
                                    'ResourceName': 'count'  # Total records per resource
                                }).round(2)
                                resource_summary.columns = ['Total Cost', 'Categories', 'Records']
                                resource_summary = resource_summary.sort_values('Total Cost', ascending=False)
                                
                                # Show top 5 resources
                                top_resources = []
                                for resource, row in resource_summary.head(5).iterrows():
                                    resource_type = "🖥️ Main" if resource == selected_machine else "🔧 Related"
                                    top_resources.append({
                                        'Type': resource_type,
                                        'Resource Name': resource,
                                        'Total Cost': f"${row['Total Cost']:,.2f}",
                                        'Categories': int(row['Categories']),
                                        'Records': int(row['Records'])
                                    })
                                
                                resource_summary_df = pd.DataFrame(top_resources)
                                st.dataframe(resource_summary_df, use_container_width=True, hide_index=True)
                                
                                if len(found_resources) > 5:
                                    st.info(f"💡 Showing top 5 of {len(found_resources)} related resources")
                            
                            else:
                                st.warning("No individual records found for this machine")
                                
                        except Exception as e:
                            st.error(f"Error in new individual records analysis: {str(e)}")
                            
                            # Fallback to old method display
                            st.markdown("**⬇️ Fallback: Old Method Results**")
                        if debug_info.get('related_resources'):
                            related_df = pd.DataFrame(debug_info['related_resources'])
                            related_df['Cost'] = related_df['Cost'].apply(lambda x: f"${x:,.2f}")
                            st.dataframe(related_df, use_container_width=True, hide_index=True)
                            st.write(f"**Total from related resources:** ${debug_info.get('related_resources_total', 0):,.2f}")

                        st.markdown("**📊 Breakdown by Category (Old Method for Comparison):**")
                        if debug_info.get('breakdown_by_category'):
                            breakdown_debug_df = pd.DataFrame(debug_info['breakdown_by_category'])
                            breakdown_debug_df['Cost'] = breakdown_debug_df['Cost'].apply(lambda x: f"${x:,.2f}")
                            st.dataframe(breakdown_debug_df, use_container_width=True, hide_index=True)
                        else:
                            st.write("No breakdown data available")

                except Exception as e:
                    st.error(f"Debug error: {str(e)}")

        else:
            st.info("👆 Please select a machine from the dropdown to see its detailed cost breakdown by category.")

    def display_traditional_analysis(self, data: AzureInvoiceData):
        """Display traditional resource group and machine analysis."""
        st.header("🏗️ Resource Analysis")

        # Get traditional data - ensure fresh calculation
        st.markdown("🔄 **Calculating fresh data...**")

        # Clear any potential caching issues by forcing fresh calculation
        cost_by_rg = data.get_cost_by_resource_group(use_classified=True)
        cost_by_machine = data.get_cost_by_machine(include_related=True)

        # Verify data freshness
        if not cost_by_rg.empty:
            st.success(f"✅ Fresh data loaded: {len(cost_by_rg)} resource groups found")

        # Add consistency validation info
        st.info("📊 **Data Source**: Using **classified data** (same as Cost Category Analysis for consistency)")

        with st.expander("🔍 **Data Source Validation**", expanded=False):
            if data.cost_analyzer:
                # Compare raw vs classified totals
                raw_total = data.df['Cost'].sum()
                classified_df = data.cost_analyzer.classify_costs()
                classified_total = classified_df['Cost'].sum()

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Raw Data Total", f"${raw_total:,.2f}")
                with col2:
                    st.metric("Classified Data Total", f"${classified_total:,.2f}")
                with col3:
                    diff = abs(raw_total - classified_total)
                    st.metric("Difference", f"${diff:,.2f}",
                              delta="✅ Consistent" if diff < 0.01 else "❌ Error")

                st.success(
                    "✅ **This section uses CLASSIFIED data** - same source as Cost Categories, Service Providers, and Interactive Drill-Down")
            else:
                st.warning("⚠️ Cost analyzer not available - using raw data")

        if cost_by_rg.empty and cost_by_machine.empty:
            st.warning("No resource data available for analysis.")
            return

        # Add debug information section
        with st.expander("🔍 **Resource Group Calculation Details**", expanded=False):
            total_invoice = data.df['Cost'].sum()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Invoice Cost", f"${total_invoice:,.2f}")
                st.caption("Sum of all Cost values")

            with col2:
                rg_total = cost_by_rg.sum()
                st.metric("Sum of RG Costs", f"${rg_total:,.2f}")
                st.caption("After grouping by RG")

            with col3:
                nan_cost = data.df[data.df['ResourceGroup'].isna()]['Cost'].sum()
                st.metric("NaN RG Costs", f"${nan_cost:,.2f}")
                st.caption("Rows without RG")

            # Show top resource groups with validation
            st.markdown("**Top 5 Resource Groups:**")
            if not cost_by_rg.empty:
                debug_df = pd.DataFrame({
                    'Resource Group': cost_by_rg.head(5).index,
                    'Cost': [f"${x:,.2f}" for x in cost_by_rg.head(5).values],
                    '% of Total': [f"{(x / total_invoice * 100):.1f}%" for x in cost_by_rg.head(5).values],
                    'Status': ['✅ Normal' if x <= total_invoice else '❌ ERROR - Exceeds Total!' for x in
                               cost_by_rg.head(5).values]
                })
                st.dataframe(debug_df, use_container_width=True, hide_index=True)

                # Check for any impossible values
                max_rg_cost = cost_by_rg.max()
                max_rg_name = cost_by_rg.idxmax()

                if max_rg_cost > total_invoice:
                    st.error(
                        f"🚨 **CRITICAL ERROR**: Resource group '{max_rg_name}' shows ${max_rg_cost:,.2f} which exceeds total invoice ${total_invoice:,.2f}")
                    st.error(
                        "This indicates a serious data calculation issue. Please use the Force Refresh button in the sidebar.")

            # Validation check
            expected_total = rg_total + nan_cost
            diff = abs(total_invoice - expected_total)

            if diff < 0.01:
                st.success(f"✅ Costs reconcile correctly (difference: ${diff:,.2f})")
            else:
                st.error(f"❌ Cost reconciliation issue! Difference: ${diff:,.2f}")
                st.info("This could indicate data processing issues or duplicate entries.")

        # Add comprehensive manual calculation for PLG-PRD-GWC-RG01 specifically
        with st.expander("🔧 **MANUAL STEP-BY-STEP Calculation: PLG-PRD-GWC-RG01**", expanded=False):
            if 'PLG-PRD-GWC-RG01' in cost_by_rg.index:
                manual_calc = data.manual_calculate_resource_group('PLG-PRD-GWC-RG01')

                if manual_calc:
                    st.markdown("### 🎯 TRUTH vs MACHINE CALCULATION")

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.markdown("**📊 TRUTH (Traditional)**")
                        st.metric("Total Cost", f"${manual_calc['truth_total']:,.2f}")
                        st.caption("Simple sum of all costs in RG")

                    with col2:
                        st.markdown("**🧮 Manual Verification**")
                        st.metric("Total Cost", f"${manual_calc['manual_total']:,.2f}")
                        st.caption("Sum of individual resources")

                    with col3:
                        st.markdown("**🖥️ Machine Assignment**")
                        st.metric("Total Cost", f"${manual_calc['machine_total']:,.2f}")
                        st.caption("Sum of machine assignments")

                    with col4:
                        st.markdown("**⚠️ Unassigned**")
                        st.metric("Total Cost", f"${manual_calc['unassigned_total']:,.2f}")
                        st.caption("Resources not assigned to machines")

                    # Validation status
                    val = manual_calc['validation']
                    if val['truth_vs_manual'] and val['truth_vs_machine'] and val['all_assigned']:
                        st.success("✅ **PERFECT**: All calculations match and all resources assigned!")
                    else:
                        st.error("❌ **CALCULATION ERRORS DETECTED**")
                        if not val['truth_vs_manual']:
                            st.error(
                                f"🚨 Truth vs Manual mismatch: ${abs(manual_calc['truth_total'] - manual_calc['manual_total']):,.2f}")
                        if not val['truth_vs_machine']:
                            st.error(
                                f"🚨 Truth vs Machine mismatch: ${abs(manual_calc['truth_total'] - manual_calc['machine_total']):,.2f}")
                        if not val['all_assigned']:
                            st.error(
                                f"🚨 {manual_calc['unassigned_resources']} resources unassigned (${manual_calc['unassigned_total']:,.2f})")

                    # Show all resources
                    st.markdown("### 📋 ALL Resources in PLG-PRD-GWC-RG01")
                    if manual_calc['all_resources']:
                        resource_df = pd.DataFrame(manual_calc['all_resources'])
                        resource_df['Cost'] = resource_df['cost'].apply(lambda x: f"${x:,.2f}")
                        resource_df['Type'] = resource_df['is_infrastructure'].apply(
                            lambda x: "Infrastructure" if x else "Main Resource")
                        resource_df = resource_df[['resource', 'Cost', 'Type', 'records']]
                        resource_df.columns = ['Resource Name', 'Cost', 'Type', 'Records']
                        st.dataframe(resource_df.head(20), use_container_width=True, hide_index=True)

                        total_shown = sum([r['cost'] for r in manual_calc['all_resources'][:20]])
                        st.info(
                            f"💡 Showing top 20 of {manual_calc['total_resources']} resources. Total shown: ${total_shown:,.2f}")

                    # Show machine assignments
                    if manual_calc['machines']:
                        st.markdown("### 🖥️ Machine Assignments")
                        machine_data = []
                        for machine_name, machine_info in manual_calc['machines'].items():
                            machine_data.append({
                                'Machine': machine_name,
                                'Cost': f"${machine_info['cost']:,.2f}",
                                'Resources': machine_info['resource_count'],
                                'Resource List': ', '.join(machine_info['resources'][:3]) + (
                                    f' (+{len(machine_info["resources"]) - 3} more)' if len(
                                        machine_info['resources']) > 3 else '')
                            })

                        machine_df = pd.DataFrame(machine_data)
                        st.dataframe(machine_df, use_container_width=True, hide_index=True)

                        st.success(
                            f"✅ **CORRECT FORMULA**: Sum of machine costs = ${manual_calc['machine_total']:,.2f}")

                    # Show unassigned if any
                    if manual_calc['unassigned']:
                        st.markdown("### ⚠️ Unassigned Resources")
                        unassigned_df = pd.DataFrame(manual_calc['unassigned'])
                        unassigned_df['Cost'] = unassigned_df['cost'].apply(lambda x: f"${x:,.2f}")
                        unassigned_df = unassigned_df[['resource', 'Cost', 'records']]
                        unassigned_df.columns = ['Resource Name', 'Cost', 'Records']
                        st.dataframe(unassigned_df, use_container_width=True, hide_index=True)
                        st.warning("These resources could not be assigned to any machine and should be investigated.")
            else:
                st.warning("PLG-PRD-GWC-RG01 not found in cost_by_rg data")

        # Resource group analysis
        if not cost_by_rg.empty:
            # Add detailed validation before creating chart
            with st.expander("🔍 **Chart Data Validation**", expanded=False):
                st.markdown("**Data being passed to Cost by Resource Group chart:**")

                # Show the exact data going to the chart
                chart_data = []
                for rg, cost in cost_by_rg.head(10).items():
                    chart_data.append({
                        'Resource Group': rg,
                        'Cost': f"${cost:,.2f}",
                        'Raw Value': cost
                    })

                chart_validation_df = pd.DataFrame(chart_data)
                st.dataframe(chart_validation_df, use_container_width=True, hide_index=True)

                # Specifically check PLG-PRD-GWC-RG01
                if 'PLG-PRD-GWC-RG01' in cost_by_rg.index:
                    plg_cost = cost_by_rg['PLG-PRD-GWC-RG01']
                    st.success(f"✅ **PLG-PRD-GWC-RG01** in chart data: ${plg_cost:,.2f}")
                else:
                    st.error("❌ PLG-PRD-GWC-RG01 not found in chart data!")

                st.markdown("**Chart Creation Details:**")
                st.markdown(f"- Total RGs in data: {len(cost_by_rg)}")
                st.markdown(f"- Data type: {type(cost_by_rg)}")
                st.markdown(f"- Chart shows top: {self.TOP_ITEMS_COUNT} items")

            fig1 = self.chart_creator.create_cost_by_resource_group_chart(cost_by_rg)
            st.plotly_chart(fig1, use_container_width=True)

        # Top machines analysis
        if not cost_by_machine.empty:
            # Add slider to control number of machines displayed
            num_machines = st.slider(
                "",
                min_value=5,
                max_value=min(50, len(cost_by_machine)),
                value=min(15, len(cost_by_machine)),
                step=5,
                key="top_machines_slider"
            )
            st.write(f"Showing top {num_machines} of {len(cost_by_machine)} total machines")
            
            fig2 = self.chart_creator.create_top_machines_chart(cost_by_machine, num_machines)
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

                # Add debugging info for the table
                with st.expander("🔍 **Table Data Debug**", expanded=False):
                    if 'PLG-PRD-GWC-RG01' in cost_by_rg.index:
                        plg_cost_table = cost_by_rg['PLG-PRD-GWC-RG01']
                        st.success(f"✅ **PLG-PRD-GWC-RG01** in table: ${plg_cost_table:,.2f}")
                    else:
                        st.error("❌ PLG-PRD-GWC-RG01 not found in table data!")

                    st.markdown(f"**Table shows {len(df_display)} resource groups**")

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
