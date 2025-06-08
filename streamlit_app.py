import streamlit as st
import pandas as pd
from typing import Optional

# Import specialized analysis modules
from complex_analysis import AzureInvoiceData, ComplexDashboard
from simple_analysis import SimpleInvoiceData, SimpleDashboard

# Page configuration
st.set_page_config(
    page_title="Azure Invoice Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stMetric > label {
        font-size: 18px !important;
        font-weight: bold !important;
    }
    .st-emotion-cache-10trblm {
        font-size: 16px;
    }
    .main-header {
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
    }
    .template-selector {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


class MainDashboard:
    """Main dashboard that handles template selection and file upload."""
    
    def __init__(self):
        self.complex_dashboard = ComplexDashboard()
        self.simple_dashboard = SimpleDashboard()
    
    def display_header(self):
        """Display the main application header."""
        st.markdown("""
        <div class="main-header">
            <h1>📊 Azure Invoice Analyzer Pro</h1>
            <p>Comprehensive Azure cost analysis with advanced insights</p>
        </div>
        """, unsafe_allow_html=True)
    
    def display_template_selection(self):
        """Display template selection and return selected template."""
        st.markdown("### 🎯 **Choose Analysis Template**")
        
        template = st.radio(
            "**Select Template:**",
            options=["Complex (Advanced Azure Invoice)", "Simple (Basic Service Usage)"],
            index=0,
            horizontal=True,
            help="Choose the analysis template that matches your CSV format"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        return template
    
    def display_file_upload(self, template: str):
        """Display file upload section and return uploaded data."""
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload your Azure invoice CSV file for analysis"
        )


        if "Complex" in template:
            st.info("""
            **Required columns for Complex template:**
            Date, SubscriptionName, SubscriptionGuid, ResourceGroup, ResourceName, 
            MeterCategory, MeterSubcategory, ConsumedService, Cost, Quantity
            """)
        else:
            st.info("""
            **Required columns for Simple template:**
            SubscriptionName, SubscriptionGuid, Date, ResourceGuid, ServiceName, 
            ServiceType, ServiceRegion, ServiceResource, Quantity, Cost
            """)
        

        if uploaded_file is not None:
            try:
                with st.spinner("🔄 Loading and processing CSV file..."):
                    # Load the CSV
                    df = pd.read_csv(uploaded_file, low_memory=False)
                    
                    # Convert Date column 
                    if 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                    
                    st.success(f"✅ File loaded successfully! {len(df):,} records found.")
                    
                    # Validate columns based on template
                    if "Complex" in template:
                        required_cols = ['Date', 'SubscriptionName', 'ResourceGroup', 'Cost', 'Quantity']
                        missing_cols = [col for col in required_cols if col not in df.columns]
                        
                        if missing_cols:
                            st.error(f"❌ Missing required columns for Complex template: {missing_cols}")
                            st.info("💡 Try using the Simple template or check your CSV format.")
                            return None
                        
                        # Create complex data object
                        return AzureInvoiceData(df)
                    
                    else:  # Simple template
                        required_cols = ['SubscriptionName', 'Date', 'ServiceName', 'Cost', 'Quantity']
                        missing_cols = [col for col in required_cols if col not in df.columns]
                        
                        if missing_cols:
                            st.error(f"❌ Missing required columns for Simple template: {missing_cols}")
                            st.info("💡 Try using the Complex template or check your CSV format.")
                            return None
                        
                        # Create simple data object
                        return SimpleInvoiceData(df)
                        
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
                st.info("💡 Please check that your file is a valid CSV format.")
                return None
        
        return None
    
    def display_welcome_message(self):
        """Display welcome message when no file is uploaded."""
        st.markdown("### 👋 Welcome to Azure Invoice Analyzer!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🎯 What This Tool Does:**
            - 📊 Analyzes Azure invoice CSV files
            - 💰 Breaks down costs by categories
            - 🏗️ Shows resource group usage
            - ⚡ Calculates efficiency metrics
            - 📈 Creates interactive visualizations
            """)
        
        with col2:
            st.markdown("""
            **🚀 How to Get Started:**
            1. Choose your analysis template above
            2. Upload your Azure invoice CSV file
            3. Explore the generated insights
            4. Export or share your analysis
            """)
        
        st.markdown("---")
        
        # Sample data info
        with st.expander("📋 **Sample Data Format**", expanded=False):
            st.markdown("""
            **Complex Template Sample:**
            ```csv
            Date,SubscriptionName,ResourceGroup,ResourceName,MeterCategory,ConsumedService,Cost,Quantity
            2024-01-01,MySubscription,RG-Production,VM-WebServer,Virtual Machines,Microsoft.Compute,100.50,24
            ```
            
            **Simple Template Sample:**
            ```csv
            SubscriptionName,Date,ServiceName,ServiceType,ServiceRegion,ServiceResource,Quantity,Cost
            MySubscription,2024-01-01,Virtual Machines,Compute,East US,VM-001,24,100.50
            ```
            """)
    
    def run(self):
        """Main application entry point."""
        # Display header
        self.display_header()
        
        # Template selection
        template = self.display_template_selection()
        
        # File upload and processing
        data = self.display_file_upload(template)
        
        if data is not None:
            
            # Route to appropriate dashboard based on template
            if "Complex" in template:
                # Complex analysis pipeline
                st.info("🔧 **Complex Template Active** - Advanced Azure invoice analysis")
                
                # Run complex analysis
                with st.container():
                    # Enhanced summary
                    self.complex_dashboard.display_enhanced_summary(data)
                    st.divider()

                    # Cost category analysis
                    self.complex_dashboard.display_cost_category_analysis(data)
                    st.divider()

                    # Service provider analysis
                    self.complex_dashboard.display_service_provider_analysis(data)
                    st.divider()

                    # Efficiency analysis
                    self.complex_dashboard.display_efficiency_analysis(data)
                    st.divider()

                    # Interactive drill-down
                    self.complex_dashboard.display_interactive_drill_down(data)
                    st.divider()

                    # Traditional analysis
                    self.complex_dashboard.display_traditional_analysis(data)
                    st.divider()

                    # Uncategorized analysis
                    self.complex_dashboard.display_uncategorized_analysis(data)
                    st.divider()

                    # Detailed tables
                    self.complex_dashboard.display_detailed_tables(data)
                    
                    st.success("✅ Complex analysis complete! All cost categories, efficiency metrics, and resource breakdowns calculated.")
            
            else:
                # Simple analysis pipeline
                self.simple_dashboard.run_simple_analysis(data)
        
        else:
            # Show welcome message
            self.display_welcome_message()


def main():
    """Main application function."""
    dashboard = MainDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
