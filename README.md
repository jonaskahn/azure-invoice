# Azure Invoice Analyzer

A comprehensive Streamlit application for analyzing Azure billing data with interactive visualizations, detailed cost breakdowns, and PDF export capabilities.

## Features

- 📊 Interactive cost and usage visualizations
- 🏗️ Resource group breakdown analysis
- 💰 Top machines by cost analysis (configurable 5-100 machines)
- 📈 Cost vs usage correlation analysis
- 🖨️ **PDF Export & Print functionality**
- 📥 Individual chart download capabilities
- 🎨 Customizable chart options
- 📱 Responsive design with print optimization

## PDF Export Features

### 🖨️ Full Report Export
- **One-click PDF export** of entire analysis
- **Professional formatting** optimized for print
- **Executive summary table** included in PDF
- **Date stamp** and report headers
- **Clean layout** with proper page breaks

### 📊 Individual Chart Downloads
- **High-quality PNG exports** (1200px width, 2x scale)
- **SVG format** available for vector graphics
- **Customizable filenames** for organization
- **Download directly from chart toolbar**

### 📋 Export Options Available:
1. **🖨️ Print/Export PDF** - Full page as PDF via browser
2. **📊 Download All Charts** - Instructions for individual charts
3. **Right-click downloads** - PNG/SVG from any chart

## Local Development

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd azure-invoice-analyzer
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`

## PDF Export Instructions

### Method 1: Full Report PDF Export
1. Click **"🖨️ Print / Export PDF"** in the sidebar
2. Browser print dialog will open
3. Choose **"Save as PDF"** as destination
4. Select **"More settings"** for layout options:
   - **Layout**: Portrait
   - **Pages**: All
   - **Options**: Check "Background graphics"
5. Click **"Save"** to download PDF

### Method 2: Individual Chart Downloads
1. Hover over any chart
2. Click the **camera icon** in the chart toolbar
3. Choose format (PNG recommended for reports)
4. File downloads automatically with descriptive name

### Method 3: High-Quality Chart Exports
1. Click **"📊 Download All Charts"** for instructions
2. Right-click any chart → **"Download plot as a png"**
3. Charts export at 1200px width for presentation quality

## Chart Configuration for PDF

### Optimized Settings:
- **Chart Height**: 400-600px recommended for PDF
- **Show Top N Machines**: 10-15 for clean PDF layout
- **Font sizes**: Automatically adjusted for print
- **Margins**: Enhanced spacing for professional appearance

## Business Use Cases

### Executive Reporting
- **Monthly cost reviews** with stakeholders
- **Board presentations** with PDF exports
- **Department cost allocation** reports
- **Budget planning** documentation

### Technical Analysis
- **Resource optimization** planning
- **Cost center** breakdowns
- **Usage efficiency** analysis
- **Capacity planning** insights