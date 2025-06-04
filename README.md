# Azure Invoice Analyzer

A comprehensive Streamlit application for analyzing Azure billing data with interactive visualizations and detailed cost breakdowns.

## Features

- 📊 Interactive cost and usage visualizations
- 🏗️ Resource group breakdown analysis
- 💰 Top machines by cost analysis
- 📈 Cost vs usage correlation analysis
- 📥 Data export capabilities
- 🎨 Customizable chart options
- 📱 Responsive design

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

## Deployment to Streamlit Community Cloud

### Prerequisites

- GitHub account
- Streamlit Community Cloud account (free at share.streamlit.io)

### Deployment Steps

1. **Create GitHub Repository:**

   - Create a new public repository on GitHub
   - Upload these files:
     - `streamlit_app.py` (main application)
     - `requirements.txt` (dependencies)
     - `.streamlit/config.toml` (configuration)
     - `README.md` (documentation)

2. **Deploy on Streamlit Cloud:**

   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository
   - Set main file path to `streamlit_app.py`
   - Click "Deploy!"

3. **Your app will be live at:**
   `https://[your-app-name].streamlit.app`

## CSV File Format

Your Azure invoice CSV should contain these columns:

| Column        | Description           | Example      |
| ------------- | --------------------- | ------------ |
| Date          | Invoice date          | 2024-01-01   |
| Cost          | Cost in USD           | 150.50       |
| Quantity      | Usage hours           | 24.5         |
| ResourceGroup | Azure resource group  | Production   |
| ResourceName  | Machine/resource name | WebServer-01 |

## Usage Instructions

1. **Upload CSV File:**

   - Click "Choose your Azure Invoice CSV file"
   - Select your Azure billing CSV file
   - Wait for processing to complete

2. **Explore Analysis:**

   - View data summary metrics
   - Analyze cost breakdowns by resource group
   - Examine top machines by cost
   - Review usage vs cost correlations

3. **Customize Views:**

   - Use sidebar controls to adjust chart height
   - Apply cost filters
   - Toggle chart value displays

4. **Export Data:**
   - Use sidebar export options
   - Download processed data as CSV
   - Save charts as images

## Features in Detail

### Interactive Visualizations

- **Cost by Resource Group:** Bar chart showing total spending per group
- **Top Machines by Cost:** Identifies your most expensive resources
- **Cost vs Usage Scatter:** Shows correlation between usage and cost
- **Resource Group Breakdown:** Detailed analysis within each group

### Data Processing

- Automatic data cleaning and validation
- Numeric conversion for cost and quantity fields
- Date parsing for time-based analysis
- Error handling for malformed data

### User Experience

- Clean, professional interface
- Responsive design for mobile and desktop
- Loading indicators and progress feedback
- Helpful error messages and guidance
