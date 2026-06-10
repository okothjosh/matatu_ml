# Multi-Page Streamlit App Structure

## Overview

This document explains the new multi-page structure of the Matatu ML dashboard.

## File Organization

```
matatu_ml/
├── app.py                          # Main entry point (home page)
├── pages/                          # Streamlit multi-page directory
│   ├── 01_home.py                 # Landing/welcome page
│   ├── 02_data_insights.py         # Data exploration page
│   ├── 03_model_performance.py     # Model comparison page
│   ├── 04_predictions.py           # Predictions display page
│   ├── 05_surge_analysis.py        # Surge pricing analysis page
│   └── 06_about.py                 # Project information page
├── utils/                          # Utility modules
│   └── data_loader.py              # Shared data loading functions
├── requirements.txt                # Python dependencies
├── .streamlit/
│   └── config.toml                 # Streamlit configuration
└── README.md                       # Project documentation
```

## How It Works

### Streamlit's Multi-Page Feature

Streamlit automatically:
1. Detects files in the `pages/` folder
2. Creates navigation buttons for each page
3. Files are named with a number prefix (01_, 02_, etc.) for ordering
4. Page name comes from the file name or `st.set_page_config(page_title=...)`

### Navigation Flow

```
User opens app → Streamlit detects pages/ → Shows sidebar menu
                            ↓
                    User clicks page
                            ↓
                  Page loads and imports data
                            ↓
                   Page renders content
```

## File Descriptions

### `app.py` (Main Entry Point)
- **Purpose**: Displays the home/landing page
- **Content**: Welcome message, project overview, quick links
- **Data Loading**: Minimal (just configuration)
- **Size**: ~80 lines

### `pages/01_home.py`
- **Purpose**: Detailed landing page with feature boxes
- **Content**: Project overview, key achievements, getting started guide
- **Data Loading**: None
- **Size**: ~80 lines

### `pages/02_data_insights.py`
- **Purpose**: Data exploration and statistics
- **Content**: Dataset overview, sample data, statistics, data types, missing values
- **Data Loading**: `load_all_data()` from utils
- **Features**: Expandable sections, data type information
- **Size**: ~100 lines

### `pages/03_model_performance.py`
- **Purpose**: Model comparison and evaluation
- **Content**: Model descriptions, comparison results, visualizations
- **Data Loading**: `ab_comparison.csv`
- **Features**: Bar charts, model insights, development process explanation
- **Size**: ~120 lines

### `pages/04_predictions.py`
- **Purpose**: Display model predictions
- **Content**: Prediction table, statistics, download button
- **Data Loading**: `predictions_test.csv`
- **Features**: CSV download, metric cards, analysis guide
- **Size**: ~110 lines

### `pages/05_surge_analysis.py`
- **Purpose**: Surge pricing analysis and opportunities
- **Content**: Top surge routes, multiplier distribution, simulation results
- **Data Loading**: `surge_ranking`, `surge_multipliers`, `simulation`
- **Features**: Horizontal bar charts, histograms, scatter plots
- **Size**: ~130 lines

### `pages/06_about.py`
- **Purpose**: Project information and metadata
- **Content**: Overview, dataset, objectives, structure, author, links
- **Data Loading**: None
- **Features**: Two-column layout, expandable sections, contact info
- **Size**: ~150 lines

### `utils/data_loader.py`
- **Purpose**: Centralized data loading
- **Functions**:
  - `load_all_data()` - Loads all CSV files with caching
  - `check_data_available()` - Checks if data exists and is not empty
- **Benefits**: 
  - Shared across all pages
  - Cached to avoid reloading
  - Error handling built-in
- **Size**: ~50 lines

## Key Benefits

### 1. **Code Organization**
- Each page is independent and focused
- Easier to maintain and debug
- Clear separation of concerns

### 2. **Reusability**
- `utils/data_loader.py` used by all pages
- Consistent data handling
- DRY (Don't Repeat Yourself) principle

### 3. **Performance**
- Streamlit's caching in `data_loader.py` prevents reloading
- Pages load only when clicked
- Reduced memory footprint

### 4. **Scalability**
- Easy to add new pages (just create new file in `pages/`)
- No need to modify main app.py
- Supports unlimited pages

### 5. **User Experience**
- Clean navigation sidebar
- Professional appearance
- Faster page transitions

## Development Workflow

### Adding a New Page

1. Create file: `pages/07_your_page.py`
2. Add page configuration:
   ```python
   import streamlit as st
   st.set_page_config(
       page_title="Your Page Name",
       page_icon="🎯",
       layout="wide"
   )
   ```
3. Add content
4. Commit and push to GitHub
5. Streamlit Cloud auto-deploys

### Modifying Existing Pages

1. Edit the page file
2. Save locally (Streamlit auto-reloads)
3. Test thoroughly
4. Commit and push
5. Auto-deployed by Streamlit Cloud

## Caching Strategy

All data loading uses `@st.cache_data` decorator:

```python
@st.cache_data
def load_all_data():
    # Data is loaded once and cached
    # All pages share the same cached data
    # Cache is cleared when file changes or after 1 hour
```

**Benefits:**
- First page load: ~2 seconds
- Subsequent pages: ~100ms
- No redundant file I/O

## Navigation Structure

### Sidebar Menu (Auto-generated)
```
🚌 Matatu ML Dashboard
├── 🏠 Home                  (app.py)
├── 🏠 Home Details          (pages/01_home.py)
├── 📊 Data Insights         (pages/02_data_insights.py)
├── 🤖 Model Performance     (pages/03_model_performance.py)
├── 🔮 Predictions           (pages/04_predictions.py)
├── ⚡ Surge Analysis         (pages/05_surge_analysis.py)
└── ℹ️ About                  (pages/06_about.py)
```

## Deployment

With Streamlit Cloud:
1. Push code to GitHub
2. Streamlit Cloud detects new pages
3. Auto-deploys in ~2 minutes
4. All pages available at public URL

## Troubleshooting

### Page Not Showing Up
- Ensure file is in `pages/` folder
- Check filename starts with number (01_, 02_, etc.)
- Restart Streamlit: `Ctrl+C` then `streamlit run app.py`

### Data Not Loading
- Check file paths are relative (not absolute)
- Verify CSV files exist in root directory
- Check `utils/data_loader.py` for errors

### Performance Issues
- Check caching is enabled in `data_loader.py`
- Avoid loading large files multiple times
- Use `st.cache_data` for expensive operations

## Future Enhancements

- [ ] Add user authentication
- [ ] Database integration
- [ ] Real-time data updates
- [ ] Export to PDF reports
- [ ] Email alerts
- [ ] API endpoints
