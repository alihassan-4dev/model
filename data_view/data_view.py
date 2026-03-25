"""
Streamlit app to view article/abstract data from data/ folder.
Run with: streamlit run data_view/data_view.py
"""

import streamlit as st
import pandas as pd
from pathlib import Path

# Data folder at project root (script lives in data_view/data_view.py)
_DATA_VIEW_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _DATA_VIEW_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

# Max rows to load for large files (train.csv is ~2GB)
MAX_ROWS_SAMPLE = 10_000
CHUNK_SIZE = 50_000


def get_data_files():
    """List available CSV files in data/."""
    if not DATA_DIR.exists():
        return []
    return sorted(f.name for f in DATA_DIR.glob("*.csv"))


def load_data(filename: str, max_rows: int | None = None) -> pd.DataFrame:
    """Load CSV with optional row limit for large files."""
    path = DATA_DIR / filename
    if not path.exists():
        return pd.DataFrame()

    try:
        # For very large files, read in chunks and take first max_rows
        if max_rows:
            return pd.read_csv(path, nrows=max_rows, on_bad_lines="skip")
        return pd.read_csv(path, on_bad_lines="skip")
    except Exception as e:
        st.error(f"Error loading {filename}: {e}")
        return pd.DataFrame()


def main():
    st.set_page_config(page_title="Data View", page_icon="📊", layout="wide")
    st.title("📊 Data view")
    st.caption("View article/abstract datasets from the `data/` folder.")

    files = get_data_files()
    if not files:
        st.warning("No CSV files found in `data/` folder.")
        return

    selected = st.sidebar.selectbox("Dataset", files, key="dataset")
    use_sample = st.sidebar.checkbox(
        "Limit rows (for large files)",
        value=True,
        help="Cap loaded rows to avoid memory issues on train/test.",
    )
    sample_size = (
        st.sidebar.number_input("Max rows", min_value=100, max_value=100_000, value=5000)
        if use_sample
        else None
    )

    with st.spinner("Loading data…"):
        df = load_data(selected, max_rows=sample_size)

    if df.empty:
        st.warning("No data loaded.")
        return

    st.sidebar.metric("Rows loaded", len(df))
    st.sidebar.metric("Columns", len(df.columns))

    st.subheader(f"`{selected}`")
    st.dataframe(df, width="stretch", height=400)

    # Column details
    st.subheader("Columns")
    for col in df.columns:
        with st.expander(f"**{col}** — dtype: {df[col].dtype}, non-null: {df[col].notna().sum()}"):
            sample_vals = df[col].dropna().head(3).tolist()
            for i, v in enumerate(sample_vals):
                text = str(v)
                if len(text) > 500:
                    text = text[:500] + "..."
                st.text(text)

    # Optional: show single row detail
    st.subheader("Row detail")
    row_idx = st.number_input("Row index", min_value=0, max_value=max(0, len(df) - 1), value=0, key="row_idx")
    if 0 <= row_idx < len(df):
        row = df.iloc[row_idx]
        for col in df.columns:
            val = row[col]
            st.markdown(f"**{col}**")
            if pd.isna(val):
                st.text("(null)")
            else:
                st.text_area(
                    label=col,
                    value=str(val),
                    height=120,
                    key=f"row_{col}_{row_idx}",
                    disabled=True,
                    label_visibility="collapsed",
                )

    st.divider()
    st.caption("Run with: `streamlit run data_view/data_view.py`")


if __name__ == "__main__":
    main()
