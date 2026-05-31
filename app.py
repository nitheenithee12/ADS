"""
=============================================================================
ADS Hackathon - Payer Policy Intelligence - Streamlit UI
=============================================================================
Displays extraction results with filters, feedback, and pipeline trigger.

Usage:
    streamlit run app.py
=============================================================================
"""

import os
import json
import time
import subprocess
import sys
from pathlib import Path

import streamlit as st
import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
RESULT_CSV = os.path.join(OUTPUT_DIR, "result_intermediate.csv")
RESULT_XLSX = os.path.join(OUTPUT_DIR, "result.xlsx")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
FEEDBACK_FILE = os.path.join(OUTPUT_DIR, "user_feedback.json")
PIPELINE_SCRIPT = os.path.join(BASE_DIR, "pipeline.py")

INPUT_DIR = os.path.join(BASE_DIR, "data", "input_pdfs")
BRANDS_FILE = os.path.join(BASE_DIR, "data", "brands.xlsx")

PARAMETERS = [
    "Age",
    "Step Therapy Requirements Documented in Policy",
    "Number of Steps through Brands",
    "Number of Steps through Generic",
    "Step through-Phototherapy",
    "TB Test required",
    "Quantity Limits",
    "Specialist Types",
    "Initial Authorization Duration(in-months)",
    "Reauthorization Duration(in-months)",
    "Reauthorization Required",
    "Reauthorization Requirements Documented in Policy",
    "Access Score",
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_results():
    """Load results from CSV/Excel."""
    if os.path.exists(RESULT_CSV):
        return pd.read_csv(RESULT_CSV)
    elif os.path.exists(RESULT_XLSX):
        return pd.read_excel(RESULT_XLSX)
    return None


def load_logs():
    """Load retrieval logs."""
    log_file = os.path.join(LOGS_DIR, "retrieval_logs.json")
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def load_judge_results():
    """Load LLM judge results."""
    judge_file = os.path.join(LOGS_DIR, "judge_results.json")
    if os.path.exists(judge_file):
        with open(judge_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_feedback():
    """Load user feedback."""
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_feedback(feedback_list):
    """Save user feedback."""
    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(feedback_list, f, indent=2, ensure_ascii=False)


def trigger_pipeline(input_dir, output_dir, brands_file, groq_key=""):
    """Trigger the extraction pipeline."""
    cmd = [
        sys.executable, PIPELINE_SCRIPT,
        "--input_dir", input_dir,
        "--output_dir", output_dir,
        "--brands_file", brands_file,
    ]
    if groq_key:
        cmd.extend(["--groq_key", groq_key])
    
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="Payer Policy Intelligence",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏥 Payer Policy Intelligence Dashboard")
    st.markdown("**Extract Access Quality Indicators from PA Policy Documents**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Check if results exist
        results_exist = os.path.exists(RESULT_CSV) or os.path.exists(RESULT_XLSX)
        
        if results_exist:
            st.success("✅ Results available")
        else:
            st.warning("⚠️ No results found. Run the pipeline first.")
        
        st.divider()
        
        # Pipeline trigger section
        st.header("🚀 Run Pipeline")
        
        input_dir = st.text_input("PDF Input Directory", INPUT_DIR)
        output_dir = st.text_input("Output Directory", OUTPUT_DIR)
        brands_file = st.text_input("Brands File", BRANDS_FILE)
        groq_key = st.text_input("Groq API Key (optional)", type="password")
        
        if st.button("▶️ Run Extraction Pipeline", type="primary"):
            with st.spinner("Running pipeline... This may take a while."):
                st.info("Pipeline started. Check terminal for detailed logs.")
                process = trigger_pipeline(input_dir, output_dir, brands_file, groq_key)
                
                # Show real-time output
                output_placeholder = st.empty()
                full_output = ""
                
                while True:
                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break
                    if line:
                        full_output += line
                        output_placeholder.code(full_output[-2000:])  # Show last 2000 chars
                
                if process.returncode == 0:
                    st.success("✅ Pipeline completed successfully!")
                    st.rerun()
                else:
                    stderr = process.stderr.read()
                    st.error(f"Pipeline failed:\n{stderr}")
        
        st.divider()
        st.header("📊 Stats")
        if results_exist:
            df = load_results()
            if df is not None:
                st.metric("Total Records", len(df))
                st.metric("Unique Files", df["Filename"].nunique())
                st.metric("Unique Brands", df["Brand"].nunique())
    
    # Main content
    if not results_exist:
        st.info("👆 Use the sidebar to run the pipeline, or place result.csv in the output directory.")
        
        # Show expected format
        st.subheader("Expected Output Format")
        sample_data = {
            "Filename": ["example.pdf"],
            "Brand": ["TREMFYA"],
            "Age": [">=18"],
            "Access Score": ["50"],
        }
        st.dataframe(pd.DataFrame(sample_data))
        return
    
    # Load data
    df = load_results()
    if df is None:
        st.error("Could not load results.")
        return
    
    # ---- TABS ----
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Results Table", "📊 Analytics", "🔍 Detailed View", "📝 Logs", "💬 Feedback"
    ])
    
    # ---- TAB 1: Results Table with Filters ----
    with tab1:
        st.subheader("Extraction Results")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            brand_filter = st.multiselect(
                "Filter by Brand",
                options=sorted(df["Brand"].unique().tolist()),
                default=[]
            )
        
        with col2:
            if "Access Score" in df.columns:
                score_range = st.slider(
                    "Access Score Range",
                    min_value=0, max_value=100,
                    value=(0, 100)
                )
        
        with col3:
            filename_search = st.text_input("Search Filename", "")
        
        # Apply filters
        filtered_df = df.copy()
        
        if brand_filter:
            filtered_df = filtered_df[filtered_df["Brand"].isin(brand_filter)]
        
        if filename_search:
            filtered_df = filtered_df[
                filtered_df["Filename"].str.contains(filename_search, case=False, na=False)
            ]
        
        if "Access Score" in filtered_df.columns:
            try:
                filtered_df["Access Score Num"] = pd.to_numeric(filtered_df["Access Score"], errors="coerce")
                filtered_df = filtered_df[
                    (filtered_df["Access Score Num"] >= score_range[0]) & 
                    (filtered_df["Access Score Num"] <= score_range[1]) |
                    filtered_df["Access Score Num"].isna()
                ]
                filtered_df = filtered_df.drop(columns=["Access Score Num"])
            except Exception:
                pass
        
        st.dataframe(
            filtered_df,
            use_container_width=True,
            height=500
        )
        
        # Download buttons
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            csv_data = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download CSV",
                csv_data,
                "result_filtered.csv",
                "text/csv"
            )
        with col_dl2:
            # Excel download
            from io import BytesIO
            excel_buffer = BytesIO()
            filtered_df.to_excel(excel_buffer, index=False)
            st.download_button(
                "📥 Download Excel",
                excel_buffer.getvalue(),
                "result_filtered.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    # ---- TAB 2: Analytics ----
    with tab2:
        st.subheader("Analytics & Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Access Score distribution
            if "Access Score" in df.columns:
                st.markdown("**Access Score Distribution**")
                scores = pd.to_numeric(df["Access Score"], errors="coerce").dropna()
                if not scores.empty:
                    score_counts = scores.value_counts().sort_index()
                    st.bar_chart(score_counts)
                    
                    st.metric("Mean Access Score", f"{scores.mean():.1f}")
                    st.metric("Median Access Score", f"{scores.median():.1f}")
        
        with col2:
            # Brand distribution
            st.markdown("**Records per Brand**")
            brand_counts = df["Brand"].value_counts()
            st.bar_chart(brand_counts)
        
        st.divider()
        
        # Parameter completeness
        st.markdown("**Parameter Completeness (% non-NA)**")
        completeness = {}
        for param in PARAMETERS:
            if param in df.columns:
                non_na = df[param].apply(
                    lambda x: str(x).strip().upper() not in ["NA", "N/A", "NAN", ""]
                ).sum()
                completeness[param] = (non_na / len(df)) * 100
        
        if completeness:
            comp_df = pd.DataFrame({
                "Parameter": list(completeness.keys()),
                "Completeness (%)": list(completeness.values())
            })
            st.dataframe(comp_df, use_container_width=True)
        
        # Step therapy analysis
        st.divider()
        st.markdown("**Step Therapy Analysis**")
        
        col3, col4 = st.columns(2)
        with col3:
            if "Number of Steps through Brands" in df.columns:
                st.markdown("Steps through Brands")
                steps_b = pd.to_numeric(df["Number of Steps through Brands"], errors="coerce").dropna()
                if not steps_b.empty:
                    st.bar_chart(steps_b.value_counts().sort_index())
        
        with col4:
            if "Number of Steps through Generic" in df.columns:
                st.markdown("Steps through Generic")
                steps_g = pd.to_numeric(df["Number of Steps through Generic"], errors="coerce").dropna()
                if not steps_g.empty:
                    st.bar_chart(steps_g.value_counts().sort_index())
    
    # ---- TAB 3: Detailed View ----
    with tab3:
        st.subheader("Detailed Record View")
        
        # Select specific record
        record_options = [f"{row['Filename']} | {row['Brand']}" for _, row in df.iterrows()]
        selected = st.selectbox("Select Record", record_options)
        
        if selected:
            filename, brand = selected.split(" | ")
            record = df[(df["Filename"] == filename) & (df["Brand"] == brand)].iloc[0]
            
            st.markdown(f"### {brand} — {filename}")
            
            # Display as cards
            col1, col2 = st.columns(2)
            
            params_left = PARAMETERS[:len(PARAMETERS)//2]
            params_right = PARAMETERS[len(PARAMETERS)//2:]
            
            with col1:
                for param in params_left:
                    if param in record:
                        value = str(record[param])
                        if value.upper() == "NA":
                            st.markdown(f"**{param}:** ⚪ {value}")
                        elif param == "Access Score":
                            score = int(float(value)) if value.replace('.', '').isdigit() else 0
                            color = "🟢" if score >= 75 else "🟡" if score >= 50 else "🟠" if score >= 25 else "🔴"
                            st.markdown(f"**{param}:** {color} {value}")
                        else:
                            st.markdown(f"**{param}:** {value}")
            
            with col2:
                for param in params_right:
                    if param in record:
                        value = str(record[param])
                        if value.upper() == "NA":
                            st.markdown(f"**{param}:** ⚪ {value}")
                        else:
                            st.markdown(f"**{param}:** {value}")
    
    # ---- TAB 4: Logs ----
    with tab4:
        st.subheader("Pipeline Logs")
        
        # Judge results
        judge_data = load_judge_results()
        if judge_data:
            st.markdown("**LLM Judge Summary**")
            st.metric("Total Judgments", judge_data.get("total", 0))
            st.metric("Average Score", f"{judge_data.get('avg_score', 0):.2f}/5")
        
        # Retrieval logs
        logs = load_logs()
        if logs:
            st.markdown(f"**Retrieval Logs ({len(logs)} entries)**")
            
            # Filter logs
            log_param_filter = st.selectbox(
                "Filter by Parameter",
                ["All"] + PARAMETERS
            )
            
            filtered_logs = logs
            if log_param_filter != "All":
                filtered_logs = [l for l in logs if l.get("parameter") == log_param_filter]
            
            # Display as table
            if filtered_logs:
                log_df = pd.DataFrame(filtered_logs)
                display_cols = ["filename", "brand", "parameter", "extracted_value", "num_chunks", "timestamp"]
                display_cols = [c for c in display_cols if c in log_df.columns]
                st.dataframe(log_df[display_cols], use_container_width=True, height=400)
        else:
            st.info("No logs available. Run the pipeline first.")
        
        # Pipeline log file
        pipeline_log = os.path.join(LOGS_DIR, "pipeline.log")
        if os.path.exists(pipeline_log):
            with st.expander("📄 Full Pipeline Log"):
                with open(pipeline_log, "r", encoding="utf-8") as f:
                    st.code(f.read()[-5000:], language="text")  # Last 5000 chars
    
    # ---- TAB 5: Feedback ----
    with tab5:
        st.subheader("User Feedback")
        st.markdown("Rate extraction quality for each record. Your feedback helps improve the pipeline.")
        
        # Load existing feedback
        existing_feedback = load_feedback()
        
        # Record selection for feedback
        record_options_fb = [f"{row['Filename']} | {row['Brand']}" for _, row in df.iterrows()]
        selected_fb = st.selectbox("Select Record for Feedback", record_options_fb, key="fb_select")
        
        if selected_fb:
            filename_fb, brand_fb = selected_fb.split(" | ")
            record_fb = df[(df["Filename"] == filename_fb) & (df["Brand"] == brand_fb)].iloc[0]
            
            # Show record
            st.markdown(f"**{brand_fb}** — {filename_fb}")
            
            # Compact display
            for param in PARAMETERS:
                if param in record_fb:
                    st.text(f"{param}: {record_fb[param]}")
            
            st.divider()
            
            # Feedback form
            col_fb1, col_fb2 = st.columns(2)
            
            with col_fb1:
                vote = st.radio(
                    "Overall Quality",
                    ["👍 Good", "👎 Needs Improvement", "❓ Uncertain"],
                    horizontal=True
                )
            
            with col_fb2:
                param_issues = st.multiselect(
                    "Parameters with Issues",
                    PARAMETERS
                )
            
            comment = st.text_area("Comments / Corrections", placeholder="Optional: Describe issues or provide correct values...")
            
            if st.button("Submit Feedback", type="primary"):
                feedback_entry = {
                    "filename": filename_fb,
                    "brand": brand_fb,
                    "vote": vote,
                    "parameters_with_issues": param_issues,
                    "comment": comment,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                existing_feedback.append(feedback_entry)
                save_feedback(existing_feedback)
                st.success("✅ Feedback submitted! Thank you.")
        
        # Show existing feedback
        if existing_feedback:
            st.divider()
            st.markdown(f"**Previous Feedback ({len(existing_feedback)} entries)**")
            fb_df = pd.DataFrame(existing_feedback)
            st.dataframe(fb_df, use_container_width=True)


if __name__ == "__main__":
    main()
