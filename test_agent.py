import streamlit as st
from PIL import Image
import io
import google.generativeai as genai
import pandas as pd
import re

# Import libraries for new features
import streamlit_cookies_manager
from st_copy_button import st_copy_button

# Page Configuration
st.set_page_config(layout="wide", page_title="Professional AI Test Case Generator")

# Initialize Cookie Manager
cookies = streamlit_cookies_manager.CookieManager()

# Session State Initialization
if 'history' not in st.session_state:
    st.session_state.history = []
if 'last_response' not in st.session_state:
    st.session_state.last_response = ""
if 'run_id' not in st.session_state:
    st.session_state.run_id = 0

# Helper Function to Parse Markdown
def parse_markdown_to_df(markdown_text):
    table_match = re.search(r'(\|.*\|(\n|\r\n)?)+(?![^|])', markdown_text)
    if not table_match:
        return pd.DataFrame()
    table_str = table_match.group(0)
    lines = table_str.strip().split('\n')
    headers = [h.strip() for h in lines[0].strip('|').split('|')]
    data = []
    for line in lines[2:]:
        if '|' in line:
            cells = [c.strip() for c in line.strip('|').split('|')]
            if len(cells) == len(headers):
                data.append(cells)
    df = pd.DataFrame(data, columns=headers)
    return df

# The Master Prompt
MASTER_PROMPT = """
You are an expert Senior QA Engineer specializing in test documentation for enterprise applications. Your task is to analyze the provided UI screenshot(s) and a user-provided description to create a detailed test case suite.
**CONTEXT:**
The user has provided a short description of this module: "{user_description}"
**YOUR TASK:**
Analyze the attached image(s) carefully. Identify all UI elements, user flows, and potential edge cases. Based on this analysis, generate the following documentation.
**OUTPUT FORMAT:**
**1. Module Description:**
Write a concise paragraph describing the purpose and main functionality of this UI module based on the image(s) and user context.
**2. Functional Test Cases:**
Create a detailed table of functional test cases in Markdown format. The table MUST have the following columns EXACTLY:
`Test Case ID`, `Module Name`, `Test Case Title`, `Test Steps`, `Expected Result`, `Severity`, `Priority`, `Actual Result`, `Status (Pass/Fail)`
**Instructions for each column:**
- **Test Case ID:** Generate a unique ID starting with "TC-" followed by the module name and a number (e.g., TC-LOGIN-001).
- **Module Name:** Use a short, relevant name based on the user's description (e.g., "Login", "User Profile").
- **Test Case Title:** Write a short, clear summary of the test's objective (e.g., "Verify successful login with valid credentials").
- **Test Steps:** Provide clear, numbered, step-by-step instructions for the tester to follow.
- **Expected Result:** Describe the specific, observable outcome that should occur if the test passes.
- **Severity:** Assign a severity level from: **Critical, High, Medium, Low**.
- **Priority:** Assign a priority level from: **High, Medium, Low**.
- **Actual Result:** Leave this column **EMPTY**.
- **Status (Pass/Fail):** Leave this column **EMPTY**.
Generate a comprehensive list of test cases, including positive tests (happy paths), negative tests (error conditions, invalid data), and UI/usability tests.
"""

# Function to reset the state
def start_new_generation():
    st.session_state.last_response = ""
    st.session_state.run_id += 1

# Main App Interface
st.title("🤖 Professional AI Test Case Generator")
st.write("Your one-stop tool to generate, edit, copy, and download test suites.")

# --- Sidebar for Configuration and History ---
st.sidebar.header("Configuration & History")

# Default value in case cookies aren't ready on the first pass
api_key_from_cookie = ""
# Check IF the cookies are ready
if cookies.ready():
    api_key_from_cookie = cookies.get('api_key')

# Now, create the text input widget.
api_key = st.sidebar.text_input(
    "Enter your Google AI API Key",
    type="password",
    value=api_key_from_cookie
)

with st.sidebar.expander("📝 View Generation History (Last 5)"):
    if not st.session_state.history:
        st.info("Your generation history will appear here.")
    else:
        for i, record in enumerate(st.session_state.history):
            st.markdown(f"**Generation {i+1}**")
            st.code(record[:200] + "...")
            if st.button(f"Restore this version", key=f"hist_btn_{i}"):
                st.session_state.last_response = record
                st.rerun()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Your Input")
    uploaded_files = st.file_uploader(
        "Upload UI Screenshot(s)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.run_id}"
    )
    user_description = st.text_area(
        "Add a short description for context",
        height=100,
        placeholder="e.g., This is the user login page for our e-commerce site.",
        key=f"description_{st.session_state.run_id}"
    )
    btn_col1, btn_col2 = st.columns(2)
    generate_button = btn_col1.button("✨ Generate Test Suite", type="primary", use_container_width=True)
    btn_col2.button("🔄 Start New Generation", on_click=start_new_generation, use_container_width=True)
    
    if uploaded_files:
        st.write("Uploaded Images:")
        for uploaded_file in uploaded_files:
            st.image(uploaded_file, width=300)

# --- THE NEW, SMARTER OUTPUT COLUMN ---
with col2:
    st.subheader("AI Generated Output")
    if not st.session_state.last_response:
        st.info("The generated Module Description and Test Cases will appear here.")
    else:
        edited_text = st.text_area(
            "You can edit the generated test cases here:",
            value=st.session_state.last_response,
            height=400,
            key="editable_output"
        )

        # Process the output for both copy and download buttons
        df = parse_markdown_to_df(st.session_state.editable_output)

        # First, create the columns for the buttons
        out_btn_col1, out_btn_col2 = st.columns(2)

        # Check if we successfully found a table to process
        if not df.empty:
            # Prepare a tab-separated string for easy pasting into Excel/Sheets
            csv_for_copy = df.to_csv(index=False, sep='\t')
            # Prepare the standard CSV for file download
            csv_for_download = df.to_csv(index=False).encode('utf-8')

            # --- Button 1: Copy as CSV (for pasting) ---
            with out_btn_col1:
                st_copy_button(csv_for_copy, "📋 Copy as CSV (for Excel)")

            # --- Button 2: Download as a CSV file ---
            with out_btn_col2:
                st.download_button(
                   label="📄 Download as CSV File",
                   data=csv_for_download,
                   file_name="test_cases.csv",
                   mime="text/csv",
                   use_container_width=True
                )
        else:
            # If no table was found, just provide a button to copy the raw text
            with out_btn_col1:
                st_copy_button(st.session_state.editable_output, "📋 Copy Raw Text")
            st.warning("Could not find a valid table in the text to create a CSV.")

# Logic to call the API
if generate_button:
    if not api_key:
        st.error("Please enter your Google AI API Key in the sidebar.")
    elif not uploaded_files:
        st.error("Please upload at least one screenshot.")
    elif not user_description:
        st.error("Please provide a description.")
    else:
        with st.spinner("🧠 The Gemini AI is analyzing and writing your test suite..."):
            try:
                genai.configure(api_key=api_key)
                image_parts = [Image.open(f) for f in uploaded_files]
                final_prompt = MASTER_PROMPT.format(user_description=user_description)
                prompt_parts = [final_prompt] + image_parts
                model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")
                response = model.generate_content(prompt_parts)
                ai_response = response.text
                st.session_state.last_response = ai_response
                st.session_state.history.insert(0, ai_response)
                st.session_state.history = st.session_state.history[:5]
                
                # Use dictionary-style assignment to set the cookie.
                cookies['api_key'] = api_key
                
                st.rerun()
            except Exception as e:
                st.error(f"An error occurred: {e}")