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
You are a Senior QA Engineer and expert in test documentation for enterprise applications. Your goal is to create a full test case suite for a form-based UI screen.

**CONTEXT:**
The user has provided a description of the module: "{user_description}"

**SCREEN ANALYSIS:**
You have access to the attached UI screenshots. From these:
- Extract all input fields, buttons, dropdowns, and validation elements.
- Identify placeholder values, error messages, and field formats.
- Consider all form behaviors (submit, cancel, autofocus, tabbing, reset).

---

**YOUR TASK:**
Based on the analysis, generate the following test documentation:

---

### 1. Module Description
Write 3–5 lines that describe the main functionality of the UI form/module, its purpose, how users interact with it, and what operations it supports.

---

### 2. Functional Test Case Suite
Generate a complete set of test cases covering all the following categories:

#### ✅ Categories to Include:
1. **Positive Scenarios** (valid data, successful submissions)
2. **Field-Level Validations** (empty fields, incorrect formats, required fields)
3. **Edge Cases** (character limits, special characters, duplicate entries, whitespace)
4. **UI/UX Scenarios** (placeholder text, field alignment, autofocus, keyboard tabbing, scroll behavior)
5. **Functional Logic** (button states, discard/reset, field persistence)
6. **System Behavior** (network failure handling, error states)
7. **Cross-field Scenarios** (combined validations, dependencies)

---

### Test Case Format (in Markdown table):

| Test Case ID      | Module Name | Test Case Title | Test Steps | Expected Result | Severity | Priority | Actual Result | Status (Pass/Fail) |
|-------------------|-------------|------------------|------------|------------------|----------|----------|----------------|--------------------|

**Rules:**
- Start Test Case ID from: `TC-<MODULE>-001`
- Use Module Name like: `"GST Update"` or `"Address Form"`
- Leave Actual Result and Status columns empty
- Keep each step clear and numbered
- Ensure **minimum 20–25 unique test cases** covering all edge and usability cases

---

### Output Format Summary:
- Module Description
- Table of 20–25 test cases in Markdown
- Focus on quality, clarity, variety, and professional QA coverage

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