import streamlit as st
from PIL import Image
import pandas as pd
import re
import io

# Import all necessary libraries for the new features
import google.generativeai as genai
import streamlit_cookies_manager
from st_copy_button import st_copy_button
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode, JsCode
import easyocr  # For OCR
from fpdf import FPDF, FontFace  # For PDF Export
from streamlit_paste_button import paste_image_button

# --- Page Configuration (MUST be the first Streamlit command) ---
st.set_page_config(layout="wide", page_title="Professional AI Test Case Generator")

# =================================================================================================
# --- CUSTOM STYLING (CSS) FOR A PROFESSIONAL LOOK ---
# =================================================================================================
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');

    /* Universal font */
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }

    /* Main App background and text */
    .stApp {
        background-color: #F0F2F6; /* Light gray background */
        color: #333333; /* Darker text for readability */
    }

    /* Sidebar styling */
    .st-emotion-cache-16txtl3 {
        background-color: #FFFFFF; /* White sidebar */
        border-right: 1px solid #E0E0E0;
        box-shadow: 2px 0px 5px rgba(0,0,0,0.05);
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #1E3A8A; /* A deep blue for headers */
        font-weight: 700;
    }

    /* Title specifically */
    h1 {
        font-size: 2.2rem; /* Adjusted title size */
        font-weight: 700;
        padding-bottom: 0.3rem;
    }

    /* Primary button - modern blue gradient */
    .stButton > button:not([kind="secondary"]) {
        background: linear-gradient(to right, #3B82F6, #2563EB);
        color: #fff;
        padding: 12px 28px;
        border-radius: 8px;
        font-weight: 500;
        border: none;
        box-shadow: 0 4px 6px rgba(59, 130, 246, 0.2);
        transition: all 0.3s ease;
    }
    .stButton > button:not([kind="secondary"]):hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(59, 130, 246, 0.3);
    }
    .stButton > button:not([kind="secondary"]):active {
        transform: translateY(0);
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
    }

    /* Secondary Button Styling (subtle outline) */
    .stButton > button[kind="secondary"] {
        background-color: transparent;
        color: #2563EB; /* Blue text */
        border: 1px solid #93C5FD; /* Light blue border */
        border-radius: 8px;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    .stButton > button[kind="secondary"]:hover {
        border-color: #2563EB;
        background-color: rgba(59, 130, 246, 0.05);
        color: #1E40AF;
    }

    /* st.info / Alert box styling */
    .stAlert {
        background-color: #E0F2FE; /* Light blue background */
        border-left: 5px solid #0EA5E9; /* Sky blue border */
        color: #0C5460;
        padding: 1rem;
        border-radius: 8px;
    }

    /* Footer styling */
    .footer {
        text-align: center;
        color: #6B7280; /* Gray for footer */
        font-size: 0.9rem;
        padding-top: 2.5rem;
    }

    /* Table headers and cells - clean and modern (for st.dataframe) */
    .stDataFrame th {
        background-color: #E5E7EB;
        color: #1F2937;
        font-weight: 600;
    }
    .stDataFrame td {
        color: #374151;
        background-color: #FFFFFF;
        border-bottom: 1px solid #F3F4F6;
    }

    /* Text input, SelectBox, etc. */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > div {
        background-color: #FFFFFF !important;
        color: #333333 !important;
        border-radius: 8px;
        border: 1px solid #D1D5DB; /* Gray border */
        padding: 10px;
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.05);
    }
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > div:focus-within {
        border-color: #3B82F6; /* Blue border on focus */
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =================================================================================================
# --- PROMPTS AND MODEL CONFIGURATION ---
# =================================================================================================

AI_MODEL_NAME = "gemini-1.5-flash-latest"

MASTER_PROMPT = """
You are a world-class Senior QA Engineer with an expert-level eye for detail and extensive experience testing complex UI workflows and data-driven systems.

Your task is to generate a **complete and accurate test case suite** that mimics the depth and professionalism of enterprise-grade QA teams.

---

**USER-PROVIDED CONTEXT:**
- Module Description: "{user_description}"
- Test Case ID Prefix: "{case_id_prefix}"

---

**SCREENSHOT ANALYSIS INSTRUCTIONS:**
You are given UI screenshots (possibly multiple, forming a flow) and OCR-extracted text to enhance accuracy. You MUST:
- Detect all visible fields, labels, buttons, tables, dropdowns, error messages, tooltips, and URLs.
- Notice **spelling mistakes**, **grammar errors**, **UI alignment issues**, **overlapping elements**, **font/color mismatches**, and **mobile-unfriendly layouts**.
- Identify broken links, missing alt text, default placeholders, poor accessibility, or contrast issues.

<OCR_TEXT>
{ocr_text}
</OCR_TEXT>

---

### 1. Module Description
Write a short 3–5 line description explaining what this module/page does, its business purpose, and how a user interacts with it.

---

### 2. Comprehensive QA Test Case Suite

Generate a complete QA suite based on the screenshots and user description. The suite **must cover** the following test types:

✅ Functional Testing  
✅ UI/UX Testing (spelling, layout, overlapping text, font issues)  
✅ Input Validation (positive, negative, edge cases)  
✅ Error Message Validation  
✅ Usability Testing (clarity, user experience, accessibility, empty state)  
✅ Navigation & Link Verification (URLs, redirects, broken links)  
✅ Responsive Design (screen size, mobile/desktop)  
✅ Visual Defect Detection (overlaps, color inconsistencies, hidden elements)  
✅ Business Logic Testing (if applicable)

**MULTIPLE SCREEN INSTRUCTION:**  
If screenshots cover multiple screens, treat them as a flow. Group test cases under headings like:

Screen 1: Login Page
Screen 2: Dashboard

---

### TEST CASE TABLE FORMAT (MANDATORY)

Output your test cases **ONLY** using this exact markdown table format:

| Test Cases ID | Module Name | Test Cases Description | Test Cases Steps | Expected Result | Actual Result | Dev_Result (PASS/FAIL) | Dev_Remarks | QA_Result (PASS/FAIL) | Screenshot | QA Comment | Date | Reported By | Recheck |
|---------------|-------------|-------------------------|------------------|------------------|---------------|-------------------------|-------------|------------------------|------------|-------------|------|--------------|---------|

---

### RULES:
- **Test Cases ID:** Start from `{case_id_prefix}-001` and increment.
- **Steps:** Be granular. Use `1.<br>2.<br>3.` etc. in each row.
- **Expected Result:** Should be behaviorally precise.
- **Actual Result:** Leave it blank (unless told to simulate bugs).
- **Screenshot:** Leave it blank unless otherwise instructed.
- **Date:** Use today's date.
- **Reported By:** Use "QA Team"
- **Recheck:** Leave blank.

---

### TONE:
You are acting like a forensic QA with deep inspection skills—nothing must escape your notice. You must inspect every corner of the UI and imagine edge cases a junior tester may miss.

Now generate the test suite.
"""

BUG_SUMMARY_PROMPT = """
You are a QA Lead. Below is a list of test cases that have been marked as bugs.
Analyze this list and provide a concise summary for a project manager.

Your summary should include:
1.  **Key Issue Categories:** Group the bugs into 2-4 high-level categories (e.g., "Input Validation Flaws", "UI/UX Inconsistencies", "Critical Login Failures").
2.  **Potential User Impact:** Briefly describe how these bugs might affect a real user.
3.  **Suggested Next Steps:** Recommend immediate actions (e.g., "Prioritize fixing the login failures", "Conduct a full regression on the user profile form").

Keep the summary brief, professional, and actionable.

**Bug List:**
{bug_list_markdown}
"""

# =================================================================================================
# --- HELPER FUNCTIONS ---
# =================================================================================================

# FIX: Replaced @st.cache with @st.cache_resource for the OCR model
@st.cache_resource
def get_ocr_reader():
    """Initializes and caches the EasyOCR reader model to prevent reloading on every script run."""
    return easyocr.Reader(['en'])

def extract_text_from_image(image_bytes):
    try:
        reader = get_ocr_reader()
        result = reader.readtext(image_bytes, detail=0, paragraph=True)
        return "\n".join(result)
    except Exception as e:
        st.warning(f"Could not perform OCR on an image. Error: {e}")
        return ""

def parse_multi_screen_markdown(markdown_text):
    def parse_table_from_chunk(chunk):
        table_match = re.search(r'(\|.*\|(\r?\n)?)+(?![^|])', chunk)
        if not table_match: return pd.DataFrame()
        table_str = table_match.group(0)
        table_io = io.StringIO(table_str)
        try:
            df = pd.read_csv(table_io, sep='|', header=0, skipinitialspace=True).dropna(axis=1, how='all').iloc[1:]
            df.columns = [col.strip() for col in df.columns]
            for col in df.columns: df[col] = df[col].str.strip()
            return df
        except Exception: return pd.DataFrame()

    screen_sections = re.split(r'(###\s*Screen\s*\d+.*)', markdown_text)
    all_dfs = []
    if len(screen_sections) == 1:
        df = parse_table_from_chunk(markdown_text)
        if not df.empty:
            df['Screen'] = 'General'
            all_dfs.append(df)
    else:
        for i in range(1, len(screen_sections), 2):
            if (i + 1) < len(screen_sections):
                screen_header, screen_content = screen_sections[i], screen_sections[i+1]
                match = re.search(r'###\s*(.*)', screen_header)
                current_screen_name = match.group(1).strip() if match else f"Screen {i//2 + 1}"
                df = parse_table_from_chunk(screen_content)
                if not df.empty:
                    df['Screen'] = current_screen_name
                    all_dfs.append(df)
    if not all_dfs: return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)

def get_spreadsheet_ready_tsv(df):
    df_for_copy = df.copy()
    columns_to_clean = ['Test Steps', 'Expected Result']
    for col in columns_to_clean:
        if col in df_for_copy.columns:
            df_for_copy[col] = df_for_copy[col].astype(str).str.replace('<br>', '\n', regex=False)
    return df_for_copy.to_csv(index=False, sep='\t')

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12); self.cell(0, 10, 'Test Case Suite', 0, 1, 'C'); self.ln(10)
    def footer(self):
        self.set_y(-15); self.set_font('Arial', 'I', 8); self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def export_df_to_pdf(df):
    pdf = PDF(); pdf.add_page(orientation='L'); pdf.set_font("Arial", size=7)
    df_pdf = df.copy()
    columns_to_clean = ['Test Steps', 'Expected Result']
    for col in columns_to_clean:
        if col in df_pdf.columns:
            df_pdf[col] = df_pdf[col].astype(str).str.replace('<br>', '\n', regex=False)
    headers = df_pdf.columns.tolist(); data = df_pdf.astype(str).values.tolist()
    known_widths = {'Test Case ID': 22, 'Module Name': 20, 'Category': 20, 'Test Case Title': 35, 'Test Steps': 65, 'Expected Result': 60, 'Priority': 15, 'Test Type': 15, 'Screen': 20, 'Status': 15, 'Mark as Bug': 15, 'Assigned To': 20, 'Actual Result': 20}
    default_width = 18; final_col_widths = [known_widths.get(h, default_width) for h in headers]
    bold_style = FontFace(emphasis="BOLD")
    with pdf.table(col_widths=final_col_widths, text_align="LEFT", line_height=pdf.font_size * 1.5, borders_layout="ALL") as table:
        header_row = table.row()
        for header in headers: header_row.cell(header, style=bold_style)
        for data_row_list in data:
            row = table.row()
            for datum in data_row_list: row.cell(datum)
    return bytes(pdf.output())

def generate_test_cases(api_key, user_description, num_cases, categories, prefix, image_parts, ocr_text):
    genai.configure(api_key=api_key); final_prompt = MASTER_PROMPT.format(user_description=user_description, num_cases=num_cases, categories_to_include=', '.join(categories), case_id_prefix=prefix, ocr_text=ocr_text)
    prompt_parts = [final_prompt] + image_parts; model = genai.GenerativeModel(model_name=AI_MODEL_NAME); response = model.generate_content(prompt_parts)
    return response.text

def generate_bug_summary(api_key, bug_df):
    genai.configure(api_key=api_key); bug_list_md = bug_df.to_markdown(index=False)
    summary_prompt_formatted = BUG_SUMMARY_PROMPT.format(bug_list_markdown=bug_list_md)
    model = genai.GenerativeModel(model_name=AI_MODEL_NAME); response = model.generate_content(summary_prompt_formatted)
    return response.text

cookies = streamlit_cookies_manager.CookieManager()
if 'history' not in st.session_state: st.session_state.history = []
if 'last_response' not in st.session_state: st.session_state.last_response = ""
if 'run_id' not in st.session_state: st.session_state.run_id = 0
if 'df_test_cases' not in st.session_state: st.session_state.df_test_cases = pd.DataFrame()
if 'bug_summary' not in st.session_state: st.session_state.bug_summary = ""
if 'last_run_inputs' not in st.session_state: st.session_state.last_run_inputs = {}

def start_new_generation():
    # Clear out the uploaded files list for the current run_id specifically
    if f"uploaded_files_{st.session_state.run_id}" in st.session_state:
        st.session_state[f"uploaded_files_{st.session_state.run_id}"] = []
    
    st.session_state.last_response, st.session_state.df_test_cases, st.session_state.bug_summary, st.session_state.last_run_inputs = "", pd.DataFrame(), "", {}
    st.session_state.run_id += 1

st.sidebar.header("⚙️ Configuration")
api_key = st.sidebar.text_input("Enter your Google AI API Key", type="password", value=cookies.get('api_key') if cookies.ready() else "")
with st.sidebar.expander("🔬 Test Generation Settings", expanded=True):
    num_cases_to_gen = st.number_input("Minimum Number of Test Cases", min_value=5, max_value=50, value=15, step=5, help="The agent will generate at least this many test cases, but may generate more if more scenarios are found.")
    case_id_prefix = st.text_input("Test Case ID Prefix", value="TC-LOGIN")
    test_categories = st.multiselect("Select Test Categories to Generate", ["Positive Scenarios", "Field-Level Validations", "Edge Cases", "UI/UX Scenarios", "Functional Logic", "System Behavior", "Accessibility"], default=["Positive Scenarios", "Field-Level Validations", "Edge Cases", "UI/UX Scenarios"])
with st.sidebar.expander("📜 View Generation History (Last 5)"):
    if not st.session_state.history: st.info("Your generation history will appear here.")
    else:
        for i, record in enumerate(st.session_state.history):
            if st.button(f"Restore Generation {i+1}", key=f"hist_btn_{i}", use_container_width=True, type="secondary"):
                st.session_state.last_response, st.session_state.df_test_cases, st.session_state.last_run_inputs = record['response'], parse_multi_screen_markdown(record['response']), record.get('inputs', {})
                st.rerun()

st.title("🚀 AI-Powered Test Case Suite Generator")
main_cols = st.columns([0.45, 0.55]) # Adjust column widths for better balance
with main_cols[0]:
    st.subheader("✍️ 1. Describe Your UI")
    
    # Initialize session state for uploaded files if it doesn't exist
    if f"uploaded_files_{st.session_state.run_id}" not in st.session_state:
        st.session_state[f"uploaded_files_{st.session_state.run_id}"] = []

    # Let users paste images from clipboard
    paste_result = paste_image_button("📋 Paste image from clipboard", key=f"paste_{st.session_state.run_id}")
    if paste_result.image_data is not None:
        # Only add if not already in the list (by object reference)
        if paste_result.image_data not in st.session_state[f"uploaded_files_{st.session_state.run_id}"]:
            st.session_state[f"uploaded_files_{st.session_state.run_id}"].append(paste_result.image_data)

    # Allow users to upload files, and append them to the session state list
    uploaded_files_from_uploader = st.file_uploader(
        "📂 Drag and drop files here",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.run_id}"
    )
    if uploaded_files_from_uploader:
        for uploaded_file in uploaded_files_from_uploader:
             if not any(f.name == uploaded_file.name for f in st.session_state[f"uploaded_files_{st.session_state.run_id}"]):
                st.session_state[f"uploaded_files_{st.session_state.run_id}"].append(uploaded_file)

    # Use the session state list as the source of truth
    uploaded_files = st.session_state[f"uploaded_files_{st.session_state.run_id}"]
    
    if uploaded_files:
        st.write("Uploaded Image Previews:")
        cols = st.columns(len(uploaded_files))
        for i, uploaded_file in enumerate(uploaded_files):
            with cols[i]: st.image(uploaded_file, use_container_width=True, caption=f"Screen {i+1}")
    user_description = st.text_area("Add a short description for context", height=120, placeholder="Example: This is the main login screen for our B2B SaaS platform...", key=f"description_{st.session_state.run_id}")
    with st.expander("💡 Need help with the description? (Prompt Tips)"):
        st.info("""
        **A good description is key to great test cases!**
        - **Weak:** "login test"
        - **Strong:** "This is the main login screen for our B2B SaaS platform. It requires a corporate email and a password. There is a 'Forgot Password' link but no 'Sign Up' option. After 3 failed attempts, the account should be locked."
        """)
    st.divider()
    st.subheader("🚀 2. Generate Your Suite")
    btn_col1, btn_col2 = st.columns(2)
    generate_button = btn_col1.button("✨ Generate Test Suite", use_container_width=True)
    btn_col2.button("🔄 Start New Generation", on_click=start_new_generation, use_container_width=True, type="secondary")

with main_cols[1]:
    st.subheader("🤖 AI Generated Output")
    if st.session_state.last_run_inputs:
        if st.button("🔁 Regenerate Last Response", type="secondary"):
            st.session_state.last_run_inputs['regenerate'] = True; st.rerun()
    if st.session_state.df_test_cases.empty:
        st.info("Your generated Module Description and Test Cases will appear here.")
    else:
        module_desc_match = re.search(r'(.*?)### 2\.', st.session_state.last_response, re.DOTALL)
        if module_desc_match: st.markdown(module_desc_match.group(1).strip())
        tab1, tab2, tab3 = st.tabs(["📝 Interactive Test Suite", "🐞 Bug Report", "📄 Raw AI Output"])
        with tab1:
            st.markdown("###### Edit, Sort, and Filter Test Cases")
            # Reorder and ensure all required columns are present
            required_columns = [
                "Test Cases ID", "Module Name", "Test Cases Description", "Test Cases Steps", "Expected Result", "Actual Result", "Dev_Result (PASS/FAIL)", "Dev_Remarks", "QA_Result (PASS/FAIL)", "Screenshot", "QA Comment", "Date", "Reported By", "Recheck"
            ]
            for col in required_columns:
                if col not in st.session_state.df_test_cases.columns:
                    st.session_state.df_test_cases[col] = ""
            st.session_state.df_test_cases = st.session_state.df_test_cases[required_columns]
            # Remove <br> from display in Test Cases Steps and Expected Result
            display_df = st.session_state.df_test_cases.copy()
            for col in ["Test Cases Steps", "Expected Result"]:
                if col in display_df.columns:
                    display_df[col] = display_df[col].astype(str).str.replace('<br>', '\n', regex=False)
            gb = GridOptionsBuilder.from_dataframe(display_df)
            gb.configure_default_column(editable=True, filterable=True, groupable=True, resizable=True)
            gb.configure_column("Dev_Result (PASS/FAIL)", cellEditor='agGridSelectCellEditor', cellEditorParams={'values': ['PASS', 'FAIL', '']})
            gb.configure_column("QA_Result (PASS/FAIL)", cellEditor='agGridSelectCellEditor', cellEditorParams={'values': ['PASS', 'FAIL', '']})
            gb.configure_column("Test Cases Steps", wrapText=True, autoHeight=True, width=350)
            gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren=True)
            aggrid_response = AgGrid(display_df, gridOptions=gb.build(), update_mode=GridUpdateMode.MODEL_CHANGED, data_return_mode=DataReturnMode.AS_INPUT, allow_unsafe_jscode=True, height=500, fit_columns_on_grid_load=True, theme='alpine')
            edited_df = pd.DataFrame(aggrid_response['data']); st.session_state.df_test_cases = edited_df
            st.divider()
            st.markdown("#### 📊 Test Suite Metrics")
            metric_cols = st.columns(2)
            metric_cols[0].metric("Total Test Cases", len(edited_df))
            if 'Dev_Result (PASS/FAIL)' in edited_df.columns:
                dev_pass = (edited_df['Dev_Result (PASS/FAIL)'] == 'PASS').sum()
                dev_fail = (edited_df['Dev_Result (PASS/FAIL)'] == 'FAIL').sum()
                metric_cols[1].metric("Dev PASS/FAIL", f"{dev_pass} / {dev_fail}")
            st.divider()
            st.markdown("#### 📤 Export Options")
            export_col1, export_col2 = st.columns(2)
            with export_col1:
                st_copy_button(get_spreadsheet_ready_tsv(edited_df), "📋 Copy for Excel/Sheets")
                st.download_button("📄 Download as CSV", edited_df.to_csv(index=False).encode('utf-8'), "test_suite.csv", "text/csv")
            with export_col2: st.download_button("📑 Download as PDF", export_df_to_pdf(edited_df), "test_suite.pdf", "application/pdf", use_container_width=True)
        with tab2:
            st.markdown("#### Test Cases Marked as Bugs")
            if "Mark as Bug" in st.session_state.df_test_cases.columns:
                bug_df = st.session_state.df_test_cases[st.session_state.df_test_cases["Mark as Bug"] == True]
                if bug_df.empty: st.info("No test cases have been marked as bugs yet.")
                else:
                    st.warning("The following test cases have been flagged as bugs:")
                    AgGrid(bug_df, fit_columns_on_grid_load=True, theme='alpine')
                    if st.button("🤖 Generate AI Bug Summary"):
                        with st.spinner("AI is analyzing the bugs..."): st.session_state.bug_summary = generate_bug_summary(api_key, bug_df)
                    if st.session_state.bug_summary:
                        st.markdown("---")
                        st.markdown("#### AI Generated Summary")
                        st.info(st.session_state.bug_summary)
        with tab3:
            st.markdown("#### Full Raw Output from AI")
            st.info("This is the complete, unedited response from the AI.")
            st_copy_button(st.session_state.last_response, "📋 Copy Raw Text")
            st.markdown(f"--- \n {st.session_state.last_response}", unsafe_allow_html=True)

def trigger_generation(run_inputs):
    with st.spinner("🧠 AI is analyzing and writing your test suite..."):
        try:
            ocr_texts, image_parts = [], []
            for uploaded_file_data in run_inputs['files']:
                ocr_texts.append(extract_text_from_image(uploaded_file_data)); image_parts.append(Image.open(io.BytesIO(uploaded_file_data)))
            ai_response = generate_test_cases(api_key=api_key, user_description=run_inputs['description'], num_cases=run_inputs['num_cases'], categories=run_inputs['categories'], prefix=run_inputs['prefix'], image_parts=image_parts, ocr_text="\n".join(ocr_texts))
            st.session_state.last_response, st.session_state.last_run_inputs = ai_response, run_inputs
            st.session_state.history.insert(0, {'response': ai_response, 'inputs': run_inputs})
            st.session_state.history = st.session_state.history[:5]
            df = parse_multi_screen_markdown(ai_response)
            if not df.empty:
                df["Mark as Bug"], df["Status"], df["Assigned To"], df["Actual Result"] = False, "Not Run", "", ""
                st.session_state.df_test_cases = df
            else:
                st.session_state.df_test_cases = pd.DataFrame(); st.warning("The AI returned a response, but no valid test case table could be parsed.")
            if cookies.ready(): cookies['api_key'] = api_key; cookies.save()
            st.session_state.bug_summary = ""
        except Exception as e: st.error(f"An error occurred: {e}")
        finally: st.rerun()

def file_to_bytes(f):
    if hasattr(f, "getvalue"):  # UploadedFile
        return f.getvalue()
    else:  # PIL Image
        buf = io.BytesIO()
        f.save(buf, format="PNG")
        return buf.getvalue()

should_run, run_inputs = False, {}
if generate_button:
    should_run = True
    run_inputs = {
        'files': [file_to_bytes(f) for f in uploaded_files],
        'description': user_description,
        'num_cases': num_cases_to_gen,
        'categories': test_categories,
        'prefix': case_id_prefix
    }
elif st.session_state.last_run_inputs.get('regenerate'):
    should_run, run_inputs = True, st.session_state.last_run_inputs
    st.session_state.last_run_inputs['regenerate'] = False

if should_run:
    if not api_key: st.error("❌ Please enter your Google AI API Key in the sidebar.")
    elif not run_inputs.get('files'): st.error("❌ Please upload at least one screenshot.")
    elif not run_inputs.get('description'): st.error("❌ Please provide a description for context.")
    else: trigger_generation(run_inputs)

# FIX: Removed extra parenthesis at the end of this line
st.markdown(f"<div class='footer'>Built with ❤️ by Jaimin Sharma | AI Model: {AI_MODEL_NAME}</div>", unsafe_allow_html=True)

