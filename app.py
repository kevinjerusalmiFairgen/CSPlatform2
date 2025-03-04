import streamlit as st
import utils.files_utils as files_utils
from modules import random_split#, targeted_split, validation


st.set_page_config(page_title="Data Splitting Tool", layout="wide")

st.sidebar.title("Global Controls")

uploaded_file = st.sidebar.file_uploader(
    "Drag & Drop or Click to Upload",
    type=["csv", "xlsx", "sav"]
)

if uploaded_file:
    try:
        file_path = files_utils.save_uploaded_file(uploaded_file)
        data, meta = files_utils.load_file(file_path)

        if data is not None:
            st.session_state["data"] = data
            st.session_state["meta"] = meta
            st.session_state["file_path"] = file_path
            print(file_path)
            file_type = st.session_state["file_type"] = file_path.split(".")[1]
            st.sidebar.success(f"✅ {file_type.upper()} file successfully loaded!")
        else:
            st.sidebar.error(f"❌ Error Uploading")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading file: {str(e)}")

else:
    st.sidebar.info("📂 Upload a file to begin.")

# Title
st.title("Data Splitting Tool 🚀")

# Create Tabs
#tab_random, tab_targeted, tab_validation = st.tabs(["Random Split", "Targeted Split", "Validation"])

# Tab Content
#with tab_random:
if "data" in st.session_state:
    random_split.app()
else:
    st.warning("⚠️ Please upload a dataset.")

# with tab_targeted:
#     if "data" in st.session_state:
#         targeted_split.app()
#     else:
#         st.warning("⚠️ Please upload a dataset.")

# with tab_validation:
#     if "data" in st.session_state:
#         validation.app()
#     else:
#         st.warning("⚠️ Please upload a dataset.")
