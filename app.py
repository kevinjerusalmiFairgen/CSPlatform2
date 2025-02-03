import streamlit as st

def page_one():
    st.title("Page One")
    st.write("Welcome to Page One!")
    st.write("Here you can add your page one content.")

def page_two():
    st.title("Page Two")
    st.write("Welcome to Page Two!")
    st.write("Here you can add your page two content.")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ("Page One", "Page Two"))

# Render the selected page
if page == "Page One":
    page_one()
elif page == "Page Two":
    page_two()
