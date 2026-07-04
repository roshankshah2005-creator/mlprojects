import streamlit as st

# Inject custom CSS to style the button
st.markdown(
    """
    <style>
    /* Target the Streamlit button */
    div.stButton > button {
        background-color: #ffffff !important; /* White background */
        color: #000000 !important;          /* Black text/emoji color */
        border: 2px solid #000000 !important; /* Dark border to match inputs */
        border-radius: 12px !important;     /* Rounded corners */
        transition: all 0.3s ease;
    }
    
    /* Optional: Add a subtle hover effect */
    div.stButton > button:hover {
        background-color: #f0f2f6 !important; /* Slight gray on hover */
        border-color: #000000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Your existing code below...
# st.button("✨ Submit")
