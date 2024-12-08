import streamlit as st

def inject_custom_navbar():
    # Inject custom CSS
        # Sidebar with updated navigation links

    st.markdown(
        """
        <style>
            /* Sidebar background with gradient and right-side border */
            [data-testid="stSidebar"] {
                background: linear-gradient(90deg, #0D0D0D, #1C1C1E); /* Gradient background */
                color: #FFFFFF;
                border-right: 2px #0D0D0D; /* Coral color for the border */
                padding-right: 10px; /* Optional: Add padding for spacing */
            }

            /* General font color for the sidebar */
            [data-testid="stSidebar"] * {
                color: #FFD700 !important; /* Gold font color */
            }

            /* Sidebar header text */
            [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
                color: #FF6F61 !important; /* Coral color for headers */
            }

            /* Highlight active sidebar item */
            [data-testid="stSidebar"] .css-qbe2hs {
                font-size: 16px;
                font-weight: bold;
                color: #FFD700 !important; /* Ensure active item matches the sidebar font color */
            }

            /* Sidebar links hover effect */
            [data-testid="stSidebar"] .css-qbe2hs:hover {
                color: #FF4B2B !important; /* Red on hover */
                text-decoration: underline;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

   