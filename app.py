import logging
from typing import Optional
import time

import streamlit as st
import PIL.Image
from google import genai
from google.genai import types

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
st.set_page_config(page_title="AI_Radiograph")
# Custom CSS for modern styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2ca02c;
    }
    .stFileUploader>div>div>div>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stFileUploader>div>div>div>button:hover {
        background-color: #2ca02c;
    }
    .stMarkdown h1 {
        color: #1f77b4;
        text-align: center;
        font-size: 36px;
        margin-bottom: 20px;
    }
    .stMarkdown h2 {
        color: #1f77b4;
        font-size: 24px;
        margin-top: 20px;
    }
    .stImage img {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        margin: 20px 0;
    }
    .stProgress>div>div>div>div {
        background-color: #2ca02c;
    }
    .stExpander {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        padding: 20px;
    }
    .footer {
        text-align: center;
        margin-top: 40px;
        color: #777;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Gemini client
def initialize_gemini_client(api_key: str) -> genai.Client:
    """Initialize and return the Gemini client."""
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("Gemini client initialized successfully.")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {e}")
        raise

# Process image and generate content
def generate_content_from_image(client: genai.Client, image: PIL.Image.Image, prompt: str) -> Optional[str]:
    """Generate content using the Gemini model based on the provided image and prompt."""
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt, image]
        )
        logger.info("Content generated successfully.")
        return response.text
    except Exception as e:
        logger.error(f"Failed to generate content: {e}")
        return None

# Streamlit app
def main():
    # App title and description
    st.markdown(
        """
        <h1>🩺 Pathology Diagnosis with AI 🔬</h1>
        <p style="text-align: center; font-size: 18px; color: #555;">
            Upload an image to diagnose potential pathology using advanced AI.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # File uploader in a centered column
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "png"], key="file_uploader")

    if uploaded_file is not None:
        try:
            # Open the image
            image = PIL.Image.open(uploaded_file)

            # Display the uploaded image
            st.markdown("### 🖼️ Uploaded Image")
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Initialize Gemini client with direct API key
            # api_key = "AIzaSyAxz3kNZLBz2PH124b-pfqVuulj960QvKo"  # Direct API key
            client = initialize_gemini_client(GEMINI_API_KEY)

            # Generate content
            prompt = """
            You are a radiologist. Analyze the provided medical image and provide the following:
            1. **Preferred Detail**: State whether the patient is "fit" or "unfit" in one point.
            2. **Explanation**: If "unfit," provide a detailed explanation of the unfit status, including potential abnormalities or pathologies observed in the image.
            Use medical terminology and provide a clear, concise, and professional response.
            """

            
            with st.spinner("🔍 Analyzing image. Please wait..."):
                progress_bar = st.progress(0)
                for percent_complete in range(100):
                    time.sleep(0.02)  # Simulate analysis progress
                    progress_bar.progress(percent_complete + 1)
                result = generate_content_from_image(client, image, prompt)

            if result:
                st.success("✅ Analysis complete!")
                st.markdown("### 🧪 Diagnosis Result")
                st.write(result)
            else:
                st.error("❌ Failed to generate analysis. Please try again.")
        except Exception as e:
            st.error(f"🚨 An error occurred: {e}")
            logger.error(f"Streamlit app error: {e}")

    # Footer with developers' credit
    st.markdown(
        """
        <div class="footer">
            <p>Developed by <strong>Iqra Rehmat Ali</strong> , <strong>Najaf Ali</strong> & <strong>Sahiba Rehmat Ali</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
