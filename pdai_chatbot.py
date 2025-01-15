import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pandasai.llm import OpenAI
from pandasai import Agent, SmartDataframe
import io

st.title("Prompt and get insights from your Data")

uploaded_file = st.file_uploader("Upload a CSV file for analysis", type=['csv'])

api_key = st.text_input("Your OpenAI API Key:", type="password")

# Create an LLM by instantiating OpenAI object, and passing API token
llm = OpenAI(api_token=api_key)

# Initialize session state for storing the messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Create PandasAI object, passing the LLM
if api_key:
    llm = OpenAI(api_token=api_key, model="gpt-3.5-turbo", temperature=0.2)
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data preview:", df.head())
        
        prompt = st.text_area("""Enter your prompt e.g:
                              \nDescribe the data? 
                              \nWhat could be happening in 'column_name'?
                              """)    

        # Generate output
        if st.button("Generate"):
            if prompt:
                try:
                    with st.spinner("Generating response, please wait..."):
                        sdf = SmartDataframe(df, config={"llm": llm, "conversational": True})
                        response = sdf.chat(prompt)
                        st.session_state.messages.append({"role":"user", "content": prompt})
                        
                        # Placeholder for image URL if generated
                        image_url = '/mount/src/prompt-data/exports/charts/temp_chart.png'  # Use the generated image URL
                        
                        st.image(image_url, caption="Generated Image")
                        
                        # Create an in-memory file for download
                        img_data = io.BytesIO()
                        plt.savefig(img_data, format='png')
                        img_data.seek(0)
                        
                        st.download_button(
                            label="Download image",
                            data=img_data,
                            file_name="generated_image.png",
                            mime="image/png"
                        )
                except Exception as e:
                    st.error(f"An error occurred: {e}")
            else:
                st.warning("Please enter a prompt.")
            
            for message in st.session_state.messages:
                st.chat_message(message["role"])
                st.markdown(message["content"])
else:
    st.warning("Please enter your OpenAI API key and press Enter to continue.")
