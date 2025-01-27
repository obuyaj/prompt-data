import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pandasai.llm import OpenAI
from pandasai import Agent, SmartDataframe
import io
import re

st.title("Prompt and get insights from your Data")

uploaded_file = st.file_uploader("Upload a CSV file for analysis", type=['csv'])

api_key = st.text_input("Your OpenAI API Key:", type="password")

try:
    llm = OpenAI(api_token=api_key)
except:
    st.warning("Enter OpenAI API Key and press Enter to continue")

if "messages" not in st.session_state:
    st.session_state.messages = []

if api_key:
    llm = OpenAI(api_token=api_key, model="gpt-3.5-turbo", temperature=0.7)
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data preview:", df.head())
        
        prompt = st.text_area("""Enter your prompt e.g:
                              \nDescribe the data? 
                              \nWhat could be happening in 'column_name'?
                              """)    
        
        if st.button("Generate"):
            if prompt:
                try:
                    with st.spinner("Generating response, please wait..."):
                        sdf = SmartDataframe(df, config={"llm": llm, "conversational": True})
                        response = sdf.chat(prompt)
                        st.session_state.messages.append({"role":"user", "content": prompt})

                        match = re.search(r"/prompt-data/.*", response)
                        
                        if match:
                            image_path = match.group(0)
                            
                            # Correctly handle the image path
                            image_path_full = f"./{image_path}"
                            st.image(image_path_full, caption="Generated Image")
                                                        
                            img_data = io.BytesIO()
                            plt.savefig(img_data, format='png')
                            img_data.seek(0)
                            
                            st.download_button(
                                label="Download image",
                                data=img_data,
                                file_name="generated_image.png",
                                mime="image/png"
                            )
                        else:
                            st.session_state.messages.append({"role":"assistant", "content": response})
                            st.write(response)
                            
                except Exception as e:
                    st.error(f"An error occurred: {e}")
            else:
                st.warning("Please enter a prompt.")
            
            for message in st.session_state.messages:
                st.chat_message(message["role"])
                st.markdown(message["content"])
