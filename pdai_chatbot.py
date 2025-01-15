import streamlit as st, pandas as pd
import seaborn as sns
from pandasai.llm import OpenAI
from pandasai import Agent, SmartDataframe



st.title("Prompt and get insights from you Data")

uploaded_file = st.file_uploader("Upload a CSV file for analysis", type=['csv'])

api_key = st.text_input("Your OpenAI API Key:", type="password")


# create an LLM by instantiating OpenAI object, and passing API token
llm = OpenAI(api_token=api_key)

if "messages" not in st.session_state:
    st.session_state.messages = []
    
# create PandasAI object, passing the LLM

if api_key:
    llm = OpenAI(api_token=api_key, model="gpt-4o-mini", temperature=0.2)
    
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
                        st.session_state.messages.append({"role":"assistant", "content": response})
                     #   st.write(response)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
            else:
                st.warning("Please enter a prompt.")
            
            for message in st.session_state.messages:
                st.chat_message(message["role"])
                st.markdown(message["content"])
else:
    st.warning("Please enter your OpenAI API key and press Enter to continue.")
 
                
            


