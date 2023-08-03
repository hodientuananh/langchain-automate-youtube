from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
import os
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

# OpenAI API key
API_KEY = st.secrets["OPEN_AI_API_KEY"]

llm = OpenAI(openai_api_key=API_KEY, temperature=0.9)

# Prompt
blog_from_template = PromptTemplate(
    input_variables = ["product_description"],
    template='Write a blog post on {product_description}'
)
# Chain
blog_chain = LLMChain(llm=llm, prompt=blog_from_template, verbose=True, output_key='blog')

youtube_script_template = PromptTemplate(
    input_variables = ["blog"],
    template = '''Write an engaging Youtube Short video script 
        for a new product based on this blog content: {blog}.'''
)

# Chain 2
youtube_script_chain = LLMChain(llm=llm, prompt=youtube_script_template,
                                verbose=True, output_key='yt_script')

# Prompt 3
youtube_visuals_template = PromptTemplate(
    input_variables=['yt_script', 'blog'],
    template='''You're an amazing director, generate the scene by scene 
Description for the Youtube video based on the following script: {yt_script}.
Here is additional blog content if additional context is needed: {blog}.'''
)

# Chain 3
youtube_visuals_chain = LLMChain(llm=llm, prompt=youtube_visuals_template,
                                 verbose=True, output_key='yt_visuals')

# Sequential Chain
marketing_automation_chain = SequentialChain(
    chains=[blog_chain, youtube_script_chain, youtube_visuals_chain],
    input_variables=['product_description'],
    output_variables=['blog', 'yt_script', 'yt_visuals'],
    verbose=True
)


# Streamlit App Front End
st.title('✨🤖 Product Marketing Assistant')
st.text(
"""Features:
    1) Blog post
    2) Youtube script
    3) Youtube visual description
Future: Instagram, Twitter, LinkedIn post generator""")

# User Input
user_input = st.text_input('Insert product description:',
                           placeholder='New recommended feature launch for photos app on phone.')
if st.button('Generate') and user_input:
    app_data = marketing_automation_chain(user_input)

    st.write(f"Generated content based on {app_data['product_description']}")

    st.write('## Blog Post')
    st.write(app_data['blog'])

    st.write('## Youtube')
    st.write('### Script')
    st.write(app_data['yt_script'])
    st.write('### Visuals')
    st.write(app_data['yt_visuals'])