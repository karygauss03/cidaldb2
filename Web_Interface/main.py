from st_on_hover_tabs import on_hover_tabs
import streamlit as st
from search import search
from predict import predict
from contact import contact
from home import home
from about import about
from documentation import documentation
import gdown
import os

# Define the path where the model should be saved
MODEL_PATH = './Web_Interface/models/Covid_chemberta_model.pkl'

# Function to download the model if it doesn't exist locally
# def download_models():
#     # Create the directory if it doesn't exist
#     os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
#     if not os.path.exists(MODEL_PATH):
#         url = 'https://drive.google.com/file/d/11N3HU8Ll0Rou-hc-yGilnz2UyQIAAMWA'
#         gdown.download(url, MODEL_PATH, quiet=False)
#     print("###################Downloaded##################")


st.set_page_config(
    page_title="CidalsDB",
    page_icon="./Web_Interface/media/logo_BIND.ico",
    layout="wide",
)

# with st.spinner("Downloading and loading the model..."):
#     download_models()

st.markdown('<style>' + open('./Web_Interface/src/style.css').read() + '</style>', unsafe_allow_html=True)
hide_streamlit_style = """
            <style>
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            div.appview-container>section {top: 0px}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
streamlit_style = """
			<style>
            @import url('https://fonts.googleapis.com/css2?family=Arimo&display=swap');
			html, body, [class*="css"]  {
            font-family: 'Noto Sans', sans-serif;
            }
            """
st.markdown(streamlit_style, unsafe_allow_html=True)

with st.sidebar:
    tabs = on_hover_tabs(tabName=['Home', 'Search', 'Predict', 'Documentation', 'About Us'], 
                         iconName=['home', 'search', 'functions', 'article', 'info'],
                         key="1",
                         default_choice=0)

if tabs == 'Home':
    home()
    about()

elif tabs == 'Search':
    search()

elif tabs == 'Predict':
    predict()

elif tabs == 'Documentation':
    documentation()

elif tabs == 'About Us':
    contact()
