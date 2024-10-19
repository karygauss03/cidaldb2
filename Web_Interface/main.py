from st_on_hover_tabs import on_hover_tabs
import streamlit as st
from search import search
from predict import predict
from contact import contact
from home import home
from about import about

st.set_page_config(
    page_title="CidalsDB",
    page_icon="./Web_Interface/media/logo_BIND.ico",
    layout="wide",
)


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
    tabs = on_hover_tabs(tabName=['Home', 'Search', 'Predict', 'About Us'], 
                         iconName=['home', 'search', 'functions', 'info'],
                         key="1",
                         default_choice=0)

if tabs == 'Home':
    home()
    about()

elif tabs == 'Search':
    search()

elif tabs == 'Predict':
    predict()

elif tabs == 'About Us':
    contact()
