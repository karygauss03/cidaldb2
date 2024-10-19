import streamlit as st
from PIL import Image


def home():
    meep = Image.open('./Web_Interface/media/Logo_MEEP.png')
    bind = Image.open('./Web_Interface/media/logo_BIND.png')
    ipt = Image.open('./Web_Interface/media/logo_IPT.png')
    cidals = Image.open('./Web_Interface/media/logo_CidalsDB.png')
    all_logos = Image.open('./Web_Interface/media/logos.png') 

    cola, colb, colc= st.columns([0.2, 0.6, 0.2])
    with cola:
        st.image("https://i.ibb.co/5Knkk4G/logo-cidals.png", width=300)
    with colb:
        pass
    with colb:
        pass
    st.markdown(
    """
    <style>
    #footnotes {
        display: none
    }
    </style>
    """,
    unsafe_allow_html=True,
)
