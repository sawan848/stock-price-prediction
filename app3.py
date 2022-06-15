from email.mime import image
import streamlit as st
import hydralit_components as hc
   
  
from PIL import Image

st.set_page_config(layout='wide')


st.markdown("""<style>#root>div:nth-child(1)>div>div>div>div>section>div {padding-top:0rem}
    </style>
    """,unsafe_allow_html=True)
st.markdown("""<style>#root>div:nth-child(1)>div>div>div>div>section>div {color:red}
        </style>
        """,unsafe_allow_html=True)


#class="block-container css-18e3th9 egzxvld2"
menu_data=[{'label':'left'},{'label':'Book'},{'label':'Components'}]

menu_id=hc.nav_bar(menu_definition=menu_data)
# st.info(f'{menu_id}')

image=Image.open('stock-market.jpg')

st.image(image,caption='share market price')
st.write('hi')
st.markdown("<h1 style='text-align:center;color:white;background-color:#e84343' >Stock Prediction APP <h1>",unsafe_allow_html=True)
