import streamlit as st
from streamlit.components.v1 import html

with open("d3_component/index.html", "r") as f:
    html_code = f.read()

html(html_code, width=960, height=600)
