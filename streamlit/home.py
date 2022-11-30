import streamlit as st

st.markdown(
    """# Welcome to my ML apps

Here I intend to gather some personal projects that involve machine learning concepts and that can be easily transformed into dockerized applications using streamlit.

I intend to briefly describe the concepts involved in each of the applications on [my medium page](https://medium.com/@arthurfmorais) or on [my linkedin profile](https://www.linkedin.com/in/arthurmorais/). 

You are invited to collaborate!

> **Note:** This is a project for learning purposes only. The concepts applied here are not necessarily proposed for a production environment.

"""
)

col1, col2, col3 = st.columns([1, 6, 1])

with col1:
    st.write("")

with col2:
    st.image("https://imgs.xkcd.com/comics/machine_learning.png")

with col3:
    st.write("")
