import streamlit.components.v1 as components

def render_slider(callback_func):
    components.declare_component("slider_component", url="http://localhost:3001")

    return components.html(
        open("slider_component/slider_component.html").read(),
        on_script_load=callback_func,
        width=300,
        height=100,
    )
