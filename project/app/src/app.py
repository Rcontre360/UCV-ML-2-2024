from st_pages import add_page_title, get_nav_from_toml 
import streamlit as st
from pathlib import Path


# Management del session state

def manage_state():
    # Esta sección de código por ahora no hace nada
    # Si quisieran compartir variables entre views deben utilizar el session_state
    # Documentación: https://docs.streamlit.io/library/api-reference/session-state
    if "active_page" not in st.session_state:
        st.session_state["active_page"] = 0
    else:
        if st.session_state["active_page"] == 1:
            # Active page was <0>

            # Delete states
            
            #del st.session_state["clinical_note"]
            pass

        st.session_state["active_page"] = 0

def ui():
    # el tercer valor es el icon que puedes usar en el sidebar de la aplicación
    # Si quieres usar otro la lista está disponible en 
    # https://raw.githubusercontent.com/omnidan/node-emoji/master/lib/emoji.json
    # Puedes sea copiar y pegar el icon como colocar el nombre entre :.
    # Ejemplo: :clipboard:
    # If you want to use the no-sections version, this
    
    # defaults to looking in .streamlit/pages.toml, so you can
    # just call `get_nav_from_toml()`

    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )

    st.write("# Welcome to Streamlit! 👋")

    st.sidebar.success("Select a demo above.")

    st.markdown(
        """
        Streamlit is an open-source app framework built specifically for
        Machine Learning and Data Science projects.
        **👈 Select a demo from the sidebar** to see some examples
        of what Streamlit can do!
        ### Want to learn more?
        - Check out [streamlit.io](https://streamlit.io)
        - Jump into our [documentation](https://docs.streamlit.io)
        - Ask a question in our [community
            forums](https://discuss.streamlit.io)
        ### See more complex demos
        - Use a neural net to [analyze the Udacity Self-driving Car Image
            Dataset](https://github.com/streamlit/demo-self-driving)
        - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
    """
    )

    # st.write(Path("src/md/Objetivo.md").read_text())

if __name__ == "__main__":
    manage_state()
    ui()
