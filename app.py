from swarms.structs.ui.ui import create_app  # Adjust import as per your directory structure

if __name__ == "__main__":
    app = create_app()  # Create the Gradio app using the function from `ui.py`
    app.launch()        # Launch the app
