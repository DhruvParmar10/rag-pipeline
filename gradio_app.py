import os
import sys

# Add the current directory to the path
sys.path.append(os.path.dirname(__file__))

# Import and launch the Gradio interface
from app import create_gradio_interface

if __name__ == "__main__":
    print("🚀 Starting RAG Blaze with Gradio interface...")
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
