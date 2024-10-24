import os
import jupytext
import fire
from functools import partial

def convert_notebooks_to_jupytext(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.ipynb'):
                notebook_path = os.path.join(root, file)
                jupytext_path = notebook_path.replace('.ipynb', '.py')  # Change to desired format
                print(f"Converting {notebook_path} to {jupytext_path}")
                jupytext.write(jupytext.read(notebook_path), jupytext_path)

def convert_jupytext_to_notebooks(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):  # Change to match the Jupytext format used
                jupytext_path = os.path.join(root, file)
                notebook_path = jupytext_path.replace('.py', '.ipynb')
                print(f"Converting {jupytext_path} to {notebook_path}")
                jupytext.write(jupytext.read(jupytext_path), notebook_path)

if __name__ == "__main__":
    fire.Fire({
        "topy": partial(convert_notebooks_to_jupytext, directory="."),
        "tonb": partial(convert_jupytext_to_notebooks, directory=".")
    })
