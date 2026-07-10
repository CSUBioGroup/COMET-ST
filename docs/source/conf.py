project = "stCOMET"
author = "CSUBioGroup"
copyright = "2026, CSUBioGroup"

extensions = [
    "myst_nb",
]

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

html_theme = "sphinx_rtd_theme"
html_title = "stCOMET Documentation"

# Do not rerun the notebook on Read the Docs.
# Display outputs already saved in tutorial.ipynb.
nb_execution_mode = "off"