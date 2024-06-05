import datetime

# -- Project information -----------------------------------------------------

project = "peytonites"
author = "peytonites"
copyright = '{}, {}'.format(datetime.datetime.now().year, "peytonites")
package = "peytonites"

# -- General configuration ---------------------------------------------------
extensions = [
    "nbsphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "myst_parser",
]
nbsphinx_execute = 'auto'
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
html_sourcelink_suffix = ""  # Avoid .ipynb.txt extensions in sources
master_doc = "index"
html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/robelgeda/peytonites2024.git",
    "repository_branch": "main",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "show_toc_level": 3,
}
html_logo = "./logo.svg"
html_favicon = "./logo.svg"
html_title = project
nbsphinx_timeout = 300
suppress_warnings = []
