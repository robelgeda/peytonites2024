import os
import sys
import datetime
from importlib import import_module


# -- Project information -----------------------------------------------------

project = "peytonites"
author = "peytonites"
copyright = '{0}, {1}'.format(datetime.datetime.now().year, "peytonites")
package = "peytonites"

# -- General configuration ---------------------------------------------------

extensions = [
    "nbsphinx",
    "sphinx.ext.mathjax",
    "nbsphinx",
    "sphinx_copybutton",
]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
# source_suffix = ['.rst', '.md', '.ipynb']
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
# html_logo = "./img/logo.svg"
# html_favicon = "./img/logo.svg"

html_title = '{0}'.format(project)

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

nbsphinx_timeout = 300


suppress_warnings = []
