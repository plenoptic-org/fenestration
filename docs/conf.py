# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pooling-windows"
copyright = "2026, Flatiron NeuroRSE"
author = "Flatiron NeuroRSE"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "matplotlib.sphinxext.plot_directive",
    "myst_nb",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The suffix(es) of source filenames.
# You can specify multiple suffixes as a list of strings:
source_suffix = [".rst", ".md"]

# The master toctree document.
master_doc = "index"

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
# when napoleon_use_rtype is true, the return type is often confused. setting this to
# false let's sphinx-autodoc-typehints handle it instead.
napoleon_use_rtype = False

# AUTOSUMMARY / AUTODOC
autodoc_default_options = {
    "members": True,
    "show-inheritance": False,
    "member-order": "groupwise",
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

# Path for static files (custom stylesheets or JavaScript)
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Enable automatic stub page generation
autosummary_generate = True
