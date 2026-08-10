# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import inspect

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "fenestration"
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
    "sphinx.ext.intersphinx",
]

intersphinx_mapping = {
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3/", None),
}

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

# SPHINX CROSS REFERENCES
add_function_parentheses = False

# Enable automatic stub page generation
autosummary_generate = True


# this sphinx event allows us to have fine-grained control over whether to document
# objects or not
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#event-autodoc-skip-member
# however, we need an extra step to determine which of *our* objects the
# object-to-document is attached to (https://github.com/sphinx-doc/sphinx/issues/9533)
def skip_torch_inherited_methods(app, obj_type, name, obj, skip, options):
    if obj_type in ("method", "property", "attribute"):
        docobj = None
        for frame in inspect.stack():
            if frame.function == "_get_members":
                docobj = frame.frame.f_locals["obj"]
        if docobj is None:
            raise Exception(
                "Stack of sphinx events has changed, so unsure how to"
                " grab object that corresponds to this method! See "
                "PR #413 for discussion."
            )
        docobj_module = getattr(docobj, "__module__", "")
        # we skip the attributes inherited from torch.nn.Module for our models and model
        # components (found in the process module, we probably never want to show these
        # attributes, but this is a more conservative way of doing this)
        if docobj_module is not None and (docobj_module.startswith("fenestration")):
            if obj_type == "method":
                obj_module = getattr(obj, "__module__", "")
                if obj_module is not None and obj_module.startswith("torch.nn.modules"):
                    return True
            else:
                # for some reason, training doesn't show up as inherited (in the
                # following set up or as part of the autodoc's inherited_members that we
                # have access to in the jinja templates), so we exclude it manually
                if name == "training":
                    return True
                # unlike methods, can't just check the module of an attribute, since it
                # will typically be a basic type (e.g., bool). instead, we go through
                # all the classes of docobj and see which one contains the attribute
                # (https://stackoverflow.com/a/42503785/4659293)
                obj_module = None
                for cls in docobj.mro():
                    if name in vars(cls):
                        obj_module = cls.__module__
                if obj_module is not None and obj_module.startswith("torch.nn.modules"):
                    return True
    return None


# connect our custom method to the sphinx events callback API:
# https://www.sphinx-doc.org/en/master/extdev/event_callbacks.html
def setup(app):
    app.connect("autodoc-skip-member", skip_torch_inherited_methods)
