# Configuration file for the Sphinx documentation builder.
import datetime
curr_year = str(datetime.date.today().year)

# -- Project information

project = 'PhySO'
copyright = '%s, Wassim Tenachi'%(curr_year)
author = 'Wassim Tenachi'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    # For latex style math (works in md too)
    'sphinx_math_dollar',
    # Maths
    'sphinx.ext.mathjax',
    # Including md files in rst
    'sphinx_mdinclude',
    # For markdown
    # 'myst_parser',  # -> incompatible with sphinx_mdinclude ?
    # For videos
    "sphinxcontrib.video",
]

mathjax_config = {
    'tex2jax': {
        'inlineMath': [ ["\\(","\\)"] ],
        'displayMath': [["\\[","\\]"] ],
    },
}

mathjax3_config = {
  "tex": {
    "inlineMath": [['\\(', '\\)']],
    "displayMath": [["\\[", "\\]"]],
  }
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'
html_logo = '../assets/logo_doc.png'


# -- Options for EPUB output
epub_show_urls = 'footnote'
