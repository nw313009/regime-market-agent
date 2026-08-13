"""The Databricks Streamlit App (spec C-5).

A package so the pages can import ``app.common`` by name, and so pytest can import a page module
to check it is free of side effects. Streamlit itself does not care: it runs ``app/app.py`` as a
script and execs each page in ``app/pages/`` under the name ``__main__``, which is exactly why
every page keeps its rendering behind an ``if __name__ == "__main__"`` guard.
"""
