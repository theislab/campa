:github_url: {{ fullname }}

{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ fullname }}

   {% if 'Enum' not in objname %}
   {% block attributes %}

   {% if attributes %}
   .. rubric:: Attributes

   .. autosummary::
      :toctree: .

   {% for item in attributes %}
      ~{{ fullname }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block methods %}
   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
      :toctree: .
   {% for item in methods %}
      {%- if item != '__init__' %}
      ~{{ fullname }}.{{ item }}
      {%- endif -%}
   {%- endfor %}
   {% endif %}
   {% endblock %}
   {% endif %}

   .. _sphx_glr_backref_{{fullname}}:
