# Models

{% for organization, models in models_by_organization().items() %}

## {{ organization }}

{% for model in models %}

### {{ model.display_name }}

- Name: `{{ model.name }}`
- Group: `{{ model.group }}`
- Tags: {{ render_model_tags(model) }}

{{ model.description }}

{% endfor %}
{% endfor %}
