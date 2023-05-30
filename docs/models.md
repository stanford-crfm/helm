# Models

{% for organization, models in models_by_organization().items() %}

## {{ organization }}

{% for model in models %}


### {{ model.display_name }}

- Name: `{{ model.name }}`
- Group: `{{ model.group }}`
- Access: `{{ model.access }}`
- Tags: {{ render_model_tags(model) }}

{{ model.description }}
{% if model.todo %}
<span style="color:red">This model is not supported yet.</span>
{% endif %}

{% endfor %}
{% endfor %}
