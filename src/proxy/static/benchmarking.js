/**
 * A very simple static way to visualize the scenarios, runs, and metrics from the benchmarking project.
 * This code doesn't really belong in `proxy`, but is there for convenience.
 */
$(function () {
  const urlParams = decodeUrlParams(window.location.search);

  ////////////////////////////////////////////////////////////
  // Main

  function renderModels(models) {
    const $table = $('<table>', {class: 'query-table'});
    models.forEach((model) => {
      const $row = $('<tr>').append($('<td>').append(`${model.description} [${model.name}]`));
      $table.append($row);
    });
    return $table;
  }

  function renderRuns(runSpecs) {
    const $table = $('<table>', {class: 'query-table'});
    const $header = $('<tr>')
        .append($('<td>').append($('<b>').append('Run')))
        .append($('<td>').append($('<b>').append('Model')))
        .append($('<td>').append($('<b>').append('Adaptation method')));
    $table.append($header);

    runSpecs.forEach((runSpec) => {
      const href = encodeUrlParams(Object.assign(urlParams, {runSpec: runSpec.name}));
      const $row = $('<tr>')
        .append($('<td>').append($('<a>', {href}).append(runSpec.name)))
        .append($('<td>').append(runSpec.adapter_spec.model))
        .append($('<td>').append(runSpec.adapter_spec.method))
      $table.append($row);
    });
    return $table;
  }

  function renderHeader(header, body) {
    return $('<div>').append($('<h4>').append(header)).append(body);
  }

  function renderMetricName(name) {
    let result = name.name;
    if (name.k) {
      result += '@' + name.k;
    }
    if (name.split) {
      result += ' on ' + name.split + (name.sub_split ? '/' + name.sub_split : '');
    }
    if (name.perturbation) {
      result += ' with ' + name.perturbation;
    }
    return result;
  }

  function renderRun(runSpec) {
    const CORRECT_TAG = 'correct';

    function instanceKey(instance) {
      return JSON.stringify(instance);
    }

    const instanceToDiv = {};

    const $root = $('<div>');
    $.getJSON(`benchmark_output/runs/${runSpec.name}/scenario.json`, {}, (scenario) => {
      console.log('scenario', scenario);

      $root.append($('<h4>').append(scenario.name));
      $root.append($('<div>').append($('<i>').append(scenario.description)));

      // Render adapter spec
      const $adapterSpec = $('<table>', {class: 'table'});
      for (let key in runSpec.adapter_spec) {
        $adapterSpec.append($('<tr>').append($('<td>').append(key)).append($('<td>').append(runSpec.adapter_spec[key])));
      }
      $root.append($('<h5>').append('Adapter specification'));
      $root.append($adapterSpec);

      // Render metrics
      const $metrics = $('<table>', {class: 'table'});
      $.getJSON(`benchmark_output/runs/${runSpec.name}/metrics.json`, {}, (metrics) => {
        console.log('metrics', metrics);
        metrics.forEach((metric) => {
          const name = renderMetricName(metric.name);
          $metrics.append($('<tr>').append($('<td>').append(name)).append($('<td>').append(round(metric.mean, 3))));
        });
      });
      $root.append($('<h5>').append('Metrics'));
      $root.append($metrics);

      $root.append($('<h5>').append('Instances'));
      const $instances = $('<div>');
      $root.append($instances);
      scenario.instances.forEach((instance, i) => {
        $instances.append($('<hr>'));
        const $instance = $('<div>');
        $instance.append($('<b>').append(`Input ${i}`));
        $instance.append(': ');
        $instance.append(multilineHtml(instance.input));
        const $references = $('<ul>');
        instance.references.forEach((reference) => {
          const isCorrect = reference.tags.includes(CORRECT_TAG);
          $references.append($('<li>').append($('<span>', {class: isCorrect ? 'correct' : ''}).append(reference.output)));
        });
        $instance.append($references);
        $instances.append($instance);
        instanceToDiv[instanceKey(instance)] = $instance;
      });

      // Get the model predictions
      $.getJSON(`benchmark_output/runs/${runSpec.name}/scenario_state.json`, {}, (scenarioState) => {
        console.log('scenarioState', scenarioState);
        scenarioState.request_states.forEach((requestState) => {
          const $instance = instanceToDiv[instanceKey(requestState.instance)];
          if (!$instance) {
            console.log('Not found: ' + instanceKey(requestState.instance));
            return;
          }

          // Create a link for the request made to the server
          const request = Object.assign({}, requestState.request);
          const prompt = request.prompt;
          delete request.prompt;
          const query = {
            prompt,
            settings: JSON.stringify(request),
            environments: '',
          };
          const href = '/static/index.html' + encodeUrlParams(query);

          // Render the prediction
          let prediction = $('<i>').append('(empty)');
          if (requestState.result) {
            prediction = requestState.result.completions[0].text.trim();
            if (requestState.output_mapping) {
              prediction = requestState.output_mapping[prediction];
            }
          }
          const isCorrect = requestState.instance.references.some((reference) => reference.tags.includes(CORRECT_TAG) && reference.output === prediction);
          $instance.append($('<a>', {href}).append($('<b>').append('Prediction'))).append(': ').append($('<span>', {class: isCorrect ? 'correct' : ''}).append(prediction));
        });
      });
    });
    return $root;
  }

  const $main = $('#main');
  let models, runSpecs;
  $.when(
    $.getJSON('benchmark_output/models.json', {}, (response) => {
      models = response;
      console.log('models', models);
    }),
    $.getJSON('benchmark_output/run_specs.json', {}, (response) => {
      runSpecs = response;
      console.log('runSpecs', runSpecs);
    }),
  ).then(() => {
    $main.empty();
    if (urlParams.models) {
      $main.append(renderHeader('Models', renderModels(models)));
    }
    if (urlParams.runSpec) {
      const matchedRunSpecs = runSpecs.filter((runSpec) => runSpec.name === urlParams.runSpec);
      if (matchedRunSpecs.length === 0) {
        $main.append(renderError('No matching runs'));
      } else {
        matchedRunSpecs.forEach((runSpec) => {
          $main.append(renderRun(runSpec));
        });
      }
    } else {
      $main.append(renderHeader('Runs', renderRuns(runSpecs)));
    }
  });
});
