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

  function renderRunsOverview(runSpecs) {
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

  function getJSONList(paths, callback) {
    // Fetch the JSON files `paths`, and pass the list of results into `callback`.
    const responses = {};
    $.when(
      ...paths.map((path) => $.getJSON(path, {}, (response) => { responses[path] = response; })),
    ).then(() => {
      callback(paths.map((path) => responses[path]));
    });
  }

  function canonicalizeList(lists) {
    // Takes as input a list of lists, and returns the list of unique elements (preserving order).
    // Example: [1, 2, 3], [2, 3, 4] => [1, 2, 3, 4]
    const result = [];
    lists.forEach((list) => {
      list.forEach((elem) => {
        if (result.indexOf(elem) === -1) {
          result.push(elem);
        }
      });
    });
    return result;
  }

  function dict(entries) {
    // Make a dictionary (object) out of the key/value `entries`
    const obj = {};
    entries.forEach(([key, value]) => {
      obj[key] = value;
    });
    return obj;
  }

  function findDiff(items) {
    // `items` is a list of dictionaries.
    // Return a corresponding list of dictionaries where all the common keys have been removed.
    const commonKeys = Object.keys(items[0]).filter((key) => items.every((item) => JSON.stringify(item[key]) === JSON.stringify(items[0][key])));
    return items.map((item) => {
      return dict(Object.entries(item).filter((entry) => commonKeys.indexOf(entry[0]) === -1));
    });
  }

  function renderRunsDetailed(runSpecs) {
    // Render all the `runSpecs`:
    // - Adapter specifictaion
    // - Metric
    // - Instances + predictions
    // For each block, we show a table and each `runSpec` is a column.
    const CORRECT_TAG = 'correct';

    // Used to hash instances.
    function instanceKey(instance) {
      return JSON.stringify(instance);
    }

    function renderDict(obj) {
      return Object.entries(obj).map(([key, value]) => `${key}=${value}`).join(',');
    }

    // Figure out short names for the runs based on where they differ
    const runDisplayNames = findDiff(runSpecs.map((runSpec) => runSpec.adapter_spec)).map(renderDict);

    // Setup the basic HTML elements
    const $root = $('<div>');
    const $scenarioInfo = $('<div>', {class: 'scenario-info'});
    $root.append($scenarioInfo);

    $root.append($('<h5>').append('Adapter specification'));
    const $adapterSpec = $('<table>', {class: 'table'});
    if (runSpecs.length > 1) {
      $adapterSpec.append($('<tr>').append($('<td>')).append(runDisplayNames.map((name) => $('<td>').append(name))));
    }
    $root.append($adapterSpec);

    $root.append($('<h5>').append('Metrics'));
    const $metrics = $('<table>', {class: 'table'});
    if (runSpecs.length > 1) {
      $metrics.append($('<tr>').append($('<td>')).append(runDisplayNames.map((name) => $('<td>').append(name))));
    }
    $root.append($metrics);

    $root.append($('<h5>').append('Instances'));
    const $instances = $('<div>');
    $root.append($instances);

    // Render adapter specs
    const keys = canonicalizeList(runSpecs.map((runSpec) => Object.keys(runSpec.adapter_spec)));
    keys.forEach((key) => {
      const $row = $('<tr>').append($('<td>').append(key));
      runSpecs.forEach((runSpec) => {
        $row.append($('<td>').append(runSpec.adapter_spec[key]));
      });
      $adapterSpec.append($row);
    });

    // Render metrics
    getJSONList(runSpecs.map((runSpec) => `benchmark_output/runs/${runSpec.name}/metrics.json`), (metricsList) => {
      console.log('metrics', metricsList);
      const displayNames = canonicalizeList(metricsList.map((metrics) => metrics.map((metric) => renderMetricName(metric.name))));

      displayNames.forEach((displayName) => {
        const $row = $('<tr>').append($('<td>').append(displayName));
        metricsList.forEach((metrics) => {
          const metric = metrics.find((metric) => renderMetricName(metric.name) === displayName);
          $row.append($('<td>').append(metric ? round(metric.mean, 3) : '?'));
        });
        $metrics.append($row);
      });
    });

    // Render scenario instances
    const instanceToDiv = {};
    getJSONList(runSpecs.map((runSpec) => `benchmark_output/runs/${runSpec.name}/scenario.json`), (scenarios) => {
      console.log('scenarios', scenarios);

      $scenarioInfo.append($('<h4>').append(scenarios[0].name));
      $scenarioInfo.append($('<div>').append($('<i>').append(scenarios[0].description)));

      scenarios.forEach((scenario) => {
        scenario.instances.forEach((instance, i) => {
          const key = instanceKey(instance);
          if (key in instanceToDiv) {
            return;
          }

          // Render instance
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
          instanceToDiv[key] = $instance;
        });
      });

      // Render the model predictions
      getJSONList(runSpecs.map((runSpec) => `benchmark_output/runs/${runSpec.name}/scenario_state.json`), (scenarioStates) => {
        console.log('scenarioStates', scenarioStates);
        scenarioStates.forEach((scenarioState, index) => {
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
            $instance.append($('<div>')
              .append($('<a>', {href}).append($('<b>').append(runSpecs.length > 1 ? `Prediction (${runDisplayNames[index]})` : 'Prediction')))
              .append(': ')
              .append($('<span>', {class: isCorrect ? 'correct' : ''}).append(prediction)));
          });
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
      const matchedRunSpecs = runSpecs.filter((runSpec) => new RegExp('^' + urlParams.runSpec + '$').test(runSpec.name));
      if (matchedRunSpecs.length === 0) {
        $main.append(renderError('No matching runs'));
      } else {
        $main.append(renderRunsDetailed(matchedRunSpecs));
      }
    } else {
      $main.append(renderHeader('Runs', renderRunsOverview(runSpecs)));
    }
  });
});
