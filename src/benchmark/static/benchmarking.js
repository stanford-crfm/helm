/**
 * A very simple static way to visualize the scenarios, runs, and metrics from the benchmarking project.
 * This code doesn't really belong in `proxy`, but is there for convenience.
 */

// Specifies all the information to help us render and understand the fields
// for adapters and metrics.
// Look at `schema.py` for the actual schema.
class Schema {
  constructor(raw) {
    this.adapter = raw.adapter;
    this.metrics = raw.metrics;
    this.perturbations = raw.perturbations;
    this.run_groups = raw.run_groups;
    this.metric_groups = raw.metric_groups;

    // Allow for quick lookup
    this.adapterFieldNames = this.adapter.map((field) => field.name);
    this.metricsFieldNames = this.metrics.map((field) => field.name);
  }

  adapterField(name) {
    const field = this.adapter.find((field) => field.name === name);
    if (!field) {
      console.error(`Adapter field ${name} not found`);
      return {};
    }
    return field;
  }

  metricsField(name) {
    const field = this.metrics.find((field) => field.name === name);
    if (!field) {
      console.error(`Metrics field ${name} not found`);
      return {};
    }
    return field;
  }

  metricGroup(name) {
    return this.metric_groups.find((group) => group.name === name);
  }
}

$(function () {
  const urlParams = decodeUrlParams(window.location.search);

  // Extract the name of the suite from the URL parameters. Default to "latest" if none is specified.
  const suite = "suite" in urlParams ? urlParams.suite : "latest";
  console.log(`Suite: ${suite}`);

  /////////////////////////////////// Pages ////////////////////////////////////

  function renderModels(models) {
    // TODO: show better information, perhaps link to ecosystem graphs
    const $table = $('<table>', {class: 'query-table'});
    models.forEach((model) => {
      const $row = $('<tr>').append([
        $('<td>').append(model.display_name),
        $('<td>').append(model.description),
        $('<td>').append(model.name),
      ]);
      $table.append($row);
    });
    return $table;
  }

  function renderRunsOverview(runSpecs) {
    let query = '';
    const $search = $('<input>', {type: 'text', size: 40, placeholder: 'Enter regex query (enter to open all)'});
    console.log(urlParams);
    $search.keyup((e) => {
      // Open up all match specs
      if (e.keyCode === 13) {
        const href = encodeUrlParams(Object.assign({}, urlParams, {runSpecRegex: '.*' + query + '.*'}));
        console.log(urlParams, href);
        window.open(href);
      }
      query = $search.val();
      renderTable();
    });

    const $table = $('<table>', {class: 'query-table'});

    function renderTable() {
      $table.empty();
      const $header = $('<tr>')
          .append($('<td>').append($('<b>').append('Run')))
          .append($('<td>').append($('<b>').append('Adaptation method')));
      $table.append($header);

      runSpecs.forEach((runSpec) => {
        if (!new RegExp(query).test(runSpec.name)) {
          return;
        }
        const href = encodeUrlParams(Object.assign({}, urlParams, {runSpec: runSpec.name}));
        const $row = $('<tr>')
          .append($('<td>').append($('<a>', {href}).append(runSpec.name)))
          .append($('<td>').append(runSpec.adapter_spec.method))
        $table.append($row);
      });
    }

    renderTable();

    return $('<div>').append([$search, $table]);
  }

  // Look at logic in `summarize.py`.
  function getMetricNames(scenarioGroup) {
    // A scenario group defines a list of metric groups, each of which defines the metrics.
    // Just pull the names from those metrics.
    const names = [];
    scenarioGroup.metric_groups.forEach((metricGroupName) => {
      const metricGroup = schema.metricGroup(metricGroupName);
      metricGroup.metrics.forEach((metric) => {
        names.push(substitute(metric.name, scenarioGroup.environment));
      });
    });
    return names;
  }

  function getStatClass(name, value, lowerIsBetter) {
    // Return the CSS class to use if a stat has `value`.

    // Based on `name` determine whether smaller or larger is better.
    if (lowerIsBetter) {
      value = 1 - value;
    }

    // Assume larger is better for now.
    if (value === 0) {
      return 'wrong';
    }
    if (value === 1) {
      return 'correct';
    }
    return '';
  }

  function renderPerInstanceStats(groups, stats, runDisplayName) {
    // This is used to render per-instance stats.
    // Groups specifies which metric names we should display.
    // Pull these out from stats and render them.
    const list = [];

    // Look for the default metrics for the group
    schema.run_groups.forEach((scenarioGroup) => {
      if (!groups.includes(scenarioGroup.name)) {
        return;
      }

      const metricNames = getMetricNames(scenarioGroup);

      // Keep only the stats that match the name
      for (let stat of stats) {
        if (!metricNames.includes(stat.name.name)) {
          continue;
        }

        const field = schema.metricsField(stat.name.name);
        list.push($('<span>', {class: getStatClass(stat.name.name, stat.mean, field.lower_is_better)}).append(field.display_name + ': ' + round(stat.mean, 3)));
      }
    });

    // String the metrics together
    const $stats = $('<div>');
    if (runDisplayName) {
      $stats.append('[' + runDisplayName + '] ');
    }
    list.forEach((item, index) => {
      if (index > 0) {
        $stats.append(', ');
      }
      $stats.append(item);
    });

    return $stats;
  }

  function metricNameCompare(k1, k2) {
    const splitCompare = (k1.split || '').localeCompare(k2.split || '');
    if (splitCompare !== 0) {
      return splitCompare;
    }
    const nameCompare = k1.name.localeCompare(k2.name);
    if (nameCompare !== 0) {
      return nameCompare;
    }
    const perturbationCompare = (k1.perturbation ? k1.perturbation.name : '').localeCompare(k2.perturbation ? k2.perturbation.name : '');
    if (perturbationCompare !== 0) {
      return perturbationCompare;
    }
    return 0;
  }

  function renderGlobalStats(query, keys, statsList, statsPaths) {
    // Render the scenario-level metrics.
    // keys: list of metric names to render (these are the rows of table)
    // statsList: for each run, list of stats
    // statsPath: for each run, list of paths to the stats files
    const $output = $('<div>');
    keys.forEach((key) => {
      // For each key (MetricName - e.g., {name: 'exact_match', ...})

      if (key.perturbation && key.perturbation.computed_on !== 'worst') {
        // Only pay attention to worst (match `summarize.py`)
        return;
      }

      const displayKey = renderMetricName(key);
      if (query !== '' && !query.split(' ').every((q) => displayKey.includes(q))) {
        return;
      }

      const field = schema.metricsField(key.name);
      const helpText = describeMetricName(field, key);
      const $key = $('<td>').append($('<span>').append(helpIcon(helpText)).append(' ').append(displayKey));
      const $row = $('<tr>').append($('<td>').append($key));
      statsList.forEach((stats) => {
        // stats: list of statistics corresponding to one run (column)
        const stat = stats.find((stat) => metricNameEquals(stat.name, key));
        $row.append($('<td>').append(stat ? renderFieldValue(field, round(stat.mean, 3)) : '?'));
      });
      $output.append($row);
    });

    // Link to the JSON file
    $output.append($('<tr>').append($('<td>'))
      .append(statsPaths.map((statsPath) => $('<td>').append($('<a>', {href: statsPath}).append('JSON')))));
    return $output;
  }

  function highlightNewWords(text, origText) {
    // Render `text`, highlighting any words that don't occur in `origText`
    // Ideally, we would form an alignment between `text` and `origText` and
    // show the full diff, but that's too expensive.
    const origWords = {};
    origText.split(' ').forEach((word) => {
      origWords[word] = true;
    });
    return text.split(' ').map((word) => origWords[word] ? word : '<u>' + word + '</u>').join(' ');
  }

  function renderRunsHeader(scenario, scenarioPath) {
    const $output = $('<div>');
    $output.append(renderGroupHeader(scenario));
    $links = $('<div>')
    if (scenario) {
      $links.append(' ').append($('<a>', {href: scenario.definition_path}).append('[code]'))
    }
    if (scenarioPath) {
      $links.append(' ').append($('<a>', {href: scenarioPath}).append('[json]'))
    }
    $links
      .append(' ').append($('<a>', {href: '#adapter'}).append('[adapter]'))
      .append(' ').append($('<a>', {href: '#instances'}).append('[instances]'))
      .append(' ').append($('<a>', {href: '#metrics'}).append('[metrics]'));
    $output.append($links);
    return $output;
  }

  function renderGroupHeader(scenario) {
    const $output = $('<div>');
    if (urlParams.group) {
      $.getJSON(`benchmark_output/runs/${suite}/groups_metadata.json`, {}, (response) => {
        if (response[urlParams.group]) {
          let groupName = response[urlParams.group].displayName;
          if (urlParams.subgroup) {
            groupName += " / " + urlParams.subgroup;
          }
          $output.append($('<h3>').append(groupName));
          $output.append($('<div>').append($('<i>').append(response[urlParams.group].description)));
        }
      });
    } else if (scenario) {
      $output.append($('<h3>').append(renderScenarioDisplayName(scenario)));
      $output.append($('<div>').append($('<i>').append(scenario.description)));
    }
    return $output;
  }

  function instanceKey(instance) {
    // The (instance id, perturbation) should be enough to uniquely identify the instance.
    return JSON.stringify([instance.id, instance.perturbation || 'original']);
  }

  function perInstanceStatsKey(entry) {
    return JSON.stringify([entry.instance_id, entry.perturbation || 'original', entry.train_trial_index]);
  }

  function instanceTrialKey(instance, trainTrialIndex) {
    return JSON.stringify([instance.id, instance.perturbation || 'original', trainTrialIndex]);
  }

  function renderScenarioInstances(scenario, $instances) {
    // Render all the instances in a scenario, outputting to $instances.
    // Return a mapping from instance key to the div where
    // we're rendering the instance, so that we can put the predictions in the
    // right spot.
    const instanceKeyToDiv = {};

    // Keep track of the original (unperturbed) instances
    const id2originalInstance = {};
    scenario.instances.forEach((instance) => {
      if (!instance.perturbation) {
        id2originalInstance[instance.id] = instance;
      }
    });

    scenario.instances.forEach((instance) => {
      const key = instanceKey(instance);
      if (key in instanceKeyToDiv) {
        console.warn(`Two instances with the same key ${key}, skipping`, instance);
        return;
      }

      // Assume original version (no perturbation shows up first)
      if (!instance.perturbation) {
        $instances.append($('<hr>'));
      } else {
        $instances.append($('<br>'));
      }

      const $instance = $('<div>');

      // For perturbations of an instance, highlight the diff between the unperturbed instance with the same ID
      const originalInstance = id2originalInstance[instance.id];

      let header;
      if (!instance.perturbation) {
        header = `Instance ${instance.id} [split: ${instance.split}]`;
      } else {
        header = '...with perturbation: ' + renderPerturbation(instance.perturbation);
      }

      $instance.append($('<b>').append(header));

      // We can hide the inputs and outputs to focus on the predictions
      if (!urlParams.hideInputOutput) {
        $instance.append('<br>');
        const input = instance.perturbation ? highlightNewWords(instance.input, originalInstance.input) : instance.input;

        // Input
        $instance.append(multilineHtml(input));

        // References
        const $references = $('<ul>');
        instance.references.forEach((reference, referenceIndex) => {
          const originalReference = instance.perturbation && originalInstance.references[referenceIndex];
          const output = instance.perturbation ? highlightNewWords(reference.output, originalReference.output) : reference.output;
          const suffix = reference.tags.length > 0 ? ' ' + ('[' + reference.tags.join(',') + ']').bold() : '';
          $references.append($('<li>').append(output + suffix));
        });
        $instance.append($references);
      }

      $instances.append($instance);
      instanceKeyToDiv[key] = $instance;
    });

    return instanceKeyToDiv;
  }

  function renderRequest(request) {
    // Render the request made to the API as a table.
    const $requestTable = $('<table>');

    const $requestTableHeader = $('<h6>').append('Request');
    $requestTable.append($requestTableHeader);

    const $promptRow = $('<tr>').append([
      $('<td>').append("prompt"),
      $('<td>').append($('<pre>').text(request.prompt)),
    ]);
    $requestTable.append($promptRow);

    for (let requestKey in request) {
      if (requestKey === 'prompt') {
        continue;
      }
      const $requestRow = $('<tr>').append([
        $('<td>').append(requestKey),
        $('<td>').append(
          typeof request[requestKey] === 'string' ? request[requestKey] : JSON.stringify(request[requestKey])
        ),
      ]);
      $requestTable.append($requestRow);
    }
    return $('<div>').append().append($requestTable);
  }

  function renderPredictions(runSpec, runDisplayName, scenarioState, perInstanceStats, instanceKeyToDiv) {
    // Add the predictions and statistics from `scenarioState` and `perInstanceStats` to the appropriate divs for each instance.
    // Each instance give rises to multiple requests (whose results are in `scenarioState`):
    //
    // Identity of the instance (instanceKey):
    // - instance_id
    // - perturbation
    // Replication:
    // - train_trial_index
    // Instance-level decompositions:
    // - for adapter method = language_modeling, a long instance is broken up into multiple requests
    // - for adapter method = multiple_choice_separate_original, have one request per reference
    // - for adapter method = multiple_choice_separate_calibrated, have two requests per reference
    const method = runSpec.adapter_spec.method;
    const numTrainTrials = scenarioState.request_states.reduce((m, request_state) => Math.max(m, request_state.train_trial_index), -1) + 1;

    // The `perInstanceStats` specifies stats for each instanceKey (instance_id, perturbation) and train_trial_index.
    const instanceKeyTrialToStats = {};
    // Whether we've already shown the stats
    const shownStats = {};
    perInstanceStats.forEach((entry) => {
      const key = perInstanceStatsKey(entry);
      instanceKeyTrialToStats[key] = (instanceKeyTrialToStats[key] || []).concat(entry.stats);
    });

    // For each request state (across all instances)...
    scenarioState.request_states.forEach((requestState) => {
      const $instance = instanceKeyToDiv[instanceKey(requestState.instance)];
      if (!$instance) {
        console.error('Not found: ' + instanceKey(requestState.instance));
        return;
      }

      // For adapter method = separate, don't show the calibration requests
      if (requestState.request_mode === 'calibration') {
        return;
      }

      // Print out instance-level statistics.
      // Show it once for each (instance id, train trial index, perturbation).
      const key = instanceTrialKey(requestState.instance, requestState.train_trial_index);
      if (!shownStats[key]) {
        const stats = instanceKeyTrialToStats[key];
        if (!stats) {
          console.warn("Cannot find stats for", key, instanceKeyTrialToStats);
        } else {
          $instance.append(renderPerInstanceStats(runSpec.groups, stats, runDisplayName));
          shownStats[key] = true;
        }
      }

      // Create a link for the request made to the API
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
      let prefix = '';
      let prediction = $('<i>').append('(empty)');
      const $logProb = $('<span>');
      if (requestState.result) {
        // Assume there is only one completion
        const completion = requestState.result.completions[0];
        prediction = completion.text.trim();

        // For adapter method = joint
        if (requestState.output_mapping) {
          prediction = requestState.output_mapping[prediction] || prediction + ' (not mappable)';
        }

        if (method.startsWith('multiple_choice_separate_')) {
          // For adapter method = separate, prediction starts with the prompt, strip it out
          if (prediction.startsWith(requestState.instance.input)) {
            prefix = '...';
            prediction = prediction.substring(requestState.instance.input.length).trim();
          }
        } else if (method === 'language_modeling') {
          // For language modeling, first token is just padding, so strip it out
          const firstToken = completion.tokens[0];
          if (Object.keys(firstToken.top_logprobs).length === 0) {
            if (!prediction.startsWith(firstToken.text)) {
              console.warning("Prediction doesn't start with first token", prediction, firstToken, completion);
            } else {
              prediction = prediction.substring(firstToken.text.length);
            }
          }

          // Prediction is a chunk of the input that's already rendered above,
          // so we just need to show the beginning and end of the chunk.
          // Ideally, we whould show all the tokens and color-code their
          // probabilities.
          prediction = truncateMiddle(prediction, 30);
        }

        $logProb.append(' ').append($('<span>', {class: 'logprob'}).append('(' + round(completion.logprob, 3) + ')'));
      }

      // Describe the prediction
      let description = '';

      // If there are multiple runs
      if (runDisplayName) {
        description += '[' + runDisplayName + '] ';
      }

      description += 'Prediction';

      // Which reference (for multiple_choice_separate_*)
      if (requestState.reference_index != null) {
        description += '[ref ' + requestState.reference_index + ']';
      }

      // If there are multiple trials
      if (numTrainTrials > 1) {
        description += '{trial ' + requestState.train_trial_index + '}';
      }

      const $request = renderRequest(requestState.request);
      $request.hide();
      $link = $('<a>', {href}).append($('<b>').append(description)).click(() => {
        $request.slideToggle();
        return false;
      });
      $instance.append($('<div>')
        .append($link)
        .append(': ')
        .append(prefix)
        .append(prediction)
        .append($logProb));
      $instance.append($request);
    });
  }

  function renderRunsDetailed(runSpecs) {
    // Render all the `runSpecs`:
    // 1. Adapter specification
    // 2. Instances + predictions
    // 3. Stats
    // For each block, we show a table and each `runSpec` is a column.
    const CORRECT_TAG = 'correct';

    // Used to hash instances.
    function instanceKey(instance) {
      return JSON.stringify(instance);
    }

    // Paths (parallel arrays corresponding to `runSpecs`)
    const statsPaths = runSpecs.map((runSpec) => {
      return `benchmark_output/runs/${suite}/${runSpec.name}/stats.json`;
    });
    const perInstanceStatsPaths = runSpecs.map((runSpec) => {
      return `benchmark_output/runs/${suite}/${runSpec.name}/per_instance_stats.json`;
    });
    const scenarioPaths = runSpecs.map((runSpec) => {
      return `benchmark_output/runs/${suite}/${runSpec.name}/scenario.json`;
    });
    const scenarioStatePaths = runSpecs.map((runSpec) => {
      return `benchmark_output/runs/${suite}/${runSpec.name}/scenario_state.json`;
    });
    const runSpecPaths = runSpecs.map((runSpec) => {
      return `benchmark_output/runs/${suite}/${runSpec.name}/run_spec.json`;
    });

    // Figure out short names for the runs based on where they differ
    const runDisplayNames = findDiff(runSpecs.map((runSpec) => runSpec.adapter_spec)).map(renderDict);

    // Setup the basic HTML elements
    const $root = $('<div>');
    const $scenarioInfo = $('<div>', {class: 'scenario-info'});
    $scenarioInfo.append('Loading...');
    $root.append($scenarioInfo);

    // Adapter
    $root.append($('<a>', {name: 'adapter'}).append($('<h5>').append('Adapter')));
    const $adapterSpec = $('<table>');
    if (runSpecs.length > 1) {
      $adapterSpec.append($('<tr>').append($('<td>'))
        .append(runDisplayNames.map((name) => $('<td>').append(name))));
    }
    $root.append($('<div>', {class: 'table-container'}).append($adapterSpec));

    // Instances
    $root.append($('<a>', {name: 'instances'}).append($('<h5>').append('Instances')));
    const $instances = $('<div>');
    $root.append($('<div>', {class: 'table-container'}).append($instances));

    // Metrics
    $root.append($('<a>', {name: 'metrics'}).append($('<h5>').append('Metrics')));
    const $stats = $('<table>');
    const $statsSearch = $('<input>', {type: 'text', size: 40, placeholder: 'Enter keywords to filter metrics'});
    if (runSpecs.length > 1) {
      $stats.append($('<tr>').append($('<td>')).append(runDisplayNames.map((name) => $('<td>').append(name))));
    }
    $root.append($('<div>', {class: 'table-container'}).append([$statsSearch, $stats]));

    // Render adapter specs
    const keys = canonicalizeList(runSpecs.map((runSpec) => Object.keys(runSpec.adapter_spec)));
    sortListWithReferenceOrder(keys, schema.adapterFieldNames);
    keys.forEach((key) => {
      const field = schema.adapterField(key);
      const helpText = describeField(field);
      const $key = $('<td>').append($('<span>').append(helpIcon(helpText)).append(' ').append(key));
      const $row = $('<tr>').append($key);
      runSpecs.forEach((runSpec) => {
        $row.append($('<td>').append(renderFieldValue(field, runSpec.adapter_spec[key])));
      });
      $adapterSpec.append($row);
    });
    $adapterSpec.append($('<tr>').append($('<td>'))
      .append(runSpecPaths.map((runSpecPath) => $('<td>').append($('<a>', {href: runSpecPath}).append('JSON')))));

    // Render metrics/stats
    getJSONList(statsPaths, (statsList) => {
      console.log('metrics', statsList);
      const keys = canonicalizeList(statsList.map((stats) => stats.map((stat) => stat.name)), metricNameCompare);
      keys.sort(metricNameCompare);

      function update() {
        $stats.empty().append(renderGlobalStats(query, keys, statsList, statsPaths));
      }

      // Filter 
      let query = '';
      $statsSearch.keyup((e) => {
        query = $statsSearch.val();
        update();
      });

      update();
    }, []);

    // Render scenario instances
    const instanceToDiv = {};  // For each instance
    getJSONList(scenarioPaths, (scenarios) => {
      console.log('scenarios', scenarios);

      const onlyOneScenario = scenarios.length && scenarios.every((scenario) => scenario.definition_path === scenarios[0].definition_path);
      const scenario = onlyOneScenario ? scenarios[0] : null;
      const scenarioPath = onlyOneScenario ? scenarioPaths[0] : null;
      $scenarioInfo.empty();
      $scenarioInfo.append(renderRunsHeader(scenario, scenarioPath));

      const instanceKeyToDiv = renderScenarioInstances(scenarios[0], $instances);

      // Render the model predictions
      getJSONList(scenarioStatePaths, (scenarioStates) => {
        console.log('scenarioStates', scenarioStates);
        getJSONList(perInstanceStatsPaths, (perInstanceStats) => {
          console.log('perInstanceStats', perInstanceStats);
          // For each run / model...
          runSpecs.forEach((runSpec, index) => {
            renderPredictions(runSpec, runDisplayNames[index], scenarioStates[index], perInstanceStats[index], instanceKeyToDiv);
          });
        });
      });
    });

    return $root;
  }

  function renderLandingPage() {
    const $intro = $('<div>').append('Welcome to the CRFM benchmarking project!');
    // TODO: put more content here.
    return $('<div>').append($intro);
  }

  function renderCell(cell) {
    const value = $('<span>', {title: cell.description}).append(cell.display_value || cell.value);
    if (cell.style) {
      value.css(cell.style);
    }
    return $('<td>').append(cell.href ? $('<a>', {href: cell.href}).append(value) : value);
  }

  function renderTable(table) {
    const $output = $('<div>');
    $output.append($('<h3>').append(table.title));
    const $table = $('<table>', {class: 'query-table results-table'});
    const $header = $('<tr>').append(table.header.map(renderCell));
    $table.append($header);

    table.rows.forEach((row) => {
      const $row = $('<tr>').append(row.map(renderCell));
      $table.append($row);
    });
    $output.append($table);
    table.links.forEach((link) => {
      $output.append(' ').append($('<a>', {href: link.href}).append('[' + link.text + ']'));
    });
    return $output;
  }

  function renderTables(tables) {
    const $output = $('<div>');
    const $links = $('<div>');
    $output.append($links);
    tables.forEach((table) => {
      $output.append($('<div>', {class: 'table-container', id: table.title}).append(renderTable(table)));
      $links.append($('<a>', {href: '#' + table.title}).append('[' + table.title + '] '));
    });
    return $output;
  }

  //////////////////////////////////////////////////////////////////////////////
  //                                   Main                                   //
  //////////////////////////////////////////////////////////////////////////////

  const $main = $('#main');
  $.when(
    $.get('schema.yaml', {}, (response) => {
      const raw = jsyaml.load(response);
      console.log('schema', raw);
      schema = new Schema(raw);
    }),
  ).then(() => {
    $main.empty();
    if (urlParams.models) {
      // Show models
      $.getJSON(`benchmark_output/runs/${suite}/models.json`, {}, (response) => {
        const models = response;
        console.log('models', models);
        $main.append(renderHeader('Models', renderModels(models)));
      });
    } else if (urlParams.runSpec || urlParams.runSpecs || urlParams.runSpecRegex) {
      // Show a set of run specs (matching a regular expression)
      $.getJSON(`benchmark_output/runs/${suite}/run_specs.json`, {}, (response) => {
        const runSpecs = response;
        console.log('runSpecs', runSpecs);
        let matcher;
        if (urlParams.runSpec) {
          // Exactly one
          matcher = (runSpec) => runSpec.name === urlParams.runSpec;
        } else if (urlParams.runSpecs) {
          // List
          const selectedRunSpecs = JSON.parse(urlParams.runSpecs);
          matcher = (runSpec) => selectedRunSpecs.includes(runSpec.name);
        } else if (urlParams.runSpecRegex) {
          // Regular expression
          const regex = new RegExp('^' + urlParams.runSpecRegex + '$');
          matcher = (runSpec) => regex.test(runSpec.name);
        } else {
          throw 'Internal error';
        }
        const matchedRunSpecs = runSpecs.filter(matcher);
        if (matchedRunSpecs.length === 0) {
          $main.append(renderError('No matching runs'));
        } else {
          $main.append(renderRunsDetailed(matchedRunSpecs));
        }
      });
    } else if (urlParams.runs) {
      // Search over all runs
      $.getJSON(`benchmark_output/runs/${suite}/run_specs.json`, {}, (response) => {
        const runSpecs = response;
        console.log('runSpecs', runSpecs);
        $main.append(renderHeader('Runs', renderRunsOverview(runSpecs)));
      });
    } else if (urlParams.groups) {
      // All groups
      $.getJSON(`benchmark_output/runs/${suite}/groups.json`, {}, (response) => {
        $main.append(renderTables(response));
      });
    } else if (urlParams.group) {
      // Specific group
      $.getJSON(`benchmark_output/runs/${suite}/groups/${urlParams.group}.json`, {}, (tables) => {
        console.log('group', tables);
        $main.append(renderGroupHeader(urlParams.group));
        $main.append(renderTables(tables));
      });
    } else {
      $main.append(renderLandingPage());
    }
  });
});
