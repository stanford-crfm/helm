/**
 * A very simple static way to visualize the scenarios, runs, and metrics from the benchmarking project.
 * This code doesn't really belong in `proxy`, but is there for convenience.
 */

// Specifies all the information to help us render and understand the fields
// for adapters and metrics.
// Look at `schema.py` for the actual schema.
class Schema {
  constructor(raw) {
    this.models = raw.models;
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
  const suite = "suite" in urlParams ? urlParams.suite : "v1.0";
  console.log(`Suite: ${suite}`);

  /////////////////////////////////// Pages ////////////////////////////////////

  function renderModels() {
    const $table = $('<table>', {class: 'query-table results-table'});
    const $header = $('<tr>').append([
      $('<td>').append('Creator'),
      $('<td>').append('Model'),
      $('<td>').append('Description'),
      $('<td>').append('Access'),
    ]);
    $table.append($header);

    schema.models.forEach((model) => {
      const $name = $('<div>').append([
        $('<div>').append(model.display_name),
        $('<div>', {class: 'technical-details'}).append(model.name),
      ]);
      const $row = $('<tr>').append([
        $('<td>').append(model.creator_organization),
        $('<td>').append($name),
        $('<td>').append(renderMarkdown(model.description)),
        $('<td>').append(renderAccess(model.access)),
      ]);
      $table.append($row);
    });
    return $table;
  }

  function renderScenarios() {
    const $table = $('<table>', {class: 'query-table results-table'});

    const $header = $('<tr>').append([
      $('<td>').append('Scenario'),
      $('<td>').append('Task'),
      $('<td>').append('What'),
      $('<td>').append('When'),
      $('<td>').append('Who'),
      $('<td>').append('Language'),
      $('<td>').append('Description'),
    ]);
    $table.append($header);

    schema.run_groups.forEach((group) => {
      if (group.category) {
        return;
      }
      const href = groupUrl(group.name);
      const $name = $('<div>').append([
        $('<div>').append($('<a>', {href}).append(group.display_name)),
        $('<div>', {class: 'technical-details'}).append(group.name),
      ]);
      const task = group.taxonomy && group.taxonomy.task;
      const what = group.taxonomy && group.taxonomy.what;
      const who = group.taxonomy && group.taxonomy.who;
      const when = group.taxonomy && group.taxonomy.when;
      const language = group.taxonomy && group.taxonomy.language;
      const $row = $('<tr>').append([
        $('<td>').append($name),
        $('<td>').append(task),
        $('<td>').append(what),
        $('<td>').append(who),
        $('<td>').append(when),
        $('<td>').append(language),
        $('<td>').append(renderMarkdown(group.description)),
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
      renderRunsTable();
    });

    const $table = $('<table>', {class: 'query-table'});

    function renderRunsTable() {
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

    renderRunsTable();

    return $('<div>').append([$search, $table]);
  }

  // Look at logic in `summarize.py`.
  function getMetricNames(scenarioGroup) {
    // A (scenario/run) group defines a list of metric groups, each of which defines the metrics.
    // Just pull the names from those metrics.
    const names = [];
    scenarioGroup.metric_groups.forEach((metricGroupName) => {
      const metricGroup = schema.metricGroup(metricGroupName);
      metricGroup.metrics.forEach((metric) => {
        // This function is supposed to return per-instance metrics, so exclude
        // metrics that mentions perturbations.
        if (metric.perturbation_name) {
          return;
        }
        names.push(substitute(metric.name, scenarioGroup.environment));
      });
    });
    return names;
  }

  let metricJudgements = null;
  function getMetricJudgements() {
    // Provide information
    // Return dictionary {metric name: {wrongThreshold, correctThreshold, lowerIsBetter}}
    // Example: {exact_match: {wrongThreshold: 0, correctThreshold: 1, lowerIsBetter: true}}
    // TODO: move the hard-coding into schema.yaml
    if (metricJudgements) {
      return metricJudgements;
    }
    metricJudgements = {};
    schema.run_groups.forEach((runGroup) => {
      const name = runGroup.environment && runGroup.environment.main_name;
      if (!["bits_per_byte"].includes(name)) {
        metricJudgements[name] = {wrongThreshold: 0, correctThreshold: 1, lowerIsBetter: false};
      }
    });
    return metricJudgements;
  }

  function getStatClass(name, value) {
    // Return the CSS class to use if a stat has `value`.
    const judgements = getMetricJudgements();
    const judgement = judgements[name];
    if (!judgement) {
      return '';
    }

    // Based on `name` determine whether smaller or larger is better.
    if (judgement.lowerIsBetter === false) {
      if (value >= judgement.correctThreshold) {
        return 'correct';
      }
      if (value <= judgement.wrongThreshold) {
        return 'wrong';
      }
    }

    if (lowerIsBetter === true) {
      if (value <= judgement.correctThreshold) {
        return 'correct';
      }
      if (value >= judgement.wrongThreshold) {
        return 'wrong';
      }
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
      for (let name of metricNames) {
        const field = schema.metricsField(name);

        const matchingStats = stats.filter((stat) => stat.name.name === name);
        if (matchingStats.length === 0) {
          // Some stats won't be here (calibration, bias) because there is no per-instance version.
          continue;
        }

        if (matchingStats.length > 1) {
          // This shouldn't happen...
          console.warn('Metric', name, 'occurs more than once', matchingStats);
        }

        const stat = matchingStats[0];
        const statClass = getStatClass(name, stat.mean);
        list.push($('<span>', {class: statClass}).append(`${field.display_name}: ${round(stat.mean, 3)}`));
      }
    });

    // String the metrics together
    const $stats = $('<div>');
    if (runDisplayName) {
      $stats.append(runDisplayName);
    }
    $stats.append(renderItems(list));

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

    // Links
    const links = [];
    if (scenario) {
      links.push($('<a>', {href: scenario.definition_path}).append('Code'))
    }
    if (scenarioPath) {
      links.push($('<a>', {href: scenarioPath}).append('Scenario JSON'));
    }
    links.push($('<a>', {href: '#adapter'}).append('Adapter specification'));
    links.push($('<a>', {href: '#instances'}).append('Instances + predictions'));
    links.push($('<a>', {href: '#metrics'}).append('All metrics'));
    $output.append(renderItems(links));

    return $output;
  }

  function renderGroupHeader(scenario) {
    const $output = $('<div>');
    if (urlParams.group) {
      $.getJSON(groupsMetadataJsonUrl(suite), {}, (response) => {
        const group = response[urlParams.group];
        if (group) {
          let groupName = group.display_name;
          if (urlParams.subgroup) {
            groupName += " / " + urlParams.subgroup;
          }
          $output.append($('<h3>').append(groupName));
          $output.append($('<div>').append($('<i>').append(renderMarkdown(group.description))));
          if (group.taxonomy) {
            const $rows = Object.entries(group.taxonomy).map(([k, v]) => {
              return $('<tr>').append([
                $('<td>').append(`<b>${k}</b>`),
                $('<td>').append(v),
              ]);
            });
            $output.append($('<table>', {class: 'taxonomy-table'}).append($rows));
          }
        }
      });
    } else if (scenario) {
      $output.append($('<h3>').append(renderScenarioDisplayName(scenario)));
      $output.append($('<div>').append($('<i>').append(renderMarkdown(scenario.description))));
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
        $instance.append($('<div>').append('Input:'));
        $instance.append($('<div>', {class: 'instance-input'}).append(multilineHtml(input)));

        // References
        if (instance.references.length > 0) {
          $instance.append($('<div>').append(instance.references.length === 1 ? 'Reference:' : 'References:'));
          const $references = $('<ul>');
          instance.references.forEach((reference, referenceIndex) => {
            const originalReference = instance.perturbation && originalInstance.references[referenceIndex];
            const output = instance.perturbation ? highlightNewWords(reference.output, originalReference.output) : reference.output;
            const suffix = reference.tags.length > 0 ? ' ' + ('[' + reference.tags.join(',') + ']').bold() : '';
            $references.append($('<li>').append([$('<span>', {class: 'instance-reference'}).append(output), suffix]));
          });
          $instance.append($references);
        }
      }
      $prediction = $('<div>');
      $prediction.addClass('prediction').text('Loading predictions...');
      $instance.append($prediction);
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
    perInstanceStats.forEach((entry) => {
      const key = perInstanceStatsKey(entry);
      instanceKeyTrialToStats[key] = (instanceKeyTrialToStats[key] || []).concat(entry.stats);
    });

    // For each request state (across all instances)...
    scenarioState.request_states.forEach((requestState) => {
      const $instanceDiv = instanceKeyToDiv[instanceKey(requestState.instance)];
      if (!$instanceDiv) {
        console.error('Not found: ' + instanceKey(requestState.instance));
        return;
      }

      // Traverse into the prediction div within the instance div
      const $instance = $instanceDiv.find('.prediction');
      $instance.empty();

      // For adapter method = separate, don't show the calibration requests
      if (requestState.request_mode === 'calibration') {
        return;
      }

      const key = instanceTrialKey(requestState.instance, requestState.train_trial_index);

      // For multiple_choice_separate_*, only render the request state for the predicted index
      if (requestState.reference_index !== undefined) {
        const predictedIndexStat = instanceKeyTrialToStats[key] ?
          instanceKeyTrialToStats[key].find((stat) => stat.name.name === "predicted_index") :
          undefined;
        if (predictedIndexStat === undefined) {
          console.warn("Cannot find predicted index for: ", key);
        } else if (requestState.reference_index !== predictedIndexStat.mean) {
          return;
        }
      }

      // Print out instance-level statistics.
      // Show it once for each (instance_id, perturbation) and train_trial_index.
      const stats = instanceKeyTrialToStats[key];
      if (!stats) {
        console.warn("Cannot find stats for", key, instanceKeyTrialToStats);
      } else {
        $instance.append(renderPerInstanceStats(runSpec.groups, stats, runDisplayName));
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
      if (requestState.reference_index !== undefined) {
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
      $prediction = $('<div>')
        .append($link)
        .append(': ')
        .append(prefix)
        .append(prediction)
        .append($logProb);
      $instance.append($prediction);
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
      return statsJsonUrl(suite, runSpec.name);
    });
    const perInstanceStatsPaths = runSpecs.map((runSpec) => {
      return perInstanceStatsJsonUrl(suite, runSpec.name);
    });
    const scenarioPaths = runSpecs.map((runSpec) => {
      return scenarioJsonUrl(suite, runSpec.name);
    });
    const scenarioStatePaths = runSpecs.map((runSpec) => {
      return scenarioStateJsonUrl(suite, runSpec.name);
    });
    const runSpecPaths = runSpecs.map((runSpec) => {
      return runSpecJsonUrl(suite, runSpec.name);
    });

    // Figure out short names for the runs based on where they differ
    const runDisplayNames = findDiff(runSpecs.map((runSpec) => runSpec.adapter_spec)).map(renderDict);

    // Setup the basic HTML elements
    const $root = $('<div>');
    const $scenarioInfo = $('<div>', {class: 'scenario-info'});
    $scenarioInfo.text('Loading scenario info...');
    $root.append($scenarioInfo);

    // Adapter
    $root.append($('<a>', {name: 'adapter'}).append($('<h5>').append('Adapter specification')));
    const $adapterSpec = $('<table>');
    if (runSpecs.length > 1) {
      $adapterSpec.append($('<tr>').append($('<td>'))
        .append(runDisplayNames.map((name) => $('<td>').append(name))));
    }
    $root.append($('<div>', {class: 'table-container'}).append($adapterSpec));

    // Instances
    $root.append($('<a>', {name: 'instances'}).append($('<h5>').append('Instances + predictions')));
    const $instancesContainer = $('<div>');
    $instancesContainer.addClass('table-container').text('Loading instances...')
    $root.append($instancesContainer);

    // Metrics
    $root.append($('<a>', {name: 'metrics'}).append($('<h5>').append('All metrics')));
    const $statsContainer = $('<div>');
    $statsContainer.addClass('table-container').text('Loading metrics...')
    $root.append($statsContainer);

    // Render adapter specs
    $adapterSpec.append($('<tr>').append($('<td>')).append(scenarioStatePaths.map((scenarioStatePath, index) => {
      return $('<td>')
        .append($('<a>', {href: runSpecPaths[index]}).append('Spec JSON'))
        .append(' | ')
        .append($('<a>', {href: scenarioStatePaths[index]}).append('Full JSON'));
    })));
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

    // Render metrics/stats
    getJSONList(statsPaths, (statsList) => {
      console.log('metrics', statsList);
      if (statsList.length && statsList.every((stat) => stat.length === 0)) {
        $statsContainer.empty().text("Metrics are currently unavailable. Please try again later.")
        return;
      }
      const $stats = $('<table>');
      const $statsSearch = $('<input>', {type: 'text', size: 40, placeholder: 'Enter keywords to filter metrics'});
      if (runSpecs.length > 1) {
        $stats.append($('<tr>').append($('<td>')).append(runDisplayNames.map((name) => $('<td>').append(name))));
      }
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
      $statsContainer.empty().append($statsSearch).append($stats);
    }, []);

    // TODO: Get all JSON files in parallel.
    // Render scenario instances
    const instanceToDiv = {};  // For each instance
    const scenariosPromise = getJSONList(scenarioPaths);
    const scenarioStatesPromise = getJSONList(scenarioStatePaths);
    const perInstanceStatsPromise = getJSONList(perInstanceStatsPaths);
    const instanceKeyToDivPromise = scenariosPromise.then((scenarios) => {
      console.log('scenarios', scenarios);
      if (scenarios.length && scenarios.every((scenario) => scenario === undefined)) {
        $instancesContainer.empty().text("Instances and predictions are currently unavailable. Please try again later.")
        return;
      }

      const onlyOneScenario = scenarios.length && scenarios.every((scenario) => scenario.definition_path === scenarios[0].definition_path);
      const scenario = onlyOneScenario ? scenarios[0] : null;
      const scenarioPath = onlyOneScenario ? scenarioPaths[0] : null;
      $scenarioInfo.empty();
      $scenarioInfo.append(renderRunsHeader(scenario, scenarioPath));

      const $instances = $('<div>');
      const instanceKeyToDiv = renderScenarioInstances(scenarios[0], $instances);
      $instancesContainer.empty().append($instances);
      return instanceKeyToDiv;
    });
    $.when(scenarioStatesPromise, perInstanceStatsPromise, instanceKeyToDivPromise).then((scenarioStates, perInstanceStats, instanceKeyToDiv) => {
      console.log('scenarioStates', scenarioStates);
      console.log('perInstanceStats', perInstanceStats);
      // For each run / model...
      runSpecs.forEach((runSpec, index) => {
        renderPredictions(runSpec, runDisplayNames[index], scenarioStates[index], perInstanceStats[index], instanceKeyToDiv);
      });
    });
    return $root;
  }

  function rootUrl(model) {
    return encodeUrlParams({});
  }

  function modelUrl(model) {
    return encodeUrlParams(Object.assign({}, urlParams, {scenarios: null, models: 1}));
  }

  function groupUrl(group) {
    return encodeUrlParams(Object.assign({}, urlParams, {scenarios: null, group}));
  }

  function metricUrl(group) {
    // e.g., Calibration
    return encodeUrlParams(Object.assign({}, urlParams, {group: 'core_scenarios'})) + '#' + group.display_name;
  }

  function groupShortDisplayName(group) {
    return group.short_display_name || group.display_name || group.name;
  }

  function metricDisplayName(metric) {
    return (metric.display_name || metric.name) + (metric.perturbation_name ? ' (perturbation: ' + metric.perturbation_name + ')' : '');
  }

  function renderModelList() {
    const $result = $('<div>', {class: 'col-sm-3'});
    const models = schema.models;
    const numModels = models.filter((model) => !model.todo).length;
    $result.append($('<div>', {class: 'list-header'}).append(`${numModels} models`));
    models.forEach((model) => {
      const extra = model.todo ? ' list-item-todo' : '';
      const display_name = model.creator_organization + ' / ' + model.display_name;
      const $item = $('<a>', {href: modelUrl(model.name), class: 'list-item' + extra, title: model.description}).append(display_name);
      $result.append($('<div>').append($item));
    });
    return $result;
  }

  function renderScenarioList() {
    const $result = $('<div>', {class: 'col-sm-3'});

    const nameToGroup = {};
    schema.run_groups.forEach((group) => {
      nameToGroup[group.name] = group;
    });

    // There are two types of groups we care about:
    // 1) Top-level groups (e.g., question_answering)
    // 2) Scenario-level groups (e.g., mmlu)
    const topGroups = schema.run_groups.filter((group) => {
      // Must have subgroups
      return group.subgroups && ['Core scenarios', 'Targeted evaluations'].includes(group.category);
    });

    const scenarioGroupNames = {};
    topGroups.forEach((group) => {
      group.subgroups.forEach((subgroupName) => {
        if (!nameToGroup[subgroupName].todo) {
          scenarioGroupNames[subgroupName] = true;
        }
      });
    });
    const numScenarios = Object.keys(scenarioGroupNames).length;

    $result.append($('<div>', {class: 'list-header'}).append(`${numScenarios} scenarios`));
    topGroups.forEach((group) => {
      const $group = $('<div>');
      $group.append($('<a>', {href: groupUrl(group.name), class: 'list-item', title: group.description}).append(groupShortDisplayName(group)));
      $group.append($('<ul>').append(group.subgroups.map((subgroupName) => {
        const subgroup = nameToGroup[subgroupName];
        const extra = subgroup.todo ? ' list-item-todo' : '';
        const $item = $('<a>', {href: groupUrl(subgroup.name), class: 'list-item' + extra, title: subgroup.description}).append(groupShortDisplayName(subgroup));
        return $('<li>').append($item);
      })));
      $result.append($group);
    });
    return $result;
  }

  function renderMetricsList() {
    const $result = $('<div>', {class: 'col-sm-3'});

    // Information about individual metrics
    const nameToMetric = {};
    schema.metrics.forEach((metric) => {
      nameToMetric[metric.name] = metric;
    });

    // Some metric groups depend on environment variables like ${main_name}
    // Look at the places where that's being used across the runs.
    // For each metric group, compute the deduped list of main_names.
    // Example: accuracy => [quasi_exact_match, f1_score, ...]
    const metricGroupToMainNames = {};
    schema.run_groups.forEach((group) => {
      if (group.metric_groups) {
        group.metric_groups.forEach((metricGroup) => {
          if (group.environment.main_name) {
            const old = metricGroupToMainNames[metricGroup] || [];
            if (!old.includes(group.environment.main_name)) {
              metricGroupToMainNames[metricGroup] = old.concat([group.environment.main_name]);
            }
          }
        });
      }
    });

    const metricGroups = schema.metric_groups.filter((group) => {
      // Skip a group if "_detailed" exists.
      return !schema.metric_groups.some((group2) => group2.name === group.name + '_detailed');
    }).map((group) => {
      // Expand the metrics for this metric group
      const newMetrics = [];
      group.metrics.forEach((metric) => {
        if (metric.name === '${main_name}') {
          (metricGroupToMainNames[group.name.replace('_detailed', '')] || []).forEach((name) => {
            newMetrics.push(Object.assign({}, metric, {name}));
          });
        } else {
          newMetrics.push(metric);
        }
      });
      return Object.assign({}, group, {metrics: newMetrics});
    });

    // Count the number of metrics
    const metricNames = {};
    metricGroups.forEach((group) => {
      group.metrics.forEach((metric) => {
        metricNames[metric.name] = true;
      });
    });
    const numMetrics = Object.keys(metricNames).length;

    $result.append($('<div>', {class: 'list-header'}).append(`${numMetrics} metrics`));
    metricGroups.forEach((group) => {
      const $group = $('<div>');
      $group.append($('<a>', {href: metricUrl(group), class: 'list-item', title: group.description}).append(groupShortDisplayName(group)));
      $group.append($('<ul>').append(group.metrics.map((metricRef) => {
        // Get the information from the metric (name, display_name, description)
        const metric = Object.assign({}, metricRef, nameToMetric[metricRef.name] || metricRef);
        const $item = $('<a>', {class: 'list-item', title: metric.description}).append(metricDisplayName(metric));
        return $('<li>').append($item);
      })));
      $result.append($group);
    });
    return $result;
  }

  function helmLogo() {
    return $('<a>', {href: rootUrl()}).append($('<img>', {src: 'images/helm-logo.png', width: '500px', class: 'mx-auto d-block'}));
  };

  function button(text, href) {
    return $('<a>', {href, class: 'main-link btn btn-lg m-5 px-5'}).append(text);
  }

  function renderMainPage() {
    const $result = $('<div>', {class: 'row'});

    $result.append($('<div>', {class: 'col-sm-12'}).append(helmLogo()));

    const $blog = button('Blog post', 'https://crfm.stanford.edu/2022/11/17/helm.html');
    const $paper = button('Paper', 'https://arxiv.org/pdf/2211.09110.pdf');
    const $code = button('GitHub', 'https://github.com/stanford-crfm/helm');
    $result.append($('<div>', {class: 'col-sm-12'}).append($('<div>', {class: 'text-center'}).append([$blog, $paper, $code])));

    const $description = $('<div>', {class: 'col-sm-8'}).append([
      'A language model takes in text and produces text:',
      $('<div>', {class: 'text-center'}).append($('<img>', {src: 'images/language-model-helm.png', width: '600px', style: 'width: 600px; margin-left: 130px'})),
      'Despite their simplicity, language models are increasingly functioning as the foundation for almost all language technologies from question answering to summarization.',
      ' ',
      'But their immense capabilities and risks are not well understood.',
      ' ',
      'Holistic Evaluation of Language Models (HELM) is a living benchmark that aims to improve the transparency of language models.'
    ]);

    function organization(src, href, height) {
      return $('<div>', {class: 'logo-item'}).append($('<a>', {href}).append($('<img>', {src, height})));
    }
    const defaultSize = 36;
    const largerSize = 50;
    const $organizations = $('<div>', {class: 'logo-container'}).append([
      organization('images/organizations/ai21.png', 'https://www.ai21.com/', defaultSize),
      organization('images/organizations/anthropic.png', 'https://www.anthropic.com/', defaultSize),
      organization('images/organizations/bigscience.png', 'https://bigscience.huggingface.co/', largerSize),
      organization('images/organizations/cohere.png', 'https://cohere.ai/', defaultSize),
      organization('images/organizations/eleutherai.png', 'https://www.eleuther.ai/', largerSize),
      organization('images/organizations/google.png', 'https://ai.google/', defaultSize),
      organization('images/organizations/meta.png', 'https://ai.facebook.com/', largerSize),
      organization('images/organizations/microsoft.png', 'https://turing.microsoft.com/', defaultSize),
      organization('images/organizations/nvidia.png', 'https://www.nvidia.com/en-us/research/machine-learning-artificial-intelligence/', largerSize),
      organization('images/organizations/openai.png', 'https://openai.com/', defaultSize),
      organization('images/organizations/tsinghua-keg.png', 'https://keg.cs.tsinghua.edu.cn/', largerSize),
      organization('images/organizations/yandex.png', 'https://yandex.com/', defaultSize),
      organization('images/organizations/together.png', 'https://together.xyz/', defaultSize),
    ]);
    $result.append($organizations);

    $description.append($('<ol>').append([
      $('<li>').append('<b>Broad coverage and recognition of incompleteness</b>. We define a taxonomy over the scenarios we would ideally like to evaluate, select scenarios and metrics to cover the space and make explicit what is missing.')
               .append($('<div>', {class: 'text-center'}).append($('<img>', {src: 'images/taxonomy-scenarios.png', width: '300px'}))),
      $('<li>').append('<b>Multi-metric measurement</b>. Rather than focus on isolated metrics such as accuracy, we simultaneously measure multiple metrics (e.g., accuracy, robustness, calibration, efficiency) for each scenario, allowing analysis of tradeoffs.')
               .append($('<div>', {class: 'text-center'}).append($('<img>', {src: 'images/scenarios-by-metrics.png', width: '300px'}))),
      $('<li>').append('<b>Standardization</b>. We evaluate all the models that we have access to on the same scenarios with the same adaptation strategy (e.g., prompting), allowing for controlled comparisons. Thanks to all the companies for providing API access to the limited-access and closed models and <a href="https://together.xyz">Together</a> for providing the infrastructure to run the open models.')
               .append($organizations),
      $('<li>').append('<b>Transparency</b>. All the scenarios, predictions, prompts, code are available for further analysis on this website. We invite you to click below to explore!'),
    ]));

    $result.append([
      $('<div>', {class: 'col-sm-2'}),
      $description,
      $('<div>', {class: 'col-sm-2'}),
    ]);

    const $models = renderModelList();
    const $scenarios = renderScenarioList();
    const $metrics = renderMetricsList();
    $result.append([
      $('<div>', {class: 'col-sm-2'}),
      $models, $scenarios, $metrics,
      $('<div>', {class: 'col-sm-1'}),
    ]);

    return $result;
  }

  function renderCell(cell) {
    let value = cell.display_value || cell.value;
    if (value == null) {
      value = '-';
    }
    if (typeof(value) === 'number') {
      value = Math.round(value * 1000) / 1000;
    }
    const $value = $('<span>');
    if (cell.markdown && value) {
      value = renderMarkdown('' + value);
      $value.append(value);
    } else {
      $value.text(value);
    }
    if (cell.style) {
      $value.css(cell.style);
    }
    if (cell.description) {
      $value.attr("title", cell.description)
    }
    const $linkedValue = cell.href ? $('<a>', {href: cell.href}).append($value) : $value;
    return $('<td>').append($linkedValue);
  }

  function renderTable(table) {
    const $output = $('<div>');
    $output.append($('<h3>').append($('<a>', {name: table.title}).append(table.title)));
    const $table = $('<table>', {class: 'query-table results-table'});
    const $header = $('<tr>').append(table.header.map(renderCell));
    $table.append($header);

    table.rows.forEach((row) => {
      const $row = $('<tr>').append(row.map(renderCell));
      $table.append($row);
    });
    $output.append($table);

    // Links
    if (table.links.length > 0) {
      $output.append(renderItems(table.links.map((link) => {
        return $('<a>', {href: link.href}).append(link.text);
      })));
    }

    return $output;
  }

  function renderTables(tables, path) {
    const $output = $('<div>');

    // Links to tables
    const $jsonLink = $('<a>', {href: path}).append('JSON');
    $output.append(renderItems(tables.map((table) => {
      return $('<a>', {href: '#' + table.title}).append(table.title);
    }).concat([$jsonLink])));

    $output.append(tables.map((table) => {
      return $('<div>', {class: 'table-container', id: table.title}).append(renderTable(table));
    }));

    return $output;
  }

  //////////////////////////////////////////////////////////////////////////////
  //                                   Main                                   //
  //////////////////////////////////////////////////////////////////////////////

  const $main = $('#main');
  const $summary = $('#summary');
  $.when(
    $.get('schema.yaml', {}, (response) => {
      const raw = jsyaml.load(response);
      console.log('schema', raw);
      schema = new Schema(raw);
    }),
    $.get(summaryJsonUrl(suite), {}, (response) => {
      console.log('summary', response);
      summary = response;
      $summary.append(`${summary.suite} (last updated ${summary.date})`);
    }),
  ).then(() => {
    if (urlParams.models) {
      // Models
      $main.empty()
      $main.append(renderHeader('Models', renderModels()));
      refreshHashLocation();
    } else if (urlParams.scenarios) {
      // Models
      $main.empty()
      $main.append(renderHeader('Scenarios', renderScenarios()));
      refreshHashLocation();
    } else if (urlParams.runSpec || urlParams.runSpecs || urlParams.runSpecRegex) {
      // Predictions for a set of run specs (matching a regular expression)
      $main.text('Loading runs...');
      $.getJSON(runSpecsJsonUrl(suite), {}, (response) => {
        $main.empty();
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
        refreshHashLocation();
      });
    } else if (urlParams.runs) {
      // All runs (with search)
      $main.text('Loading runs...');
      $.getJSON(runSpecsJsonUrl(suite), {}, (runSpecs) => {
        $main.empty();
        console.log('runSpecs', runSpecs);
        $main.append(renderHeader('Runs', renderRunsOverview(runSpecs)));
      });
    } else if (urlParams.groups) {
      // All groups
      $main.text('Loading groups...');
      const path = groupsJsonUrl(suite);
      $.getJSON(path, {}, (tables) => {
        $main.empty();
        console.log('groups', tables);
        $main.append(renderTables(tables, path));
        refreshHashLocation();
      });
    } else if (urlParams.group) {
      // Specific group
      $main.text('Loading group...');
      const path = groupJsonUrl(suite, urlParams.group);
      $.getJSON(path, {}, (tables) => {
        $main.empty();
        console.log('group', tables);
        $main.append(renderGroupHeader(urlParams.group));
        $main.append(renderTables(tables, path));
        refreshHashLocation();
      });
    } else {
      // Main landing page
      $main.empty()
      $main.append(renderMainPage());
    }
  });
});
