/**
 * A very simple static way to visualize the scenarios, runs, and metrics from the benchmarking project.
 * This code doesn't really belong in `proxy`, but is there for convenience.
 */
$(function () {
  const urlParams = decodeUrlParams(window.location.search);
  // Extract the name of the suite from the URL parameters. Default to "latest" if none is specified.
  const suite = "suite" in urlParams ? urlParams.suite : "latest";
  console.log(`Suite: ${suite}`);

  //////////////////////////////// Schema //////////////////////////////////////

  // Captures information about a field in the schema.
  class Field {
    constructor(raw) {
      this.name = raw.name;
      this.display_name = raw.display_name;
      this.description = raw.description;
    }
  }

  // Captures information about a field of an adapter (e.g.,
  // max_train_instances) or a metric name (e.g., exact_match).
  class AdapterField extends Field {
    constructor(raw) {
      super(raw);
      this.values = this.readValues(raw.values);
    }

    readValues(values) {
      // Read the values field.
      // Note: We are using `Field` to represent the schema for a field value too.
      if (Array.isArray(values)) {
        // If the values field is an array, read each element as a Field.
        return values.map((valueRaw) => new Field(valueRaw));
      } else if (values === undefined) {
        return undefined;
      }
      // If no matching schema is found, raise an error!
      console.error(`The values field of ${this.name} should be an array or an object. Instead found: ${values}.`);
    }
  }

  // Specifies all the information to help us render and understand the fields
  // for adapters and metrics.
  class Schema {
    constructor(raw) {
      this.adapterFields = raw.adapter.map((fieldRaw) => new AdapterField(fieldRaw));
      this.metricsFields = raw.metrics.map((fieldRaw) => new Field(fieldRaw));

      this.scenario_groups = raw.scenario_groups;
      this.metric_groups = raw.metric_groups;

      // Allow convenient access
      this.adapterFieldNames = this.adapterFields.map((field) => field.name);
      this.metricsFieldNames = this.metricsFields.map((field) => field.name);
    }

    adapterField(name) {
      // Return the adapter field with the given `name`.
      const field = this.adapterFields.find((field) => field.name === name);
      return field || new Field({name});
    }

    metricsField(name) {
      // Return the metrics field with the given `name`.
      const field = this.metricsFields.find((field) => field.name === name);
      return field || new Field({name});
    }

    metricGroup(name) {
      return this.metric_groups.find((group) => group.name === name);
    }
  }

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

  function renderGroups(groups) {
    const $table = $('<table>', {class: 'query-table'});
    groups.forEach((group) => {
      const params = encodeUrlParams(Object.assign({}, {group: group.name}));
      const href = `benchmarking.html${params}`;
      const $row = $('<tr>').append([
        $('<td>').append($('<a>', {href: href}).append(group.display_name)),
        $('<td>').append(group.description),
      ]);
      $table.append($row);
    });
    return $table;
  }

  function renderRunsOverview(runSpecs) {
    let query = '';
    const $search = $('<input>', {type: 'text', size: 40, placeholder: 'Enter regex query (enter to open all)'});
    $search.keyup((e) => {
      // Open up all match specs
      if (e.keyCode === 13) {
        const href = encodeUrlParams(Object.assign(urlParams, {runSpec: '.*' + query + '.*'}));
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
        // To maintain backward compatibility, as `scenario` in `RunSpec` was renamed to `scenario_spec`.
        const scenario_spec = runSpec.hasOwnProperty('scenario_spec') ? runSpec.scenario_spec : runSpec.scenario;
        const href = encodeUrlParams(Object.assign(urlParams, {runSpec: runSpec.name}));
        const $row = $('<tr>')
          .append($('<td>').append($('<a>', {href}).append(runSpec.name)))
          .append($('<td>').append(runSpec.adapter_spec.method))
        $table.append($row);
      });
    }

    renderTable();

    return $('<div>').append([$search, $table]);
  }

  function substitute(str, environment) {
    if (!str) {
      return str;
    }
    for (let key in environment) {
      str = str.replace('${' + key + '}', environment[key]);
    }
    return str;
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

  function getStatClass(name, value) {
    // Return the CSS class to use if a stat has `value`.

    // Based on `name` determine whether smaller or larger is better.
    // Assume larger is better for now.
    if (value === 0) {
      return 'wrong';
    }
    if (value === 1) {
      return 'correct';
    }
    return '';
  }

  function renderStats(groups, stats) {
    // This is used to render per-instance stats (which is why we only care
    // about the metric name).
    // Groups specifies which metric names we should display.
    // Pull these out from stats and render them.
    const list = [];

    // Look for the default metrics for the group
    schema.scenario_groups.forEach((scenarioGroup) => {
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
        list.push($('<span>', {class: getStatClass(stat.name.name, stat.mean)}).append(field.display_name + ': ' + stat.mean));
      }
    });

    // String the metrics together
    const $stats = $('<div>');
    //$stats.append('Metrics: ');
    if (list.length == 0) {
      $stats.append(' (none)');
    } else {
      list.forEach((item, index) => {
        if (index > 0) {
          $stats.append(', ');
        }
        $stats.append(item);
      });
    }

    return $stats;
  }

  function renderRunsDetailed(runSpecs) {
    // Render all the `runSpecs`:
    // - Instances + predictions
    // - Adapter specification
    // - Stats
    // For each block, we show a table and each `runSpec` is a column.
    const CORRECT_TAG = 'correct';

    // Used to hash instances.
    function instanceKey(instance) {
      return JSON.stringify(instance);
    }

    // Paths (parallel arrays corresponding to `runSpecs`)
    const metricsPaths = runSpecs.map((runSpec) => {
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
    const $metrics = $('<table>');
    const $metricsSearch = $('<input>', {type: 'text', size: 40, placeholder: 'Enter keywords to filter metrics'});
    if (runSpecs.length > 1) {
      $metrics.append($('<tr>').append($('<td>')).append(runDisplayNames.map((name) => $('<td>').append(name))));
    }
    $root.append($('<div>', {class: 'table-container'}).append($metricsSearch).append($metrics));

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

    // Render metrics
    getJSONList(metricsPaths, (metricsList) => {
      console.log('metrics', metricsList);
      const keys = canonicalizeList(metricsList.map((metrics) => metrics.map((metric) => metric.name)));

      // Sort
      keys.sort((k1, k2) => {
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
      });

      // Filter 
      let query = '';
      $metricsSearch.keyup((e) => {
        query = $metricsSearch.val();
        renderMetrics();
      });

      function renderMetrics() {
        $metrics.empty();
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
          metricsList.forEach((metrics) => {
            // metrics is a list of statistics corresponding to one run (column)
            const metric = metrics.find((metric) => metricNameEquals(metric.name, key));
            $row.append($('<td>').append(metric ? renderFieldValue(field, round(metric.mean, 3)) : '?'));
          });
          $metrics.append($row);
        });
        $metrics.append($('<tr>').append($('<td>'))
          .append(metricsPaths.map((metricsPath) => $('<td>').append($('<a>', {href: metricsPath}).append('JSON')))));
      }

      renderMetrics();

    }, []);

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

    // Render scenario instances
    const instanceToDiv = {};
    getJSONList(scenarioPaths, (scenarios) => {
      console.log('scenarios', scenarios);

      // Only grab the first scenario
      const i = 0;
      $scenarioInfo.empty();
      $scenarioInfo.append($('<h3>').append(scenarios[i].name));
      $scenarioInfo.append($('<div>').append($('<i>').append(scenarios[i].description)));
      $scenarioInfo.append($('<div>')
        .append($('<a>', {href: scenarios[i].definition_path}).append('[code]'))
        .append(' ').append($('<a>', {href: scenarioPaths[i]}).append('[JSON]'))
        .append(' ').append($('<a>', {href: '#adapter'}).append('[adapter]'))
        .append(' ').append($('<a>', {href: '#instances'}).append('[instances]'))
        .append(' ').append($('<a>', {href: '#metrics'}).append('[metrics]'))
      );

      scenarios.forEach((scenario) => {
        // Keep track of the original (unperturbed) instances
        const id2originalInstance = {};
        scenario.instances.forEach((instance) => {
          if (!instance.perturbation) {
            id2originalInstance[instance.id] = instance;
          }
        });

        scenario.instances.forEach((instance, instanceIndex) => {
          const key = instanceKey(instance);
          if (key in instanceToDiv) {
            return;
          }

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
            $instance.append(multilineHtml(input));
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
          instanceToDiv[key] = $instance;
        });
      });

      // Render the model predictions
      getJSONList(scenarioStatePaths, (scenarioStates) => {
        console.log('scenarioStates', scenarioStates);
        getJSONList(perInstanceStatsPaths, (perInstanceStats) => {
          console.log('perInstanceStats', perInstanceStats);

          scenarioStates.forEach((scenarioState, index) => {
            // Build mapping to stats
            const instanceTrialToStats = {};
            perInstanceStats[index].forEach((instanceTrialStats) => {
              const key = [instanceTrialStats.instance_id, instanceTrialStats.trial_index];
              instanceTrialToStats[key] = (instanceTrialToStats[key] || []).concat(instanceTrialStats.stats);
            });

            // Go through all the request states
            scenarioState.request_states.forEach((requestState) => {
              const $instance = instanceToDiv[instanceKey(requestState.instance)];
              if (!$instance) {
                console.log('Not found: ' + instanceKey(requestState.instance));
                return;
              }

              // For adapter method = separate, don't show the calibration
              if (requestState.request_mode === 'calibration') {
                return;
              }

              // Print out statistics
              if (requestState.reference_index == null || requestState.reference_index === 0) {
                // Keep only stats that match instance ID, train trial index, and perturbatation
                const stats = instanceTrialToStats[[requestState.instance.id, requestState.train_trial_index]].filter((stat) => {
                  const p1 = requestState.instance.perturbation;
                  const p2 = stat.name.perturbation;
                  return (p1 && p1.name) === (p2 && p2.name);
                });
                $instance.append(renderStats(runSpecs[index].groups, stats));
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
              let prefix = '';
              let prediction = $('<i>').append('(empty)');
              const $logProb = $('<span>');
              if (requestState.result) {
                // Assume there is only one completion
                const completion = requestState.result.completions[0];
                prediction = completion.text.trim();
                if (requestState.output_mapping) {
                  prediction = requestState.output_mapping[prediction];
                }

                if (requestState.request_mode === 'original') {
                  // For adapter method = separate, prediction starts with the prompt, strip it out
                  if (prediction.startsWith(requestState.instance.input)) {
                    prefix = '...';
                    prediction = prediction.substring(requestState.instance.input.length).trim();
                  }
                }

                $logProb.append(' ').append($('<span>', {class: 'logprob'}).append('(' + round(completion.logprob, 3) + ')'));
              }

              let description = '';
              if (requestState.reference_index !== null) {
                description += '[' + requestState.reference_index + ']';
              }
              if (runSpecs.length > 1) {
                description += '(' + runDisplayNames[index] + ')';
              }
              $instance.append($('<div>')
                .append($('<a>', {href}).append($('<b>').append('Prediction' + description)))
                .append(': ')
                .append(prefix + prediction)
                .append($logProb));
            });
          });
        });
      });
    });

    return $root;
  }

  function renderLandingPage() {
    const $intro = $('<div>').append('Welcome to the CRFM benchmarking project!');
    const $links = $('<ul>').append(
      $('<li>').append($('<a>', {href: '?models'}).append('Models')),
      $('<li>').append($('<a>', {href: '?groups'}).append('Scenario groups')),
      $('<li>').append($('<a>', {href: '?runs'}).append('Runs')),
    );
    return $('<div>').append($intro).append($links);
  }

  function renderCell(cell) {
    const value = $('<span>', {title: cell.description}).append(cell.display_value || cell.value);
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
    $output.append($('<a>', {href: '?latex=' + table.title.replaceAll(" ", "_").replace("/", "_")}).append('[latex]'));
    return $output;
  }

  function renderTables(tables) {
    const $output = $('<div>');
    tables.forEach((table) => {
      $output.append($('<div>', {class: 'table-container'}).append(renderTable(table)));
    });
    return $output;
  }

  function renderLatex(latex) {
    const $output = $('<div>', {class: 'latex'}).append(latex);
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
      $.getJSON(`benchmark_output/runs/${suite}/models.json`, {}, (response) => {
        const models = response;
        console.log('models', models);
        $main.append(renderHeader('Models', renderModels(models)));
      });
    } else if (urlParams.runSpec) {
      // Display a set of run specs
      $.getJSON(`benchmark_output/runs/${suite}/run_specs.json`, {}, (response) => {
        const runSpecs = response;
        console.log('runSpecs', runSpecs);
        const matchedRunSpecs = runSpecs.filter((runSpec) => new RegExp('^' + urlParams.runSpec + '$').test(runSpec.name));
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
        $main.append(renderTable(response));
      });
    } else if (urlParams.group) {
      // Specific group
      $.getJSON(`benchmark_output/runs/${suite}/groups/${urlParams.group}.json`, {}, (tables) => {
        console.log('group', tables);
        $main.append(renderTables(tables));
      });
    } else if (urlParams.latex) {
      // Tex corresponding to a group
      $.get(`benchmark_output/runs/${suite}/groups/latex/${urlParams.latex}.tex`, {}, (latex) => {
        console.log('latex', latex);
        $main.append(renderLatex(latex));
      });
    } else {
      $main.append(renderLandingPage());
    }
  });
});
