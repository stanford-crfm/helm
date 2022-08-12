/**
 * A very simple static way to visualize the scenarios, runs, and metrics from the benchmarking project.
 * This code doesn't really belong in `proxy`, but is there for convenience.
 */
$(function () {
  const urlParams = decodeUrlParams(window.location.search);
  // Extract the name of the suite from the URL parameters. Default to "latest" if none is specified.
  const suite = "suite" in urlParams ? urlParams.suite : "latest";
  console.log(`Suite: ${suite}`);

  ////////////////////////////////// Main //////////////////////////////////////

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
      // Possible values this field can take
      this.values = this.readValues(raw.values);
    }

    readValues(values) {
      // Read the values field.
      // Note: We are using field to represent the schema for a field value too.
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

  // Captures information about a field of a group.
  class GroupField extends Field {
    constructor(raw) {
      super(raw);
      // Possible values this field can take
      this.display = raw.display;
    }
  }

  // Captures information about a field of a metric group.
  class MetricGroupField extends Field {
    constructor(raw) {
      super(raw);
      // Possible values this field can take
      this.display_k = raw.display_k;
      this.stat_names = raw.stat_names;
      this.perturbation_names = raw.perturbation_names;
    }
  }

  // Specifies all the information to help us render and understand the fields
  // for adapters and metrics.
  class Schema {
    constructor(raw) {
      this.adapterFields = raw.adapter.map((fieldRaw) => new AdapterField(fieldRaw));
      this.metricsFields = raw.metrics.map((fieldRaw) => new Field(fieldRaw));
      this.perturbationsFields = raw.perturbations.map((fieldRaw) => new Field(fieldRaw));
      this.groupsFields = raw.groups.map((fieldRaw) => new GroupField(fieldRaw));
      this.metricGroupsFields = raw.metric_groups.map((fieldRaw) => new MetricGroupField(fieldRaw));

      // Allow convenient access
      this.adapterFieldNames = this.adapterFields.map((field) => field.name);
      this.metricsFieldNames = this.metricsFields.map((field) => field.name);
      this.perturbationsFieldNames = this.perturbationsFields.map((field) => field.name);
      this.groupsFieldNames = this.groupsFields.map((field) => field.name);
      this.metricGroupsFieldNames = this.metricGroupsFields.map((field) => field.name);
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

    groupsField(name) {
      // Return the group field with the given `name`.
      const field = this.groupsFields.find((field) => field.name === name);
      return field || new Field({name});
    }

    metricGroupsField(name) {
      // Return the group field with the given `name`.
      const field = this.metricGroupsFields.find((field) => field.name === name);
      return field || new Field({name});
    }

    tableSettingsField(name) {
      // Return the group field with the given `name`.
      const field = this.tableSettingsFields.find((field) => field.name === name);
      return field || new Field({name});
    }

    perturbationsField(name) {
      // Return the group field with the given `name`.
      const field = this.perturbationsFields.find((field) => field.name === name);
      return field || new Field({name});
    }

  }

  function renderModels(models) {
    const $table = $('<table>', {class: 'query-table'});
    models.forEach((model) => {
      const $row = $('<tr>').append($('<td>').append(`${model.description} [${model.name}]`));
      $table.append($row);
    });
    return $table;
  }
  
  function renderGroups(groups) {
    const $table = $('<table>', {class: 'query-table'});
    groups.forEach((group) => {
      const params = encodeUrlParams(Object.assign({}, {group: group.name}));
      const href = `benchmarking.html${params}`;
      const $row = $('<tr>').append($('<td>').append($('<a>', {href: href}).append(group.name)));
      $table.append($row);
    });
    return $table;
  }

  /////////////////////////////////// Pages ////////////////////////////////////

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
          .append($('<td>').append($('<b>').append('Scenario')))
          .append($('<td>').append($('<b>').append('Model')))
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
          .append($('<td>').append(renderScenarioSpec(scenario_spec)))
          .append($('<td>').append(runSpec.adapter_spec.model))
          .append($('<td>').append(runSpec.adapter_spec.method))
        $table.append($row);
      });
    }

    renderTable();

    return $('<div>').append([$search, $table]);
  }

  function renderRunsDetailed(runSpecs) {
    // Render all the `runSpecs`:
    // - Adapter specification
    // - Metric
    // - Instances + predictions
    // For each block, we show a table and each `runSpec` is a column.
    const CORRECT_TAG = 'correct';

    // Used to hash instances.
    function instanceKey(instance) {
      return JSON.stringify(instance);
    }

    // Paths (parallel arrays corresponding to `runSpecs`)
    const metricsPaths = runSpecs.map((runSpec) => {
      return `benchmark_output/runs/${suite}/${runSpec.name}/metrics.json`;
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
    $root.append($scenarioInfo);

    $root.append($('<h5>').append('Adapter specification'));
    const $adapterSpec = $('<table>', {class: 'table'});
    if (runSpecs.length > 1) {
      $adapterSpec.append($('<tr>').append($('<td>'))
        .append(runDisplayNames.map((name) => $('<td>').append(name))));
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

      keys.forEach((key) => {
        // For each key (MetricName - e.g., {name: 'exact_match', ...})
        const field = schema.metricsField(key.name);
        const helpText = describeMetricName(field, key);
        const $key = $('<td>').append($('<span>').append(helpIcon(helpText)).append(' ').append(renderMetricName(key)));
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
    }, []);

    // Render scenario instances
    const instanceToDiv = {};
    getJSONList(scenarioPaths, (scenarios) => {
      console.log('scenarios', scenarios);

      // Only grab the first scenario
      const i = 0;
      $scenarioInfo.append($('<h3>').append(scenarios[i].name));
      $scenarioInfo.append($('<div>').append($('<i>').append(scenarios[i].description)));
      $scenarioInfo.append($('<div>')
        .append($('<a>', {href: scenarios[i].definition_path}).append('[code]'))
        .append(' ').append($('<a>', {href: scenarioPaths[i]}).append('[JSON]'))
      );

      scenarios.forEach((scenario) => {
        scenario.instances.forEach((instance, instanceIndex) => {
          const key = instanceKey(instance);
          if (key in instanceToDiv) {
            return;
          }

          // Render instance
          $instances.append($('<hr>'));
          const $instance = $('<div>');
          $instance.append($('<b>').append(`Input ${instanceIndex} (${instance.split} - ${instance.id} ${renderPerturbation(instance.perturbation)})`));
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
      getJSONList(scenarioStatePaths, (scenarioStates) => {
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

  function renderGroupsPage(runs, groups, models) {
    // Page showing aggregate stats for the passed groups.

    // Groups page information panel
    const $root = $('<div>');
    const groupsPageTitle = groups.map(s => s.name).join(", ");
    $root.append($('<h1>').append(groupsPageTitle));
    
    // Table column information
    const columnSpecs = getColumnSpecs(schema, groups);
    const headerColumnName = 'Model';

    // Main table for the groups
    const mainTableTitle = 'Aggregated Results';
    const modelRunGroups = groupByModel(models, runs);
    const $table = renderStatTable(modelRunGroups, columnSpecs, mainTableTitle, headerColumnName);
    $root.append($table);

    // Individual scenario spec tables
    const scenarioRunGroups = groupByScenarioSpec(runs);
    Object.entries(scenarioRunGroups).forEach(([scenarioName, scenarioRuns] = entry) => {
      const scenarioModelRunGroups = groupByModel(models, scenarioRuns);
      const $subTableContainer = renderStatTable(scenarioModelRunGroups, columnSpecs, scenarioName, headerColumnName);
      $root.append($subTableContainer);
    });

    return $root;
  }

  //////////////////////////////////////////////////////////////////////////////
  //                                   Main                                   //
  //////////////////////////////////////////////////////////////////////////////

  const $main = $('#main');
  let models, runSpecs, runs, schema;
  $.when(
    $.getJSON(`benchmark_output/runs/${suite}/models.json`, {}, (response) => {
      models = response;
      console.log('models', models);
    }),
    $.getJSON(`benchmark_output/runs/${suite}/run_specs.json`, {}, (response) => {
      runSpecs = response;
      console.log('runSpecs', runSpecs);
    }),
     $.getJSON(`benchmark_output/runs/${suite}/runs.json`, {}, (response) => {
      runs = response;
      console.log('runs', runs);
    }),
    $.get('schema.yaml', {}, (response) => {
      const raw = jsyaml.load(response);
      console.log('schema', raw);
      schema = new Schema(raw);
    }),
  ).then(() => {
    $main.empty();
    if (urlParams.models) {
      $main.append(renderHeader('Models', renderModels(models)));
    } else if (urlParams.runSpec) {
      const matchedRunSpecs = runSpecs.filter((runSpec) => new RegExp('^' + urlParams.runSpec + '$').test(runSpec.name));
      if (matchedRunSpecs.length === 0) {
        $main.append(renderError('No matching runs'));
      } else {
        $main.append(renderRunsDetailed(matchedRunSpecs));
      }
    } else if (urlParams.groups) {
      $main.append(renderHeader('Groups', renderGroups(schema.groupsFields)));
    } else if (urlParams.group) {
      // TODO: The groups page can display multiple groups at the same time as
      // To realize this, we originally intended to use regular expressions
      // here. However, we can't allow all regular expressions because our group
      // names include () and - characters. We should either change our group
      // names or rely on string parsing of the URL to allow multiple groups to
      // be displayed at the same time.
      // TODO: We have removed the spaces/() from the group names, so we can
      // switch back to the RegEx match after the next run.
      const matchedGroups = schema.groupsFields.filter((group) => urlParams.group === group.name);
      const matchedGroupNames = matchedGroups.map((group) => group.name);
      const matchedRuns = filterByGroupNames(runs, matchedGroupNames);
      if (matchedGroupNames.length === 0) {
        $main.append(renderError('No matching groups'));
      } else if (matchedRuns.length === 0) {
        $main.append(renderError('No matching runs'));
      } else {
        $main.append(renderGroupsPage(matchedRuns, matchedGroups, models));
      }
    } else {
      $main.append(renderHeader('Runs', renderRunsOverview(runSpecs)));
    }
  });
});
