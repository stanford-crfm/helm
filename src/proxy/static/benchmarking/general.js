function describeField(field) {
  let result = field.name + ": " + field.description;
  if (field.values) {
    result += '\nPossible values:\n' + field.values.map(value => `- ${value.name}: ${value.description}`).join('\n');
  }
  return result;
}

function renderStopSequence(value) {
  return JSON.stringify(value);
}

function renderFieldValue(field, value) {
  if (!field.values) {
    if (field.name === 'stop_sequences') {
      return renderStopSequence(value);
    }
    return value;
  }
  const valueField = field.values.find(valueField => valueField.name === value);
  return $('<a>', {title: valueField ? valueField.description : '(no description)'}).append(value);
}

function perturbationEquals(perturbation1, perturbation2) {
  if (perturbation1 == null) {
    return perturbation2 == null;
  }
  if (perturbation2 == null) {
    return perturbation1 == null;
  }
  return renderDict(perturbation1) === renderDict(perturbation2);
}

function metricNameEquals(name1, name2) {
  return name1.name === name2.name &&
         name1.k === name2.k &&
         name1.split === name2.split &&
         name1.sub_split === name2.sub_split &&
         perturbationEquals(name1.perturbation, name2.perturbation);
}

function renderPerturbation(perturbation) {
  if (!perturbation) {
    return 'original';
  }
  // The perturbation field must have the "name" subfield
  const fields_str = Object.keys(perturbation)
                     .filter(key => key !== 'name')
                     .map(key => `${key}=${perturbation[key]}`)
                     .join(', ');
  return perturbation.name + (fields_str ? '(' + fields_str + ')' : '');
}

function renderMetricName(name) {
  // Return a short name (suitable for a cell of a table)
  // Example: name = {name: 'exact_match'}
  let result = name.name.bold();
  if (name.k) {
    result += '@' + name.k;
  }
  if (name.split) {
    result += ' on ' + name.split + (name.sub_split ? '/' + name.sub_split : '');
  }
  if (name.perturbation) {
    result += ' with ' + renderPerturbation(name.perturbation);
  }
  return result;
}

function describeMetricName(field, name) {
  // Return a longer description that explains the name
  let result = describeField(field);
  if (name.k) {
    result += `\n@${name.k}: consider the best over the top ${name.k} predictions`;
  }
  if (name.split) {
    result += `\non ${name.split}: evaluated on the subset of ${name.split} instances`;
  }
  if (name.perturbation) {
    result += `\nwith ${renderPerturbation(name.perturbation)}: applied this perturbation (worst means over all perturbations of an instance)`;
  }
  return result;
}

function renderStatName(schema, statName) {
  // TODO: Should we ensure all stats have a display name?
  // TODO: Should display name be a default field in schemas? (Rather than being a part of the values)
  const metric = schema.metricsField(statName);
  if (metric.values && metric.values.display_name) {
    return metric.values.display_name;
  } else {
    const formattedName = statName.replaceAll('_', ' ');
    const capitalizedName = formattedName.charAt(0).toUpperCase() + formattedName.slice(1);
    return capitalizedName;
  }
}

function renderPerturbationName(schema, perturbationName) {
  return schema.perturbationsField(perturbationName).values.display_name;
}

function getLast(l) {
  return l[l.length - 1];
}

function renderScenarioSpec(spec) {
  // Example: benchmark.mmlu_scenario.MMLUScenario => MMLU
  const name = getLast(spec.class_name.split('.')).replace('Scenario', '');
  const args = Object.keys(spec.args).length > 0 ? `(${renderDict(spec.args)})` : null;
  return args ? `${name} ${args}` : name;
}

function renderHeader(header, body) {
  return $('<div>').append($('<h4>').append(header)).append(body);
}

function getJSONList(paths, callback, defaultValue) {
  // Fetch the JSON files `paths`, and pass the list of results into `callback`.
  const responses = {};
  $.when(
    ...paths.map((path) => $.getJSON(path, {}, (response) => { responses[path] = response; })),
  ).then(() => {
    callback(paths.map((path) => responses[path] || defaultValue));
  }, (error) => {
    console.error('Failed to load / parse:', paths.filter((path) => !(path in responses)));
    console.error(error.responseText);
    JSON.parse(error.responseText);
    callback(paths.map((path) => responses[path] || defaultValue));
  });
}

function sortListWithReferenceOrder(list, referenceOrder) {
  // Return items in `list` based on referenceOrder.
  // Example:
  // - list = [3, 5, 2], referenceOrder = [2, 5]
  // - Returns [2, 5, 3]
  function getKey(x) {
    const i = referenceOrder.indexOf(x);
    return i === -1 ? 9999 : i;  // Put unknown items at the end
  }
  list.sort(([a, b]) => getKey(a) - getKey(b));
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

function renderDict(obj) {
  return Object.entries(obj).map(([key, value]) => `${key}=${value}`).join(',');
}

function toDecimalString(value, numDecimalPlaces) {
  // Convert the number in `value` to a decimal string with `numDecimalPlaces`.
  // The `value` must be of type `number`.
  return value.toLocaleString("en-US", {
    maximumFractionDigits: numDecimalPlaces,
    minimumFractionDigits: numDecimalPlaces
  });
}

function renderRunSpecLink(runs) {
  // Render a runSpec link for the given `runs`. The string returned will be 
  // in the following format: 
  //   '?runSpec={run_spec_name1}|{run_spec_name1}|...'
  const value = runs.map(r => r.run_spec.name).join('|');
  const params = encodeUrlParams(Object.assign({}, {runSpec: value}));
  return `benchmarking.html${params}`;
}

function checkRunGroupNameMatch(run, groupName) {
  // Check whether the `run` belongs to a group with the given `groupName`.
  return run.run_spec.groups.includes(groupName);
}

function filterByGroup(runs, groupName) {
  // Filter runs to those that belong to the group specified by `groupName`.
  return runs.filter(run => checkRunGroupNameMatch(run, groupName));
}

function filterByGroupNames(runs, possibleGroupNames) {
  // Filter runs to those that belong to one of the groups specified in `possibleGroupNames`.
  return runs.filter(run => {
    var match = false;
    possibleGroupNames.forEach(groupName => {
      match = match || checkRunGroupNameMatch(run, groupName);
    });
    return match;
  });
}

function groupByModel(runs) {
  // Group `runs` by models. Return a dictionary mapping each model name to a list of runs.
  var modelToRuns = {};
  for (let run of runs) {
    const model = run.run_spec.adapter_spec.model;
    modelToRuns[model] = (modelToRuns[model] || []).concat([run]);
  }
  return modelToRuns;
}

function groupByScenarioSpec(runs) {
  // Group `runs` by scenario specs. Return a dictionary mapping each scenario spec string to a list of runs.
  return runs.reduce((acc, run) => {
    // To maintain backward compatibility, as `scenario` in `run_spec` was renamed to `scenario_spec`.
    // TODO: Remove the fallback option once we run all the run specs again.
    const scenarioSpec = renderScenarioSpec(
        run.run_spec.hasOwnProperty('scenario_spec') ? run.run_spec.scenario_spec : run.run_spec.scenario
    );
    acc[scenarioSpec] = acc[scenarioSpec] || [];
    acc[scenarioSpec].push(run);
    return acc;
  }, {});
}

function getUniqueValue(arr, messageType) {
  const arrUnique = new Set(arr);
  // TODO: Double check that the assert statement is throwing an error as expected.
  assert(arrUnique.size == 1, `The groups have incompatible ${messageType}.`);
  return arr[0];
}

function getTableSetting(schema, groups) {
  const tableSettingNames = groups.map(group => group.values.tableSetting || 'default');
  const tableSettingsName = getUniqueValue(tableSettingNames, 'table settings');
  const tableSetting = schema.tableSettingsField(tableSettingsName);
  return tableSetting;
}

function getDisplayK(groups) {
  const displayKArr = groups.map(group => group.values.display.k);
  const displayK = getUniqueValue(displayKArr, 'display k');
  return displayK;
}

function getDisplaySplit(groups) {
  const displaySplitArr = groups.map(group => group.values.display.split);
  const displaySplit = getUniqueValue(displaySplitArr, 'display k');
  return displaySplit;
}

function getStatNames(groups) {
  const statNamesArr = groups.map(group => group.values.display.stat_names);
  const statNames = [].concat(...statNamesArr);
  return statNames;
}
