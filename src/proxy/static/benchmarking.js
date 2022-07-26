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

  // Captures information about a field of an adapter (e.g.,
  // max_train_instances) or a metric name (e.g., exact_match).
  class Field {
    constructor(raw) {
      this.name = raw.name;
      this.description = raw.description;
      // Possible values this field can take
      this.values = this.readValues(raw.values);
    }

    readValues(values) {
      // Read the values field.
      // Note: We are using field to represent the schema for a field value too.
      if (Array.isArray(values)) {
        // If the values field is an array, read each element as a Field.
        return values.map((valueRaw) => new Field(valueRaw));
      } else if (values === undefined || typeof(values) === 'object') {
        // If the values field is an object, read it as is. Note that an object can be null.
        return values;
      }
      // If no matching schema is found, raise an error!
      console.error(`The values field of ${this.name} should be an array or an object. Instead found: ${values}.`);
    }
  }

  // Specifies all the information to help us render and understand the fields
  // for adapters and metrics.
  class Schema {
    constructor(raw) {
      this.adapterFields = raw.adapter.map((fieldRaw) => new Field(fieldRaw));
      this.metricsFields = raw.metrics.map((fieldRaw) => new Field(fieldRaw));
      this.perturbationsFields = raw.perturbations.map((fieldRaw) => new Field(fieldRaw));
      this.groupsFields = raw.groups.map((fieldRaw) => new Field(fieldRaw));
      this.tableSettingsFields = raw.tableSettings.map((fieldRaw) => new Field(fieldRaw));
      this.statGroupsFields = raw.statGroups.map((fieldRaw) => new Field(fieldRaw));

      // Allow convenient access
      this.adapterFieldNames = this.adapterFields.map((field) => field.name);
      this.metricsFieldNames = this.metricsFields.map((field) => field.name);
      this.perturbationsFieldNames = this.perturbationsFields.map((field) => field.name);
      this.groupsFieldNames = this.groupsFields.map((field) => field.name);
      this.tableSettingsFieldNames = this.tableSettingsFields.map((field) => field.name);
      this.statGroupsFieldNames = this.statGroupsFields.map((field) => field.name);
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

    statGroupsField(name) {
      // Return the group field with the given `name`.
      const field = this.statGroupsFields.find((field) => field.name === name);
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

  ///////////////////// Utility Functions and Classes //////////////////////////

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

  function renderStatName(statName) {
    // TODO: Should we ensure all stats have a display name?
    // TODO: Should display name be a default field in schemas? (Rather than being a part of the values)
    // TODO: Clean up this function.
    const alternativeName = statName.replace('_', ' ');
    const capitalized = alternativeName.charAt(0).toUpperCase() + alternativeName.slice(1);
    const metric = schema.metricsField(statName);
    return metric.values && metric.values.display_name || capitalized;
  }

  function renderPerturbationName(perturbationName) {
    return schema.perturbationsField(perturbationName).values.display_name;
  }

  function renderModels(models) {
    const $table = $('<table>', {class: 'query-table'});
    models.forEach((model) => {
      const $row = $('<tr>').append($('<td>').append(`${model.description} [${model.name}]`));
      $table.append($row);
    });
    return $table;
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

  function toDecimalString(value, fractionDigits) {
    if (typeof value === 'number') {
      return value.toLocaleString("en-US", {
        maximumFractionDigits: fractionDigits,
        minimumFractionDigits: fractionDigits
      });
    }
    return value;
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
    return runs.reduce((acc, run) => {
      model = run.run_spec.adapter_spec.model;
      acc[model] = acc[model] || [];
      acc[model].push(run);
      return acc;
    }, {});
  }

  function groupByScenarioSpec(runs) {
    // Group `runs` by scenario specs. Return a dictionary mapping each scenario spec string to a list of runs.
    return runs.reduce((acc, run) => {
      const scenarioSpec = renderScenarioSpec(run.run_spec.scenario);
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

  function getTableSetting(groups) {
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

  class WrappedStat {

    constructor(runSpecName, stat) {
      this.runSpecName = runSpecName;
      this.stat = stat;
    }

  }

  class StatNameFilter {

    constructor(statNames, ks, splits, perturbations) {
      this.keyToOptions = {};
      if (!(statNames === undefined)) {this.keyToOptions.name = statNames}
      if (!(ks === undefined)) {this.keyToOptions.k = ks}
      if (!(splits === undefined)) {this.keyToOptions.split = splits}
      if (!(perturbations === undefined)) {this.keyToOptions.perturbation = perturbations}
    }

    test(statName) {
      // TODO: Can be improved.
      var match = true;
      Object.entries(this.keyToOptions).forEach(([key, options] = entry) => {
        if (key === 'perturbation') {
          // Compute perturbation matches
          var perturbationMatches = options.map(target => {
            // If the target or stat perturbation is null, we check whether the
            // other one is null and return.
            if ([target, statName.perturbation].includes(null)) {
              return statName.perturbation === target;
            }
            // If the target perturbation is an object, we iterate through each
            // field and value.
            var perturbationMatch = true;
            Object.entries(target).forEach(([pertKey, pertValue] = entry) => {
              if (!(pertKey in statName.perturbation) || !(statName.perturbation[pertKey] === pertValue)) {
                perturbationMatch = false;
              }
            });
            return perturbationMatch;     
          });
          // Check if there is at least one matching perturbation
          match = match && perturbationMatches.some(m => m);
        } else {
          match = match && options.includes(statName[key]);
        }
      });
      return match;
    }

  }

  class Cell {

    constructor() {
      this.type = 'cell';
      this.data = null;
      this.hoverData = null;
    }

    renderData() {
      return this.data;
    }

    renderHoverData() {
      return this.hoverData;
    }

  }

  class EmptyCell extends Cell {

    constructor() {
      super();
    }

  }

  class HeaderCell extends Cell {
    
    constructor(name, href) {
      super();
      this.data = {name: name, href: href};
      this.hoverData = null;
    }

    renderData() {
      return $('<a>', {href: this.data.href}).append(this.data.name);
    }

    renderHoverData() {
      // TODO: Determine hover value for models.
      return this.hoverData;
    }

  }

  class StatTableCell extends Cell {

    constructor(wrappedStats, defaultK, defaultSplit) {
      // Call super's constructor
      super();

      // User provided fields
      this.wrappedStats = wrappedStats;
      
      // Data fields used in table
      this.data = {value: null, stddev: null};
      this.hoverData = {runSpecNameToStatsDict: []};

      // Class variables
      this.getAggFunc = {
        average: this.average,
      }

      // Update data fields
      this.updateDataFields([defaultK], [defaultSplit], 'average');
    }

    updateDataFields(kArr, splitArr, aggFuncName) {
      // Update this.data and this.hoverData
      const statFilter = new StatNameFilter(undefined, kArr, splitArr, undefined);
      var filteredWrappedStats = this.wrappedStats.filter(ws => statFilter.test(ws.stat.name));
      this.updateData(filteredWrappedStats, aggFuncName);
      this.updateHoverData(filteredWrappedStats);
    }

    updateData(runSpecStats, aggFuncName) {
      this.data = this.getAggFunc[aggFuncName](runSpecStats);
    }

    updateHoverData(wrappedStats) {
      var runSpecNameToStatsDict = {};
      wrappedStats.forEach(ws => {
        runSpecNameToStatsDict[ws.name] = runSpecNameToStatsDict[ws.name] || [];
        runSpecNameToStatsDict[ws.name].push(ws.stat);
      });
      this.hoverData.runSpecNameToStatsDict = runSpecNameToStatsDict;
    }

    average(wrappedStats) {
      // Grab the means.
      const means = wrappedStats.map(ws => ws.stat.mean);
      const count = means.length; // Number of means.

      // Compute the overall mean, Note that we are computing the mean of the
      // means, which doesn't account for the number of values that contributed
      // to each mean.
      const sum = means.reduce((partial, num) => partial + num, 0);
      const mean = sum / count; // This can be undefined.

      // Computed stddev of the means.
      const sumSquared = means.reduce((partial, num) => partial + Math.pow(num, 2), 0);
      const variance = sumSquared / count - Math.pow(mean, 2);
      const stddev = variance < 0 ? 0 : Math.sqrt(variance);

      // Return
      return {value: mean, stddev: stddev};
    }

    renderData() {
      // TODO: Adding stddev to the string makes the table look crowded.
      // Not included for now, let's think of how/when we want to display it.
      // It can also be a part of the hover data.
      const valueString = toDecimalString(this.data.value, 3);
      const stddevString = toDecimalString(this.data.stddev, 1);
      const dataString = `${valueString}`;
      return dataString;
    }

    renderHoverData() {
      // TODO: Format hover data.
      return null;
    }

  }

  class Column {

    constructor(group, name, rowKeyToCell) {
      this.type = 'column';
      this.group = group;
      this.name = name;
      this.rowKeyToCell = rowKeyToCell;
    }

    renderName() {
      return this.name;
    }

    getCell(rowKey) {
      if (rowKey in this.rowKeyToCell) {
        return this.rowKeyToCell[rowKey];
      }
      return new EmptyCell();
    }

  }

  class StatTableColumnSpec {

    constructor(group, name, statNameFilter, defaultK, defaultSplit) {
      this.group = group;
      this.name = name;
      this.statNameFilter = statNameFilter;
      this.defaultK = defaultK;
      this.defaultSplit = defaultSplit;
    }

    makeColumn(groupedRuns) {

      var rowKeyToCell = {};

      Object.entries(groupedRuns).forEach(([groupName, runs] = entry) => {
        const wrappedStats = getWrappedStatsFromRuns(runs);
        const filteredWrappedStats = wrappedStats.filter(ws => this.statNameFilter.test(ws.stat.name));
        if (filteredWrappedStats.length > 0) {
          rowKeyToCell[groupName] = new StatTableCell(filteredWrappedStats, this.defaultK, this.defaultSplit);
        }
      });

      var column = null;
      if (Object.keys(rowKeyToCell).length > 0) {
        column = new Column(this.group, this.name, rowKeyToCell);
      }

      return column;
    }

  }

  class TableData {

    constructor(name, headerColumn, dataColumns) {
      this.type = 'tableData';
      this.name = name;
      this.headerColumn = headerColumn;
      this.dataColumns = dataColumns;
    }

  }

  function getWrappedStatsFromRun(run) {
    return run.stats.map(stat => new WrappedStat(run.run_spec.name, stat));
  }

  function getWrappedStatsFromRuns(runs) {
    var wrappedStats = [];
    runs.forEach(run => wrappedStats.push(...getWrappedStatsFromRun(run)));
    return wrappedStats;
  }

  function getColumnSpecs(groups) {
    // Get table setting for the groups
    const tableSetting = getTableSetting(groups);
    const defaultK = getDisplayK(groups);
    const defaultSplit = getDisplaySplit(groups);
    var columnSpecs = [];

    tableSetting.values.stat_groups.forEach(statGroupName => {
      // Get stat names and perturbations.
      const statGroup = schema.statGroupsField(statGroupName);
      var perturbationNames = statGroup.values.perturbation_names;
      var statNames = statGroup.values.stat_names;
      if (statGroupName === 'accuracy') {
        // Infer the stat names for the group.
        statNames = getStatNames(groups);
      }

      // Create filters
      statNames.forEach(statName => {
        perturbationNames.forEach(perturbationName => {
          const statNameFilter = new StatNameFilter([statName], undefined, undefined, [{name: perturbationName}]);
          // TODO: Wrap the rendering below in a function.
          const columnName = perturbationName === 'identity' ? renderStatName(statName) : renderStatName(statName) + ' (' + renderPerturbationName(perturbationName) + ')';
          const columnSpec = new StatTableColumnSpec(statGroupName, columnName, statNameFilter, defaultK, defaultSplit);
          columnSpecs.push(columnSpec);
        });
      })
    });

    return columnSpecs;
  }

  function makeStatTableData(groupedRuns, columnSpecs, title, headerColumnName) {

    // Header column.
    var headerRowKeyToCell = {};
    Object.entries(groupedRuns).map(([groupName, runs] = entry) => {
      const href = renderRunSpecLink(runs);
      headerRowKeyToCell[groupName] = new HeaderCell(groupName, href);
    });
    const headerColumn = new Column(null, headerColumnName, headerRowKeyToCell);

    // Data columns.
    const dataColumns = columnSpecs.map(cs => cs.makeColumn(groupedRuns)).filter(cs => cs);
                           
    // Create table data.
    const tableData = new TableData(title, headerColumn, dataColumns);
    return tableData;
  }

  function renderTableData(tableData, headerColumnName) {
    const $table = $('<table>'); // TODO: Customize css.
    
    // TODO: Add group row.

    // Header row.
    const $headerRow = $('<tr>');
    $headerRow.append($('<th>').append(headerColumnName));
    tableData.dataColumns.forEach(column => {
      const $th = $('<th>').append(column.name);
      $headerRow.append($th);
    });
    $table.append($headerRow);

    // Data rows
    const dataRows = Object.entries(tableData.headerColumn.rowKeyToCell).map(([rowKey, headerCell] = entry) => {
      const $row = $('<tr>');
      const $th =  $('<th>').append(headerCell.renderData());
      $row.append($th);
      tableData.dataColumns.forEach(dataColumn => {
        const dataCell = dataColumn.getCell(rowKey);
        const $td =  $('<td>').append(dataCell.renderData());
        $row.append($td);
      });
      $table.append($row);
    });

    return $table;
  }

  function renderStatTableExplainer(groupedRuns, title) {
    // mightdo: Table information can be moved to the table caption.
    const $tableExplainer = $('<div>');
    $tableExplainer.append($('<h2>').append(title)); // Title
    const runs = [].concat(...Object.values(groupedRuns));
    const predictionsHref = renderRunSpecLink(runs);
    $tableExplainer.append($('<a>', {href: predictionsHref}).append(`All predictions for the table`)); // Predictions
    return $tableExplainer;
  }

  function renderStatTable(groupedRuns, columnSpecs, title, headerColumnName) {
    const tableData = makeStatTableData(groupedRuns, columnSpecs, title, headerColumnName);
    const $tableContainer = $('<div>'); // mightdo: we can add a custom class. 
    $tableContainer.append(renderStatTableExplainer(groupedRuns, title));
    $tableContainer.append(renderTableData(tableData, headerColumnName));
    return $tableContainer;
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
        const href = encodeUrlParams(Object.assign(urlParams, {runSpec: runSpec.name}));
        const $row = $('<tr>')
          .append($('<td>').append($('<a>', {href}).append(runSpec.name)))
          .append($('<td>').append(renderScenarioSpec(runSpec.scenario)))
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

  function renderGroupsPage(runs, groups) {
    // Page showing aggregate stats for the passed groups.

    // Groups page information panel
    const $root = $('<div>');
    const groupsPageTitle = groups.map(s => s.name).join(", ");
    $root.append($('<h1>').append(groupsPageTitle));
    
    // Table column information
    const columnSpecs = getColumnSpecs(groups);
    const headerColumnName = 'Model';

    // Main table for the groups
    const mainTableTitle = 'Aggregated Results';
    const modelRunGroups = groupByModel(runs);
    const $table = renderStatTable(modelRunGroups, columnSpecs, mainTableTitle, headerColumnName);
    $root.append($table);

    // Individual scenario spec tables
    const scenarioRunGroups = groupByScenarioSpec(runs);
    Object.entries(scenarioRunGroups).forEach(([scenarioName, scenarioRuns] = entry) => {
      const scenarioModelRunGroups = groupByModel(scenarioRuns);
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
    } else if (urlParams.group) {
      const matchedGroups = schema.groupsFields.filter((group) => new RegExp('^' + urlParams.group + '$').test(group.name));
      const matchedGroupNames = matchedGroups.map((group) => group.name);
      const matchedRuns = filterByGroupNames(runs, matchedGroupNames);
      if (matchedGroupNames.length === 0) {
        $main.append(renderError('No matching groups'));
      } else {
        $main.append(renderGroupsPage(matchedRuns, matchedGroups));
      }
    } else {
      $main.append(renderHeader('Runs', renderRunsOverview(runSpecs)));
    }
  });
});
