/**
 * This file contains functions and classes used for rendering UI tables.
 * 
 * A table is backed by a `TableData`, which consists of one header `Column` and
 * any number of data `Column`s. Each `Column` consists of `Cell`s. Cells in a
 * column are stored in a sparse dictionary named `rowKeyToCell`.
 * 
 * `renderTableData` renders a given `TableData` object, by first creating a
 * header column. It then uses the keys in `rowKeyToCell` as row keys to get
 * the corresponding cells in data columns for each row. It then calls
 * `renderData` and `renderHoverData` methods on each cell and wraps the results
 * in HTML table elements.
 * 
 * Both `Column` and `Cell` classes can be extended for custom use, as long as
 * the children class overrides the methods defined in the base classes.
 */

/**
 * WrappedStat is a wrapper class enhancing stat information with extra
 * information related to the run that the stat belongs to.
*/
class WrappedStat {

  constructor(runSpecName, stat) {
    this.runSpecName = runSpecName;
    this.stat = stat;
  }

}

/**
 * StatNameFilter is a filter that filters `statName`s. It can take a list of
 * options for each of the following stat name fields: `name`, `k`, `split`, and
 * `perturbation`.
 */
class StatNameFilter {

  constructor(statNames, ks, splits, perturbations) {
    this.keyToOptions = {};
    if (!(statNames === undefined)) { this.keyToOptions.name = statNames };
    if (!(ks === undefined)) { this.keyToOptions.k = ks };
    if (!(splits === undefined)) { this.keyToOptions.split = splits };
    if (!(perturbations === undefined)) { this.keyToOptions.perturbation = perturbations };
  }

  test(statName) {
    // Test whether the values of the `statName` keys correspond to one of the
    // options specified in this.keyToOptions. Return a boolean.
    let match = true;
    Object.entries(this.keyToOptions).forEach(([key, options] = entry) => {
      if (key === 'perturbation') {
        // Compute perturbation matches
        const perturbationMatches = options.map(target => {
          // If the target or stat perturbation is null, we check whether the
          // other one is null and return.
          if ([target, statName.perturbation].includes(null)) {
            return statName.perturbation === target;
          }
          // If the target perturbation is an object, we iterate through each
          // field and value.
          let perturbationMatch = true;
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

/**
 * `Cell` represents a single cell in a table `Column`.
 */
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

/**
 * `EmptyCell` represents a Cell subclass that is empty.
 */
class EmptyCell extends Cell {

  constructor() {
    super();
  }

}

/**
 * `HeaderCell` is a specific type of `Cell` to be used in header columns. In
 * addition to the regular cells, it stores an optional link that is rendered
 * when `renderData` function is called.
 */
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

/**
 * `StatTableCell` is a type of `Cell` that stores a list of `wrappedStats`
 * that is used to compute the cell's value.
 */
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
    const filteredWrappedStats = this.wrappedStats.filter(ws => statFilter.test(ws.stat.name));
    this.updateData(filteredWrappedStats, aggFuncName);
    this.updateHoverData(filteredWrappedStats);
  }

  updateData(runSpecStats, aggFuncName) {
    this.data = this.getAggFunc[aggFuncName](runSpecStats);
  }

  updateHoverData(wrappedStats) {
    const runSpecNameToStatsDict = {};
    for (let ws of wrappedStats) {
      runSpecNameToStatsDict[ws.name] = runSpecNameToStatsDict[ws.name] || [];
      runSpecNameToStatsDict[ws.name].push(ws.stat);
    };
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
    const valueString = this.data.value ? toDecimalString(this.data.value, 3) : this.data.value;
    const stddevString = this.data.stddev ? toDecimalString(this.data.stddev, 1) : this.data.stddev;
    const dataString = `${valueString}`;
    return dataString;
  }

  renderHoverData() {
    // TODO: Format hover data.
    return null;
  }

}

/**
 * `Column` represents a column in a `TableData` and stores cells.
 */
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

/**
 * `StatTableColumnSpec` contains a spec to create a `Column` object.
 */
class StatTableColumnSpec {

  constructor(group, name, statNameFilter, defaultK, defaultSplit) {
    this.group = group;
    this.name = name;
    this.statNameFilter = statNameFilter;
    this.defaultK = defaultK;
    this.defaultSplit = defaultSplit;
  }

  makeColumn(groupedRuns) {

    const rowKeyToCell = {};

    Object.entries(groupedRuns).forEach(([groupName, runs] = entry) => {
      const wrappedStats = getWrappedStatsFromRuns(runs);
      const filteredWrappedStats = wrappedStats.filter(ws => this.statNameFilter.test(ws.stat.name));
      if (filteredWrappedStats.length > 0) {
        rowKeyToCell[groupName] = new StatTableCell(filteredWrappedStats, this.defaultK, this.defaultSplit);
      }
    });

    let column = null;
    if (Object.keys(rowKeyToCell).length > 0) {
      column = new Column(this.group, this.name, rowKeyToCell);
    }

    return column;
  }

}

/**
 * `TableData` class contains a header column and a list of data columns.
 */
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
  const wrappedStats = [];
  runs.forEach(run => wrappedStats.push(...getWrappedStatsFromRun(run)));
  return wrappedStats;
}

function getColumnSpecs(schema, groups) {
  // Get table setting for the groups
  const tableSetting = getTableSetting(schema, groups);
  const defaultK = getDisplayK(groups);
  const defaultSplit = getDisplaySplit(groups);
  const columnSpecs = [];

  tableSetting.values.stat_groups.forEach(statGroupName => {
    // Get stat names and perturbations.
    const statGroup = schema.statGroupsField(statGroupName);
    const perturbationNames = statGroup.values.perturbation_names;
    let statNames = statGroup.values.stat_names;
    if (statGroupName === 'accuracy') {
      // Infer the stat names for the group.
      statNames = getStatNames(groups);
    }

    // Create filters
    statNames.forEach(statName => {
      perturbationNames.forEach(perturbationName => {
        const statNameFilter = new StatNameFilter([statName], undefined, undefined, [{name: perturbationName}]);
        // TODO: Wrap the rendering below in a function.
        const columnName = perturbationName === 'identity' ? renderStatName(schema, statName) : renderStatName(schema, statName) + ' (' + renderPerturbationName(schema, perturbationName) + ')';
        const columnSpec = new StatTableColumnSpec(statGroupName, columnName, statNameFilter, defaultK, defaultSplit);
        columnSpecs.push(columnSpec);
      });
    })
  });

  return columnSpecs;
}

function makeStatTableData(groupedRuns, columnSpecs, title, headerColumnName) {

  // Header column.
  const headerRowKeyToCell = {};
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
