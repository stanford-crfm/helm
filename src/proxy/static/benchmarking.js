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

  function renderRun(runSpec) {
    const CORRECT_TAG = 'correct';

    const $root = $('<div>');
    $.getJSON(`benchmark_output/runs/${runSpec.name}/scenario.json`, {}, (scenario) => {
      console.log('scenario', scenario);
      const $instances = $('<div>');
      $root.append($instances);
      scenario.instances.forEach((instance, i) => {
        const $instance = $('<div>');
        $instance.append(`[${i}] ${multilineHtml(instance.input)}`);
        const $references = $('<ul>');
        instance.references.forEach((reference) => {
          const isCorrect = reference.tags.includes(CORRECT_TAG);
          $references.append($('<li>').append($('<span>', {class: isCorrect ? 'correct' : ''}).append(reference.output)));
        });
        $instance.append($references);
        $instances.append($instance);
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
          $main.append(renderHeader(runSpec.name, renderRun(runSpec)));
        });
      }
    } else {
      $main.append(renderHeader('Runs', renderRuns(runSpecs)));
    }
  });
});
