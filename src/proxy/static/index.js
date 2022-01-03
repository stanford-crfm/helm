/**
 * This is a very quick and dirty frontend for just interacting with the models.
 * Please refrain from adding additional functionality to this.
 * TODO: Write this in React.
 */
$(function () {
  const urlParams = decodeUrlParams(window.location.search);
  const rootUrl = '/static/index.html';
  let auth = null;

  function censor(api_key) {
    // Show only the first k letters
    const k = 2;
    if (api_key.length <= k) {
      return api_key;
    }
    return api_key.substring(0, k) + '*'.repeat(api_key.length - k);
  }

  function helpIcon(help, link) {
    // Show a ?
    return $('<a>', {href: link, target: 'blank_', class: 'help-icon'}).append($('<img>', {src: 'info-icon.png', width: 15, title: help}));
  }

  // Logging in and out
  function updateLogin() {
    const $loginInfo = $('#loginInfo');
    $loginInfo.empty();
    let api_key = readCookie('api_key');
    if (api_key) {
      auth = {api_key};
      $loginInfo.append($('<a>', {class: 'nav-link', href: '#'}).append('Logout of API key ' + censor(api_key)).click(() => {
        eraseCookie('api_key');
        updateLogin();
      }));
    } else {
      auth = null;
      $loginInfo.append($('<a>', {class: 'nav-link', href: '#'}).append('Login').click(() => {
        api_key = prompt('Enter your API key:');
        if (!api_key) {
          return;
        }
        createCookie('api_key', api_key);
        updateLogin();
      }));
    }
  }

  updateLogin();

  ////////////////////////////////////////////////////////////
  // Rendering functions

  function multilineHtml(s) {
    return s.replace(/\n/g, '<br>');
  }

  function renderError(e) {
    return $('<div>').addClass('alert alert-danger').append(multilineHtml(e));
  }

  function renderExampleQueries(updateQuery) {
    // Show links for each example query, so when you click on them, they populate the textboxes.
    const $examplesBlock = $('<div>');
    $examplesBlock.append($('<span>').append('Examples:'));
    generalInfo.example_queries.forEach((query, i) => {
      const href = '#';
      const title = '[Prompt]\n' + query.prompt + '\n[Settings]\n' + query.settings + '\n[Environments]\n' + query.environments;
      const $link = $('<a>', {href, title}).append(`[${i}]`);
      $link.click(() => {
        // Populate the query with the example
        updateQuery(query);
        urlParams.prompt = query.prompt;
        urlParams.settings = query.settings;
        urlParams.environments = query.environments;
        updateBrowserLocation(urlParams);
      });
      $examplesBlock.append('&nbsp;');
      $examplesBlock.append($link);
    });
    return $examplesBlock;
  }

  function renderQuery(handleQueryResult) {
    // Render the textboxes for entering the query (which includes the prompt, settings, and environment)
    const $queryBlock = $('<div>', {class: 'block'});
    const $prompt = $('<textarea>', {cols: 90, rows: 7, placeholder: 'Enter prompt'}).val(urlParams.prompt);
    const $settings = $('<textarea>', {cols: 90, rows: 5, placeholder: 'Enter settings (e.g., model: openai/davinci for GPT-3)'}).val(urlParams.settings);
    const $environments = $('<textarea>', {cols: 90, rows: 3, placeholder: 'Enter environment variables (e.g., city: [Boston, New York])'}).val(urlParams.environments);

    $queryBlock.data('prompt', $prompt);
    $queryBlock.data('settings', $settings);
    $queryBlock.data('environments', $environments);

    function bindSubmit($text) {
      $text.keypress((e) => {
        if ((event.keyCode === 10 || event.keyCode === 13) && event.ctrlKey) {
          submit();
        }
      });
    }

    bindSubmit($prompt);
    bindSubmit($settings);
    bindSubmit($environments);

    function submit() {
      if (!auth) {
        alert('You must log in first.');
        return;
      }

      const query = {
        prompt: $prompt.val(),
        settings: $settings.val(),
        environments: $environments.val(),
      };

      urlParams.prompt = query.prompt;
      urlParams.settings = query.settings;
      urlParams.environments = query.environments;
      updateBrowserLocation(urlParams);

      $.getJSON('/api/query', query, handleQueryResult);
    }

    // Show examples of queries
    const $exampleQueries = renderExampleQueries((query) => {
      $queryBlock.data('prompt').val(query.prompt);
      $queryBlock.data('settings').val(query.settings);
      $queryBlock.data('environments').val(query.environments);
      urlParams.prompt = query.prompt;
      urlParams.settings = query.settings;
      urlParams.environments = query.environments;
      updateBrowserLocation();
    });

    const promptHelp = 'This is the text you feed into the language model to complete.\nExample:\n  Life is like';
    const settingsHelp = 'Specifies what information we want from the language model (see [Help] for more details).\nExample:\n  temperature: ${temperature}\n  model: openai/davinci\n  max_tokens: 10\n  num_completions: 5';
    const environmentsHelp = 'Specifies a list of values to try for each variable that appears in the prompt or settings.\nExample:\n  temperature: [0, 0.5, 1]';

    const $promptLabel = $('<span>').append(helpIcon(promptHelp, 'help.html#query')).append('Prompt');
    const $settingsLabel = $('<span>').append(helpIcon(settingsHelp, 'help.html#query')).append('Settings');
    const $environmentsLabel = $('<span>').append(helpIcon(environmentsHelp, 'help.html#query')).append('Environments');

    $queryBlock.append($('<h4>').append('Query'));
    $queryBlock.append($exampleQueries);
    const $table = $('<table>', {class: 'query-table'});
    $table.append($('<tr>').append($('<td>').append($promptLabel)).append($('<td>').append($prompt)));
    $table.append($('<tr>').append($('<td>').append($settingsLabel)).append($('<td>').append($settings)));
    $table.append($('<tr>').append($('<td>').append($environmentsLabel)).append($('<td>').append($environments)));
    $queryBlock.append($table);
    $queryBlock.append($('<button>').append('Submit').click(submit));

    return $queryBlock;
  }

  function getChangingKeys(items) {
    // Return the list of keys whose values vary across `items.`
    if (items.length === 0) {
      return [];
    }
    return Object.keys(items[0]).filter((key) => {
      return !items.every((item) => JSON.stringify(item[key]) === JSON.stringify(items[0][key]));
    });
  }

  function renderRequest(changingKeys, request) {
    // Render the request metadata (e.g., temperature if it is changing)
    const title = JSON.stringify(request);
    // Always include model, never prompt (since that's shown right after).
    const showKeys = ['model'].concat(changingKeys.filter((key) => key !== 'prompt' && key !== 'model'));
    const summary = '[' + showKeys.map(key => key + ':' + request[key]).join(', ') + ']';
    return $('<div>', {title}).append(summary + ' ' + multilineHtml(request.prompt));
  }

  function renderTime(time) {
    return (Math.round(time * 10) / 10) + 's';
  }

  function renderTokens(tokens) {
    // Render text as a sequence of tokens that you can interact with to see more information (e.g., logprobs)
    const $result = $('<div>');
    for (const token of tokens) {
      // When mouse over token, show the alternative tokens and their log probabilities (including the one that's generated)
      const entries = Object.entries(token.top_logprobs);
      if (!(token.text in token.top_logprobs)) {
        entries.push([token.text, token.logprob]);
      }
      entries.sort((a, b) => b[1] - a[1]);
      function marker(text) { return text === token.text ? ' [selected]' : ''; }
      const title = 'Candidates (with logprobs):\n' + entries.map(([text, logprob]) => `${text}: ${logprob}${marker(text)}`).join('\n');
      const $token = $('<span>', {class: 'token', title}).append(multilineHtml(token.text));
      $result.append($token);
    }
    return $result;
  }

  function renderRequestResult(requestResult) {
    // Render the list of completions.
    if (requestResult.error) {
      return renderError(requestResult.error);
    }
    const $result = $('<div>');
    requestResult.completions.forEach((completion) => {
      $result.append($('<div>', {class: 'completion', title: `logprob: ${completion.logprob}`}).append(renderTokens(completion.tokens)));
    });
    $result.append($('<i>').append(renderTime(requestResult.request_time)));
    return $result;
  }

  function renderAccount() {
    // Render the account information (usage, quotas).
    if (!auth) {
      return null;
    }

    const $accountBlock = $('<div>', {class: 'block'});
    const args = {auth: JSON.stringify(auth)};
    $.getJSON('/api/account', args, (account) => {
      console.log('/api/account', account);
      const items = [];
      for (modelGroup in account.usages) {
        for (granularity in account.usages[modelGroup]) {
          const usage = account.usages[modelGroup][granularity];
          // Only print out usage for model groups and granularities where there is a quota
          if (usage.quota) {
            const percent = Math.round(usage.used / usage.quota * 100);
            items.push(`<b>${modelGroup}</b>: ${usage.period} (${usage.used} / ${usage.quota} = ${percent}%)`);
          }
        }
      }
      if (items.length === 0) {
        items.push('no restrictions');
      }
      $accountBlock.empty()
        .append(helpIcon('Specifies your usage/quota (321/10000) for each model group (e.g., gpt3) for the current period (e.g., 2022-1-2).', 'help.html#quotas'))
        .append('Usage')
        .append(': ')
        .append(items.join(' | '));
    });
    return $accountBlock;
  }

  ////////////////////////////////////////////////////////////
  // For index.html

  function renderQueryInterface() {
    // For index.html
    const $accountBlock = $('<div>').append(renderAccount());

    // Allow editing the query
    const $queryBlock = renderQuery((queryResult) => {
      // Create requests
      console.log('/api/query', queryResult);
      $requestsBlock.empty();

      if (queryResult.error) {
        $requestsBlock.append(renderError(queryResult.error));
        return;
      }

      $requestsBlock.append($('<h4>').append(`Requests (${queryResult.requests.length})`));
      if (queryResult.error) {
        $requestsBlock.append(renderError(queryResult.error));
        return;
      }
      const changingKeys = getChangingKeys(queryResult.requests);
      queryResult.requests.forEach((request) => {
        const $request = $('<div>', {class: 'request'}).append(renderRequest(changingKeys, request));
        const $requestResult = $('<div>').append($('<i>').append('(waiting)'));
        const args = {
          auth: JSON.stringify(auth),
          request: JSON.stringify(request),
        };
        $.getJSON('/api/request', args, (requestResult) => {
          console.log('/api/request', request, requestResult);
          $requestResult.empty().append(renderRequestResult(requestResult));
          if (!requestResult.cached) {
            $accountBlock.empty().append(renderAccount());
          }
        });
        $request.append($requestResult);
        $requestsBlock.append($request);
      });
    });

    // Where the requests and responses come in
    const $requestsBlock = $('<div>', {class: 'block'});

    const $group = $('<div>');
    $group.append($accountBlock);
    $group.append($queryBlock);
    $group.append($requestsBlock);
    return $group;
  }

  ////////////////////////////////////////////////////////////
  // For help.html

  function renderModelsTable() {
    // Render the list of models
    const $table = $('<table>', {class: 'table'});
    const $header = $('<tr>')
      .append($('<td>').append('group'))
      .append($('<td>').append('name'))
      .append($('<td>').append('description'));
    $table.append($header);
    generalInfo.all_models.forEach((model) => {
      const $row = $('<tr>')
        .append($('<td>').append($('<tt>').append(model.group)))
        .append($('<td>').append($('<tt>').append(model.name)))
        .append($('<td>').append(model.description));
      $table.append($row);
    });
    return $table;
  };

  ////////////////////////////////////////////////////////////
  // Main

  let generalInfo;

  $.getJSON('/api/general_info', (response) => {
    generalInfo = response;
    console.log('/api/general_info', generalInfo);

    // For index.html
    const $main = $('#main');
    if ($main.length > 0) {
      $main.empty().append(renderQueryInterface());
    }

    // For help.html
    const $helpModels = $('#help-models');
    if ($helpModels.length > 0) {
      $helpModels.empty().append(renderModelsTable());
    }
  });
});
