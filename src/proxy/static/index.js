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

  // Logging in and out
  function updateLogin() {
    const $loginInfo = $('#loginInfo');
    $loginInfo.empty();
    let api_key = readCookie('api_key');
    if (api_key) {
      auth = {api_key};
      $loginInfo.append($('<span>').append('You are logged in with API key ' + censor(api_key) + '&nbsp;'));
      $loginInfo.append($('<button>').append('Logout').click(() => {
        eraseCookie('api_key');
        updateLogin();
      }));
    } else {
      auth = null;
      $loginInfo.append($('<button>').append('Login').click(() => {
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

  function multiline(s) {
    return s.replace(/\n/g, '<br>');
  }

  function renderError(e) {
    return $('<div>').addClass('alert alert-danger').append(multiline(e));
  }

  function renderExampleQueries(updateQuery) {
    const $examplesBlock = $('<div>');
    $examplesBlock.append($('<span>').append('Example queries:'));
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
    // A query that we're making.
    const $queryBlock = $('<div>');
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

    $queryBlock.append($('<h4>').append('Query'));
    $queryBlock.append($('<div>').append($prompt));
    $queryBlock.append($('<div>').append($settings));
    $queryBlock.append($('<div>').append($environments));
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
    const title = JSON.stringify(request);
    // Always include model, never prompt (since that's shown right after).
    const showKeys = ['model'].concat(changingKeys.filter((key) => key !== 'prompt' && key !== 'model'));
    const summary = '[' + showKeys.map(key => key + ':' + request[key]).join(', ') + ']';
    return $('<div>', {title}).append(summary + ' ' + multiline(request.prompt));
  }

  function renderTime(time) {
    return (Math.round(time * 10) / 10) + 's';
  }

  function renderTokens(tokens) {
    // Render text as a sequence of tokens
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
      const $token = $('<span>', {class: 'token', title}).append(multiline(token.text));
      $result.append($token);
    }
    return $result;
  }

  function renderRequestResult(requestResult) {
    if (requestResult.error) {
      return renderError(requestResult.error);
    }
    const $result = $('<div>');
    requestResult.completions.forEach((completion) => {
      $result.append($('<ul>').append($('<li>')
        .append(renderTokens(completion.tokens))));
    });
    $result.append($('<i>').append(renderTime(requestResult.request_time)));
    return $result;
  }

  function renderAccount() {
    if (!auth) {
      return null;
    }

    const $accountBlock = $('<div>');
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
        .append($('<a>', {href: 'help.html#quotas'}).append('Usage'))
        .append(': ')
        .append(items.join(' | '));
    });
    return $accountBlock;
  }

  function renderQueryInterface() {
    const $accountBlock = $('<div>').append(renderAccount());

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
    const $requestsBlock = $('<div>');

    const $group = $('<div>');
    $group.append($accountBlock);
    $group.append($exampleQueries);
    $group.append($queryBlock);
    $group.append($requestsBlock);
    return $group;
  }

  ////////////////////////////////////////////////////////////
  // Main

  let generalInfo;

  $.getJSON('/api/general_info', (response) => {
    generalInfo = response;
    console.log('/api/general_info', generalInfo);
    $('#main').empty().append(renderQueryInterface());

    const $helpModels = $('#help-models');
    $helpModels.empty().append(renderModelsTable);
  });

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

});
