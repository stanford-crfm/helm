$(function () {
  const urlParams = decodeUrlParams(window.location.search);
  const rootUrl = '/static/index.html';
  let auth = null;

  // Logging in and out
  function updateLogin() {
    const $loginInfo = $('#loginInfo');
    $loginInfo.empty();
    let username = readCookie('username');
    let password = readCookie('password');
    auth = {username, password};
    if (username) {
      $loginInfo.append($('<span>').append('You are logged in as ' + username + '&nbsp;'));
      $loginInfo.append($('<button>').append('Logout').click(() => {
        eraseCookie('username');
        eraseCookie('password');
        updateLogin();
      }));
    } else {
      $loginInfo.append($('<button>').append('Login').click(() => {
        username = prompt('Enter your username:');
        if (!username) {
          return;
        }
        password = prompt('Enter your password (NOT SECURE):');
        if (!password) {
          return;
        }
        createCookie('username', username);
        createCookie('password', password);
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
    $examplesBlock.append($('<span>').append('Examples:'));
    generalInfo.exampleQueries.forEach((query, i) => {
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

      $.getJSON('/api/expandQuery', query, handleQueryResult);
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

  function renderRequestResult(requestResult) {
    if (requestResult.error) {
      return renderError(requestResult.error);
    }
    const $result = $('<div>');
    requestResult.completions.forEach((completion) => {
      $result.append($('<ul>').append($('<li>')
        .append(multiline(completion.text))));
    });
    $result.append($('<i>').append(renderTime(requestResult.requestTime)));
    return $result;
  }

  function renderUser() {
    const $userBlock = $('<div>');
    const args = {auth: JSON.stringify(auth)};
    $.getJSON('/api/getUser', args, (user) => {
      console.log('getUser', user);
      const items = [];
      items.push('Usage');
      ['daily', 'monthly', 'total'].forEach((granularity) => {
        const usages = user[granularity];
        for (const group in usages) {
          const usage = usages[group];
          if (usage.quota != null) {
            const percent = Math.round(usage.used / usage.quota * 100);
            items.push(`<b>${group}</b>: ${usage.period} (${usage.used} / ${usage.quota} = ${percent}%)`);
          }
        }
      });
      $userBlock.empty().append(items.join(' | '));
    });
    return $userBlock;
  }

  function renderQueryInterface() {
    const $userBlock = $('<div>').append(renderUser());

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
      console.log('expandQuery', queryResult);
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
        $.getJSON('/api/makeRequest', args, (requestResult) => {
          console.log('makeRequest', request, requestResult);
          $requestResult.empty().append(renderRequestResult(requestResult));
          if (!requestResult.cached) {
            $userBlock.empty().append(renderUser());
          }
        });
        $request.append($requestResult);
        $requestsBlock.append($request);
      });
    });

    // Where the requests and responses come in
    const $requestsBlock = $('<div>');

    const $group = $('<div>');
    $group.append($userBlock);
    $group.append($exampleQueries);
    $group.append($queryBlock);
    $group.append($requestsBlock);
    return $group;
  }

  ////////////////////////////////////////////////////////////
  // Main

  let generalInfo;

  $.getJSON('/api/getGeneralInfo', (response) => {
    generalInfo = response;
    console.log('generalInfo', generalInfo);
    $('#main').empty().append(renderQueryInterface());

    const $helpModels = $('#help-models');
    $helpModels.empty().append(renderModelsTable);
  });

  function renderModelsTable() {
    // Render the list of models
    const $table = $('<table>', {class: 'table'});
    const $header = $('<tr>')
      .append($('<td>').append('name'))
      .append($('<td>').append('description'));
    $table.append($header);
    generalInfo.allModels.forEach((model) => {
      const $row = $('<tr>')
        .append($('<td>').append($('<tt>').append(model.name)))
        .append($('<td>').append(model.description));
      $table.append($row);
    });
    return $table;
  };

});
