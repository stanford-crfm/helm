/**
 * This is a very quick and dirty frontend for just interacting with the models.
 * Please refrain from adding additional functionality to this.
 * TODO: Write this in React.
 */
$(function () {
  const urlParams = decodeUrlParams(window.location.search);
  let auth = null;

  function censor(api_key) {
    // Show only the first k letters
    const k = 2;
    if (api_key.length <= k) {
      return api_key;
    }
    return api_key.substring(0, k) + "*".repeat(api_key.length - k);
  }

  // Logging in and out
  function updateLogin() {
    const $loginInfo = $("#loginInfo");
    $loginInfo.empty();
    let api_key = readCookie("api_key");
    if (api_key) {
      auth = { api_key };
      $loginInfo.append(
        $("<a>", { class: "nav-link", href: "#" })
          .append("Logout of API key " + censor(api_key))
          .click(() => {
            eraseCookie("api_key");
            updateLogin();
          }),
      );
    } else {
      auth = null;
      $loginInfo.append(
        $("<a>", { class: "nav-link", href: "#" })
          .append("Login")
          .click(() => {
            api_key = prompt("Enter your API key:");
            if (!api_key) {
              return;
            }

            // Check the API key the user entered using endpoint /api/account
            const args = { auth: JSON.stringify({ api_key }) };
            $.getJSON("/api/account", args, (response) => {
              console.log("/api/account", response);
              if ("error" in response) {
                alert("The API key you've entered is invalid. Try again.");
              } else {
                createCookie("api_key", api_key);
                updateLogin();
              }
            });
          }),
      );
    }
  }

  updateLogin();

  ////////////////////////////////////////////////////////////
  // Rendering functions

  function renderExampleQueries(updateQuery) {
    // Show links for each example query, so when you click on them, they populate the textboxes.
    const $examplesBlock = $("<div>", { class: "examples-block" });
    $examplesBlock.append($("<span>").append("Examples:"));
    generalInfo.example_queries.forEach((query, i) => {
      const href = "#";
      const title =
        "[Prompt]\n" +
        query.prompt +
        "\n[Settings]\n" +
        query.settings +
        "\n[Environments]\n" +
        query.environments;
      const $link = $("<a>", { href, title }).append(`[${i}]`);
      $link.click(() => {
        // Populate the query with the example
        updateQuery(query);
        urlParams.prompt = query.prompt;
        urlParams.settings = query.settings;
        urlParams.environments = query.environments;
        updateBrowserLocation(urlParams);
      });
      $examplesBlock.append("&nbsp;");
      $examplesBlock.append($link);
    });
    return $examplesBlock;
  }

  function renderQuery(handleQueryResult) {
    // Render the textboxes for entering the query (which includes the prompt, settings, and environment)
    const $queryBlock = $("<div>", { class: "block" });
    const $prompt = $("<textarea>", {
      cols: 90,
      rows: 7,
      placeholder: "Enter prompt",
    }).val(urlParams.prompt);
    const $settings = $("<textarea>", {
      cols: 90,
      rows: 5,
      placeholder:
        "Enter settings (e.g., model: openai/text-davinci-002 for Instruct GPT-3); click Help at the top to learn more",
    }).val(urlParams.settings);
    const $environments = $("<textarea>", {
      cols: 90,
      rows: 3,
      placeholder:
        "Enter environment variables (e.g., city: [Boston, New York]); click Help at the top to learn more",
    }).val(urlParams.environments);

    $queryBlock.data("prompt", $prompt);
    $queryBlock.data("settings", $settings);
    $queryBlock.data("environments", $environments);

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
        alert("You must log in first.");
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

      $.getJSON("/api/query", query, handleQueryResult);
    }

    // Show examples of queries
    const $exampleQueries = renderExampleQueries((query) => {
      $queryBlock.data("prompt").val(query.prompt);
      $queryBlock.data("settings").val(query.settings);
      $queryBlock.data("environments").val(query.environments);
      urlParams.prompt = query.prompt;
      urlParams.settings = query.settings;
      urlParams.environments = query.environments;
      updateBrowserLocation();
    });

    const promptHelp =
      "This is the text you feed into the language model to complete.\nExample:\n  Life is like";
    const settingsHelp =
      "Specifies what information we want from the language model (see [Help] for more details).\nExample:\n  temperature: ${temperature}\n  model: openai/davinci\n  max_tokens: 10\n  num_completions: 5";
    const environmentsHelp =
      "Specifies a list of values to try for each variable that appears in the prompt or settings.\nExample:\n  temperature: [0, 0.5, 1]";

    const $promptLabel = $("<span>")
      .append(helpIcon(promptHelp, "help.html#query"))
      .append("Prompt");
    const $settingsLabel = $("<span>")
      .append(helpIcon(settingsHelp, "help.html#query"))
      .append("Settings");
    const $environmentsLabel = $("<span>")
      .append(helpIcon(environmentsHelp, "help.html#query"))
      .append("Environments");

    $queryBlock.append($("<h4>").append("Query"));
    $queryBlock.append($exampleQueries);
    const $table = $("<table>", { class: "query-table" });
    $table.append(
      $("<tr>")
        .append($("<td>").append($promptLabel))
        .append($("<td>").append($prompt)),
    );
    $table.append(
      $("<tr>")
        .append($("<td>").append($settingsLabel))
        .append($("<td>").append($settings)),
    );
    $table.append(
      $("<tr>")
        .append($("<td>").append($environmentsLabel))
        .append($("<td>").append($environments)),
    );
    $queryBlock.append($table);
    $queryBlock.append($("<button>").append("Submit").click(submit));

    return $queryBlock;
  }

  function getChangingKeys(items) {
    // Return the list of keys whose values vary across `items.`
    if (items.length === 0) {
      return [];
    }
    return Object.keys(items[0]).filter((key) => {
      return !items.every(
        (item) => JSON.stringify(item[key]) === JSON.stringify(items[0][key]),
      );
    });
  }

  function renderRequest(changingKeys, request) {
    // Render the request metadata (e.g., temperature if it is changing)
    const title = JSON.stringify(request);
    // Always include model, never prompt (since that's shown right after).
    const showKeys = ["model"].concat(
      changingKeys.filter((key) => key !== "prompt" && key !== "model"),
    );
    const summary =
      "[" + showKeys.map((key) => key + ":" + request[key]).join(", ") + "]";
    return $("<div>", { title }).append(
      summary + " " + multilineHtml(request.prompt),
    );
  }

  function renderTime(time) {
    return Math.round(time * 10) / 10 + "s";
  }

  function constructTokenGroups(tokens) {
    // Note: sometimes multiple tokens correspond to one character, for example:
    // ["bytes:\xe2\x80", "bytes:\x99"] => ’
    // For these, we keep these in the buffer and collapse them, and concatenate the entries.
    //
    // get_num_bytes() and convert_tokens_to_text() in src/helm/benchmark/basic_metrics.py are adapted from this function.
    const groups = [];
    for (let i = 0; i < tokens.length;) {
      // Aggregate consecutive tokens while they're "bytes:..."
      const group = { tokens: [] };
      if (tokens[i].text.startsWith("bytes:")) {
        let bytestring = "";
        while (i < tokens.length && tokens[i].text.startsWith("bytes:")) {
          group.tokens.push(tokens[i]);
          // Extract part after : (e.g., \xe2\x80)
          bytestring += tokens[i].text.split(":")[1];
          i++;
        }
        // Convert to encoded URI (e.g., %e2%80%99) and decode
        group.text = decodeURIComponent(bytestring.replaceAll("\\x", "%"));
      } else {
        group.tokens.push(tokens[i]);
        group.text = tokens[i].text;
        i++;
      }
      groups.push(group);
    }
    return groups;
  }

  function renderTokens(tokens) {
    // Render text as a sequence of tokens that you can interact with to see more information (e.g., logprobs)
    const $result = $("<div>");
    const groups = constructTokenGroups(tokens);
    for (const group of groups) {
      const $group = $("<span>", { class: "token" }).append(
        multilineHtml(group.text),
      );
      $result.append($group);
    }
    return $result;
  }

  function renderRequestResult(requestResult) {
    // Render the list of completions.
    if (requestResult.error) {
      return renderError(requestResult.error);
    }
    const $result = $("<div>");
    requestResult.completions.forEach((completion) => {
      const $contents = $("<span>", {
        title: `logprob: ${completion.logprob}`,
      }).append(renderTokens(completion.tokens));
      const $metadata = $("<span>", { class: "metadata" });
      $metadata.append(
        $("<span>", { title: "Log probability" }).append(
          round(completion.logprob, 2),
        ),
      );
      if (completion.finish_reason) {
        const title =
          "Generation finished because of this reason: " +
          JSON.stringify(completion.finish_reason);
        $metadata
          .append(" ")
          .append(
            $("<span>", { title }).append(completion.finish_reason.reason),
          );
      }
      $result.append(
        $("<div>", { class: "completion" }).append($metadata).append($contents),
      );
    });
    $result.append($("<i>").append(renderTime(requestResult.request_time)));
    return $result;
  }

  function renderAccount() {
    // Render the account information (usage, quotas).
    if (!auth) {
      return null;
    }

    const $accountBlock = $("<div>", { class: "block" });
    const args = { auth: JSON.stringify(auth) };
    $.getJSON("/api/account", args, ([account]) => {
      console.log("/api/account", account);
      const items = [];
      for (modelGroup in account.usages) {
        for (granularity in account.usages[modelGroup]) {
          const usage = account.usages[modelGroup][granularity];
          // Only print out usage for model groups and granularities where there is a quota
          if (usage.quota) {
            const percent = Math.round((usage.used / usage.quota) * 100);
            items.push(
              `<b>${modelGroup}</b>: ${usage.period} (${usage.used} / ${usage.quota} = ${percent}%)`,
            );
          }
        }
      }
      if (items.length === 0) {
        items.push("no restrictions");
      }
      $accountBlock
        .empty()
        .append(
          helpIcon(
            "Specifies your usage/quota (321/10000) for each model group (e.g., gpt3) for the current period (e.g., 2022-1-2).",
            "help.html#quotas",
          ),
        )
        .append("Usage")
        .append(": ")
        .append(items.join(" | "));
    });
    return $accountBlock;
  }

  ////////////////////////////////////////////////////////////
  // For index.html

  function renderQueryInterface() {
    // For index.html
    const $accountBlock = $("<div>").append(renderAccount());

    // Allow editing the query
    const $queryBlock = renderQuery((queryResult) => {
      // Create requests
      console.log("/api/query", queryResult);
      $requestsBlock.empty();

      if (queryResult.error) {
        $requestsBlock.append(renderError(queryResult.error));
        return;
      }

      $requestsBlock.append(
        $("<h4>").append(`Requests (${queryResult.requests.length})`),
      );
      if (queryResult.error) {
        $requestsBlock.append(renderError(queryResult.error));
        return;
      }
      const changingKeys = getChangingKeys(queryResult.requests);
      queryResult.requests.forEach((request) => {
        const $request = $("<div>", { class: "request" }).append(
          renderRequest(changingKeys, request),
        );
        const $requestResult = $("<div>").append($("<i>").append("(waiting)"));
        const args = {
          auth: JSON.stringify(auth),
          request: JSON.stringify(request),
        };
        $.getJSON("/api/request", args, (requestResult) => {
          console.log("/api/request", request, requestResult);
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
    const $requestsBlock = $("<div>", { class: "block" });

    const $group = $("<div>");
    $group.append($accountBlock);
    $group.append($queryBlock);
    $group.append($requestsBlock);
    return $group;
  }

  ////////////////////////////////////////////////////////////
  // For help.html

  function renderModelsTable() {
    // Render the list of models
    const $table = $("<table>", { class: "table" });
    const $header = $("<tr>")
      .append($("<td>").append("group"))
      .append($("<td>").append("name"))
      .append($("<td>").append("description"))
      .append($("<td>").append("tags"));
    $table.append($header);
    generalInfo.all_models.forEach((model) => {
      const $row = $("<tr>")
        .append($("<td>").append($("<tt>").append(model.group)))
        .append($("<td>").append($("<tt>").append(model.name)))
        .append($("<td>").append(model.description))
        .append($("<td>").append(model.tags.join(" ")));
      $table.append($row);
    });
    return $table;
  }

  ////////////////////////////////////////////////////////////
  // Main

  let generalInfo;

  $.getJSON("/api/general_info", (response) => {
    generalInfo = response;
    console.log("/api/general_info", generalInfo);
    if (generalInfo.error) {
      alert(generalInfo.error);
      return;
    }

    // For index.html
    const $main = $("#main");
    if ($main.length > 0) {
      $main.empty().append(renderQueryInterface());
    }

    // For help.html
    const $helpModels = $("#help-models");
    if ($helpModels.length > 0) {
      $helpModels.empty().append(renderModelsTable());
    }
  });
});
