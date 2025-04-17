# Credentials

## Credentials file

You should create a `credentials.conf` file in your local configuration folder, which is `./prod_env/` by default, unless you have overridden it using the `--local-path` flag to `helm-run`. This file should be in HOCON format. Example:

```
platformOneApiKey: sk-abcdefgh
platformTneApiKey: sk-ijklmnop
```

Here are the keys that must be set for to access these platforms:

- AI21: `ai21ApiKey`
- Aleph Alpha: `AlephAlphaApiKey`
- Anthropic: `anthropicApiKey`
- Cohere: `cohereApiKey`
- Google: `googleProjectId`, `googleLocation`, also see Additional Setup below
- GooseAI: `gooseApiKey`
- Hugging Face Hub: None, but see Additional Setup below
- Mistral AI: `mistralaiApiKey`
- OpenAI: `openaiApiKey`, `openApiOrgId`
- Perspective: `perspectiveApiKey`
- Writer: `writerApiKey`

## Additional setup

### Google

You will need to install the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install). Then, as the user that will be running `helm-run`, run:

```
gcloud auth application-default login
gcloud auth application-default set-quota-project 123456789012
```

Replace `123456789012` with your actual _numeric_ project ID.

### Hugging Face Hub

If you are attempting to access models that are private, restricted, or require signing an agreement (e.g. Llama 2) through Hugging Face, you need to be authenticated to Hugging Face through the CLI. As the user that will be running `helm-run`, run:

```
huggingface-cli login
```

Refer to [Hugging Face's documentation](https://huggingface.co/docs/huggingface_hub/en/quick-start#login-command) for more information.