# Deploying to production (for maintainers)

The production version of the proxy is running on `crfm-models.stanford.edu`;
you need to get permission to get ssh access.

## One-time setup

This is done, but just for the record:

    laptop:$ ssh crfm-models.stanford.edu
    crfm-models:$ cd /home
    crfm-models:$ git clone git@github.com:stanford-crfm/benchmarking
    crfm-models:$ cd benchmarking
    crfm-models:$ mkdir prod_env
    crfm-models:$ echo '{"api_key": "crfm"}' > prod_env/accounts.jsonl
    laptop:$ rsync -arvz prod_env/credentials.conf crfm-models.stanford.edu:/home/benchmarking/prod_env

## Perspective API

We use Google's [Perspective API](https://www.perspectiveapi.com) to calculate the toxicity of completions.
To send requests to PerspectiveAPI, we need to generate an API key from GCP. Follow the
[Get Started guide](https://developers.perspectiveapi.com/s/docs-get-started)
to request the service and the [Enable the API guide](https://developers.perspectiveapi.com/s/docs-enable-the-api)
to generate the API key. Once you have a valid API key, add an entry to `credentials.conf`:

```
perspectiveApiKey: <Generated API key>
```

By default, Perspective API allows only 1 query per second. Fill out this
[form](https://developers.perspectiveapi.com/s/request-quota-increase) to increase the request quota.

The [current API key](https://console.cloud.google.com/apis/api/commentanalyzer.googleapis.com/overview?authuser=1&project=hai-gcp-models)
we are using in production was created with the `hai-gcp-models` account and allows 200 queries per second.
**The API key expires on 7/30/2022.**

## SSL

The SSL certificate, CSR and private key for crfm-models.stanford.edu is stored at `/home/ssl`.
**The current SSL certificate expires on 12/30/2022.**

To renew the SSL certificate, follow these steps:

1. Fill out this [form](https://certificate.stanford.edu/cert-request):
    1. Log on with your SUNet ID. You must be an admin in order to submit a request.
    1. For `Server Name`, put `crfm-models.stanford.edu`.
    1. For `Server type`, select `OTHER`.
    1. For `Contact group/mailman address`, enter your Stanford email address.
    1. Under `Copy and paste your CSR`, paste the content of `/home/ssl/public.csr`.
    1. Leave the optional fields blank and click `Submit`.
    1. You should receive your certificate by email within 2 business days.
2. Once you receive the SSL cert, concatenate the contents of `X509 Certificate only, Base64 encoded`
   with the contents of `X509 Intermediates/root only Reverse, Base64 encoded`
   and place it at path `/home/ssl/crfm-models.crt`. `crfm-models.crt` should look something like this:

   ```text
    -----BEGIN CERTIFICATE-----
    (Your Primary SSL certificate: .crt)
    -----END CERTIFICATE-----
    -----BEGIN CERTIFICATE-----
    (Your Intermediate certificate: reversed.crt)
    -----END CERTIFICATE-----
   ```
3. Restart the server.
4. Open the [website](https://crfm-models.stanford.edu) in a browser and verify the connection is secure.

### Misplaced private key or CSR

If, for whatever reason, the private key or CSR is misplaced, generate new ones by running:

`sudo openssl req -new -nodes -newkey rsa:2048 -keyout private.key -out public.csr`

and fill out the form:

```text
Country Name (2 letter code) [AU]:US
State or Province Name (full name) [Some-State]:California
Locality Name (eg, city) []:Stanford
Organization Name (eg, company) [Internet Widgits Pty Ltd]:Stanford University
Organizational Unit Name (eg, section) []:CRFM
Common Name (e.g. server FQDN or YOUR name) []:crfm-models.stanford.edu
Email Address []:

Please enter the following 'extra' attributes
to be sent with your certificate request
A challenge password []:
An optional company name []:
```

Then, follow the steps above to request for a new SSL certificate.

## Deployment

Every time we need to deploy, do the following.

Update the code:

    laptop:$ ssh crfm-models.stanford.edu
    crfm-models:$ cd /home/benchmarking
    crfm-models:$ git pull
    crfm-models:$ ./pre-commit.sh

If everything looks okay:

    ssh crfm@crfm-models.stanford.edu

    # Switch into the screen session
    crfm-models:$ screen -r deploy

    # Hit ctrl-c to kill the existing process
    # Restart the server
    sudo venv/bin/crfm-proxy-server -p 443 --ssl-key-file /home/ssl/private.key --ssl-cert-file /home/ssl/crfm-models.crt --workers 16 &> server.log

    # Exit the screen session: ctrl-ad

The recommended number of Gunicorn workers is twice the number of cores.
crfm-models.stanford.edu has 8 cores (verified with `nproc`) * 2 = 16 workers.

Double check that the [website](https://crfm-models.stanford.edu) still works.
The server logs can be streamed by running: `tail -f /home/benchmarking/server.log`.
