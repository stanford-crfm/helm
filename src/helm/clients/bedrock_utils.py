"""Helper utilities for working with Amazon Bedrock."""

import os
from typing import Optional

from helm.common.hierarchical_logger import hlog
from helm.common.optional_dependencies import handle_module_not_found_error

try:
    import boto3
    from boto3 import Session
    from botocore.config import Config
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["aws"])


# From https://github.com/aws-samples/amazon-bedrock-workshop/blob/main/01_Generation/00_generate_w_bedrock.ipynb
# MIT-0 Licensed
def get_bedrock_client(
    assumed_role: Optional[str] = None,
    region: Optional[str] = None,
    runtime: Optional[bool] = True,
):
    """Create a boto3 client for Amazon Bedrock, with optional configuration overrides

    Parameters
    ----------
    assumed_role :
        Optional ARN of an AWS IAM role to assume for calling the Bedrock service. If not
        specified, the current active credentials will be used.
    region :
        Optional name of the AWS Region in which the service should be called (e.g. "us-east-1").
        If not specified, AWS_REGION or AWS_DEFAULT_REGION environment variable will be used.
    runtime :
        Optional choice of getting different client to perform operations with the Amazon Bedrock service.
    """
    if region is None:
        target_region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION"))
    else:
        target_region = region

    session_kwargs = {"region_name": target_region}
    client_kwargs = {**session_kwargs}

    profile_name = os.environ.get("AWS_PROFILE")
    if profile_name:
        session_kwargs["profile_name"] = profile_name

    retry_config = Config(
        region_name=target_region,
        retries={
            "max_attempts": 10,
            "mode": "standard",
        },
    )
    session = boto3.Session(**session_kwargs)

    if assumed_role:
        sts = session.client("sts")
        response = sts.assume_role(RoleArn=str(assumed_role), RoleSessionName="crfm-helm")
        client_kwargs["aws_access_key_id"] = response["Credentials"]["AccessKeyId"]
        client_kwargs["aws_secret_access_key"] = response["Credentials"]["SecretAccessKey"]
        client_kwargs["aws_session_token"] = response["Credentials"]["SessionToken"]

    if runtime:
        service_name = "bedrock-runtime"
    else:
        service_name = "bedrock"

    bedrock_client = session.client(service_name=service_name, config=retry_config, **client_kwargs)

    hlog(f"Amazon Bedrock client successfully created with endpoint {bedrock_client._endpoint}")
    return bedrock_client


def get_bedrock_client_v1(
    region: Optional[str] = None,
    service_name: str = "bedrock-runtime",
    assumed_role: Optional[str] = None,
    read_timeout: int = 5000,
    connect_timeout: int = 5000,
    max_attempts: int = 10,
):
    boto_config = Config(
        read_timeout=read_timeout, connect_timeout=connect_timeout, retries={"max_attempts": max_attempts}
    )

    if assumed_role:
        session = boto3.Session(region_name=region)
        # Assume role and get credentials
        sts = session.client("sts")
        creds = sts.assume_role(RoleArn=str(assumed_role), RoleSessionName="crfm-helm")["Credentials"]
        session = Session(
            aws_access_key_id=creds["AccessKeyId"],
            aws_secret_access_key=creds["SecretAccessKey"],
            aws_session_token=creds["SessionToken"],
        )
        return session.client(
            service_name=service_name,
            region_name=region,
            config=boto_config,
        )

    # default to instance role to get the aws credentials or aws configured credentials
    return boto3.client(service_name=service_name, region_name=region, config=boto_config)
