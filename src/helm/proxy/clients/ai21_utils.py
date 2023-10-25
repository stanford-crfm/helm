from typing import Dict


class AI21RequestError(Exception):
    pass


def handle_failed_request(api_type: str, response: Dict):
    error_message: str = f"AI21 {api_type} API error -"

    # Error messages are returned via 'detail' or 'Error' in response
    if "detail" in response:
        error_message += f" Detail: {response['detail']}"
    if "Error" in response:
        error_message += f" Error: {response['Error']}"

    raise AI21RequestError(error_message)
