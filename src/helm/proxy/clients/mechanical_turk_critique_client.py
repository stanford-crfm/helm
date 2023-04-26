import textwrap
import threading

import boto3
from helm.common.cache import Cache, CacheConfig
from helm.common.critique_request import (
    CritiqueQuestionTemplate,
    CritiqueRequest,
    CritiqueRequestResult,
    CritiqueTaskTemplate,
    QuestionType,
    populate_template_with_fields,
)
from helm.common.hierarchical_logger import hlog
from helm.proxy.clients.critique_client import CritiqueClient
from typing import Dict, List, Optional


_mechanical_turk_cache_lock = threading.Lock()


_DEFAULT_ASSIGNMENT_DURATION_IN_SECONDS = 28 * 24 * 60 * 60  # 28 days
"""An amount of time, in seconds, after which the HIT is no longer available for users to accept.

After the lifetime of the HIT elapses, the HIT no longer appears in HIT searches, even if not all
of the assignments for the HIT have been accepted."""

_DEFAULT_LIFETIME_IN_SECONDS = 24 * 60 * 60  # 1 day
"""An amount of time, in seconds, after which the HIT is no longer available for users to accept.

After the lifetime of the HIT elapses, the HIT no longer appears in HIT searches, even if not all
of the assignments for the HIT have been accepted."""

_XML_NAMESPACE = "http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2017-11-06/QuestionForm.xsd"
"""XML schema for Mechanical Turk.

See: https://docs.aws.amazon.com/AWSMechTurk/latest/AWSMturkAPI/ApiReference_SchemaLocationArticle.html"""


def _indent_to_level(text: str, level: int) -> str:
    """Helper for indenting XML to the same level as the external template."""
    return textwrap.indent(text, " " * 4 * level).lstrip()


def _render_option_xml(index: int, option: str, fields: Dict[str, str]) -> str:
    # The SelectionIdentifier must be alphanumeric.
    # Since the content may contain non-alphanumeric characters, we set the SelectionIdentifier to
    # the option index instead.
    return textwrap.dedent(
        f"""\
        <Selection>
            <SelectionIdentifier>{index}</SelectionIdentifier>
            <FormattedContent><![CDATA[{populate_template_with_fields(option, fields)}]]></FormattedContent>
        </Selection>"""
    )


def _render_options_xml(options: List[str], fields: Dict[str, str]) -> str:
    return "\n".join([_render_option_xml(index, option, fields) for index, option in enumerate(options)])


def _render_multiple_choice_answers_xml(options: List[str], fields: Dict[str, str]) -> str:
    options_xml_list = _render_options_xml(options, fields)
    return textwrap.dedent(
        f"""\
        <SelectionAnswer>
            <StyleSuggestion>radiobutton</StyleSuggestion>
            <Selections>
                {_indent_to_level(options_xml_list, 4)}
            </Selections>
        </SelectionAnswer>"""
    )


def _render_checkbox_answers_xml(options: List[str], fields: Dict[str, str]) -> str:
    options_xml_list = _render_options_xml(options, fields)
    return textwrap.dedent(
        f"""\
        <SelectionAnswer>
            <MinSelectionCount>0</MinSelectionCount>
            <MaxSelectionCount>{len(options)}</MaxSelectionCount>
            <Selections>
                {_indent_to_level(options_xml_list, 4)}
            </Selections>
        </SelectionAnswer>"""
    )


def _render_question_xml(question: CritiqueQuestionTemplate, fields: Dict[str, str]) -> str:
    answers_xml: str
    if question.question_type == QuestionType.FREE_RESPONSE:
        answers_xml = textwrap.dedent(
            """\
            <FreeTextAnswer></FreeTextAnswer>"""
        )
    elif question.question_type == QuestionType.MULTIPLE_CHOICE:
        answers_xml = _render_multiple_choice_answers_xml(question.options, fields)
    elif question.question_type == QuestionType.CHECKBOX:
        answers_xml = _render_checkbox_answers_xml(question.options, fields)
    return textwrap.dedent(
        f"""\
        <Question>
            <QuestionIdentifier>{question.name}</QuestionIdentifier>
            <DisplayName>{populate_template_with_fields(question.text, fields)}</DisplayName>
            <IsRequired>true</IsRequired>
            <QuestionContent>
                <FormattedContent><![CDATA[{populate_template_with_fields(question.text, fields)}]]></FormattedContent>
            </QuestionContent>
            <AnswerSpecification>
                {_indent_to_level(answers_xml, 4)}
            </AnswerSpecification>
        </Question>"""
    )


def _render_overview_xml(instructions: str, fields: Dict[str, str]) -> str:
    return textwrap.dedent(
        f"""\
        <Overview>
            <FormattedContent><![CDATA[{populate_template_with_fields(instructions, fields)}]]></FormattedContent>
        </Overview>"""
    )


def _render_question_form_xml(request: CritiqueRequest) -> str:
    overview_xml = _render_overview_xml(request.template.instructions, request.fields)
    questions_xml = "\n".join(
        [_render_question_xml(question, request.fields) for question in request.template.questions]
    )
    return textwrap.dedent(
        f"""\
        <QuestionForm xmlns="{_XML_NAMESPACE}">
            {_indent_to_level(overview_xml, 3)}
            {_indent_to_level(questions_xml, 3)}
        </QuestionForm>"""
    )


class MechanicalTurkCritiqueClient(CritiqueClient):
    """A CritiqueClient that creates HITs for workers in Mechanical Turk."""

    SANDBOX_ENDPOINT_URL = "https://mturk-requester-sandbox.us-east-1.amazonaws.com"
    """Endpoint URL for the Mechanical Turk sandbox.

    See: https://docs.aws.amazon.com/AWSMechTurk/latest/AWSMechanicalTurkRequester/mturk-use-sandbox.html"""

    def __init__(
        self,
        cache_config: CacheConfig,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        endpoint_url: Optional[str] = None,
    ):
        """Create a CritiqueClient.

        To authenticate with AWS, either specify both aws_access_key_id and aws_secret_access_key, or
        specify only aws_session_token. Alternatively, use another method supported by boto3, such as
        running `aws configure` on the command-line.
        See: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html

        endpoint_url can be set to the sendbox URL to make sandbox requests.
        See: https://docs.aws.amazon.com/AWSMechTurk/latest/AWSMechanicalTurkRequester/mturk-use-sandbox.html
        """
        client_kwargs = {}
        if aws_access_key_id:
            client_kwargs["aws_access_key_id"] = aws_access_key_id
        if aws_secret_access_key:
            client_kwargs["aws_secret_access_key"] = aws_secret_access_key
        if aws_session_token:
            client_kwargs["aws_session_token"] = aws_session_token
        if endpoint_url:
            client_kwargs["endpoint_url"] = endpoint_url
        self._mturk_client = boto3.Session().client("mturk", **client_kwargs)
        self._cache = Cache(cache_config)

    def _get_or_create_hit_type(self, template: CritiqueTaskTemplate) -> str:
        """Get or create a HIT Type on Mechanical Turk and return the HITTypeId.

        Attempt to find a HIT Type for the template in the cache. If one exists, reuse that HIT Type.
        Otherwise, create a HIT Type using the template. Return the HITTypeId."""

        with _mechanical_turk_cache_lock:
            raw_request = {
                "AssignmentDurationInSeconds": _DEFAULT_ASSIGNMENT_DURATION_IN_SECONDS,
                "Reward": template.reward,
                "Title": template.name,
                "Description": template.description,
            }

            def create_hit_type():
                raw_response = self._mturk_client.create_hit_type(**raw_request)
                return raw_response

            cache_key = {"create_hit_type": raw_request}
            create_hit_type_response, is_cached = self._cache.get(cache_key, create_hit_type)
        hit_type_id = create_hit_type_response["HITTypeId"]
        if is_cached:
            hlog(f"Reusing existing Mechanical Turk HIT Type: {hit_type_id}")
        else:
            hlog(f"Creating new Mechanical Turk HIT Type: {hit_type_id}")
        return hit_type_id

    def _get_or_create_hit(self, request: CritiqueRequest) -> str:
        """Get or create a HIT on Mechanical Turk and return the HITId.

        Attempt to find a HIT for the HITTypeId and the fields from the cache.
        If one exists, reuse that HIT. Otherwise, create a new HIT with the given HIT Type using the fields.
        and save it to the cache. Return the HITId.

        Similarly, attempt to find a HIT Type for the template in the cache and reuse it.
        Otherwise, create a HIT Type using the tepmlate"""
        hit_type_id = self._get_or_create_hit_type(request.template)

        with _mechanical_turk_cache_lock:
            question_form_xml = _render_question_form_xml(request)
            raw_request = {
                "HITTypeId": hit_type_id,
                "MaxAssignments": request.template.num_respondents,
                "LifetimeInSeconds": _DEFAULT_LIFETIME_IN_SECONDS,
                "Question": question_form_xml,
            }

            def create_hit_with_hit_type():
                raw_response = self._mturk_client.create_hit_with_hit_type(**raw_request)
                return raw_response

            cache_key = {"create_hit_with_hit_type": raw_request}
            create_hit_with_hit_type_response, is_cached = self._cache.get(cache_key, create_hit_with_hit_type)
        hit_id = create_hit_with_hit_type_response["HIT"]["HITId"]
        if is_cached:
            hlog(f"Reusing existing Mechanical Turk HIT: {hit_id}")
        else:
            hlog(f"Creating new Mechanical Turk HIT: {hit_id}")
        return hit_id

    def make_critique_request(self, request: CritiqueRequest) -> CritiqueRequestResult:
        self._get_or_create_hit(request)
        return CritiqueRequestResult([])
