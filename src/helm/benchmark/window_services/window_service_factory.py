from typing import Optional

from helm.benchmark.model_deployment_registry import ModelDeployment, WindowServiceSpec, get_model_deployment
from helm.benchmark.tokenizer_config_registry import TokenizerConfig, get_tokenizer_config
from helm.benchmark.window_services.window_service import WindowService
from helm.benchmark.window_services.tokenizer_service import TokenizerService
from helm.common.object_spec import create_object, inject_object_spec_args


class WindowServiceFactory:
    @staticmethod
    def get_window_service(model_deployment_name: str, service: TokenizerService) -> WindowService:
        """
        Returns a `WindowService` given the name of the model.
        Make sure this function returns instantaneously on repeated calls.
        """
        model_deployment: Optional[ModelDeployment] = get_model_deployment(model_deployment_name)
        if model_deployment:
            # If the model deployment specifies a WindowServiceSpec, instantiate it.
            window_service_spec: WindowServiceSpec
            if model_deployment.window_service_spec:
                window_service_spec = model_deployment.window_service_spec
            else:
                window_service_spec = WindowServiceSpec(
                    class_name="helm.benchmark.window_services.default_window_service.DefaultWindowService", args={}
                )

            # If provided, look up special tokens from TokenizerConfig.
            end_of_text_token: Optional[str] = None
            prefix_token: Optional[str] = None
            if model_deployment.tokenizer_name:
                tokenizer_config: Optional[TokenizerConfig] = get_tokenizer_config(model_deployment.tokenizer_name)
                if tokenizer_config:
                    end_of_text_token = tokenizer_config.end_of_text_token
                    prefix_token = tokenizer_config.prefix_token

            # Perform dependency injection to fill in remaining arguments.
            # Dependency injection is needed here for these reasons:
            #
            # 1. Different window services have different parameters. Dependency injection provides arguments
            #    that match the parameters of the window services.
            # 2. Some arguments, such as the tokenizer service, are not static data objects that can be
            #    in the users configuration file. Instead, they have to be constructed dynamically at runtime.
            window_service_spec = inject_object_spec_args(
                window_service_spec,
                constant_bindings={
                    "service": service,
                    "tokenizer_name": model_deployment.tokenizer_name,
                    "max_sequence_length": model_deployment.max_sequence_length,
                    "max_request_length": model_deployment.max_request_length,
                    "max_sequence_and_generated_tokens_length": model_deployment.max_sequence_and_generated_tokens_length,  # noqa
                    "end_of_text_token": end_of_text_token,
                    "prefix_token": prefix_token,
                },
                provider_bindings={
                    "gpt2_window_service": lambda: WindowServiceFactory.get_window_service("huggingface/gpt2", service)
                },
            )
            return create_object(window_service_spec)

        raise ValueError(f"Unhandled model deployment name: {model_deployment_name}")
