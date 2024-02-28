import os
import tempfile

from helm.common.optional_dependencies import handle_module_not_found_error


class PretrainedFromWandbMixin:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Initializes from a wandb artifact or delegates loading to the superclass.
        """
        try:
            import wandb
        except ModuleNotFoundError as e:
            handle_module_not_found_error(e, ["heim"])

        with tempfile.TemporaryDirectory() as tmp_dir:  # avoid multiple artifact copies
            if ":" in pretrained_model_name_or_path and not os.path.isdir(pretrained_model_name_or_path):
                # wandb artifact
                if wandb.run is not None:
                    artifact = wandb.run.use_artifact(pretrained_model_name_or_path)
                else:
                    artifact = wandb.Api().artifact(pretrained_model_name_or_path)
                pretrained_model_name_or_path = artifact.download(tmp_dir)

            return super(PretrainedFromWandbMixin, cls).from_pretrained(
                pretrained_model_name_or_path, *model_args, **kwargs
            )
