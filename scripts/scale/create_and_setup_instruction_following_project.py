import argparse
import json
from scale_utils import get_scale_client
from scaleapi.tasks import TaskType
from scaleapi.exceptions import ScaleDuplicateResource

parser = argparse.ArgumentParser()
parser.add_argument("--project_name", type=str, help="Name of the project to create")
parser.add_argument(
    "--credentials_path", type=str, default="prod_env/credentials.conf", help="Path to the credentials file"
)
args = parser.parse_args()
project_name = args.project_name
client = get_scale_client(args.credentials_path)

print("\nGetting project...")
try:
    print(f"Trying to create project {project_name} ...")
    project = client.create_project(
        project_name=project_name,
        task_type=TaskType.TextCollection,
        rapid=True,
        params={},
    )
    print("Project created.")
except ScaleDuplicateResource as err:
    print(f"Project {project_name} already exists. Using existing project. Error: {err}")
    project = client.get_project(project_name)


# Create a calibration batch
print("\nCreating calibration batch...")
try:
    calib_batch_name = project_name + "_calibration"
    batch = client.create_batch(
        project=project_name,
        batch_name=calib_batch_name,
        calibration_batch=True,
    )
    print("Calibration batch created.")
    # Create calibration tasks
    with open("scripts/scale/instruction_following_calibration_instances.jsonl", "r") as f:
        instances = json.load(f)["instances"]
    for i in range(len(instances)):
        instance: dict = instances[i]
        payload = dict(
            project=project_name,
            batch=calib_batch_name,
            instruction="Evaluate the AI model generated output following the instructions below",
            attachment_type="text",
            attachments=[
                {
                    "type": "text",
                    "content": "<p>Rate the response to the instruction. Please read the <a href=https://docs.google.com/document/d/1tWArTQiuuM44v4Db85C638i7fkHLTP_fXpGaxiS8c5M/edit?usp=sharing>tutorial and examples</a> before starting.</p>"  # noqa: E501
                    "<h4>Instruction</h4>"
                    f'<p style="white-space: pre-wrap;">{instance["instruction"]}</p>'
                    "<h4>Response</h4>"
                    f'<p style="white-space: pre-wrap;">{instance["response"]}</p>',
                }
            ],
            fields=[
                {
                    "type": "category",
                    "field_id": question["criterion_name"],
                    "title": question["criterion_name"],
                    "description": question["description"],
                    "choices": [
                        {"label": question["choices"][i], "value": i + 1} for i in range(len(question["choices"]))
                    ],
                }
                for question in instance["multiple_choice_questions"]
            ]
            + [
                {
                    "type": "text",
                    "field_id": question["name"],
                    "title": question["name"],
                    "description": question["description"],
                    "max_characters": 500,
                    "required": True,
                }
                for question in instance["text_questions"]
            ],
        )
        client.create_task(TaskType.TextCollection, **payload)
        print(f"    Calibration task {i} created.")
    print("Finalizing calibration batch...")
    client.finalize_batch(calib_batch_name)
    print("Calibration batch finalized.")
except ScaleDuplicateResource as err:
    print(f"Calibration batch {calib_batch_name} already exists. It will not be recreated. Error: {err}")
