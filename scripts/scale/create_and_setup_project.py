import argparse
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
    # Create 10 tasks in the calibration batch
    for i in range(10):
        payload = dict(
            project=project_name,
            batch=calib_batch_name,
            instruction="This is a fake calibration task to bypass the API. Please simply answer Yes.",
            attachment_type="text",
            attachments=[
                {
                    "type": "text",
                    "content": "This is a fake calibration task to bypass the API. "
                    "We do not need calibration but would like to be able to send actual task. "
                    "In order to do this, we need to finish calibration. Please simply answer Yes.",
                }
            ],
            fields=[
                {
                    "type": "category",
                    "field_id": "answer",
                    "title": "Continue to the next task?",
                    "choices": [{"label": "Yes", "value": "yes"}, {"label": "No", "value": "no"}],
                }
            ],
        )
        client.create_task(TaskType.TextCollection, **payload)
        print(f"    Calibration task {i} created.")
    print("Finalizing calibration batch...")
    client.finalize_batch(calib_batch_name)
    print("Calibration batch finalized.")
except ScaleDuplicateResource as err:
    print(f"Calibration batch {calib_batch_name} already exists. It will not be recreated. Error: {err}")


# Create evaluation tasks
expected_response = {
    "annotations": {"answer_reasonable": {"type": "category", "field_id": "answer", "response": [["no"]]}}
}
initial_response = {
    "annotations": {"answer_reasonable": {"type": "category", "field_id": "answer", "response": [["yes"]]}}
}
attachments = [
    {
        "type": "text",
        "content": "Please Answer Yes to this question. This is simply a way to bypass the need for evaluation tasks.",
    },
]
payload = dict(
    project=project_name,
    rapid=True,
    attachments=attachments,
    initial_response=initial_response,
    expected_response=expected_response,
    fields=[
        {
            "type": "category",
            "field_id": "answer",
            "title": "Continue to the next task?",
            "choices": [{"label": "Yes", "value": "yes"}, {"label": "No", "value": "no"}],
        }
    ],
)
print("\nCreating evaluation tasks...")
for i in range(10):
    evaluation_task = client.create_evaluation_task(TaskType.TextCollection, **payload)
    print(f"    Evaluation task {i} created.")
print("Evaluation tasks created.")

# Create a test batch
print("\nCreating test batch...")
try:
    test_batch_name = project_name + "_test"
    batch = client.create_batch(
        project=project_name,
        batch_name=test_batch_name,
        calibration_batch=False,
    )
    print("Test batch created.")
except ScaleDuplicateResource as err:
    print(f"Test batch {test_batch_name} already exists. It will not be recreated. Error: {err}")
# Try to create a single task in the test batch
payload = dict(
    project=project_name,
    batch=test_batch_name,
    instruction="This is a test task to check that we can create tasks. If you are a worker please simply answer Yes.",
    attachment_type="text",
    attachments=[
        {
            "type": "text",
            "content": "This is a placeholder for the test task. If you are a worker please simply answer Yes.",
        }
    ],
    fields=[
        {
            "type": "category",
            "field_id": "answer",
            "title": "Finish?",
            "choices": [{"label": "Yes", "value": "yes"}, {"label": "No", "value": "no"}],
        }
    ],
)
print("Creating test task...")
client.create_task(TaskType.TextCollection, **payload)
print("Test task created.")
print("The test batch is not going to be finalized so that it does not get sent to workers.")

# If we are here, it means that the project is ready.
# Print the project_name and a success message.
print(f"\n\nProject {project_name} is ready.")
print("Please go to https://app.scale.com/projects to check that the project is ready.")
