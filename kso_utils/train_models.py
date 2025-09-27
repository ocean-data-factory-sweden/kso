import kso_utils.project_utils as p_utils
import kso_utils.server_utils as s_utils
from kso_utils.MLProjectProcessor import MLProjectProcessor


def train_models(
    project_name="Template project",
    data_path="ml-template-data",
    exp_name="test_template",
    model_name="yolov8m-base-model",
    model_download_dir=".",
    batch_size=1,
    epochs=1,
    img_h=128,
):
    # Find project
    project = p_utils.find_project(project_name=project_name)
    # Initialise mlp
    mlp = MLProjectProcessor(project)

    # Download the data for the template projects, other projects the computer should have the data
    if project_name == "Template project":
        s_utils.get_ml_data(project)

    # Configure the data paths
    mlp.output_path = data_path
    mlp.setup_paths()

    # Get the baseline model from the registry, specified by model_name
    model_path = mlp.get_model(model_name, model_download_dir, baseline=True)

    mlp.train_yolo(
        exp_name=exp_name,
        weights=str(model_path),
        project=mlp.project_name,
        epochs=epochs,
        batch_size=batch_size,
        img_size=img_h,  # this requires an int
    )

    return mlp  # To allow for doing more manual analysis in the notebook afterwards, for example: enhance_yolo
