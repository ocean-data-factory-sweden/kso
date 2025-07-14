import wandb
import logging
import ultralytics
from pathlib import Path
import os

# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


def init():
    """
    WandB traking of the training is managed via YOLO, so we don't need many wandb statements in our own code.
    When for example YOLO.train is called, it automatically logs in and initiates the tracking, and closes the run.
    It can happen that the closing fails, when the training itself fails, or if it was a very short session. For this
    we should have a check: close_run() and call it after every yolo function.

    Only if we would want to log any extra analysis, we would need to open up a certain run again
    (wandb.init(resume="must", id=run.id)), log it there with wandb.log and finish the session with wandb.finish()

    WandB will automatically log in by using the environment variable WANDB_API_KEY, so this should be specified.

    For future development: If we want to support other model packages than YOLO, we might need to specify the start and
    end of a run (tracking) ourselves and not rely on this init() function anymore.
    """
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key and not wandb.api.api_key:
        raise AssertionError(
            "No WandB login provided. Set the WANDB_API_KEY in your environment, or if you are running locally, run wandb.login() and enter your credentials., This will be saved for future runs."
        )
    if api_key:
        wandb.login(key=api_key)

    ultralytics.settings.update({"wandb": True})


def start_run(self, project, name):
    self.run = wandb.init(
        entity=self.team_name,
        project=project,
        name=name,
        settings=wandb.Settings(start_method="thread"),
    )


def close_run():
    if wandb.run is not None:
        wandb.finish()


def _get_full_path(self, custom_project: str, baseline: bool):
    if len(custom_project) > 0:
        logging.info(
            "Please note: Using models from custom project, please ensure that you have access."
        )
        full_path = custom_project
    elif baseline:
        full_path = f"koster/model-registry"
    else:
        if self.team_name == "wildlife-ai":
            logging.info("Please note: Using models from adi-ohad-heb-uni account.")
            full_path = "adi-ohad-heb-uni/project-wildlife-ai"
        elif self.project_name == "template_project":
            full_path = f"{self.team_name}/spyfish_aotearoa"
        else:
            full_path = f"{self.team_name}/{self.project_name.lower()}"
    return full_path


def show_available_models(self, custom_project: str = "", baseline: bool = False):
    """
    List all the models that are available on WandB. By default it shows it for the set team and project name.
    If you want to look at all the baseline models, set baseline to true.
    If you have a different path to your project on WandB, specify that path as custom_project.
    """
    full_path = _get_full_path(self, custom_project, baseline)
    api = wandb.Api()
    model_collections = set()

    # Go through each run in the project and check for logged model artifacts
    for run in api.runs(full_path):
        for artifact in run.logged_artifacts():
            if artifact.type == "model":
                # Only keep the base name, not version suffix
                model_name = artifact.name.split(":")[0]
                model_collections.add(model_name)

    return sorted(model_collections)


def get_model(
    self,
    model_name: str,
    download_path: str,
    custom_project: str = "",
    baseline: bool = False,
):
    """
    This function retrieves the latest model for model_name from WandB.
    model_name: the name of the model we want to retrieve
    download_path: the path on the local computer that the model should be downloaded to
    custom_project: the path to the location of the model on WandB for custom projects, for others, it is build uip from the team name and project name.
    baseline: to access baseline models in the koster/model_registry.
    Returns a string to the location of the model at the local computer after download.
    """
    full_path = _get_full_path(self, custom_project, baseline)
    wandb_model = f"{full_path}/{model_name}"
    try:
        api = wandb.Api()
        artifacts = api.artifact_collection(
            type_name="model", name=wandb_model
        ).artifacts()
        latest = artifacts[
            0
        ]  # This is the latest model, since artifact_collections returns a sorted list
        model_dir = latest.download(
            root=download_path
        )  # only gets downloaded if the file does not exist yet
        model_filename = list(latest.manifest.entries.keys())[0]
        return f"{model_dir}/{model_filename}"
    except Exception as e:
        # WandB is not specific with the errors they return when the model does not exist, for example:
        # 404 Client Error: Not Found for url: https://api.wandb.ai/graphql"
        # But it can also happen that it passes and returns an empty list, in which case this code fails with an
        # indexerror in line 91. The behaviour is not consistent. Therefore catch all exceptions and print the error.
        raise AttributeError(
            f"Error when trying to retrieve the model '{wandb_model}' from WandB. Error raied is: {e}"
        )


def get_dataset(self, model: str, team_name: str = "koster"):
    api = wandb.Api()
    if "_" in model:
        run_id = model.split("_")[1]
        try:
            run = api.run(f"{team_name}/{self.project_name}/runs/{run_id}")
        except wandb.CommError:
            logging.error("Run data not found")
            return "", ""
        datasets = [
            artifact for artifact in run.used_artifacts() if artifact.type == "dataset"
        ]
        if len(datasets) == 0:
            logging.error(
                "No datasets are linked to these runs. Please try another run."
            )
            return "", ""
        dirs = []
        for i in range(len(["train", "val"])):
            artifact = datasets[i]
            logging.info(f"Downloading {artifact.name} checkpoint...")
            artifact_dir = artifact.download()
            logging.info(f"{artifact.name} - Dataset downloaded.")
            dirs.append(artifact_dir)
        return dirs
    else:
        logging.error("Externally trained model. No data available.")
        return "", ""


def set_config(**kwargs):
    """
    `set_config` takes in a confidence threshold, model name, and evaluation directory and returns a
    configuration object.

    :param conf_thres: This is the confidence threshold for the bounding boxes
    :type conf_thres: float
    :param model: The name of the model you want to use
    :type model: str
    :param eval_dir: The directory where the evaluation images are stored
    :type eval_dir: str
    :return: The config object is being returned.
    """
    config = wandb.config
    for key, value in kwargs.items():
        if key == "model_name" and "model_name" in config:
            pass
        else:
            setattr(config, key, value)
    return config


def add_data(path: str, name: str, registry: str, run):
    """
    > The function `add_data` takes a path to a directory, a name for the directory, and a run
    object, and adds the directory to the run as an artifact

    :param path: the path to the directory you want to upload
    :type path: str
    :param name: The name of the artifact
    :type name: str
    :param run: The run object that you get from calling wandb.init()
    """
    my_data = wandb.Artifact(name, type="raw_data")
    if Path(path).is_dir():
        my_data.add_dir(path)
        run.log_artifact(my_data)
    else:
        my_data.add_file(path)
        run.log_artifact(my_data)
