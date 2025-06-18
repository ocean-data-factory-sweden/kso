import wandb
import ipywidgets as widgets
from IPython.display import display, clear_output
from pathlib import Path
import logging
import itertools

# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


def choose_baseline_model(download_path: str):
    api = wandb.Api()
    # weird error fix (initialize api another time)
    api.runs(path="koster/model-registry")
    api = wandb.Api()
    collections = [
        coll
        for coll in api.artifact_type(
            type_name="model", project="koster/model-registry"
        ).collections()
    ]

    model_dict = {}
    for artifact in collections:
        model_dict[artifact.name] = artifact

    model_widget = widgets.Dropdown(
        options=[(name, model) for name, model in model_dict.items()],
        value=None,
        description="Select model:",
        ensure_option=False,
        disabled=False,
        layout=widgets.Layout(width="50%"),
        style={"description_width": "initial"},
    )

    main_out = widgets.Output()
    display(model_widget, main_out)

    def on_change(change):
        with main_out:
            clear_output()
            try:
                for af in model_dict[change["new"].name]:
                    artifact_dir = af.download(download_path)
                    artifact_file = [
                        str(i)
                        for i in Path(artifact_dir).iterdir()
                        if str(i).endswith(".pt")
                    ][-1]
                    logging.info(
                        f"Baseline {af.name} successfully downloaded from WANDB"
                    )
                    model_widget.artifact_path = artifact_file
            except Exception as e:
                logging.error(
                    f"Failed to download the baseline model. Please ensure you are logged in to WANDB. {e}"
                )
                model_widget.artifact_path = "yolov8m.pt"

    model_widget.observe(on_change, names="value")
    return model_widget


def choose_model(self, model_dict: dict, custom_project: str = ""):
    # TODO: Remove hardcoded API key from Zenodo
    model_info = {v: {"data": "No model info"} for k, v in model_dict.items()}
    data_info = {v: {"data": "No data info"} for k, v in model_dict.items()}

    api = wandb.Api()
    # weird error fix (initialize api another time)
    if len(custom_project) > 0:
        logging.info(
            "Please note: Using models from custom project, please ensure that you have access."
        )
        full_path = custom_project
        api.runs(path=full_path).objects
    elif self.project_name == "template_project":
        full_path = f"{self.team_name}/spyfish_aotearoa"

    else:
        full_path = f"{self.team_name}/{self.project_name}"

    runs = api.runs(full_path)

    if len(runs) > 100:
        runs = list(runs)[:100]

    for run in runs:
        model_artifacts = [
            artifact
            for artifact in itertools.chain(
                run.logged_artifacts(), run.used_artifacts()
            )
            if artifact.type == "model"
        ]
        if len(model_artifacts) > 0:
            model_dict[run.name] = model_artifacts[0].name.split(":")[0]
            model_info[model_artifacts[0].name.split(":")[0]] = run.summary
            data_info[model_artifacts[0].name.split(":")[0]] = run.config

    # Add "no movie" option to prevent conflicts
    # models = np.append(list(model_dict.keys()),"No model")

    model_widget = widgets.Dropdown(
        options=[(name, model) for name, model in model_dict.items()],
        description="Select model:",
        ensure_option=False,
        disabled=False,
        value=None,
        layout=widgets.Layout(width="50%"),
        style={"description_width": "initial"},
    )

    main_out = widgets.Output()
    display(model_widget, main_out)

    # Display model metrics
    def on_change(change):
        with main_out:
            clear_output()
            if change["new"] == "No file":
                logging.info("Choose another file")
            else:
                if self.project_name == "model-registry":
                    logging.info("No metrics available")
                else:
                    self.data_path = data_info[change["new"]]["data"]
                    logging.info(
                        {
                            k: v
                            for k, v in model_info[change["new"]].items()
                            if "metrics" in k
                        }
                    )

    model_widget.observe(on_change, names="value")
    return model_widget


def get_model(self, model_name: str, download_path: str, custom_project: str = ""):
    # weird error fix (initialize api another time)
    if len(custom_project) > 0:
        logging.info(
            "Please note: Using models from custom project, please ensure that you have access."
        )
        full_path = custom_project
    else:
        if self.team_name == "wildlife-ai":
            logging.info("Please note: Using models from adi-ohad-heb-uni account.")
            full_path = "adi-ohad-heb-uni/project-wildlife-ai"
        elif self.project_name == "template_project":
            full_path = f"{self.team_name}/spyfish_aotearoa"
        else:
            full_path = f"{self.team_name}/{self.project_name.lower()}"
    api = wandb.Api()
    try:
        api.artifact_type(type_name="model", project=full_path).collections()
    except Exception as e:
        logging.error(f"No model collections found. No artifacts have been logged. {e}")
        return None
    collections = [
        coll
        for coll in api.artifact_type(
            type_name="model", project=full_path
        ).collections()
    ]
    model = [i for i in collections if i.name == model_name]
    if len(model) > 0:
        model = model[0]
    else:
        logging.error("No model found")
    artifact = api.artifact(full_path + "/" + model.name + ":latest")
    logging.info("Downloading model checkpoint...")
    artifact_dir = artifact.download(root=download_path)
    logging.info("Checkpoint downloaded.")
    return str(Path(artifact_dir).resolve())


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
