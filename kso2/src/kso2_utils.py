from __future__ import annotations
from pathlib import Path
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from ultralytics import YOLO





@dataclass
class Project:
    Project_name: str
    project_path: str | Path | None = None
    data: Optional[Dict[str, Any]] = None
    tracking: Optional[Dict[str, Any]] = None
    weights_path: str = None
    model_name:str=None
    metadata:str=None





def create_project(
    Project_name: str,
    project_path: str | Path | None = None,
    *,
    data: Optional[Dict[str, Any]] = None,
    tracking: Optional[Dict[str, Any]] = None,
    weights_path: str = None,
    model_name:str=None,
    metadata:str=None,
    overwrite: bool = False,
) -> Path:
    """Create a YAML file describing a KSO project.
    
    """
    # user mistakes.
    if not Project_name or not isinstance(Project_name, str):
        raise ValueError("'Project_name' must be a non-empty string.")

    if project_path and not isinstance(project_path, (str, Path)):
        raise ValueError("'Project_path' must be a non-empty string.")        

    if project_path is None:
        sanitized = "".join(c.lower() if c.isalnum() else "_" for c in Project_name).strip("_")
        project_path = Path.cwd().parent / "projects"/ sanitized

    # Ensure the target directory exists.
    project_path = Path(project_path).expanduser().resolve()
    project_path.mkdir(parents=True, exist_ok=True)
    
    yaml_path = project_path / f"{sanitized}.project.yaml"
    if yaml_path.exists() and not overwrite:
        raise FileExistsError(
            f"{yaml_path} already exists.  Pass overwrite=True to replace it."
        )
    # Assemble the YAML structure.
    yaml_dict: Dict[str, Any] = {
        "Project_name": Project_name,
        "Project_path": str(project_path),
        "Data": {"path":str(data)},
        "model": {"weights_path":weights_path,"model_name":model_name},
        "tracking": tracking,
        "metadata": metadata,
    }
    print(yaml_dict)
   # print(yaml_path)
    with yaml_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(yaml_dict, fh, sort_keys=False, default_flow_style=False)

    logging.info(f"Project YAML created at {yaml_path}")
    return yaml_path



def setup_data(project_path: str | Path, data: str | Path) -> Path:


    if not project_path or not isinstance(project_path,(str,Path)):
        raise ValueError("'Project_path' must be a non-empty string.")
    if not data or not isinstance(data,(str,Path)):
        raise ValueError("'Project_path' must be a non-empty string.")
    data_path=Path(data).expanduser().resolve()
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"The data_path {data_path} does not exist."
        )
    else:
        base_dir =Path(__file__).resolve().parent.parent
        project=base_dir / "projects" / project_path
        yaml_path = project / f"{project_path}.project.yaml"
        if not yaml_path.exists():
            raise FileExistsError(
                f"{yaml_path} does not exist."
            )

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
        
        data["Data"]["path"]=str(data_path)
        with open(yaml_path, "w", encoding="utf-8") as d:
            yaml.safe_dump(data, d, sort_keys=False, default_flow_style=False)

        logging.info(f"Project YAML data path updated at {yaml_path}")
    return yaml_path









def setup_model(project_path: str | Path, weights_path: str | Path=None, model_name:str=None) -> Path:


    if not project_path or not isinstance(project_path,(str,Path)):
        raise ValueError("'Project_path' must be a non-empty string.")
        
    base_dir =Path(__file__).resolve().parent.parent
    project=base_dir / "projects" / project_path
    
    model_path=Path(weights_path).expanduser().resolve()

    yaml_path = project / f"{project_path}.project.yaml"
    if not yaml_path.exists():
        raise FileExistsError(
            f"{yaml_path} does not exist."
        )

    if  model_path:
        if not model_path.exists():
           raise FileNotFoundError(
               f"The weights_path {model_path} does not exist."
           )
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
    
        data["model"]["weights_path"] =str(model_path)
    
        with open(yaml_path, "w", encoding="utf-8") as d:
            yaml.safe_dump(data, d, sort_keys=False, default_flow_style=False)
    
        logging.info(f"Project YAML model weights path updated at {yaml_path}")
    
    if model_name:
        if not isinstance(model_name,str):
            raise ValueError("'model_name' must be a non-empty string.")

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
        
        data["model"]["model_name"] =model_name
        
        with open(yaml_path, "w", encoding="utf-8") as d:
            yaml.safe_dump(data, d, sort_keys=False, default_flow_style=False)
        
        logging.info(f"Project YAML model name updated at {yaml_path}")
    return yaml_path





















def load_project(project_name: str | Path):



    if not project_name or not isinstance(project_name,(str,Path)):
            raise ValueError("'project_name' must be a non-empty string.")

    base_dir =Path(__file__).resolve().parent.parent
    project=base_dir / "projects" / project_name
    
    yaml_path = project / f"{project_name}.project.yaml"
    
    with open(yaml_path, mode="r", newline="", encoding="utf-8") as file:

        reader = yaml.load(file,Loader=yaml.SafeLoader)
        logging.info(f"{project_name} loaded successfully")
        
        # Convert yaml into a project instance
        project = Project(
            Project_name=reader["Project_name"],
            project_path=reader["Project_path"],
            data=reader["Data"]["path"],
            tracking=reader["tracking"],
            weights_path=reader["model"]["weights_path"],
            model_name=reader["model"]["model_name"],
            metadata=reader["metadata"],
        )
    return project





def training_model(project_name:str, epochs:int=100, imgsz:int=640):

    project=load_project(project_name=project_name)
    if not isinstance(project.data,str):
            raise ValueError("'model' must be a non-empty string.")

    if project.weights_path :
        
        model = YOLO(project.weights_path)
        results = model.train(
            data=project.data,
            epochs=epochs,
            imgsz=imgsz
        )
        
    elif project.model_name:
        
        model = YOLO(project.model_name)
        results = model.train(
            data=project.data,
            epochs=epochs,
            imgsz=imgsz
        )
        
    else:
        raise ValueError("'model' must be a non-empty string.")
    return results







