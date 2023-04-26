import uvicorn
from typing import Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
from datasets import Dataset, load_dataset
import numpy as np
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer, sample_dataset
from huggingface_hub import login as huggingface_login
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError
import os
import shutil
from dotenv import load_dotenv
from enum import Enum
load_dotenv()
organization = os.getenv('ORGANIZATION')
base_model = os.getenv('BASE_MODEL')
huggingface_login(token=os.getenv('HUGGINGFACE_TOKEN'))
huggingface_client = HfApi()


class Visibility(str, Enum):
    private = 'private'
    public = 'public'


class TrainPayload(BaseModel):
    texts: list
    labels: list
    model_name: str
    model_visibility: Union[Visibility, None] = Visibility.public
    retrain_model: Union[bool, None] = False
    texts_eval: Union[list, None] = None
    labels_eval: Union[list, None] = None
    multi_label: Union[bool, None] = True


class ClassifyPayload(BaseModel):
    texts: list
    model_name: str
    multi_label: Union[bool, None] = True


# load environment variables
port = os.environ["PORT"]

# initialize FastAPI
app = FastAPI(
    title="few-shot-classification-api",
    description="Few-shot classification with SetFit. \n"
                "Built with love by [NLRC 510](https://www.510.global/). "
                "See [the project on GitHub](https://github.com/rodekruis/text-generation-app).",
    version="0.0.1",
    license_info={
        "name": "AGPL-3.0 license",
        "url": "https://www.gnu.org/licenses/agpl-3.0.en.html",
    },
)


@app.get("/")
def index():
    return {"data": "Welcome to few-shot-classification-api!"}


@app.post("/train")
async def train_model(payload: TrainPayload):
    output = {"model": payload.model_name}
    model_path = os.path.join(organization, payload.model_name).replace('\\', '/')

    # Define training and evaluation set
    if len(payload.texts) != len(payload.labels):
        raise HTTPException(status_code=400, detail="number of texts and labels must be the same")
    texts = np.asarray(payload.texts)
    unique_labels = list(set(payload.labels))
    unique_labels.sort()
    labels = np.asarray([unique_labels.index(label) for label in payload.labels])
    train_dataset = Dataset.from_dict({'text': texts, 'label': labels})

    if payload.texts_eval:
        if len(payload.texts_eval) != len(payload.labels_eval):
            raise HTTPException(status_code=400, detail="number of texts_eval and labels_eval must be the same")
        texts_eval = np.asarray(payload.texts_eval)
        unique_labels_eval = list(set(payload.labels_eval))
        unique_labels_eval.sort()
        if unique_labels_eval != unique_labels:
            raise HTTPException(status_code=400, detail="labels_eval does not contain the same unique values as labels")
        labels_eval = np.asarray([unique_labels.index(label) for label in payload.labels_eval])
        eval_dataset = Dataset.from_dict({'text': texts_eval, 'label': labels_eval})
    else:
        eval_dataset = train_dataset

    # Create model and train
    if not payload.retrain_model:
        model = SetFitModel.from_pretrained(base_model)
    else:
        try:
            model = SetFitModel.from_pretrained(model_path)
        except RepositoryNotFoundError:
            raise HTTPException(status_code=404, detail=f"model {payload.model_name} not found.")
    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss_class=CosineSimilarityLoss,
        metric="accuracy",
        batch_size=16,
        num_iterations=20,  # The number of text pairs to generate for contrastive learning
    )
    trainer.train()

    if payload.texts_eval:
        metrics = trainer.evaluate()
        output['evaluation'] = metrics

    # save model
    is_private = payload.model_visibility == Visibility.private
    try:
        huggingface_client.create_repo(repo_id=model_path, private=is_private)
    except HfHubHTTPError:
        pass
    trainer.push_to_hub(model_path)

    # save label dict
    label_dict = {v: k for v, k in enumerate(unique_labels)}
    os.makedirs(model_path, exist_ok=True)
    with open(f"{model_path}/label_dict.json", "w") as outfile:
        json.dump(label_dict, outfile)
    huggingface_client.upload_file(
        path_or_fileobj=f"{model_path}/label_dict.json",
        path_in_repo="label_dict.json",
        repo_id=model_path,
        repo_type="model",
    )

    return output


@app.post("/classify")
async def classify_text(payload: ClassifyPayload):
    output = {"model": payload.model_name}
    organization = os.getenv('ORGANIZATION')
    model_path = os.path.join(organization, payload.model_name).replace('\\', '/')
    shutil.rmtree(model_path)
    try:
        # load model and run inference
        model = SetFitModel.from_pretrained(model_path)
        texts_pred = list(payload.texts)
        predictions = model(texts_pred).numpy()
        # download label names and convert predictions
        os.makedirs(model_path, exist_ok=True)
        hf_hub_download(
            repo_id=model_path,
            filename="label_dict.json",
            local_dir=model_path,
            local_dir_use_symlinks="auto"
        )
        with open(os.path.join(model_path, "label_dict.json")) as infile:
            label_dict = json.load(infile)
            output["predictions"] = []
            for text, prediction in zip(texts_pred, predictions):
                output["predictions"].append({"text": text, "label": label_dict[str(prediction)]})
    except RepositoryNotFoundError:
        raise HTTPException(status_code=404, detail=f"model {payload.model_name} not found.")
    return output


# @app.get("/models")
# async def get_models():
#     container = os.getenv('CONTAINER')
#     organization = os.getenv('ORGANIZATION')
#     models = list_blobs(container, organization)
#     models = [model.replace(f"{directory}/", "") for model in models]
#     return {"models": models}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
