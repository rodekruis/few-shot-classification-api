import uvicorn
from typing import Union
from fastapi.responses import RedirectResponse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import requests
from setfit import SetFitModel, SetFitTrainer, sample_dataset
from huggingface_hub import login as huggingface_login
from huggingface_hub import HfApi, hf_hub_download, ModelFilter
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError
import os
import shutil
import re
from cleantext import clean
from dotenv import load_dotenv
from enum import Enum
load_dotenv()

# load environment variables
port = os.environ["PORT"]
organization = os.getenv('ORGANIZATION')
trainer_url = os.getenv('TRAINER_URL')
base_model = os.getenv('BASE_MODEL')
admin_key = str(os.getenv('ADMIN_KEY')).strip()
user_key = str(os.getenv('USER_KEY')).strip()
huggingface_login(token=os.getenv('HUGGINGFACE_TOKEN'))
huggingface_client = HfApi()


class Visibility(str, Enum):
    private = 'private'
    public = 'public'


class TrainPayload(BaseModel):
    key: Union[str, None]
    texts: Union[list, str]
    labels: Union[list, str]
    model_name: str
    model_visibility: Visibility


class ClassifyPayload(BaseModel):
    key: Union[str, None]
    texts: Union[list, str]
    model_name: str
    predict_proba: Union[bool, None] = False


class DeleteModelPayload(BaseModel):
    key: Union[str, None]
    model_name: str


tags_metadata = [
    {
        "name": "train",
        "description": "Create a new model and train it with some examples",
        "externalDocs": {
            "description": "More info",
            "url": "https://github.com/rodekruis/few-shot-classification-api#prerequisites",
        },
    },
    {
        "name": "classify",
        "description": "Use an existing model to classify some text(s)",
    },
]
description = """
Create and use your own text classification model, starting with only a few examples.

Based on [SetFit](https://arxiv.org/abs/2209.11055) and [Transformers](https://huggingface.co/docs/transformers/index).
All models are hosted at [huggingface.co/rodekruis](https://huggingface.co/rodekruis).

Built with love by [NLRC 510](https://www.510.global/). See [the project on GitHub](https://github.com/rodekruis/few-shot-classification-api).
"""

# initialize FastAPI
app = FastAPI(
    title="few-shot-classification-api",
    description=description,
    version="0.0.1",
    license_info={
        "name": "AGPL-3.0 license",
        "url": "https://www.gnu.org/licenses/agpl-3.0.en.html",
    },
    openapi_tags=tags_metadata
)


@app.get("/")
async def docs_redirect():
    return RedirectResponse(url='/docs')


@app.post("/train", tags=["train"])
async def train_model(payload: TrainPayload):
    key = str(payload.key).strip()
    if key != admin_key and key != user_key:
        raise HTTPException(status_code=401, detail="unauthorized")
    output = {
        "model_name": payload.model_name,
        "model_url": f"https://huggingface.co/{organization}/{payload.model_name}"
    }

    if type(payload.texts) == str:
        texts = payload.texts.split(";")
    elif type(payload.texts) == list:
        texts = payload.texts
    else:
        raise HTTPException(status_code=400, detail="texts must be a list or a string with semicolon-separated items")
    texts = [re.sub(r'(!|\$|#|&|\"|\'|\(|\)|\||<|>|`|\\\|;)', "", t).strip() for t in texts]  # cleaning
    texts = [clean(t, lower=False, no_line_breaks=True, no_emoji=True) for t in texts]
    texts = [t[:500] for t in texts]  # 06-06-2023 training stops without error if text too long, to be investigated

    if type(payload.labels) == str:
        labels = payload.labels.split(";")
    elif type(payload.labels) == list:
        labels = payload.labels
    else:
        raise HTTPException(status_code=400, detail="labels must be a list or a string with semicolon-separated items")
    labels = [re.sub(r'(!|\$|#|&|\"|\'|\(|\)|\||<|>|`|\\\|;)', "", t).strip() for t in labels]  # cleaning
    labels = [clean(t, lower=False, no_line_breaks=True, no_emoji=True) for t in labels]
    labels = [t[:500] for t in labels]

    if len(texts) != len(labels):
        raise HTTPException(status_code=400, detail=f"number of texts and labels must be the same"
                                                    f" (received {len(texts)} texts and {len(labels)} labels)")

    texts = ";".join(texts)
    labels = ";".join(labels)
    payload = {
        'texts': texts,
        'labels': labels,
        'model_visibility': payload.model_visibility,
        'model_name': payload.model_name
    }
    response = requests.post(trainer_url, json=payload)
    if response.status_code == 202:
        return output
    else:
        raise HTTPException(status_code=response.status_code, detail="training failed, check the logs")


@app.post("/classify", tags=["classify"])
async def classify_text(payload: ClassifyPayload):
    key = str(payload.key).strip()
    if key != admin_key and key != user_key:
        raise HTTPException(status_code=401, detail="unauthorized")
    output = {"model_name": payload.model_name}
    model_path = os.path.join(organization, payload.model_name).replace('\\', '/')

    if type(payload.texts) == str:
        texts = payload.texts.split(";")
    elif type(payload.texts) == list:
        texts = payload.texts
    else:
        raise HTTPException(status_code=400, detail="texts must be a list or a string with semicolon-separated items")

    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    try:
        # load model and run inference
        model = SetFitModel.from_pretrained(model_path)
        predictions = model(texts).numpy()
        if payload.predict_proba:
            predict_probabilities = model.predict_proba(texts).numpy()
            max_probabilities = [max(predict_probabilities[ix]) for ix in range(predict_probabilities.shape[0])]

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
            if payload.predict_proba:
                for text, prediction, proba in zip(texts, predictions, max_probabilities):
                    output["predictions"].append({"text": text, "label": label_dict[str(prediction)], "probability": proba})
            else:
                for text, prediction in zip(texts, predictions):
                    output["predictions"].append({"text": text, "label": label_dict[str(prediction)]})
    except RepositoryNotFoundError:
        raise HTTPException(status_code=404, detail=f"model {payload.model_name} not found.")
    return output


@app.get("/list_models")
async def list_models():
    models = huggingface_client.list_models(
        filter=ModelFilter(
            task="text-classification",
            author=organization
        )
    )
    models_list = []
    for model in models:
        models_list.append({
            'modelId': model.modelId.replace(f"{organization}/", ""),
            'lastModified': model.lastModified,
            'tags': model.tags
        })
    return {"models": models_list}


@app.post("/delete_model")
async def delete_model(payload: DeleteModelPayload):
    key = str(payload.key).strip()
    if key != admin_key:
        raise HTTPException(status_code=401, detail="unauthorized")
    huggingface_client.delete_repo(
        repo_id=f"{organization}/{payload.model_name}"
    )
    return {"model deleted": payload.model_name}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(port), reload=True)
