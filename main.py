import uvicorn
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
import json
import shutil
from datasets import Dataset, load_dataset
import numpy as np
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer, sample_dataset
import os
from enum import Enum
from dotenv import load_dotenv
import traceback

load_dotenv()

# Initialize model
modelname_ = os.environ["MODEL_ORG"] + "/" + os.environ["MODEL_NAME"]


class ModelName(str, Enum):
    modelname = modelname_

    @classmethod
    def exists(cls, key):
        return key in cls.__members__


class TrainPayload(BaseModel):
    texts: list
    labels: list
    save_as: str
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
                "Built with love by [NLRC 510](https://www.510.global/). See [the project on GitHub](https://github.com/rodekruis/text-generation-app).",
    version="0.0.1",
    license_info={
        "name": "AGPL-3.0 license",
        "url": "https://www.gnu.org/licenses/agpl-3.0.en.html",
    },
)


@app.get("/")
def index():
    return {"data": "Welcome to few-shot-classification-api!"}


@app.post("/train/")
async def train_model(payload: TrainPayload):
    output = {}
    try:
        # Define training and evaluation set
        texts = np.asarray(payload.texts)
        unique_labels = list(set(payload.labels))
        unique_labels.sort()
        labels = np.asarray([unique_labels.index(label) for label in payload.labels])
        train_dataset = Dataset.from_dict({'text': texts, 'label': labels})

        if payload.texts_eval:
            texts_eval = np.asarray(payload.texts_eval)
            unique_labels_eval = list(set(payload.labels_eval))
            unique_labels_eval.sort()
            if unique_labels_eval != unique_labels:
                return {"error": "'labels_eval' does not contain the same unique values as 'labels'"}
            labels_eval = np.asarray([unique_labels.index(label) for label in payload.labels_eval])
            eval_dataset = Dataset.from_dict({'text': texts_eval, 'label': labels_eval})
        else:
            eval_dataset = train_dataset

        # Create model and train
        model = SetFitModel.from_pretrained(modelname_)
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

        # if model already exists, delete old version
        os.makedirs("./models", exist_ok=True)
        if os.path.exists(f"./models/{payload.save_as}"):
            shutil.rmtree(f"./models/{payload.save_as}")

        # save model
        trainer.model._save_pretrained(f"./models/{payload.save_as}")
        label_dict = {v: k for v, k in enumerate(unique_labels)}
        with open(f"./models/{payload.save_as}/label_dict.json", "w") as outfile:
            json.dump(label_dict, outfile)
        output['model_saved_as'] = payload.save_as

    except Exception as e:
        output = {"error": traceback.format_exc()}
    return output


@app.post("/classify/")
async def classify_text(payload: ClassifyPayload):
    output = {}
    # check if model exists
    if not os.path.exists(f"./models/{payload.model_name}"):
        return {"error": f"model {payload.model_name} not found."}
    try:
        # load model and run inference
        model = SetFitModel.from_pretrained(f"./models/{payload.model_name}")
        texts_pred = list(payload.texts)
        predictions = model(texts_pred)

        # map to label names
        with open(f"./models/{payload.model_name}/label_dict.json", 'r') as openfile:
            label_dict = json.load(openfile)
        output["predictions"] = []
        for text, prediction in zip(texts_pred, predictions):
            output["predictions"].append({"text": text, "class": label_dict[str(prediction)]})
    except Exception as e:
        output = {"error": traceback.format_exc()}
    return output


@app.get("/model")
async def get_model():
    return {"model_name": modelname_, "source": f"https://huggingface.co/{modelname_}"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
