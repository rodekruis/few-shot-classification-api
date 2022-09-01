import uvicorn
from typing import Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline, AutoModelForCausalLM
from transformers import GPT2Tokenizer, OPTForCausalLM
import os
from enum import Enum
from dotenv import load_dotenv
load_dotenv()

# Initialize model
modelname = "facebook/opt-13b"
tokenizer = GPT2Tokenizer.from_pretrained(modelname)
model = OPTForCausalLM.from_pretrained(modelname)


class ModelName(str, Enum):
    modelname = modelname

    @classmethod
    def exists(cls, key):
        return key in cls.__members__


class GeneratePayload(BaseModel):
    text: str
    model: str = ModelName.modelname


# load environment variables
port = os.environ["PORT"]

# initialize FastAPI
app = FastAPI(
    title="language model",
    description="Generate text from a prompt. \n"
                "Built with love by [NLRC 510](https://www.510.global/).",
    version="0.0.1",
    license_info={
        "name": "AGPL-3.0 license",
        "url": "https://www.gnu.org/licenses/agpl-3.0.en.html",
    },
)


@app.get("/")
def index():
    return {"data": "Welcome to language-model-app!"}


@app.post("/generate/")
async def generate_text(payload: GeneratePayload):
    text = ""
    try:
        inputs = tokenizer(payload.text, return_tensors="pt")
        print(inputs)
        generate_ids = model.generate(inputs.input_ids, max_length=30)
        print(generate_ids)
        output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(output)
        if len(output) > 0:
            text = output[0].replace(payload.text, "")
    except:
        pass
    return {"generated_text": text}


@app.get("/model")
async def get_model():
    return {"model_name": modelname, "source": f"https://huggingface.co/{modelname}"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)