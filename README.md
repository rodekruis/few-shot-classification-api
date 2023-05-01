# few-shot-classification-api
Create and use your own text classification model, starting with only a few examples.

Based on [SetFit](https://arxiv.org/abs/2209.11055) and [Transformers](https://huggingface.co/docs/transformers/index).
All models are hosted at [huggingface.co/rodekruis](https://huggingface.co/rodekruis).

The API is publicly accessible at [few-shot-classification-api.azurewebsites.net](https://few-shot-classification-api.azurewebsites.net/docs).

## Description

Synopsis: a [dockerized](https://www.docker.com/) [python](https://www.python.org/) API that removes personally identifiable information (PII) from text.

Workflow: 

1. Create a new model and train it with N examples (N >= 10)
2. wait M = 0.2 * N minutes, your model will appear [here](https://huggingface.co/rodekruis) when ready
3. Classify a given piece of text with your model

## API Usage

See [the documentation](https://few-shot-classification-api.readthedocs.io/).

## Models

Sentence transformers ([all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)) fine-tuned using [SetFit](https://arxiv.org/abs/2209.11055).

## Setup

From project root, run locally with `pipenv run python main.py`.

Deploy with [Azure Web Apps](https://azure.microsoft.com/en-us/services/app-service/web/) to serve publicly, for example as explained [here](https://medium.com/nerd-for-tech/deploying-a-simple-fastapi-in-azure-79c59c430064).
