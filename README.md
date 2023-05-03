# few-shot-classification-api
Create and use your own text classification model, starting with only a few examples.

Based on [SetFit](https://arxiv.org/abs/2209.11055) and [Transformers](https://huggingface.co/docs/transformers/index).
All models are hosted at [huggingface.co/rodekruis](https://huggingface.co/rodekruis).

The API is publicly accessible at [few-shot-classification-api.azurewebsites.net](https://few-shot-classification-api.azurewebsites.net/docs).

## Prerequisites

* A user-level or admin-level API key: see BitWarden > Servers or ask Jacopo.
* Some examples of what you want to classify.

N.B.: the examples must be formatted as a semicolon-separated string (or a [Python list](https://www.w3schools.com/python/python_lists.asp)) of **texts** and corresponding **labels**. For example:
* texts = "Cats make great pets; They hunt small rodents; The dog is a domesticated descendant of the wolf"
* labels = "cat; cat; dog;"

**Order matters**: in this example, the model will learn that the first text corresponds to the first label, 
the second text to the second label, etc. If you provide texts and labels in random order, your model will not work as intended.

**Tips and tricks**:
* Provide at least **10 examples per label**; the more examples you provide, the higher the accuracy of the model will be.
* If providing examples as semicolon-separated string: remove semicolons from your texts and labels, to ensure that they are correctly parsed

## Usage

1. Create and train a new model using ``/train``.
2. Wait about 0.2 minutes per example provided, where one example is one text-label pair. Your model will appear at [huggingface.co/rodekruis](https://huggingface.co/rodekruis) when ready.
3. Use your model to classify new text using ``/classify``
4. \[OPTIONAL\] Delete your model using ``/delete_model`` (requires admin-level API key)

See [the documentation](https://few-shot-classification-api.readthedocs.io/).

## Models

Sentence transformers ([all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)) fine-tuned using [SetFit](https://arxiv.org/abs/2209.11055).

## Setup

From project root, run locally with `pipenv run python main.py`.

Deploy with [Azure Web Apps](https://azure.microsoft.com/en-us/services/app-service/web/) to serve publicly, for example as explained [here](https://medium.com/nerd-for-tech/deploying-a-simple-fastapi-in-azure-79c59c430064).
