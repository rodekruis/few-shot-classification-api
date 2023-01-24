# python base image in the container from Docker Hub
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

#set up environment
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install unzip
RUN apt-get -y install python3
RUN apt-get -y install python3-pip

# copy files to the /app folder in the container
COPY ./main.py /app/main.py
COPY ./Pipfile /app/Pipfile

# set the working directory in the container to be /app
WORKDIR /app

# install the packages from the Pipfile in the container
RUN pip install pipenv
RUN pipenv install --system --deploy --ignore-pipfile

# expose the port that uvicorn will run the app on
ENV PORT=8000
EXPOSE 8000

# define the model that will be loaded
ENV MODEL_ORG="sentence-transformers"
ENV MODEL_NAME="all-MiniLM-L6-v2"

# execute the command python main.py (in the WORKDIR) to start the app
CMD ["python3", "main.py"]