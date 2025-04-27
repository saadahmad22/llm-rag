# A tutorial on how to use RAG on LLMs 

## Table of contents
1. [Introduction](#introduction)
2. [Setting up an LLM](#llm)
3. [Setting up a Database](#db)
4. [Setting up a RAG app](#rag)

### Introduction <a name="introduction"></a>
First of all, what is RAG? Simply put, RAG is a technique that uses data (in the form of text) to improve results from an LLM model. What this might look like, for example, might be a training chatbot for new hires which uses RAG to sort of "feed" training data (e.g., employee manuals, training video transcripts, etc.) to an LLM. This drastically improves the chatbot's performance, giving it a huge edge over a "base" LLM, which is just using general-purpose information and can take a lot of liberties in interpreting questions and possible answers, since it has not been "trained/taught" on that specific company's training procedure. This is the motivation behind RAG, and below will be a discussion on how to get started. The tutorial will go over doing it locally on a single machine, but bear in mind that scaling it is as simple as deploying the LLM and the RAG app on separate containers.

1. The first order of business is to install [Python](https://www.python.org/downloads/) and [pip](https://pip.pypa.io/en/stable/installation/) for package management.
2. Then, launch a [virtual environment](https://docs.python.org/3/library/venv.html) (to not interfere with global packages).
3. Now,  install the required packages by using pip (or pip3 if you installed that instead): `pip install -r requirements.txt`

### Setting up an LLM <a name="llm"></a>
As with anything, there are many different LLMs to choose from. The key is to have it compatible (or integrate it) with an established library (Langchain in this case). [Langchain has a ton of different integrations for LLMs](https://python.langchain.com/docs/integrations/chat/), and for the sake of this tutorial, we will use [Llama CPP](https://python.langchain.com/docs/integrations/llms/llamacpp/). The reasoning behind this choice is that this specific library containerizes very easily, has support for many features, and most importantly, is bare-bones enough that it reduces a lot of overhead while allowing one to run a model locally.

The code for the LLM is located inside [`src`](src/models). The way the project is configured, you should put the Llama model into [`model`](model) while following the instructions listed there, and then [update the configurations](src/config). 

Here is a step-by-step explanation of [`llama.py`](src/models/llama.py):
1. We create a wrapper class to interact with the LLM, `LlamaModel`
2. When we construct an instance of the class, it takes the `model_path` and constructs the model. Currently, it is only tuning two hyperparameters, but that can/should be changed as needed for any use case, as [detailed by the library](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#high-level-api:~:text=for%20llama.cpp.-,llama_cpp.Llama,-High%2Dlevel%20Python).
3. This specific tutorial uses [Chat Histories](https://python.langchain.com/api_reference/core/chat_history.html), and provides an unauthenticated method (`get_user_history`) to access those from a locally stored dictionary. On a production-scale application, this might not make sense. The storage location can differ as needed, and authentication should be considered as needed since it is open access right now, being a tutorial.
4. Queries are made using `generate_response`, which takes in a string (the assumption is that the RAG information has already been included) and Chat History, and runs it through the model before returning the results. Note that **minimal** preprocessing (`preprocess_input`) and postprocessing (`postprocess_output`) have been done on the data. In production, potential escape sequences and injection attacks should be handled and accounted for (other libraries such as [`rebuff`](https://pypi.org/project/rebuff/) exist for this). 

### Setting up a Database <a name="db"></a>

### Setting up a RAG app <a name="rag"></a>
