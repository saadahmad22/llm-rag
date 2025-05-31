# A tutorial on how to use RAG on LLMs 

## Table of contents
0. [Introduction](#introduction)
1. [System Setup](#setup)
1. [Setting up an LLM](#llm)
2. [Setting up a Database](#db)
3. [Setting up a RAG app](#rag)
4. [Conclusion](#end)

### Introduction <a name="introduction"></a>
First of all, what is Retrieval-Augmented Generation, a.k.a RAG? Simply put, RAG is a technique that uses data (in the form of text) to improve results from an LLM model. What this might look like, for example, might be a training chatbot for new hires which uses RAG to sort of "feed" training data (e.g., employee manuals, training video transcripts, etc.) to an LLM. This drastically improves the chatbot's performance, giving it a huge edge over a "base" LLM, which is just using general-purpose information and can take a lot of liberties in interpreting questions and possible answers, since it has not been "trained/taught" on that specific company's training procedure. This is the motivation behind RAG, and below will be a discussion on how to get started. The tutorial will go over doing it locally on a single machine, but bear in mind that scaling it is as simple as deploying the LLM and the RAG app on separate containers.

### System Setup <a name="setup"></a>
1. The first order of business is to install [Python](https://www.python.org/downloads/) and [pip](https://pip.pypa.io/en/stable/installation/) for package management.
2. Then, launch a [virtual environment](https://docs.python.org/3/library/venv.html) (to not interfere with global packages).
3. Now,  install the required packages by using pip (or pip3 if you installed that instead): `pip install -r requirements.txt`

### Setting up an LLM <a name="llm"></a>
As with anything, there are many different LLMs to choose from. The key is to have it compatible (or integrate it) with an established library (Langchain in this case). [Langchain has a ton of different integrations for LLMs](https://python.langchain.com/docs/integrations/chat/), and for the sake of this tutorial, we will use [Llama CPP](https://python.langchain.com/docs/integrations/llms/llamacpp/). The reasoning behind this choice is that this specific library containerizes very easily, has support for many features, and most importantly, is bare-bones enough that it reduces a lot of overhead while allowing one to run a model locally.

The code for the LLM is located inside [`src`](src/models). The way the project is configured, you should put the Llama model into [`model`](model) while following the instructions listed there, and then [update the configurations](src/config). 

Here is a step-by-step explanation of [`llama.py`](src/models/llama.py):
1. We create a wrapper class to interact with the LLM, `LlamaModel`
2. When we construct an instance of the class, it takes the `model_path` and constructs the model. Currently, it is only tuning two hyperparameters, but that can/should be changed as needed for any use case, as [detailed by the library](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#high-level-api:~:text=for%20llama.cpp.-,llama_cpp.Llama,-High%2Dlevel%20Python).
3. This specific tutorial uses [Chat Histories](https://python.langchain.com/api_reference/core/chat_history.html), and provides an unauthenticated method (`get_user_history`) to access those from a locally stored dictionary. On a production-scale application, this might not make sense. The storage location can differ as needed, and authentication should be considered as needed, since it is open access right now, being a tutorial.
4. Queries are made using `generate_response`, which takes in a string (the assumption is that the RAG information has already been included) and Chat History, and runs it through the model before returning the results. Note that **minimal** preprocessing (`preprocess_input`) and postprocessing (`postprocess_output`) have been done on the data. In production, potential escape sequences and injection attacks should be handled and accounted for (other libraries such as [`rebuff`](https://pypi.org/project/rebuff/) exist for this). 

### Setting up a Database <a name="db"></a>
As the name suggests, Retrieval-Augmented Generation requires, well, retrieval. It is beyond the scope of this tutorial, but there are many different ways to store the documents that you will retrieve. It can range from retrieving them from local storage, storage in the cloud, web crawling, and so much more. However, by far the best method as of now is to maintain a database where some representation of the document will live. This can be anything from a binary data Blob storage to storing a textual representation of the document (for non-text documents, techniques such as image/video captioning, OCR, etc., may be necessary). However, by far the best method is to store them as vectors of ["embeddings"](https://python.langchain.com/docs/concepts/embedding_models/). Embeddings, on a use-case level, are a list of numbers (generally floating point) that represent the "meaning" behind a document, whether it be text, image, or anything else (a [more detailed explanation](https://huggingface.co/learn/cookbook/en/faiss_with_hf_datasets_and_clip) can be found here, which goes into how this might look under the hood of LangChain). 

The reason embeddings are so useful is that they allow for quick comparisons of text/documents, such that the most relevant documents can be passed to the LLM when conducting RAG. On an aside, this also makes them useful for information retrieval. LangChain currently has about 80 different embedding models, of which we will use HuggingFace, which is more diverse than enough on its own. 

Although it is not ideal, this tutorial creates a database (FAISS, though there are [many other options](https://python.langchain.com/docs/integrations/vectorstores/)) on initialization. Ideally, it behaves like a normal database and does not need to be recomputed frequently. 

Here is a step-by-step explanation of [`vector_store.py`](src/retrievers/vector_store.py):
1. We create a wrapper class to interact with the LLM, `VectorStore`
2. When we construct an instance of the class, it takes some text-only documents ([LangChain supports other types as well](https://python.langchain.com/docs/integrations/document_loaders/), but this tutorial will use text only for simplicity) and create a FAISS database from them. There are unused parameters relating to chunks, and those are optionally used by the reader in `create_from_documents` when they get passed to the [`TextSplitter`](src/utils/text_splitter.py). The idea is that smaller documents contain more targeted information, much of which is far more useful to the LLM than one big, confusing document.
3. There is a retrieve function that allows retrieving documents from the FAISS database
NOTE: An in-depth look at the [FAISS LangChain documentation](https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.faiss.FAISS.html#langchain_community.vectorstores.faiss.FAISS) is quite beneficial for understanding more deeply what is going on.

### Setting up a RAG app <a name="rag"></a>
All the building blocks are in place. All that remains is to construct an API with them for the user to be able to interface with. This tutorial approaches that task by constructing a simple [Flask server](src/app.py) with the route logic (as per convention) in the [`routes` folder ](src/routes). [`init_system.py`](src/routes/init_system.py) initializes the tools discussed above in addition to another tool, [`RAGChain`](src/chains/rag_chain.py), which just serves as a bridge between the RAG and the normal LLM; it adds "retrieval" to normal chat queries in a fairly simple manner. This can/should be fine-tuned using targeted prompting techniques based on the task at hand. This means figuring out what specific text command (e.g., "You are the world's best engineer...`query`....here are some things you know...`documents`"), rather than just the simple concatenation as of now, gives the best results for the specific use case.

### Conclusion <a name="end"></a>
I hope this tutorial was helpful. If something is not working or you have further questions, feel free to contact me or open up an issue thread on GitHub.
