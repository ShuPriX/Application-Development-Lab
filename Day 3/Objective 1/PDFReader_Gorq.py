import nest_asyncio
from llama_index.core import SimpleDirectoryReader
nest_asyncio.apply()

# from llama_parse import LlamaParse

# parser = LlamaParse(
#     api_key="llx-hG7xdLPbx3NFilFYJgqhwB72WOLs7ile4AswQd7GvVUfgQxN",  # can also be set in your env as LLAMA_CLOUD_API_KEY
#     result_type="text",  # "markdown" and "text" are available
#     num_workers=4,  # if multiple files passed, split in `num_workers` API calls
#     verbose=True,
#     language="en",  # Optionally you can define a language, default=en
# )

# # sync
# document = parser.load_data("./sample.pdf")
document = SimpleDirectoryReader(input_files=["pdf1.pdf"]).load_data()

# ---- cell break ----

from llama_index.core import VectorStoreIndex,  Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.groq import Groq
from llama_index.llms.ollama import Ollama

# ---- cell break ----

embed_model = HuggingFaceEmbedding(model_name = "BAAI/bge-small-en-v1.5")
# embed_model = OllamaEmbedding(model_name='llama3.2')
llm = Ollama(model="gemma3:4b")

# ---- cell break ----

llm = Groq(model="llama-3.3-70b-versatile",api_key = "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")
#llm = Groq(model="llama-3.1-8b-instant",api_key = "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")

# ---- cell break ----

index = VectorStoreIndex.from_documents(document,embed_model = embed_model)

# ---- cell break ----

query_engine = index.as_query_engine(llm = llm)
response = query_engine.query("Write a section on An Overview of Spiking Neural Networks. First begin by describing in brief about the origins of the spiking neural networks based on biological phenomenon")
file = open('review_SNN.docx','a')
file.write(response.response)
file.close()

# ---- cell break ----

response = query_engine.query("Next add a section on the different encoding schemes for SNNs")
file = open('review_SNN.docx','a')
file.write(response.response)
file.close()

# ---- cell break ----

response = query_engine.query("Next add a section on the different SNN models. Also write the relevant equations in word document format")
file = open('review_SNN.docx','a')
file.write(response.response)
file.close()

# ---- cell break ----

response = query_engine.query("Next add a section on the different learning rules")
file = open('review_SNN.docx','a')
file.write(response.response)
file.close()

# ---- cell break ----

response = query_engine.query("Next add a table showing whcih encoding scheme is used for which application")
file = open('review_SNN.docx','a')
file.write(response.response)
file.close()

# ---- cell break ----

response = query_engine.query("Next describe in detail the different SNN models")
file = open('review_SNN.docx','a')
file.write(response.response)
file.close()

# ---- cell break ----

response = query_engine.query("Next describe in detail different learning rules")
file = open('review_SNN.docx','a')
file.write(response.response)
file.close()

# ---- cell break ----

response = query_engine.query("Next describe in detail the different SNN models based on the brain")
file = open('review_SNN.docx','a')
file.write(response.response)
file.close()

# ---- cell break ----

document1 = SimpleDirectoryReader(input_files=["pdf1.pdf"]).load_data()

# ---- cell break ----

index1 = VectorStoreIndex.from_documents(document1,embed_model = embed_model)

# ---- cell break ----

llm1 = Groq(model="llama-3.3-70b-versatile",api_key = "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")

# ---- cell break ----

query_engine = index1.as_query_engine(llm = llm1)
response = query_engine.query("Keep your temperature 0.7. Write a detailed section of atleast 500 words on different encoding schemes used by SNNs. Expalin all the encoding schemes and their types")
file = open('review_SNN.docx','a')
file.write(response.response)
file.close()
