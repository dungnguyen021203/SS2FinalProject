from langchain_community.document_loaders import TextLoader
import textwrap
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import HuggingFaceHub
from langchain_community.llms import HuggingFaceEndpoint
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

loader = TextLoader("data.txt")
document = loader.load()


# This is a testing environment


# Preprocessing
def wrap_text_preserve_newlines(text, width=110):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text


# Text Splitting (Chunks)
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(document)

# Embedding
embeddings = HuggingFaceEmbeddings()
db = FAISS.from_documents(docs, embeddings)


# Q-A
llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.8, "max_length": 512})
chain = load_qa_chain(llm, chain_type="stuff")

queryText = "What is HCI"
docsResult = db.similarity_search(queryText)
answer = chain.run(input_documents=docsResult, question=queryText)
print(answer)
