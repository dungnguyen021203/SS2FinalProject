from flask import Flask, render_template, request, jsonify
import textwrap
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import HuggingFaceHub

import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Setup
app = Flask(__name__)
loader = TextLoader("data.txt")
document = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
docs = text_splitter.split_documents(document)
embeddings = HuggingFaceEmbeddings()
db = FAISS.from_documents(docs, embeddings)
llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.8, "max_length": 512})
chain = load_qa_chain(llm, chain_type="stuff")


# Web endpoints
@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    user_query = request.form["msg"]
    docs_result = db.similarity_search(user_query)
    answer = chain.run(input_documents=docs_result, question=user_query)
    return jsonify(answer)

if __name__ == '__main__':
    app.run(debug=True)
