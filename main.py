# loading all the important libraries

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
import os
from llama_cpp import Llama

# setting path
data_path = "data/"

#loading the ollama model
llm = Llama(
    model_path="mistral-7b-instruct-v0.2.Q8_0.gguf",
    n_ctx=4096,
    n_threads=8,
    verbose=False
)

#loading the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

#creating text splitter 
splitter = CharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
)

#splitting the documents
chunks_dict = {}
speeches = os.listdir(data_path)

for speech in speeches:
    if os.path.splitext(speech)[1] == ".txt":
        loader = TextLoader(os.path.join(data_path, speech))
        documents = loader.load()
        chunks = splitter.split_documents(documents)
        chunks_dict[speech] = chunks

# Converting the chunks into embeddings and storing them in chroma db
chroma_store = Chroma.from_documents(
    documents= [chunk for chunk_list in list(chunks_dict.values()) for chunk in chunk_list],
    embedding=embeddings,
    persist_directory="chroma_db"
)

#  creating prompt template
prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="""
You must answer the question strictly and exclusively using the information provided in the context below.

If the context does NOT contain information that directly answers the question, reply exactly with:
"No relevant information available."

Context:
{context}

Question:
{question}

Answer:
""",
)



#function to get output
def get_response(query):
    results = chroma_store.similarity_search(query, k=2)
    context = ""
    fetched_documents = []
    for r in results:
        context += r.page_content + "\n"
        fetched_documents.append(r.metadata["source"].split("/")[-1])
  
    inp =  prompt.format(
        question=query,
        context=context)
  
    response = llm(
        inp,
        max_tokens=200,
        temperature=0.7)

    output = response["choices"][0]["text"]
    return output, fetched_documents


if __name__ == "__main__":
  
    # get user input

    question = input("Enter your question: ")
    # or
    # question = ""


    response, docs_name = get_response(question)

    print(response)
    print(docs_name)

    llm.close()
