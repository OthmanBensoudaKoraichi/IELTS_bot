## Split text, create vector embeddings with OpenAI embeddings, and store these embeddings on Pinecone.

# Import libraries

from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
from langchain_community.document_loaders import PyPDFLoader
from tqdm.auto import tqdm
from uuid import uuid4

# Every record contains a lot of text.
# Our first task is therefore to identify a good preprocessing methodology for chunking these articles into more "concise" chunks
# to later be embedding and stored in our Pinecone vector database.
# For this we use LangChain's RecursiveCharacterTextSplitter to split our text into chunks of a specified max length.

tiktoken.encoding_for_model("gpt-3.5-turbo")

tokenizer = tiktoken.get_encoding("cl100k_base")


# create the length function
def tiktoken_len(text):
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)


# Create a text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""],
)

# Create a Pinecone object
pc = Pinecone(api_key="TA CLEF PINECONE ICI")

# Set index to your index, which is the name of our index on Pinecone
index = pc.Index("LE NOM DE TON INDEX ICI")

# Initialize OpenAI Embeddings
embed = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key="TA CLEF OPENAI ICI")

# Paths to your PDF files
pdf_paths = [""]

def process_batch(texts, metadatas):
    try:
        ids = [str(uuid4()) for _ in range(len(texts))]
        embeds = embed.embed_documents(texts)
        index.upsert(vectors=zip(ids, embeds, metadatas))
        return True
    except openai.error.RateLimitError as e:
        print("Rate limit exceeded. Waiting before retrying...")
        time.sleep(60)  # Wait for 60 seconds before retrying
        return False

for pdf_path in pdf_paths:
    # Load and split PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()

    texts = []
    metadatas = []

    for page_num, page in enumerate(tqdm(pages)):

        # Split text into chunks
        record_texts = text_splitter.split_text(page.page_content)
        record_metadatas = [{"page": page_num, "chunk": i, "text": chunk} for i, chunk in enumerate(record_texts)]

        texts.extend(record_texts)
        metadatas.extend(record_metadatas)

        if len(texts) >= 100:
            success = process_batch(texts, metadatas)
            if success:
                texts, metadatas = [], []

    # Process remaining texts
    if texts:
        ids = [str(uuid4()) for _ in range(len(texts))]
        embeds = embed.embed_documents(texts)
        index.upsert(vectors=zip(ids, embeds, metadatas))

print("Processing complete.")