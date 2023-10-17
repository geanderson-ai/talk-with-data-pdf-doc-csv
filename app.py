import os
from langchain.agents import create_pandas_dataframe_agent
from langchain.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
import chainlit as cl
from chainlit.types import AskFileResponse
import pandas as pd
import io
from langchain.llms import OpenAI



text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
embeddings = OpenAIEmbeddings()

welcome_message = """Welcome to the Chainlit PDF QA demo! To get started:
1. Upload a PDF or text file
2. Ask a question about the file
"""
def create_agent(data: str, llm):
    """Create a Pandas DataFrame agent."""
    return create_pandas_dataframe_agent(llm, data, verbose=False)

def process_file(file: AskFileResponse):
    import tempfile

    if file.type == "text/plain":
        Loader = TextLoader
    if file.type == "text/csv":
        Loader = CSVLoader
    elif file.type == "application/pdf":
        Loader = PyPDFLoader

    with tempfile.NamedTemporaryFile() as tempfile:
        tempfile.write(file.content)
        loader = Loader(tempfile.name)
        documents = loader.load()
        docs = text_splitter.split_documents(documents)
        for i, doc in enumerate(docs):
            doc.metadata["source"] = f"source_{i}"
        return docs


def get_docsearch(file: AskFileResponse):
    docs = process_file(file)

    # Save data in the user session
    cl.user_session.set("docs", docs)

    # Create a unique namespace for the file

    docsearch = Chroma.from_documents(
        docs, embeddings
    )
    return docsearch


@cl.on_chat_start
async def start():
    # Sending an image with the local file path
    await cl.Message(content="Hello there, Welcome to AskAnyQuery related to Data!").send()
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content=welcome_message,
            accept=["text/plain", "application/pdf", "text/csv"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]

    if file.type == "application/pdf" or file.type == "text/plain":
        msg = cl.Message(content=f"Processing `{file.name}`...")
        await msg.send()

        # No async implementation in the Pinecone client, fallback to sync
        docsearch = await cl.make_async(get_docsearch)(file)

        chain = RetrievalQAWithSourcesChain.from_chain_type(
            ChatOpenAI(temperature=0, streaming=True),
            chain_type="stuff",
            retriever=docsearch.as_retriever(max_tokens_limit=4097),
        )

        # Let the user know that the system is ready
        msg.content = f"`{file.name}` processed. You can now ask questions!"
        await msg.update()

        cl.user_session.set("chain", chain)
        @cl.on_message
        async def main(message):
            chain = cl.user_session.get("chain")  # type: RetrievalQAWithSourcesChain
            cb = cl.AsyncLangchainCallbackHandler(
                stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
            )
            cb.answer_reached = True
            res = await chain.acall(message, callbacks=[cb])

            answer = res["answer"]
            sources = res["sources"].strip()
            source_elements = []

            # Get the documents from the user session
            docs = cl.user_session.get("docs")
            metadatas = [doc.metadata for doc in docs]
            all_sources = [m["source"] for m in metadatas]

            if sources:
                found_sources = []

                # Add the sources to the message
                for source in sources.split(","):
                    source_name = source.strip().replace(".", "")
                    # Get the index of the source
                    try:
                        index = all_sources.index(source_name)
                    except ValueError:
                        continue
                    text = docs[index].page_content
                    found_sources.append(source_name)
                    # Create the text element referenced in the message
                    source_elements.append(cl.Text(content=text, name=source_name))

                if found_sources:
                    answer += f"\nSources: {', '.join(found_sources)}"
                else:
                    answer += "\nNo sources found"

            if cb.has_streamed_final_answer:
                cb.final_stream.elements = source_elements
                await cb.final_stream.update()
            else:
                await cl.Message(content=answer, elements=source_elements).send()

    else:
        await cl.Message(content="Unsupported file type").send()
        # implement logic to chat with csv in langchain
            # Read csv file with pandas
        csv_file = io.BytesIO(file.content)
        df = pd.read_csv(csv_file, encoding="utf-8")

        # creating user session to store data
        cl.user_session.set('data', df)

        # Send response back to user
        # Let the user know that the system is ready
        msg.content = f"Processing `{file.name}` done. You can now ask questions!"
        await msg.update()
        @cl.on_message
        async def main(message: str):

            # Get data
            df = cl.user_session.get('data')
            llm = OpenAI()


            # Agent creation
            agent = create_agent(df, llm)

            # Run model 
            response = agent.run(message)

            # Send a response back to the user
            await cl.Message(
                content=response,
            ).send()




