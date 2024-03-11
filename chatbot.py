import os
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI API KEY') # Enter your openai api key
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from operator import itemgetter
import tempfile
import whisper
from pytube import YouTube
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

class YoutubeChatBot:
  def __init__(self , youtube_url , model = 'gpt-3.5-turbo' , whisper_model = 'base' ):
    self.model = model
    self.transcript(youtube_url , whisper_model)
    self.define_chain()

  # This function will transcript the video (It may take some time depending on the video time)
  def transcript(self , youtube_url , whisper_model):
    
    # Let's check if the video is transcripted already, this will only happen if you terminate the program and run it again without changing the youtube video. If you changed the video make sure to delete the transcript.txt file before runing the program again
    if not os.path.exists('transcript.txt'):
      
      youtube = YouTube(youtube_url)
      audio = youtube.streams.filter(only_audio=True).first()
      
      whisper_model = whisper.load_model("base") # You can use better models for transcription but it may take longer time to transcript than the one I used

      with tempfile.TemporaryDirectory() as temp_dir:
        file = audio.download(output_path=temp_dir)
        transciption = whisper_model.transcribe(file , fp16 = False)['text'].strip() 

        with open('transcript.txt', 'w') as f:
          f.write(transciption)
  
    
  def load_and_split_documents(self):
    loader = TextLoader('transcript.txt')
    text_documents = loader.load()
    
    # Splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(text_documents)
    
    return chunks
  
  def embed_documents(self):
    chunks = self.load_and_split_documents()
    
    # Embedding
    embeddings = OpenAIEmbeddings()
    vectore_store = Chroma.from_documents(chunks, embeddings)
    retriever = vectore_store.as_retriever(search_kwargs = {'k' : 5})
    
    return retriever
  
  def define_chain(self):
    template = """
Answer the following question based on the context provided.
If you don't know the answer from the given context just replay with "I don't know".

Context: {context}

Question: {question}
"""
    self.prompt = ChatPromptTemplate.from_template(template)
    self.retriever = self.embed_documents()
    
    self.model = ChatOpenAI(model_name = self.model)
    
    # First define RAG chain
    self.rag = ({"context": itemgetter('question')|self.retriever, "question": itemgetter('question')}
                | self.prompt
                | self.model
                | StrOutputParser()
                )
    # Then define the translation chain based on RAG chain answer
    translation_prompt = ChatPromptTemplate.from_template("Translate {answer} to {language}")
    self.chain = ({'answer' : self.rag , "language" : itemgetter('language')} | translation_prompt | self.model | StrOutputParser())
    

  # This function will receive a question and will answer in any language of your choice
  def ask(self , question, language = "English"):
    return self.chain.invoke({'question':question , 'language':language}) # Usually best language will be english

