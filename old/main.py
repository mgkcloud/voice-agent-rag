import asyncio
import os
import sys
import time
import aiohttp
import requests
import logging
from loguru import logger
from multiprocessing import Process
from pipecat.frames.frames import LLMMessagesFrame, EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator, LLMUserResponseAggregator)
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.vad.silero import SileroVADAnalyzer
from pipecat.vad.vad_analyzer import VADParams
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Redis
from langchain.chains.combine_documents import create_stuff_documents_chain
from helpers import (
    AudioVolumeTimer,
    TranscriptionTimingLogger,
    LangchainRAGProcessor,
    ElevenLabsTurbo
)
from redisvl.schema import IndexSchema
from elevenlabs import VoiceSettings

from dotenv import load_dotenv
from functools import lru_cache
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from flask import Flask, jsonify

app = Flask(__name__)

@lru_cache()
def get_env_variables():
    load_dotenv(override=True)
    return {
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "REDIS_URL": os.getenv("REDIS_URL"),
        "ELEVENLABS_API_KEY": os.getenv("ELEVENLABS_API_KEY"),
        "DEEPGRAM_API_KEY": os.getenv("DEEPGRAM_API_KEY"),
        "DAILY_TOKEN": os.getenv("DAILY_TOKEN"),          
    }

get_env_variables.cache_clear()
env_vars = get_env_variables()


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ['SSL_CERT'] = ''
os.environ['SSL_KEY'] = ''


message_store = {}
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=env_vars["GOOGLE_API_KEY"])

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in message_store:
        message_store[session_id] = ChatMessageHistory()
    return message_store[session_id]

async def main(room_url: str, token: str):
    async with aiohttp.ClientSession() as session:
        
        
        deepgramkey = env_vars["DEEPGRAM_API_KEY"]
        
        transport = DailyTransport(
            room_url,
            token,
            "feisty",
            DailyParams(
                audio_out_enabled=True,
                transcription_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            )
        )
        stt = DeepgramSTTService(
            name="STT",
            api_key=deepgramkey,
            url='https://api.deepgram.com/v1/listen'
        )
        
    
        tts = CartesiaTTSService(
            aiohttp_session=session,
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",  # British Lady
        )
        
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=env_vars["GOOGLE_API_KEY"], convert_system_message_to_human=True)
        
        custom_schema = {
            "index": {"name": "gdrive", "prefix": "doc"},
            "fields": [
                {"type": "tag", "name": "id"},
                {"type": "tag", "name": "doc_id"},
                {"type": "text", "name": "text"},
                {
                    "type": "vector",
                    "name": "vector",
                    "attrs": {
                        "dims": 768,  # Gemini embedding dimension
                        "algorithm": "hnsw",
                        "distance_metric": "cosine",
                    },
                },
                {
                    "type": "numeric",
                    "name": "distance",
                },
            ],
        }
                
        vectorstore = Redis.from_existing_index(
            GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=env_vars["GOOGLE_API_KEY"]),
            index_name="gdrive",
            redis_url="redis://172.17.0.4:6379",
            schema=custom_schema,
        )
        retriever = vectorstore.as_retriever()
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a conversational AI bot with access to a vast knowledge base. Your goal is to engage in meaningful conversations with users, providing helpful and informative responses. \
                    Utilize the context and information available through RAG techniques to create succinct and relevant answers. Be personable, friendly, and ask for clarification if a user's question is ambiguous. \
                    Ensure your responses contain only words and avoid using special characters other than '?' or '!'. {context}""",
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm, answer_prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        history_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            history_messages_key="chat_history",
            input_messages_key="input",
            output_messages_key="answer"
        )
        lc = LangchainRAGProcessor(chain=history_chain)
        avt = AudioVolumeTimer()
        tl = TranscriptionTimingLogger(avt)
        
        
        messages = [
                {
                    "role": "system",
                    "content": "You are a fast, low-latency chatbot. Your goal is to demonstrate voice-driven AI capabilities at human-like speeds. Respond to what the user said in a creative and helpful way, but keep responses short and legible. Ensure responses contain only words. Keep responses under 2 sentences. Check again that you have not included special characters other than '?' or '!'.",
                },
        ]
        
        tma_in = LLMUserResponseAggregator(messages)
        tma_out = LLMAssistantResponseAggregator()
        pipeline = Pipeline([
            transport.input(),
            avt,
            stt,
            tl,
            tma_in,
            lc,
            tts,
            transport.output(),
            tma_out,
        ])
        task = PipelineTask(pipeline, PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            report_only_initial_ttfb=True,
        ))
        

        # @transport.event_handler("on_first_participant_joined")
        # async def on_first_participant_joined(transport, participant):
      
            
        @transport.event_handler("on_participant_joined")
        async def on_participant_joined(transport, participant):
            # Kick off the conversation.
            time.sleep(1.5)
            messages.append(
                {
                    "role": "system",
                    "content": "Introduce yourself by saying 'hey, William, whats up?'",
                }
            )
            print(participant["id"])
            transport.capture_participant_transcription(participant["id"])
            lc.set_participant_id(participant["id"])
          
            await task.queue_frame(LLMMessagesFrame(messages))    

        #@transport.event_handler("on_participant_left")
        #async def on_participant_left(transport, participant, reason):
            # await task.queue_frame(EndFrame())

        #@transport.event_handler("on_call_state_updated")
        # async def on_call_state_updated(transport, state):
            # if state == "left":
                # await task.queue_frame(EndFrame())

        runner = PipelineRunner()
        await runner.run(task)
        # await session.close()
        
        
        
        return True

async def start_bot(room_url: str, token: str = None):
    await check_deepgram_model_status()

    try:
        await main(room_url, token)
    except Exception as e:
        logger.error(f"Exception in main: {e}")
        sys.exit(1)  # Exit with a non-zero status code
    
    return {"message": "session finished"}

def create_room():
    
    daily_token = env_vars["DAILY_TOKEN"]
    url = "https://api.daily.co/v1/rooms/"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {daily_token}"
    }
    data = {
        "properties": {
            "exp": int(time.time()) + 60*180, ##5 mins
            "eject_at_room_exp" : True,
        }
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        room_info = response.json()
        token = create_token(room_info['name'])
        if token and 'token' in token:
            room_info['token'] = token['token']
        else:
            print("Failed to create token")
            return {"message": 'There was an error creating your room', "status_code": 500}
        return room_info
    else:
        data = response.json()
        if data.get("error") == "invalid-request-error" and "rooms reached" in data.get("info", ""):
            print("We are currently at capacity for this demo. Please try again later.")
            return {"message": "We are currently at capacity for this demo. Please try again later.", "status_code": 429}
        print(f"Failed to create room: {response.status_code}")
        return {"message": 'There was an error creating your room', "status_code": 500}

def create_token(room_name: str):
    url = "https://api.daily.co/v1/meeting-tokens"
    daily_token = env_vars["DAILY_TOKEN"]
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {daily_token}"
    }
    data = {
        "properties": {
            "room_name": room_name,
            "is_owner": True,
        }
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        token_info = response.json()
        return token_info
    else:
        print(f"Failed to create token: {response.status_code}")
        return None

async def check_deepgram_model_status():
    deepgramkey = env_vars["DEEPGRAM_API_KEY"]
    url = "https://api.deepgram.com/v1/status/engine"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Token {deepgramkey}"
    }
    max_retries = 10
    async with aiohttp.ClientSession() as session:
        for _ in range(max_retries):
            print("Trying Deepgram local server")
            # try:
            #     async with session.get(url, headers=headers) as response:
            #         if response.status == 200:
            #             json_response = await response.json()
            #             print(json_response)
            #             if json_response.get('engine_connection_status') == 'Connected':
            #                 print("Connected to deepgram local server")
            return True
            # except aiohttp.ClientConnectionError:
            #     print("Connection refused, retrying...")
            await asyncio.sleep(10)
    return False


if __name__ == "__main__":
    room_info = create_room()
    if room_info.get("status_code", 200) == 200:
        room_url = room_info["url"]
        token = room_info["token"]
        
        # Write the room URL and token to a txt file
        with open("daily_room_info.txt", "w") as file:
            file.write(f"VITE_DAILY_URL={room_url}\n")
            file.write(f"VITE_DAILY_TOKEN={token}\n")
        
        print("\n\n\n\n\n\n\n\n\n\nRoom URL and token have been written to daily_room_info.txt\n\n\n\n\n\n\n\n\n\n")
        asyncio.run(start_bot(room_url, token))
    else:
        print(room_info.get("message", "Failed to create room"))