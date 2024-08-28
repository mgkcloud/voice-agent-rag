# bot.py
import asyncio
import aiohttp
import os
import sys
import argparse
from loguru import logger
from dotenv import load_dotenv
from pipecat.frames.frames import EndFrame, LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator,
    LLMUserResponseAggregator
)
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.vad.silero import SileroVADAnalyzer
from pipecat.vad.vad_analyzer import VADParams
from helpers import (
    AudioVolumeTimer,
    TranscriptionTimingLogger,
    LangchainRAGProcessor,
    ElevenLabsTurbo
)
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.vectorstores import Redis
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from elevenlabs import VoiceSettings
from twilio.rest import Client

from dotenv import load_dotenv
from functools import lru_cache
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv(override=True)
logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

@lru_cache()
def get_env_variables():
    return {
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "REDIS_URL": os.getenv("REDIS_URL"),
        "ELEVENLABS_API_KEY": os.getenv("ELEVENLABS_API_KEY"),
        "DEEPGRAM_API_KEY": os.getenv("DEEPGRAM_API_KEY"),
        "DAILY_TOKEN": os.getenv("DAILY_TOKEN"),
        "TWILIO_ACCOUNT_SID": os.getenv("TWILIO_ACCOUNT_SID"),
        "TWILIO_AUTH_TOKEN": os.getenv("TWILIO_AUTH_TOKEN"),
        "CARTESIA_API_KEY": os.getenv("CARTESIA_API_KEY"),
    }

get_env_variables.cache_clear()
env_vars = get_env_variables()

twilio_account_sid = os.getenv('TWILIO_ACCOUNT_SID')
twilio_auth_token = os.getenv('TWILIO_AUTH_TOKEN')
twilioclient = Client(twilio_account_sid, twilio_auth_token)

message_store = {}
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", api_key=env_vars["GOOGLE_API_KEY"])

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in message_store:
        message_store[session_id] = ChatMessageHistory()
    return message_store[session_id]

async def run_bot(room_url: str, token: str, callId: str, sipUri: str):
    async with aiohttp.ClientSession() as session:
        daily_api_key = env_vars["DAILY_TOKEN"]
        if not daily_api_key:
            raise ValueError("DAILY_TOKEN environment variable is not set")

        logger.info(f"Initializing DailyTransport with room_url: {room_url}, call_id: {callId}, sip_uri: {sipUri}")
        transport = DailyTransport(
            room_url,
            token,
            "feisty",
            DailyParams(
                api_key=daily_api_key,
                dialin_settings=None,  # Not required for Twilio
                audio_in_enabled=True,
                audio_out_enabled=True,
                camera_out_enabled=False,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(
                    params=VADParams(stop_secs=0.2)),
                transcription_enabled=True,
            )
        )
        logger.info("DailyTransport initialized successfully")

        stt = DeepgramSTTService(
            name="STT",
            api_key=env_vars["DEEPGRAM_API_KEY"],
            url='https://api.deepgram.com/v1/listen'
        )
        tts = CartesiaTTSService(
            aiohttp_session=session,
            api_key=env_vars["CARTESIA_API_KEY"],
            voice_id="41f3c367-e0a8-4a85-89e0-c27bae9c9b6d",  
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
            GoogleGenerativeAIEmbeddings(
                model="models/embedding-001", api_key=env_vars["GOOGLE_API_KEY"]),
            index_name="gdrive",
            redis_url=env_vars["REDIS_URL"],
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

        @transport.event_handler("on_participant_joined")
        async def on_participant_joined(transport, participant):
            messages.append(
                {
                    "role": "system",
                    "content": "Add punct. Dates: MM/DD/YYYY. Pauses: '-'. Match voice to lang. Use continuations. Phone #s: words+commas. ?? for emphasis. No quotes. Fast chat. Creative+helpful. Short (<2 sent). Words only. No special chars except ?!. Intro: 'hey, William, whats up?'",
                }
            )
            logger.info(f"Participant joined: {participant['id']}")
            transport.capture_participant_transcription(participant["id"])
            lc.set_participant_id(participant["id"])

            await task.queue_frame(LLMMessagesFrame(messages))

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            await task.queue_frame(EndFrame())


        @transport.event_handler("on_dialin_ready")
        async def on_dialin_ready(transport, cdata):
            logger.info(f"Forwarding call: {callId} {sipUri}")

            try:
                # The TwiML is updated using Twilio's client library
                call = twilioclient.calls(callId).update(
                    twiml=f'<Response><Dial><Sip>{sipUri}</Sip></Dial></Response>'
                )
            except Exception as e:
                raise Exception(f"Failed to forward call: {str(e)}")


        runner = PipelineRunner()
        await runner.run(task)

        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipecat Bot")
    parser.add_argument("-u", type=str, help="Room URL")
    parser.add_argument("-t", type=str, help="Token")
    parser.add_argument("-i", type=str, help="Call ID")
    parser.add_argument("-s", type=str, help="SIP URI")
    config = parser.parse_args()

    if not config.s:
        logger.error("SIP URI is missing. Make sure it's passed correctly from main.py")
        sys.exit(1)

    asyncio.run(run_bot(config.u, config.t, config.i, config.s))