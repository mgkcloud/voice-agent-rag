import asyncio
import os
import sys
import time
import aiohttp
import requests
from loguru import logger
from dotenv import load_dotenv
from functools import lru_cache

from pipecat.frames.frames import LLMMessagesFrame, EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_response import LLMAssistantResponseAggregator, LLMUserResponseAggregator
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.vad.silero import SileroVADAnalyzer
from pipecat.vad.vad_analyzer import VADParams
from pipecat.services.llm import LLMService

from elevenlabs import VoiceSettings

# Custom helpers (assuming these are defined in a separate file)
from helpers import AudioVolumeTimer, TranscriptionTimingLogger, ElevenLabsTurbo


@lru_cache()
def get_env_variables():
    load_dotenv(override=True)
    return {
        "DAILY_TOKEN": os.getenv("DAILY_TOKEN"),
        "ELEVENLABS_API_KEY": os.getenv("ELEVENLABS_API_KEY"),
        "DEEPGRAM_API_KEY": os.getenv("DEEPGRAM_API_KEY"),
        "EXTERNAL_API_KEY": os.getenv("EXTERNAL_API_KEY"),
        "EXTERNAL_ASSISTANT_ID": os.getenv("EXTERNAL_ASSISTANT_ID"),
    }


env_vars = get_env_variables()


class ExternalAPILLMService(LLMService):
    def __init__(self, api_key, assistant_id):
        super().__init__()
        self.api_key = api_key
        self.assistant_id = assistant_id
        self.session_id = None
        self.base_url = "https://agentivehub.com/api/chat"

    async def create_session(self):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/session",
                json={"api_key": self.api_key,
                      "assistant_id": self.assistant_id}
            ) as response:
                data = await response.json()
                self.session_id = data["session_id"]

    async def process_frame(self, frame):
        if not self.session_id:
            await self.create_session()

        if isinstance(frame, LLMMessagesFrame):
            messages = frame.messages
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/message",
                    json={
                        "api_key": self.api_key,
                        "session_id": self.session_id,
                        "assistant_id": self.assistant_id,
                        "messages": messages
                    }
                ) as response:
                    data = await response.json()
                    response_content = data["response"]

            # Instead of yielding multiple frames, we'll return a single LLMMessagesFrame
            return LLMMessagesFrame([{"role": "assistant", "content": response_content}])


async def main(room_url: str, token: str):
    async with aiohttp.ClientSession() as session:
        transport = DailyTransport(
            room_url,
            token,
            "feisty",
            DailyParams(
                audio_out_enabled=True,
                transcription_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(
                    params=VADParams(stop_secs=0.2)),
            )
        )

        stt = DeepgramSTTService(
            name="STT",
            api_key=env_vars["DEEPGRAM_API_KEY"],
            url='https://api.deepgram.com/v1/listen'
        )

        # tts = ElevenLabsTurbo(
        #     aiohttp_session=session,
        #     api_key=env_vars["ELEVENLABS_API_KEY"],
        #     voice_id="WLKp2jV6nrS8aMkPPDRO",
        #     voice_settings=VoiceSettings(
        #         stability=0.6,
        #         similarity_boost=0.5,
        #         style=0.2,
        #     ),
        # )

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",  # British Lady
        )

        llm = ExternalAPILLMService(
            api_key=env_vars["EXTERNAL_API_KEY"],
            assistant_id=env_vars["EXTERNAL_ASSISTANT_ID"]
        )

        messages = [
            {
                "role": "system",
                "content": "You are a fast, low-latency chatbot. Your goal is to demonstrate voice-driven AI capabilities at human-like speeds. Respond to what the user said in a creative and helpful way, but keep responses short and legible. Ensure responses contain only words. Keep responses under 2 sentences. Check again that you have not included special characters other than '?' or '!'.",
            },
        ]

        tma_in = LLMUserResponseAggregator(messages)
        tma_out = LLMAssistantResponseAggregator()
        avt = AudioVolumeTimer()
        tl = TranscriptionTimingLogger(avt)

        pipeline = Pipeline([
            transport.input(),
            avt,
            stt,
            tl,
            tma_in,
            llm,
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
            time.sleep(1.5)
            messages.append(
                {
                    "role": "system",
                    "content": "Introduce yourself by saying 'hey, William, whats up?'",
                }
            )
            print(participant["id"])
            transport.capture_participant_transcription(participant["id"])
            await task.queue_frame(LLMMessagesFrame(messages))

        runner = PipelineRunner()
        await runner.run(task)

        return True


async def start_bot(room_url: str, token: str = None):
    await check_deepgram_model_status()

    try:
        await main(room_url, token)
    except Exception as e:
        logger.error(f"Exception in main: {e}")
        sys.exit(1)

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
            "exp": int(time.time()) + 60*180,
            "eject_at_room_exp": True,
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
            return True
            await asyncio.sleep(10)
    return False

if __name__ == "__main__":
    room_info = create_room()
    if room_info.get("status_code", 200) == 200:
        room_url = room_info["url"]
        token = room_info["token"]

        with open("daily_room_info.txt", "w") as file:
            file.write(f"VITE_DAILY_URL={room_url}\n")
            file.write(f"VITE_DAILY_TOKEN={token}\n")

        print("\n\n\n\n\n\n\n\n\n\nRoom URL and token have been written to daily_room_info.txt\n\n\n\n\n\n\n\n\n\n")
        asyncio.run(start_bot(room_url, token))
    else:
        print(room_info.get("message", "Failed to create room"))
