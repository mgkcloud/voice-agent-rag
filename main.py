# main.py
import asyncio
import aiohttp
import os
import logging
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from twilio.twiml.voice_response import VoiceResponse
from bot import run_bot
from helpers import create_room, create_token, get_env_variables
from dotenv import load_dotenv
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
import uvicorn

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv(override=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

async def create_and_write_room_info():
    room_info = create_room()
    if room_info.get("status_code", 200) == 200:
        room_url = room_info["url"]
        token = room_info["token"]
        sip_endpoint = room_info["sip_endpoint"]
        
        with open("daily_room_info.txt", "w") as file:
            file.write(f"VITE_DAILY_URL={room_url}\n")
            file.write(f"VITE_DAILY_TOKEN={token}\n")
        
        logger.info("\n\n\n\n\n\n\n\n\n\nRoom URL and token have been written to daily_room_info.txt\n\n\n\n\n\n\n\n\n\n")
        return room_url, token, sip_endpoint
    else:
        logger.error(room_info.get("message", "Failed to create room"))
        return None, None, None

async def run_bot_process(cmd: str):
    process = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    async def log_output(stream, prefix):
        while True:
            line = await stream.readline()
            if line:
                logger.info(f"{prefix}: {line.decode().strip()}")
            else:
                break

    try:
        await asyncio.gather(
            log_output(process.stdout, "BOT STDOUT"),
            log_output(process.stderr, "BOT STDERR")
        )
    except asyncio.CancelledError:
        logger.warning("Bot process logging was cancelled")
    finally:
        return_code = await process.wait()
        logger.info(f"Bot process exited with return code: {return_code}")
        return return_code


twilio_client = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))

@app.post("/start_bot", response_class=PlainTextResponse)
async def start_bot(request: Request, background_tasks: BackgroundTasks):
    logger.info("Received request to /start_bot")
    
    # Log the request headers
    logger.debug("Request headers:")
    for name, value in request.headers.items():
        logger.debug(f"{name}: {value}")
    
    # Log the request body
    body = await request.body()
    logger.debug(f"Request body: {body.decode()}")
    
    try:
        form_data = await request.form()
        data = dict(form_data)
        logger.debug(f"Parsed form data: {data}")
    except Exception as e:
        logger.error(f"Error parsing form data: {str(e)}")
        data = {}

    call_id = data.get('CallSid')
    if not call_id:
        logger.error("Missing 'CallSid' in request")
        raise HTTPException(status_code=400, detail="Missing 'CallSid' in request")

    logger.info(f"Received call with CallSid: {call_id}")

    try:
        room_url, token, sip_endpoint = await create_and_write_room_info()
        if not room_url or not token or not sip_endpoint:
            logger.error("Failed to create room or get necessary information")
            raise HTTPException(status_code=500, detail="Failed to create room or get necessary information")

        # Start the bot in a background task
        cmd = f"python3 bot.py -u {room_url} -t {token} -i {call_id} -s {sip_endpoint}"
        logger.info(f"Starting bot with command: {cmd}")
        background_tasks.add_task(run_bot_process, cmd)

        resp = VoiceResponse()
        resp.play(url="http://com.twilio.sounds.music.s3.amazonaws.com/MARKOVICHAMP-Borghestral.mp3", loop=10)
        return str(resp)
    except Exception as e:
        logger.exception("Error in start_bot endpoint")
        raise HTTPException(status_code=500, detail=str(e))

async def main():
    room_url, token, sip_endpoint = await create_and_write_room_info()
    if room_url and token and sip_endpoint:
        async with aiohttp.ClientSession() as session:
            await run_bot(room_url, token, None, sip_endpoint)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pipecat Bot Runner")
    parser.add_argument("--serve", action="store_true", help="Run as a server")
    args = parser.parse_args()

    if args.serve:
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
    else:
        asyncio.run(main())
