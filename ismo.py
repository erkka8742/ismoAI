import os
import io
import re
import sys
import time
import signal
import queue
import random
import threading
import asyncio
import numpy as np
import scipy.signal
import webrtcvad
import discord
from faster_whisper import WhisperModel
from groq import Groq
from dotenv import load_dotenv
from cerebras.cloud.sdk import Cerebras

load_dotenv()

# --- 1. SETUP & CONFIG ---
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
if os.path.exists(cuda_path):
    try:
        os.add_dll_directory(cuda_path)
    except Exception as e:
        print(f"Warning: Could not add DLL directory: {e}")

# Use Pycord's Bot (supports voice receiving)
intents = discord.Intents.default()
intents.voice_states = True
intents.guilds = True
bot = discord.Bot(intents=intents)
print(f"[DEBUG] Using py-cord version: {discord.__version__}")
BOT_TOKEN = os.environ.get("DISCORD_TOKEN")

# Global State
current_voice_client = None
playback_queue = queue.Queue()
transcription_queue = queue.Queue()

# --- 2. LOAD MODELS ---
print("Loading Whisper model...")
try:
    # 'large-v3' is heavy; ensure your GPU can handle it alongside the bot
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    exit()

client = Cerebras(api_key=os.environ.get("API_KEY"))

# Load Prompts & Voice Lines
with open("prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

voice_lines = [f[:-4] for f in os.listdir("voice_lines") if f.endswith(".wav")]
system_prompt += "\n" + ", ".join([f"{i+1}. {line}" for i, line in enumerate(voice_lines)])

convoContext = [{"role": "system", "content": system_prompt}]

# Greeting voice lines (played randomly when joining)
GREETING_FILES = [
    "voice_lines/miten_voin_palvella.wav",
    "voice_lines/täältä_tullaan.wav",
    "voice_lines/tadaa.wav",
    "voice_lines/täällä_tehdään_työt_niin_kun_mä_haluan.wav",
    "voice_lines/terve.wav",
    "voice_lines/toivottavasti_en_ammu_kaksijalkasta.wav",
    "voice_lines/uh.wav",
    "voice_lines/viitta_päälle_ja_menoks.wav",
    "voice_lines/wow.wav",
    "voice_lines/lallallaa.wav",
    "voice_lines/katos_katos.wav"
]

# --- 3. VAD SINK (The Ears) ---
class VadSink(discord.sinks.Sink):
    """Captures audio, filters silence using WebRTC VAD, and buffers speech."""
    def __init__(self):
        super().__init__()
        self.vad = webrtcvad.Vad(2)  # Mode 2 = Moderately aggressive
        self.user_data = {}
        self._running = True
        # Start background monitor thread to flush stale buffers
        self._monitor_thread = threading.Thread(target=self._monitor_buffers, daemon=True)
        self._monitor_thread.start()
    
    def _monitor_buffers(self):
        """Background thread that flushes buffers if no packets received for a while."""
        STALE_TIMEOUT = 1.2  # Flush if no packets for 1.2 seconds
        while self._running:
            time.sleep(0.1)  # Check every 100ms
            current_time = time.time()
            
            for user_id, state in list(self.user_data.items()):
                if state['is_speaking'] and len(state['buffer']) > 0:
                    time_since_last = current_time - state['last_packet_time']
                    if time_since_last > STALE_TIMEOUT:
                        print(f"[DEBUG] No packets for {time_since_last:.1f}s, flushing buffer for user {user_id}")
                        self.flush_audio(user_id, state)

    def write(self, data, user_id):
        # Skip oversized packets (Discord burst buffers) - they break processing
        if len(data) > 10000:
            return
            
        if user_id not in self.user_data:
            self.user_data[user_id] = {
                'buffer': bytearray(),
                'silence_frames': 0,
                'is_speaking': False,
                'warmup_frames': 0,  # Skip first few frames (connection noise)
                'last_packet_time': time.time()
            }
        
        state = self.user_data[user_id]
        
        # Skip first 25 frames (~0.5 sec) to avoid connection noise
        if state['warmup_frames'] < 25:
            state['warmup_frames'] += 1
            return
        
        # Convert PCM (48kHz Stereo) to 16kHz Mono for VAD
        try:
            audio_np = np.frombuffer(data, dtype=np.int16).reshape(-1, 2)
            mono_audio = audio_np.mean(axis=1).astype(np.int16)
            
            # Calculate energy for debugging
            energy = np.abs(mono_audio).mean()
            
            # Downsample 48kHz -> 16kHz (take every 3rd sample)
            mono_16k = mono_audio[::3]
            
            # WebRTC VAD needs exactly 10/20/30ms frames at 16kHz
            # 20ms at 16kHz = 320 samples = 640 bytes
            FRAME_SIZE = 320
            is_speech = False
            
            # Only run VAD if energy is above minimum threshold
            MIN_ENERGY_THRESHOLD = 50
            if energy > MIN_ENERGY_THRESHOLD:
                # Process in 20ms chunks
                for i in range(0, len(mono_16k) - FRAME_SIZE + 1, FRAME_SIZE):
                    frame = mono_16k[i:i + FRAME_SIZE].tobytes()
                    if len(frame) == 640:  # 320 samples * 2 bytes
                        if self.vad.is_speech(frame, 16000):
                            is_speech = True
                            break
            
            # Debug: print energy level periodically (every ~1 second)
            if not hasattr(self, '_debug_counter'):
                self._debug_counter = 0
            self._debug_counter += 1
            if self._debug_counter % 50 == 0:  # Every 50 frames = 1 second
                print(f"[DEBUG] Audio energy: {energy:.1f}, VAD detected speech: {is_speech}")
                
        except Exception as e:
            print(f"[DEBUG] VAD exception: {e}")
            # If VAD fails, use simple energy detection as fallback
            try:
                audio_np = np.frombuffer(data, dtype=np.int16)
                energy = np.abs(audio_np).mean()
                is_speech = energy > 500  # Threshold
            except:
                is_speech = False

        # Pause Threshold = 1.1 seconds (55 frames at 20ms each)
        PAUSE_THRESHOLD_FRAMES = 55
        # Max buffer duration before forced flush (3 seconds of audio at 48kHz stereo)
        MAX_BUFFER_BYTES = 3 * 48000 * 4  # ~576000 bytes
        
        current_time = time.time()
        
        if is_speech:
            if not state['is_speaking']:
                print(f"[DEBUG] Speech STARTED for user {user_id}")
            state['silence_frames'] = 0
            state['is_speaking'] = True
            state['buffer'].extend(data)
            state['last_packet_time'] = current_time
            
            # Force flush if buffer gets too large (prevents very long recordings)
            if len(state['buffer']) > MAX_BUFFER_BYTES:
                print(f"[DEBUG] Buffer full, forcing flush for user {user_id}")
                self.flush_audio(user_id, state)
                
        elif state['is_speaking']:
            state['buffer'].extend(data)
            state['silence_frames'] += 1
            state['last_packet_time'] = current_time
            
            if state['silence_frames'] > PAUSE_THRESHOLD_FRAMES:
                print(f"[DEBUG] Speech ENDED for user {user_id}, buffer size: {len(state['buffer'])} bytes")
                self.flush_audio(user_id, state)

    def flush_audio(self, user_id, state):
        if len(state['buffer']) > 0:
            print(f"[DEBUG] Flushing {len(state['buffer'])} bytes to transcription queue")
            # Send copy of buffer to worker thread
            transcription_queue.put(bytes(state['buffer']))
            # Reset
            state['buffer'] = bytearray()
            state['silence_frames'] = 0
            state['is_speaking'] = False

    def cleanup(self):
        # Stop the monitor thread
        self._running = False
        # Flush any remaining audio when recording stops
        try:
            for user_id, state in list(self.user_data.items()):
                if len(state['buffer']) > 0:
                    print(f"[DEBUG] Cleanup: flushing remaining {len(state['buffer'])} bytes for user {user_id}")
                    self.flush_audio(user_id, state)
        except Exception as e:
            print(f"[DEBUG] Cleanup error (non-critical): {e}")

# --- 4. WORKER THREADS ---
ismoTalking = True

async def set_mute_state(muted: bool):
    """Toggle bot's self-mute state in voice channel."""
    global current_voice_client
    if current_voice_client and current_voice_client.is_connected():
        try:
            guild = current_voice_client.guild
            channel = current_voice_client.channel
            await guild.change_voice_state(channel=channel, self_mute=muted)
            print(f"[DEBUG] Bot self-mute set to: {muted}")
        except Exception as e:
            print(f"[DEBUG] Failed to change mute state: {e}")

def process_audio():
    """Reads completed phrases from queue and sends to Whisper + LLM."""
    global ismoTalking
    print("[DEBUG] Transcription worker started, waiting for audio...")
    while True:
        raw_bytes = transcription_queue.get()
        print(f"[DEBUG] Transcription worker received {len(raw_bytes)} bytes")
        
        try:
            # Convert raw PCM (48k Stereo) to Whisper (16k Mono Float32)
            audio_np = np.frombuffer(raw_bytes, dtype=np.int16).reshape(-1, 2)
            audio_np = audio_np.mean(axis=1) # Stereo -> Mono
            
            # Decimate 48k -> 16k (Take every 3rd sample)
            audio_np = audio_np[::3]
            
            # Normalize to -1.0 to 1.0
            final_audio = audio_np.astype(np.float32) / 32768.0
            
            duration_sec = len(final_audio) / 16000
            print(f"[DEBUG] Audio duration: {duration_sec:.2f}s, samples: {len(final_audio)}, max amplitude: {np.abs(final_audio).max():.4f}")

            # Transcribe
            print("[DEBUG] Starting Whisper transcription...")
            segments, info = model.transcribe(
                final_audio, 
                language="fi", 
                beam_size=5,
                condition_on_previous_text=False
            )
            
            # Collect all segments
            segment_list = list(segments)
            print(f"[DEBUG] Whisper returned {len(segment_list)} segments")
            
            full_text = "".join([segment.text for segment in segment_list]).strip()
            print(f"[DEBUG] Transcription result: '{full_text}'")
            
            # Skip common Whisper hallucinations
            text_clean_check = re.sub(r'[^\w\s]', '', full_text.lower()).strip()
            if text_clean_check in ["kiitos", "kiitos kiitos"]:
                print("[DEBUG] Skipping Whisper hallucination")
                continue
            
            if full_text:
                print(f"User: {full_text}")
                # Remove punctuation for wake word detection
                text_clean = re.sub(r'[^\w\s]', '', full_text.lower())
                
                if any(word in text_clean for word in ["hiljaa"]):
                    ismoTalking = False
                    print("[DEBUG] Ismo going quiet")
                    # Set mute icon on Discord
                    if bot.loop and bot.loop.is_running():
                        asyncio.run_coroutine_threadsafe(set_mute_state(True), bot.loop)
                    askIsmo(full_text)
                elif any(word in text_clean for word in ["puhutaan"]):
                    ismoTalking = True
                    print("[DEBUG] Ismo resuming conversation")
                    # Remove mute icon on Discord
                    if bot.loop and bot.loop.is_running():
                        asyncio.run_coroutine_threadsafe(set_mute_state(False), bot.loop)
                    askIsmo(full_text)
                    print("[DEBUG] Processing complete, ready for next input")
                elif ismoTalking and any(name in text_clean for name in ["ismo", "isma", "ismoo", "ismaa", "osmo", "osmoo"]):
                    askIsmo(full_text)
                    print("[DEBUG] Processing complete, ready for next input")
                else:
                    print("[DEBUG] 'Ismo' not mentioned, ignoring")
            else:
                print("[DEBUG] Empty transcription - no speech detected by Whisper")

        except Exception as e:
            import traceback
            print(f"Error in processing: {e}")
            traceback.print_exc()

def askIsmo(prompt):
    """Sends text to LLM and queues the response audio."""
    convoContext.append({"role": "user", "content": prompt})
    
    # Maintain context window
    if len(convoContext) > 5:
        convoContext.pop(1)
        convoContext.pop(1)

    try:
        completion = client.chat.completions.create(
            model="zai-glm-4.7",
            messages=convoContext,
            temperature=1.2,
            #max_completion_tokens=1024,
            top_p=1,
            #logprobs=True,
            disable_reasoning=True,
            #reasoning_format="hidden"
        )
        
        response_text = completion.choices[0].message.content
        if not response_text:
            print("[DEBUG] LLM returned empty response")
            return
        response_text = response_text.strip()
        numbers = re.findall(r'\d+', response_text)
        print(f"AI response: '{numbers}'")
        
        # Discard responses with more than 5 numbers
        if len(numbers) > 5:
            print(f"[DEBUG] Discarding response - too many numbers ({len(numbers)})")
            return
        
        convoContext.append({"role": "assistant", "content": str(numbers)})

        if numbers:
            for num in numbers:
                index = int(num) - 1
                if 0 <= index < len(voice_lines):
                    # Queue audio safely from this thread
                    file_path = f"voice_lines/{voice_lines[index]}.wav"
                    try:
                        if bot.loop and bot.loop.is_running():
                            bot.loop.call_soon_threadsafe(queue_audio, file_path)
                    except RuntimeError:
                        print("[DEBUG] Event loop closed, skipping audio playback")
    except Exception as e:
        print(f"LLM Error: {e}")

# --- 5. AUDIO PLAYBACK ---
def queue_audio(file_path):
    """Adds file to queue and starts playback if idle."""
    if not os.path.exists(file_path):
        print(f"Missing file: {file_path}")
        return

    playback_queue.put(file_path)
    if current_voice_client and not current_voice_client.is_playing():
        play_next()

def play_next():
    """Plays the next item in the queue."""
    if playback_queue.empty():
        return

    file_path = playback_queue.get()
    print(f"Playing: {file_path}")
    
    source = discord.FFmpegPCMAudio(file_path)
    
    # The 'after' callback triggers the next song when this one finishes
    current_voice_client.play(source, after=lambda e: play_next())

# --- 6. DISCORD EVENTS ---
@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}')
    # Start the transcription worker in the background
    threading.Thread(target=process_audio, daemon=True).start()

@bot.command()
async def join(ctx):
    global current_voice_client
    if ctx.author.voice:
        channel = ctx.author.voice.channel
        current_voice_client = await channel.connect()
        
        # Wait for voice connection to be fully ready
        await asyncio.sleep(1)
        
        print(f"[DEBUG] Voice client connected: {current_voice_client.is_connected()}")
        print(f"[DEBUG] Voice channel: {current_voice_client.channel}")
        
        # Start listening/recording
        async def recording_finished(sink, *args):
            print(f"[DEBUG] Recording finished callback triggered")
        
        try:
            sink = VadSink()
            print(f"[DEBUG] Starting recording with VadSink...")
            current_voice_client.start_recording(
                sink, 
                recording_finished,
                ctx.channel
            )
            print(f"[DEBUG] Recording started successfully")
            print(f"[DEBUG] Voice client recording: {current_voice_client.recording}")
            
            # Play random greeting when joining
            if GREETING_FILES:
                greeting_file = random.choice(GREETING_FILES)
                if os.path.exists(greeting_file):
                    queue_audio(greeting_file)
            
            await ctx.respond("Täältä tullaan", ephemeral=True)
        except Exception as e:
            print(f"Recording error: {e}")
            await ctx.respond(f"Joku meni vikaan", ephemeral=True)
    else:
        await ctx.respond("You need to be in a voice channel.")

@bot.command()
async def leave(ctx):
    global current_voice_client
    if ctx.voice_client:
        ctx.voice_client.stop_recording()
        await ctx.voice_client.disconnect()
        current_voice_client = None
        await ctx.respond("Hyvää päivänjatkoa", ephemeral=True)

# Signal handler for clean shutdown
def signal_handler(sig, frame):
    print("\n[DEBUG] Shutting down...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Start Bot
bot.run(BOT_TOKEN)