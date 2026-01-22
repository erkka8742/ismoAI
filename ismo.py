import os
import io
import re
import winsound
import speech_recognition as sr
from faster_whisper import WhisperModel
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"

if os.path.exists(cuda_path):
    try:
        os.add_dll_directory(cuda_path)
        print("Success: Linked NVIDIA CUDA 12.4 libraries.")
    except Exception as e:
        print(f"Warning: Could not add DLL directory. Error: {e}")
else:
    print(f"Error: Could not find CUDA folder at: {cuda_path}")
    print("Did you install CUDA 12.4 to the default location?")

# --- 2. LOAD MODEL ---
print("Loading model to GPU...")
try:
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
except Exception as e:
    print("\nCRITICAL ERROR: The model failed to load.")
    print("Most likely cause: 'zlibwapi.dll' is missing from the CUDA bin folder.")
    print(f"Exact error: {e}")
    exit()

print("Listening...")

recognizer = sr.Recognizer()
recognizer.pause_threshold = 1.5 
recognizer.energy_threshold = 300
recognizer.dynamic_energy_threshold = True


client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# Load base prompt
with open("prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

# Load voice line names
voice_lines = [f[:-4] for f in os.listdir("voice_lines") if f.endswith(".wav")]

system_prompt = system_prompt + "\n" + ", ".join(voice_lines)


def clean_text(text):
    # Remove "vastaus:" prefix if present
    if "vastaus:" in text.lower():
        text = text.lower().split("vastaus:")[-1]
    return text.replace("'", "").replace('"', "").replace(".", "").replace(",", "").strip().lower()

def play_voice_line(voice_line):
    print(f"Playing voice line: {voice_line}")
    filename = f"voice_lines/{voice_line}.wav"
    if os.path.exists(filename):
        winsound.PlaySound(filename, winsound.SND_FILENAME)
        print(f"✅ Voice line played: {filename}")
    else:
        print(f"❌ Error: Could not find voice line at: {filename}")

def askIsmo(prompt):
    completion = client.chat.completions.create(
    model="qwen/qwen3-32b",
    messages=[
      {
        "role": "system",
        "content": system_prompt
      },
      {
        "role": "user",
        "content": prompt
      }
    ],
    temperature=1.2,
    max_completion_tokens=1024,
    top_p=1,
    stream=False,
    stop=None,
    include_reasoning=False,
    reasoning_effort="none"

    )       
    response = clean_text(completion.choices[0].message.content.strip())
    print(f"AI response: '{response}'")
    if response in voice_lines:
        play_voice_line(response)
    else:
        print(f"❌ '{response}' not found in voice_lines.")






try:
    with sr.Microphone(sample_rate=16000) as source:
        recognizer.adjust_for_ambient_noise(source, duration=1.0)
        
        while True:
            print("\nListening...", end="", flush=True)
            audio_data = recognizer.listen(source)
            print(" Processing...", end="", flush=True)
            
            wav_stream = io.BytesIO(audio_data.get_wav_data())
            segments, info = model.transcribe(
            wav_stream, 
            language="fi", 
            beam_size=5,
            vad_filter=True,                       
            vad_parameters=dict(min_silence_duration_ms=500), 
            condition_on_previous_text=False,      
            no_speech_threshold=0.6                
)
            
            full_text = "".join([segment.text for segment in segments])
            if full_text.strip():
                print(f"\rUser: {full_text}")
                askIsmo(full_text)
                

except KeyboardInterrupt:
    print("\nStopping...")