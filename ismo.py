import os
import io
import re
import winsound
import speech_recognition as sr
from faster_whisper import WhisperModel
from groq import Groq
from dotenv import load_dotenv
from cerebras.cloud.sdk import Cerebras

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
recognizer.pause_threshold = 1.1
recognizer.energy_threshold = 200
recognizer.dynamic_energy_threshold = True


client = Cerebras(
    api_key=os.environ.get("API_KEY"),
)

# Load base prompt
with open("prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

# Load voice line names
voice_lines = [f[:-4] for f in os.listdir("voice_lines") if f.endswith(".wav")]

system_prompt = system_prompt + "\n" + ", ".join([f"{i+1}. {line}" for i, line in enumerate(voice_lines)])


def clean_text(text):
    # Extract all numbers from the text
    matches = re.findall(r'\d+', text)
    return matches

def play_voice_line(voice_line):
    print(f"Playing voice line: {voice_line}")
    filename = f"voice_lines/{voice_line}.wav"
    if os.path.exists(filename):
        winsound.PlaySound(filename, winsound.SND_FILENAME)
        print(f"✅ Voice line played: {filename}")
    else:
        print(f"❌ Error: Could not find voice line at: {filename}")

convoContext = [
      {
        "role": "system",
        "content": system_prompt
      },
]
       

def askIsmo(prompt):
    convoContext.append({
        "role": "user",
        "content": prompt
    })
    if len(convoContext) > 10:
        convoContext.pop(1)
        convoContext.pop(1)
    completion = client.chat.completions.create(
    #model="qwen/qwen3-32b",
    #model="llama-3.1-8b-instant",
    model="llama-3.3-70b",
    messages=convoContext,
    temperature=1.5,
    max_completion_tokens=1024,
    top_p=1,
    stream=False,
    stop=None,
    #reasoning_format="none"

    )       
    numbers = clean_text(completion.choices[0].message.content.strip())
    print(f"AI response: '{numbers}'")
    convoContext.append({
        "role": "assistant",
        "content": str(numbers)
    })
    if numbers:
        for num in numbers:
            try:
                index = int(num) - 1  # Convert to 0-based index
                if 0 <= index < len(voice_lines):
                    play_voice_line(voice_lines[index])
                else:
                    print(f"❌ Number '{num}' is out of range. Valid numbers: 1-{len(voice_lines)}")
            except ValueError:
                print(f"❌ Could not parse number: '{num}'")
    else:
        print(f"❌ No numbers found in response.")

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
                print(f"\r{' ' * 50}\rUser: {full_text}")  # Clear line before printing
                askIsmo(full_text)
                

except KeyboardInterrupt:
    print("\nStopping...")