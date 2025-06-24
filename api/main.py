from fastapi import FastAPI, UploadFile, HTTPException, Path
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List
import io
import librosa
import numpy as np
import matplotlib.pyplot as plt

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/pitch")
async def pitch(file: UploadFile):
    contents = await file.read()
    audio, sr = librosa.load(io.BytesIO(contents))
    
    # Calculate pitch using YIN
    pitches = librosa.yin(audio, fmin=65, fmax=2093, sr=sr)
    times = librosa.times_like(pitches, sr=sr)
    
    # Create plot
    plt.figure(figsize=(10, 4))
    plt.plot(times, pitches)
    plt.title('Pitch Tracking')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    
    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    # Return the image as a streaming response
    return StreamingResponse(buf, media_type="image/png")
