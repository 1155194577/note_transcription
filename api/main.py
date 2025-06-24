from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str = None):
    item = {"item_id": item_id}
    if q:
        item.update({"q": q})
    return item

@app.post("/pitch")
async def pitch(audio: bytes):
    
    return {"message": "Pitch"}