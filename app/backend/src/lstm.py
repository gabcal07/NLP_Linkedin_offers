import fastapi
from models.src.generation.lstm import lstm_generate_description
app = fastapi.FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.post("/lstm_job_description")
async def lstm_job_description(request: fastapi.Request):
    data = await request.json()
    title = data["title"]
    description = lstm_generate_description(title)
    return {"description": description}


