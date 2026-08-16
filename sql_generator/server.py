from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from agent import run_sql_agent

app = FastAPI()
templates = Jinja2Templates(directory="templates")

class QueryRequest(BaseModel):
    question: str

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/query")
def process_query(req: QueryRequest):
    return run_sql_agent(req.question)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
