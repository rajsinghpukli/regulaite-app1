# placeholder FastAPI main for health
from fastapi import FastAPI
from .pipeline import ask
app = FastAPI()
@app.get("/healthz")
def health(): return {"ok": True}
