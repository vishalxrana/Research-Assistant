from fastapi import FastAPI, APIRouter, Body, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from app import services, config

app = FastAPI()
router = APIRouter()

app.include_router(router, prefix=config.API_PREFIX)


@app.get("/")
def read_root():
    return {"Hello": "World"}