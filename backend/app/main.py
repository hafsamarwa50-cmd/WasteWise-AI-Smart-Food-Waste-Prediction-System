from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes.dashboard import router as dashboard_router
from backend.app.api.routes.expiry import router as expiry_router

app = FastAPI(
    title="WasteWise AI API",
    description="Food inventory and expiry-risk prediction API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(expiry_router)
app.include_router(dashboard_router)


@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "application": "WasteWise AI",
    }