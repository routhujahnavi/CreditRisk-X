import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api.router import router

app = FastAPI(
    title="CreditRisk-X API",
    description="Backend API for Credit Default Risk Prediction & Threshold Optimization",
    version="1.0.0"
)

# Configure CORS to allow the frontend dashboard to communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins in development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the ML API routes
app.include_router(router)

from fastapi.staticfiles import StaticFiles
import os

# Resolve path to frontend/dist statically and mount it
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
frontend_dist = os.path.join(base_dir, "frontend", "dist")

if os.path.exists(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
