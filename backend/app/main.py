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

@app.get("/")
def read_root():
    return {
        "message": "Welcome to CreditRisk-X API",
        "docs_url": "/docs",
        "health_url": "/api/health"
    }

if __name__ == "__main__":
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
