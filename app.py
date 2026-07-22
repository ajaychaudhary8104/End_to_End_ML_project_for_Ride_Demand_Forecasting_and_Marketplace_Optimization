from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

import pandas as pd
import uvicorn
import subprocess

from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
    Request
)

from fastapi.responses import (
    HTMLResponse,
    JSONResponse
)

from fastapi.staticfiles import StaticFiles

from fastapi.middleware.cors import CORSMiddleware

from fastapi.exceptions import (
    RequestValidationError
)

from pydantic import (
    BaseModel,
    ConfigDict
)

from src.ride_demand_forecasting_and_marketplace_optimization import logger

from src.ride_demand_forecasting_and_marketplace_optimization.config.configuration import (
    ConfigurationManager
)

from src.ride_demand_forecasting_and_marketplace_optimization.components.inference import (
    ModelInference
)
from src.ride_demand_forecasting_and_marketplace_optimization.utils.s3_loader import download_artifacts

# ==========================================================
# PATHS
# ==========================================================

ROOT_DIR = Path(__file__).resolve().parent

TEMPLATE_DIR = ROOT_DIR / "templates"
STATIC_DIR = ROOT_DIR / "static"

INDEX_HTML = TEMPLATE_DIR / "index.html"

# ==========================================================
# REQUEST MODELS
# ==========================================================

class ForecastRequest(BaseModel):

    zone_id: int

    timestamp: str


class BatchForecastRequest(BaseModel):

    inputs: List[Dict[str, Any]]


# ==========================================================
# RESPONSE MODELS
# ==========================================================

class ForecastResponse(BaseModel):

    forecast_demand: float

    available_drivers: float

    required_drivers: int

    driver_gap: float

    forecast_supply_ratio: float

    recommended_surge: float

    marketplace_status: str

    shortage_pct: float

    predicted_wait_time: float

    forecast_revenue: float

    risk_score: float

    zone_id: int

    timestamp: str

    success: bool

    model_path: Optional[str] = None


class BatchForecastResponse(BaseModel):

    predictions: List[Dict[str, Any]]

    total_predictions: int

    success: bool


class HealthResponse(BaseModel):

    model_config = ConfigDict(
        protected_namespaces=()
    )

    status: str

    model_loaded: bool

    version: str

    model_path: Optional[str]


# ==========================================================
# INITIALIZATION
# ==========================================================

def initialize_inference():

    config = ConfigurationManager()

    inference_config = (
        config.get_model_inference_config()
    )

    return ModelInference(
        config=inference_config
    )


# ==========================================================
# LIFESPAN
# ==========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    try:
        logger.info("Downloading Artifacts...")
        download_artifacts()
        logger.info("Artifacts Downloaded...")

        logger.info(
            "Initializing Model Inference..."
        )

        subprocess.run(
            [
                "feast",
                "apply"
            ],
            cwd="src/ride_demand_forecasting_and_marketplace_optimization/feature_repo",
            check=True
        )
        
        inference = initialize_inference()

        app.state.inference = inference

        app.state.model_loaded = True

        app.state.model_path = str(
            inference.config.model_path
        )

        logger.info(
            "Inference Ready"
        )

    except Exception as e:

        logger.exception(e)

        app.state.model_loaded = False

        app.state.inference = None

        app.state.model_path = None

    yield


# ==========================================================
# APP
# ==========================================================

app = FastAPI(

    title="Ride Demand Forecasting & Marketplace Optimization API",

    description=(
        "Production-grade Ride Demand Forecasting "
        "and Marketplace Optimization API"
    ),

    version="1.0.0",

    lifespan=lifespan,

    docs_url="/api/docs",

    redoc_url="/api/redoc"
)

# ==========================================================
# MIDDLEWARE
# ==========================================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]
)

# ==========================================================
# STATIC FILES
# ==========================================================

if STATIC_DIR.exists():

    app.mount(
        "/static",
        StaticFiles(directory=str(STATIC_DIR)),
        name="static"
    )

# ==========================================================
# HOME
# ==========================================================

@app.get("/",response_class=HTMLResponse)
async def home():

    if INDEX_HTML.exists():

        return HTMLResponse(
            INDEX_HTML.read_text(
                encoding="utf-8"
            )
        )

    return HTMLResponse(
        """
        <html>
        <head>
            <title>Ride Marketplace Forecasting</title>
        </head>
        <body>
            <h1>Ride Demand Forecasting API</h1>
            <p>API Running Successfully</p>
            <a href="/api/docs">Swagger Docs</a>
        </body>
        </html>
        """
    )


# ==========================================================
# HEALTH
# ==========================================================

@app.get(
    "/health",
    response_model=HealthResponse
)
async def health():

    return HealthResponse(

        status="healthy",

        model_loaded=getattr(
            app.state,
            "model_loaded",
            False
        ),

        version=app.version,

        model_path=getattr(
            app.state,
            "model_path",
            None
        )
    )


@app.get("/health/live")
async def liveness():

    return {
        "status": "alive"
    }


@app.get("/health/ready")
async def readiness():

    if not getattr(
        app.state,
        "model_loaded",
        False
    ):
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    return {
        "status": "ready"
    }


# ==========================================================
# METADATA
# ==========================================================

@app.get("/metadata/zones")
async def zones():

    return {
        "zones": list(range(1, 264))
    }


@app.get("/metadata/marketplace-status")
async def marketplace_status():

    return {
        "status": [
            "Balanced",
            "Driver Shortage",
            "Driver Oversupply",
            "High Surge",
            "Critical Shortage"
        ]
    }


# ==========================================================
# FORECAST
# ==========================================================

@app.post("/forecast", response_model=ForecastResponse)
async def forecast(request: ForecastRequest):

    if not getattr(
        app.state,
        "model_loaded",
        False
    ):
        raise HTTPException(
            status_code=503,
            detail="Model unavailable"
        )

    try:

        payload = [{
            "zone_id": request.zone_id,
            "timestamp": request.timestamp,
            # "zone_timestamp_key": request.zone_timestamp_key
        }]

        results = (
            app.state
            .inference
            .predict_records(payload)
        )

        row = results.iloc[0]

        return ForecastResponse(

            forecast_demand=float(
                row.get(
                    "forecast_demand",
                    0
                )
            ),

            available_drivers=float(
                row.get(
                    "available_drivers",
                    0
                )
            ),

            required_drivers=int(
                row.get(
                    "required_drivers",
                    0
                )
            ),

            driver_gap=float(
                row.get(
                    "driver_gap",
                    0
                )
            ),

            forecast_supply_ratio=float(
                row.get(
                    "forecast_supply_ratio",
                    0
                )
            ),

            recommended_surge=float(
                row.get(
                    "recommended_surge",
                    1.0
                )
            ),

            marketplace_status=str(
                row.get(
                    "marketplace_status",
                    "Unknown"
                )
            ),

            shortage_pct=float(
                row.get(
                    "shortage_pct",
                    0
                )
            ),

            predicted_wait_time=float(
                row.get(
                    "predicted_wait_time",
                    0
                )
            ),

            forecast_revenue=float(
                row.get(
                    "forecast_revenue",
                    0
                )
            ),

            risk_score=float(
                row.get(
                    "risk_score",
                    0
                )
            ),

            zone_id=int(
                row.get(
                    "zone_id",
                    request.zone_id
                )
            ),

            timestamp=str(
                row.get(
                    "timestamp",
                    request.timestamp
                )
            ),

            success=True,

            model_path=getattr(
                app.state,
                "model_path",
                None
            )
        )

    except Exception as e:

        logger.exception(e)

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ==========================================================
# BATCH FORECAST
# ==========================================================

@app.post("/forecast/batch", response_model=BatchForecastResponse)
async def forecast_batch(request: BatchForecastRequest):

    if not getattr(
        app.state,
        "model_loaded",
        False
    ):
        raise HTTPException(
            status_code=503,
            detail="Model unavailable"
        )

    try:

        results = (
            app.state
            .inference
            .predict_records(
                request.inputs
            )
        )

        return BatchForecastResponse(

            predictions=results.to_dict(
                orient="records"
            ),

            total_predictions=len(
                results
            ),

            success=True
        )

    except Exception as e:

        logger.exception(e)

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ==========================================================
# CSV UPLOAD
# ==========================================================

@app.post("/forecast/upload")
async def upload_csv(file: UploadFile = File(...)):

    if not getattr(
        app.state,
        "model_loaded",
        False
    ):
        raise HTTPException(
            status_code=503,
            detail="Model unavailable"
        )

    try:

        df = pd.read_csv(
            file.file
        )

        results = (
            app.state
            .inference
            .predict_records(
                df.to_dict(
                    orient="records"
                )
            )
        )

        return {

            "success": True,

            "rows": len(results),

            "predictions": (
                results.to_dict(
                    orient="records"
                )
            )
        }

    except Exception as e:

        logger.exception(e)

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ==========================================================
# MODEL INFO
# ==========================================================

@app.get("/model/info")
async def model_info():

    return {

        "project":
        "Ride Demand Forecasting & Marketplace Optimization",

        "model_type":
        "XGBoost Regressor",

        "target":
        "ride_requests",

        "feature_store":
        "Feast",

        "online_inference":
        True,

        "version":
        app.version
    }


# ==========================================================
# VALIDATION HANDLER
# ==========================================================

@app.exception_handler(
    RequestValidationError
)
async def validation_handler(
    request: Request,
    exc: RequestValidationError
):

    return JSONResponse(

        status_code=422,

        content={
            "success": False,
            "errors": exc.errors()
        }
    )


# ==========================================================
# GLOBAL EXCEPTION HANDLER
# ==========================================================

@app.exception_handler(Exception)
async def global_handler(
    request: Request,
    exc: Exception
):

    logger.exception(exc)

    return JSONResponse(

        status_code=500,

        content={
            "success": False,
            "detail": "Internal Server Error"
        }
    )


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )