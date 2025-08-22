from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from pathlib import Path
import shutil
import re
import time
import os

ROOT = Path(__file__).parent
UPLOADS = ROOT / "uploads"
CLIENT_DIST = ROOT.parent / "client" / "dist"
UPLOADS.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Gyro Perlin Backend")

# If your client runs on a different origin during dev, enable CORS:
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

SAFE_NAME = re.compile(r"[^a-zA-Z0-9_.-]")

@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...)):
    name = file.filename or f"{int(time.time()*1000)}.csv"
    if not name.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are allowed")

    try:
        with name.open("wb") as out:
            shutil.copyfileobj(file.file, out)
    except Exception as e:
        if name.exists():
            name.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse({"url": f"/uploads/{name}", "name": name})

@app.get("/api/files")
async def list_files():
    files = []
    for f in sorted(UPLOADS.iterdir(), key=os.path.getmtime, reverse=True):
        if f.is_file() and f.suffix.lower() == ".csv":
            files.append({
                "name": f.name,
                "url": f"/api/uploads/{f.name}",
                "size": f.stat().st_size,
                "modified": f.stat().st_mtime
            })
    return {"files": files}

# Serve uploads and the built client
app.mount("/api/uploads", StaticFiles(directory=str(UPLOADS)), name="uploads")
if CLIENT_DIST.exists():
    app.mount("/", StaticFiles(directory=str(CLIENT_DIST), html=True), name="static")