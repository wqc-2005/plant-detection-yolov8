import io
import base64
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pathlib import Path

from .inference import predict, predict_with_image
from . import database

app = FastAPI(title="植物类别识别系统")

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# 启动时初始化数据库
@app.on_event("startup")
def startup():
    database.init_db()


def _get_session_id(request: Request, response: Response) -> str:
    """获取或创建 session_id（基于 Cookie）。"""
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = uuid.uuid4().hex[:16]
        response.set_cookie("session_id", session_id, max_age=30 * 24 * 3600)
    return session_id


def _save_image(image: Image.Image, prefix: str) -> str:
    """保存图片到磁盘，返回文件名。"""
    filename = f"{prefix}_{uuid.uuid4().hex[:8]}.jpg"
    filepath = UPLOAD_DIR / filename
    image.save(str(filepath), format="JPEG", quality=90)
    return filename


@app.get("/")
async def index():
    return FileResponse(str(BASE_DIR / "static" / "index.html"))


@app.post("/api/predict")
async def predict_plant(request: Request, response: Response, file: UploadFile = File(...), conf: float = 0.25):
    """上传图片，返回检测结果（JSON），有结果时保存到数据库。"""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="请上传图片文件")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")

        detections = predict(image, conf=conf)
        session_id = _get_session_id(request, response)

        # 只有检测到目标时才保存
        if detections:
            original_filename = _save_image(image, "original")
            database.save_record(session_id, original_filename, "", detections)

        return {"detections": detections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict/image")
async def predict_plant_with_image(request: Request, response: Response, file: UploadFile = File(...), conf: float = 0.25):
    """上传图片，返回检测结果 + 标注后的图片，有结果时保存记录。"""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="请上传图片文件")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")

        detections, annotated_img = predict_with_image(image, conf=conf)

        # 只有检测到目标时才保存
        if detections:
            session_id = _get_session_id(request, response)
            original_filename = _save_image(image, "original")
            annotated_filename = _save_image(annotated_img, "annotated")
            database.save_record(session_id, original_filename, annotated_filename, detections)

        buf = io.BytesIO()
        annotated_img.save(buf, format="JPEG", quality=90)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "detections": detections,
            "annotated_image": f"data:image/jpeg;base64,{img_b64}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/history")
async def get_history(request: Request, limit: int = Query(default=20, le=100), page: int = Query(default=1, ge=1)):
    """分页查询当前 session 的检测历史记录。"""
    session_id = request.cookies.get("session_id")
    if not session_id:
        return {"total": 0, "page": page, "limit": limit, "records": []}

    total = database.get_record_count(session_id)
    offset = (page - 1) * limit
    records = database.get_records(session_id, limit=limit, offset=offset)
    return {
        "total": total,
        "page": page,
        "limit": limit,
        "records": records,
    }


@app.get("/api/classes")
async def list_classes():
    """返回模型支持的所有植物类别列表。"""
    from .inference import get_model
    model = get_model()
    return {"classes": model.names}
