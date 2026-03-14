import os
import uuid
import pandas as pd

from app.agents.schema_agent import detect_schema
from app.agents.cleaning_agent import basic_clean
from app.agents.eda_agent import run_eda
from app.agents.model_agent import train_and_evaluate
from app.agents.insight_agent import generate_insight
from app.agents.feature_plot_agent import plot_feature_importance
from app.agents.report_download_agent import build_text_report
from app.rag.build_knowledge import build_knowledge_file
from app.rag.chunker import build_chunks_file
from app.rag.chat_agent import answer_question
from app.rag.vector_store import build_faiss_index


from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


UPLOAD_DIR = "app/storage/uploads"
OUTPUT_DIR = "app/storage/outputs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI()

app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
templates = Jinja2Templates(directory="app/templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "step": 1})


@app.post("/upload", response_class=HTMLResponse)
async def upload_csv(request: Request, file: UploadFile = File(...), step: int = 2):
    if not file.filename.endswith(".csv"):
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "Please upload a CSV file"}
        )

    file_id = str(uuid.uuid4())
    path = os.path.join(UPLOAD_DIR, f"{file_id}.csv")

    contents = await file.read()
    with open(path, "wb") as f:
        f.write(contents)

    df = pd.read_csv(path)
    columns = df.columns.tolist()

    return templates.TemplateResponse(
        "configure.html",
        {
            "request": request,
            "file_id": file_id,
            "columns": columns,
            "step": step
        }
    )


@app.post("/analyze", response_class=HTMLResponse)
def analyze(
    request: Request,
    file_id: str = Form(...),
    target: str = Form(...),
    model_choice: str = Form("rf"),
    step: int = 3
):
    csv_path = os.path.join(UPLOAD_DIR, f"{file_id}.csv")
    df = pd.read_csv(csv_path)

    # Agent 1: schema
    schema = detect_schema(df)

    # Agent 2: cleaning
    cleaned_df, cleaning_report = basic_clean(df, target)

    # Agent 3: EDA
    plots = run_eda(cleaned_df, target, file_id)

    # Agent 4: model
    model_info = train_and_evaluate(cleaned_df, target, model_choice)

    # Agent 5: insight
    insight = generate_insight(model_info)

    # Feature importance chart
    feature_importance = model_info.get("feature_importance", [])
    fi_plot_path = plot_feature_importance(feature_importance, file_id)

    # Build downloadable report text
    report_text = build_text_report(schema, cleaning_report, model_info, insight, target)

    # Build RAG knowledge file
    knowledge_path = build_knowledge_file(
        file_id,
        schema,
        cleaning_report,
        model_info,
        feature_importance,
        insight
    )
    print("Knowledge file created:", knowledge_path)

    chunks_path = build_chunks_file(file_id, knowledge_path)
    print("Chunks file created:", chunks_path)

    faiss_path = build_faiss_index(file_id, chunks_path)
    print("FAISS index created:", faiss_path)

    # Save report
    report_path = os.path.join(OUTPUT_DIR, file_id, "report.txt")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "file_id": file_id,
            "target": target,
            "schema": schema,
            "cleaning_report": cleaning_report,
            "plots": plots,
            "model_info": model_info,
            "insight": insight,
            "fi_plot_available": fi_plot_path is not None,
            "report_ready": True,
            "model_choice": model_choice,
            "step": step
        }
    )

@app.post("/chat")
def chat_with_dataset(
    file_id: str = Form(...),
    question: str = Form(...)
):
    result = answer_question(file_id, question)

    return JSONResponse({
        "question": question,
        "answer": result["answer"],
        "sources": result["sources"]
    })


@app.get("/download_report/{file_id}", response_class=PlainTextResponse)
def download_report(file_id: str):
    report_path = os.path.join(OUTPUT_DIR, file_id, "report.txt")
    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()

    return PlainTextResponse(
        content,
        headers={"Content-Disposition": f"attachment; filename=report_{file_id}.txt"}
    )
