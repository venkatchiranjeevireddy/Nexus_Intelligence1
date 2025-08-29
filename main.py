from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.responses import FileResponse
from business_intelligence import run_pipeline  # Import from the module we created
from pydantic import BaseModel
from typing import List
import os

app = FastAPI(
    title="Business Intelligence API",
    description="Multi-agent business intelligence analysis API",
    version="1.0.0"
)



class AnalysisInput(BaseModel):
    task: str
    competitors: List[str]

def verify_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

@app.get("/")
def root():
    """Root endpoint to confirm API is running"""
    return {
        "message": "Business Intelligence API is running",
        "endpoints": [
            "/get_file_json - POST: Run analysis and get results as JSON",
            "/download_file - POST: Run analysis and download report file",
            "/docs - GET: Interactive API documentation"
        ]
    }

@app.post("/get_file_json")
def get_file_json(data: AnalysisInput, api_key: str = Depends(verify_key)):
    """Run business intelligence analysis and return results as JSON"""
    try:
        # Call the pipeline function with the correct parameters
        filename = run_pipeline(data.task, data.competitors)
                
        if not filename or not os.path.exists(filename):
            raise HTTPException(status_code=500, detail="Analysis failed or file not created")
                
        # Read file content
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()
                
        return {
            "status": "success",
            "file_name": filename,
            "content": content,
            "task": data.task,
            "competitors": data.competitors
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/download_file")
def download_file(data: AnalysisInput, api_key: str = Depends(verify_key)):
    """Run business intelligence analysis and return file for download"""
    try:
        # Call the pipeline function with the correct parameters
        filename = run_pipeline(data.task, data.competitors)
                
        if not filename or not os.path.exists(filename):
            raise HTTPException(status_code=500, detail="Analysis failed or file not created")
                
        return FileResponse(
            filename, 
            filename=os.path.basename(filename),
            media_type="text/markdown"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "API is running properly"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)