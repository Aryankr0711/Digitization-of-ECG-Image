import asyncio
from pathlib import Path
from web.backend.services.ecg_pipeline import pipeline_service

async def run_test():
    # Pick a sample image from the dataset
    sample_img = Path("d:/mdm_proj/Digitization-of-ECG-Image/train/1006427285/1006427285-0001.png")
    if not sample_img.exists():
        print("Sample image not found:", sample_img)
        return
            
    out_dir = Path("d:/mdm_proj/Digitization-of-ECG-Image/web/backend/outputs/test_pipeline")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    async def mock_report(stage, idx, total, progress, msg):
        print(f"[{stage}] {progress*100:.0f}%: {msg}")
        
    print(f"Running pipeline on {sample_img}")
    results = await pipeline_service.run_pipeline(sample_img, out_dir, mock_report)
    
    print("\nPipeline finished successfully!")
    print(f"Extracted {len(results['leads'])} leads.")
    print("Metrics generated:", results['metrics']['average_metrics'])
    
if __name__ == "__main__":
    asyncio.run(run_test())
