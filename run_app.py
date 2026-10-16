"""Launch both local services. Train separately before using this command."""
from pathlib import Path
import subprocess, sys, time, urllib.request, json, os
ROOT=Path(__file__).resolve().parent

def main():
    if not (ROOT/'models/random_forest.joblib').exists():
        raise SystemExit('Train first: python training/train_model.py')
    processes=[]
    try:
        processes.append(subprocess.Popen([sys.executable,'-m','uvicorn','backend.main:app','--host','127.0.0.1','--port','8000'],cwd=ROOT))
        for _ in range(120):
            if processes[0].poll() is not None:raise RuntimeError('Backend exited; check whether port 8000 is occupied')
            try:
                with urllib.request.urlopen('http://127.0.0.1:8000/ready',timeout=1) as response:
                    if json.load(response)['status']=='ready':break
            except Exception:time.sleep(.5)
        else:raise RuntimeError('Backend not ready. Check artifacts and installed dependency versions')
        processes.append(subprocess.Popen([sys.executable,'-m','streamlit','run','frontend/app.py','--server.address','127.0.0.1','--server.port','8501'],cwd=ROOT,env={**os.environ,'SEPSIS_API_URL':'http://127.0.0.1:8000'}))
        print('Dashboard: http://localhost:8501\nAPI docs: http://localhost:8000/docs')
        while all(p.poll() is None for p in processes):time.sleep(.5)
    except KeyboardInterrupt:pass
    finally:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill()

if __name__=='__main__':main()