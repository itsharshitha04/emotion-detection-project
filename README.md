# Facial Emotion Recognition

## Run

Open PowerShell in this folder and run:

```powershell
.\venv\Scripts\Activate.ps1
python main.py
```

The webcam window will show the detected emotion and confidence. Press `q` to
close it.

If the virtual environment has not been set up yet:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe main.py
```

Close other applications using the webcam before starting the program.