# Pipe Counter

Streamlit app that counts pipes in photos, camera snapshots, a live webcam and videos using your trained YOLO model. It also names each pipe's color and can estimate diameter in mm after calibration.

## Folder structure

```
pipe_counter/
├── app.py                 # main UI
├── requirements.txt
├── .streamlit/config.toml # theme
├── models/                # put your best.pt here
└── utils/
    ├── detector.py        # model loading, inference, drawing
    ├── analysis.py        # color + diameter helpers
    └── inventory.py       # size grouping, colored drawing, godown totals
```

## Run in VS Code

1. Open the `pipe_counter` folder in VS Code.
2. Open a terminal (Ctrl + `) and create a virtual environment:
   - Windows: `python -m venv venv` then `venv\Scripts\activate`
   - Mac/Linux: `python3 -m venv venv` then `source venv/bin/activate`
3. Install packages: `pip install -r requirements.txt`
4. Copy your trained `best.pt` from Colab into the `models/` folder.
5. Start the app: `streamlit run app.py`
6. The browser opens at http://localhost:8501

## Notes

- Live camera only works when the app runs on the computer that has the webcam.
- Diameter in mm needs the calibration in the sidebar: photograph a ruler at the same distance as the pipes, then enter pixels per mm.
- The box-based diameter is approximate. For ±0.1 mm accuracy, use a fixed camera distance and circle fitting on segmented end-faces.
- Color detection uses HSV on the pipe center and works best with even lighting.

## Godown inventory tab (main feature)

1. Type the pipe sizes your godown stocks, e.g. `20, 25, 32, 40, 50`.
2. Add one photo per rack or bay, taken straight on to the pipe ends.
3. Calibrate each photo: open "Calibrate this photo", enter the number of one pipe whose real diameter you know and that diameter in mm. (Or use the sidebar px/mm value.)
4. Read the result: colored pipes on the photo, count per size, and a godown total across all photos.
5. Download the inventory as Excel or CSV.

Pipes that look squashed (cut off by the photo edge or seen at a steep angle) are counted but marked "Uncertain (partial)" instead of getting a guessed size.
