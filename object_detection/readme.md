
### Datasets preparation

- run `split_data.py` to split the dataset into training, validation, and test sets. Make sure to set the correct `SOURCE_DIR` in the script.

```python

python split_data.py
```

### Training YOLO

```python

from ultralytics import YOLO
# Load a model
model = YOLO("yolo11m.pt")  # load a pretrained model (recommended for training)
# Train using the single most idle GPU
results = model.train(data="coco8.yaml", epochs=300, imgsz=640, device=-1)

```
