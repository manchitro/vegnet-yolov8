# %%
from ultralytics import YOLO


def vegnet_val(model_path="yolov8s", split="val"):
    model = YOLO(model_path)
    # model.load(f'{model_name}.pt')
    # print(model)

    # Validate the model
    metrics = model.val(
        split=split, plots=True
    )  # no arguments needed, dataset and settings remembered
    metrics.box.map  # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps  # a list contains map50-95 of each category
