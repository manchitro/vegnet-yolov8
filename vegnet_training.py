from ultralytics import YOLO
import os


def vegnet_training(model_name="yolov8s", exp_name="original_custom_head", resume=False, weights=None):
    if resume and weights:
        model = YOLO(weights)
    else:
        model = YOLO(model_name)
        model.load(f"{model_name}.pt")
    # print(model)

    results = model.train(
        data="/content/datasets/vegnet_yolo.yaml",
        epochs=200,
        batch=32,
        imgsz=256,
        device=0,
        workers=16,
        cache=True,
        # optimizer = 'Adam',
        # freeze=[
        #     "0",
        #     "1",
        #     "2",
        #     "3",
        #     "4",
        #     "5",
        #     "6",
        #     "7",
        #     "8",
        #     "9",
        #     "10",
        #     "11",
        #     "12",
        #     "13",
        #     "14",
        #     "15",
        #     "16",
        #     "17",
        #     "18",
        #     "19",
        #     "20",
        #     "21",
        #     "22.cv2.0.0",
        #     "22.cv2.1.0",
        #     "22.cv2.2.0",
        #     "22.cv2.0.1",
        #     "22.cv2.1.1",
        #     "22.cv2.2.1",
        #     "22.cv3.0.0",
        #     "22.cv3.1.0",
        #     "22.cv3.2.0",
        #     "22.cv3.0.1",
        #     "22.cv3.1.1",
        #     "22.cv3.2.1",
        # ],
        val=True,
        plots=True,
        save=True,
        save_period=-1,
        show=True,
        exist_ok=True,
        name=f"{model_name}_{exp_name}",
        project="/content/drive/MyDrive/vegnet_yolo/out_new",
    )
