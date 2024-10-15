from ultralytics import YOLOv10
# Load a model
#model = YOLO('yolov8s.yaml')
#model = YOLO('/content/gdrive/MyDrive/Train_Yolov8/Ketqua_ppe/last_org_yolov8s.pt')
model = YOLOv10('/content/Yolov10_1_1/yolov10-1.1/ultralytics/cfg/models/v10/yolov10s_SimAM.yaml').load("/content/gdrive/MyDrive/TrainYolov10/last_0_100_WIOU.pt")
# Use the model
results = model.train(data="data1.yaml", project='/content/gdrive/MyDrive/TrainYolov10/Ketqua_sintering', imgsz=640 , epochs=2 )  # train the model