from ultralytics import YOLOv10

# Load a model
#model = YOLO('yolov8s.yaml')
#model = YOLO('/content/gdrive/MyDrive/Train_Yolov8/Ketqua_ppe/last_org_yolov8s.pt')
model = YOLOv10('/content/gdrive/MyDrive/TrainYolov10/Ketqua_sintering/train5/weights/last.pt')

# Use the model
results = model.train(data="data1.yaml", project='/content/gdrive/MyDrive/TrainYolov10/Ketqua_sintering', pretrained= "/content/gdrive/MyDrive/TrainYolov10/Ketqua_sintering/train5/weights/last.pt",imgsz=640 , epochs=100 )  # train the model
