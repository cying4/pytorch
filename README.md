# pytorch
Use CNN to predict the blood cell type
The raw dataset can be downloaded by:
if "train" not in os.listdir():
    print('A')
    os.system("wget https://storage.googleapis.com/exam-deep-learning/train-Exam2.zip")
    os.system("unzip train-Exam2.zip")
DATA_DIR = os.getcwd() + "/train/"
x = []
y = []
for path in [f for f in os.listdir(DATA_DIR) if f[-4:] == ".png"]:
    x.append(cv2.resize(cv2.imread(DATA_DIR + path), (60, 60)))
    with open(DATA_DIR + path[:-4] + ".txt", "r") as s:
        label = s.read()
        y.append(label)
