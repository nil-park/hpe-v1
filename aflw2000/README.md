# Head Pose Estimation Benchmark for AFLW2000 Dataset

레퍼런스: https://paperswithcode.com/sota/head-pose-estimation-on-aflw2000

# 개발 환경
```
pip install opencv-python scipy numpy

뷰어용:
pip install pyrender
```

# 사용법 예시
```
from benchmark import AFLW2000
aflw = AFLW2000()

# preprocessing
def preprocessor(images):
    return images / 128 - 1
aflw.preprocess(preprocessor)

# prediction
results = []
for x in aflw.processed:
    y = model.predict(x)
    results.append(y)
results = np.array(results)

# calculate MAEs
yaw, pitch, roll, mean = aflw.rotation_mae_in_degree(results)
```
