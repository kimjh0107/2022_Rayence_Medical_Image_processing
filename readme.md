## 2022 Yonsei Univ. Rayence Lung and Heart Segmentation

<img width="612" alt="image" src="https://github.com/kimjh0107/2022_Rayence_Medical_Image_processing/assets/83206535/d5744bae-2f80-43a5-82d9-77aa5b5d279d">


## 0. Setting Environments
python version : 3.9.7
```
pip install -r requirements.txt
```

## 1. Train Directory에 있는 데이터를 분할합니다.
- rayence에서 제공한 validation dataset을 train, valid로 8:2로 나누어 저장합니다.
- 이 때 Mask에 있는 값들을 Lung, Heart, Background로 쪼개어 저장합니다.
- Configure.yaml 파일 수정 필요
    - data_root_path : 데이터가 저장되어 루트 경로 (없으면 .) EX ) data
    - raw_data_name  : 원본 데이터가 저장되어있는 경로 EX ) raw_data
    - new_data_name  : 새로 데이터를 저장할 경로 EX) new_data
    - train_ratio    : Train : Valid를 나눌 비율
    - random_seed    : Train-Valid를 나누어 저장할 때의 랜덤 시드

```python
python split_data.py
```

## 2. 주어진 Task를 학습하도록 합니다.
- Image Size : [256, 512, 1024]
- Model : [UNet, Unet2Plus, Unet3Plu]
- Image Size x Model의 조합으로 총 9개의 결과를 냅니다.
```python3
python train_model.py
```

- Result
    - UNet
        - IMG_이미지사이즈_BS_배치크기
            - Prediction : Dice Score가 가장 높을 때의 Weight를 이용해 Validation 이미지를 예측한 결과
            - LOG.txt : 학습 로그
            - Model_Weight.pth : 모델의 Weight
    - UNet2Plus
    - UNet3Plus

## 3. Test Data를 예측합니다.
- Configure.yaml 파일 수정 필요
    - inference_model : 확인해보고싶은 모델
    - inference_img_size : 확인해보고싶은 이미지 크기

```python3
python inference.py
```

- Infernce
    - 모델 명
        - IMG_이미지사이즈
            - 결과 1
            - 결과 2
            ...
---

### Model Information
<img src = 'https://velog.velcdn.com/images/kbm970709/post/d10c28a8-104b-4a1b-9577-280737b52651/image.png' width = '500px'>

**Information for UNet, UNet2Plus and UNet3Plus**

[[2015] U-Net: Convolutional Networks for Biomedical Image Segmentation (MICCAI)](https://arxiv.org/pdf/1505.04597.pdf)

[[2017] Road Extraction by Deep Residual U-Net](https://arxiv.org/pdf/1711.10684v1.pdf)

[[2018] UNet++: A Nested U-Net Architecture for Medical Image Segmentation (MICCAI)](https://arxiv.org/pdf/1807.10165.pdf)

[[2019] Bi-Directional ConvLSTM U-Net with Densley Connected Convolutions](https://arxiv.org/pdf/1909.00166v1.pdf)

[[2020] UNET 3+: A Full-Scale Connected UNet for Medical Image Segmentation (ICASSP 2020)](https://arxiv.org/pdf/2004.08790.pdf)


THX to 
- https://github.com/avBuffer/UNet3plus_pth
- https://github.com/mmheydari97/BCDU-Net
- https://github.com/rishikksh20/ResUnet



### Model Result
**Resized (1024 px, 1024 px) + UNet3Plus**

- Image 0152.png (Resized Input, True Label, Predicted)
<img src = 'Inference/UNet3Plus/IMG_1024/0152.png' width = '500px'>

- Image 0161.png (Resized Input, True Label, Predicted)
<img src = 'Inference/UNet3Plus/IMG_1024/0161.png' width = '500px'>
