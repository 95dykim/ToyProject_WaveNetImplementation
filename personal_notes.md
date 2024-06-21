# Audio Synthesis - WaveNet 구현과 학습

[Github Link](https://github.com/95dykim/TF_WaveNet)

## 모델 구현

모델 구현은 별로 어렵지 않았다. 하지만 하이퍼패러미터가 문제였다.

데이터셋이 다른 경우 dilation limit을 몇으로 설정해야 하나? 채널 사이즈는 몇으로 잡나? 마지막 두 레이어의 채널은 몇이 되어야 하나? 훈련은 또 어떻게 하지? 이러한 의문은 논문을 읽어도 해결할 수 없어 시행착오를 겪어야 했다.

최대한 다른 사람의 구현을 참고하지 않고 구현하다 보니 고생을 했지만 그만큼 보람이 있었다. 오디오 생성 모델 전반이 어떻게 돌아가는지 이해가 되는 느낌이다.

### a. Skip Connection

Skip connection 채널 사이즈의 경우 max_n으로 나가기 전 채널이 극단적으로 작으면 예상대로 성능에 문제가 느껴졌다.
또 더하는 대신 DenseNet과 같이 skip connetions를 쌓는 방법으로도 접근해보았는데 비대해지는 패러미터 대비 성능 이득은 없다고 느껴졌다.

### c. Dilation Limit 및 Input Size

처음에는 입력 크기로 그냥 적당한 수를 넣었지만 이후에는 dilation으로 context가 어디까지 주어지는지를 계산하고, 이 값을 기반으로 입력 값을 설정했다.

Dilation limit의 경우 개인 컴퓨터로 실험하다보니 자세한 실험은 불가능했기에 논문의 값을 따랐다.

### d. Normalization

학습에 앞서 분절되지 않은 상태의 음원들을 normalize 하는 경우 생성된 음원들이 덜 만족스러웠다.
조용해야 할 음원이 normalize로 인해 음량이 커지는 케이스가 발생하는 것으로 추측된다.

## 모델 학습

Global Conditional WaveNet 역시 구현했지만, 모든 학습은 Unconditional WaveNet으로 진행하였다.

학습 초기에도 어느정도는 작동하는 듯 보이는 이미지 분류 태스크와는 달리 초기 epoch에서는 노이즈만을 생성하다보니, 훈련을 중단하고 구현에 오류가 없는지 확인하는 과정을 너무 자주 거쳤다. 혹시라도 이 글을 읽고 WaveNet을 훈련하시는 분이 계신다면 epoch을 길게 잡고 오래 기다리는 것을 추천드립니다!

개인 PC로 학습하다보니 리소스의 제약이 있어 대규모 데이터셋은 사용할 수 없었다. 정확히 말하면 사용할 수는 있지만 답답했을 것이다. 하지만 spoken_digit이나 gtzan등 여러 데이터셋을 둘러보아도 적당하다 느껴지는 데이터셋이 없어서 (전체 음원 대비 소리가 너무 다양하거나, 음원이 너무 짧거나, 음원이 너무 적거나 등등) groove 데이터셋으로 한번 시도해보고, 그 다음으로는 개인적으로 좋아하는 뮤지션인 Jeff Rosenstock이 몸담았던 밴드들의 앨범으로 학습을 했다.

총 길이는 8시간이 안되어서 불안했지만, 그래도 데이터간의 통일성은 있으리라 판단했다. 어쨌든 곡간 장르가 유사하고, Jeff Rosenstock이라는 공통점이 있으니 말이다. 해당 데이터셋은 rosenstock 데이터셋으로 칭할 것이다.

몇일동안 컴퓨터를 켜놓고 팬이 돌아가는 소리를 들으며 잠을 제대로 못잤지만 새로운걸 시도하고 배우는데 이정도는 당연히 감안할 수 있었다. 대학원 다닐 때 이런것들을 좀 더 많이 해봤다면 하는 약간의 후회도 들었다.

## 훈련 결과

### a. groove 데이터셋

[Epoch 10000](https://user-images.githubusercontent.com/115688680/218380283-0825a8e1-5ec8-4833-bb7b-c5bf73402e9f.mp4)

[Epoch 10000](https://user-images.githubusercontent.com/115688680/218380290-81c3d0e6-a46e-4d74-a25d-ad41b90f8e3a.mp4)

groove 데이터셋으로 훈련시킨 결과는 그럴듯하지만 다양성이 부족하다고 느껴졌다. 사실 드럼 소리라고 해서 한 소리만 나는 것이 아니다보니 Unconditional WaveNet에서는 이렇게 훈련되는 것이 당연할 것이다. 아마 피아노 소리로 테스트를 해봐야 하지 않았나 싶다.

### b. rosenstock 데이터셋

[Epoch 5000](https://user-images.githubusercontent.com/115688680/218375592-0b791144-1646-40aa-9cb4-5bf8275f2e78.mp4)

[Epoch 35000](https://user-images.githubusercontent.com/115688680/218375736-d7e3e0ab-1d92-4f1e-a696-8523dd4e1578.mp4)

[Epoch 50000](https://user-images.githubusercontent.com/115688680/218379203-4b0c6a44-479f-42eb-8faa-d7723aa07464.mp4)

[Epoch 50000](https://user-images.githubusercontent.com/115688680/218381676-750ed747-98b5-4013-9c8e-d833e8b6e892.mp4)

[비교대상](https://www.youtube.com/watch?v=G9oJBatX0pk)


기대한것 만큼 좋은 결과가 나지 않아서 아쉽다. 악기 소리와 목소리와 비슷한 소리들이 나오지만 잡음이 너무 끼어있다. 모델 훈련에 시간이 너무 오래 걸리고 또 음악이 드럼 소리보다는 더 복잡하다 보니 데이터의 샘플링 레이트를 10000, quantization 비트를 64로 설정했었는데 이것이 오히려 잡음만 내는 결과를 준 것 같다.
또한 음악 데이터 자체가 본질적으로 복잡하기 때문에 비슷한 소리를 학습하기 위해서는 데이터가 지금보다 더 많이 필요했을 것으로 예상된다.

따라서,

1) 대규모의 데이터셋으로 모델을 훈련 후 transfer learning

2) 학습할 데이터 자체도 현재보다 더 많은 상태에서 데이터셋을 unconditional 방식이 아닌 각 악기(및 목소리)에 대한 길이, 음높이 등의 정보를 conditional 하게 학습

3) 음원에 대해 source separation을 수행하고 각 개별 모델은 하나의 악기 또는 목소리만 생성하게 한 후 최종적으로는 여러 생성 모델 결과의 합을 사용

이러한 접근을 했다면 훨씬 더 좋은 결과가 나왔을 것이다.

## 노트

[뒤로가기](./)