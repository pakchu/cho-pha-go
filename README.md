alphago zero 에서 영감을 받아 강화 학습으로 학습된 모델과 바둑을 둘 수 있는 프로그램입니다.

`python12.5` 환경에서 테스트 되었습니다. 

`python main.py --train -b 5 --num-iterations 5000 --games-per-iteration 10 --num-simulations 200 --batch-size 16 --epochs 100 --capacity 1000 --ai-vs-ai -lr 1e-4 -t 1 --exploration 0.5 --network-trust 0.1`로 적당한 능력의 에이전트를 학습시켜 저장할 수 있습니다.

`python main.py --play -b 5`로 pretrained 5x5 모델과 바둑을 둘 수 있습니다.

`python validate.py` 로 기본적인 사활 문제의 통과율을 확인할 수 있습니다.
2025-03-30 모델 기준 통과율 99%입니다.

`python vs_random_bot_validate.py (--ai-black)` 로 랜덤으로 합법 착수를 두는 봇과 겨뤄 승률을 구할 수 있습니다.
2025-03-30 모델 기준 승률 95 ~ 98%입니다.

![](https://github.com/pakchu/cho-pha-go/blob/main/learning_curve_5x5.png?raw=true)