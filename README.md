alphago zero 에서 영감을 받아 강화 학습으로 학습된 모델과 바둑을 둘 수 있는 프로그램입니다.

`python12.5` 환경에서 테스트 되었습니다. 

`python main.py --train --board-size 5 --num-iterations 30 --games-per-iteration 3 --num-simulations 100 --batch-size 16 --epochs 100 --capacity 100 `로 적당한 능력의 에이전트를 학습시켜 저장할 수 있습니다.

`python main.py`로 pretrained 5x5 모델과 바둑을 둘 수 있습니다.