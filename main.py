import argparse
import os
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description="Cho Pha Go Main Script")

    # 공통 옵션
    parser.add_argument("--board-size", '-b', type=int, default=5, choices=range(3, 20),
                        help="Set the board size (default=5)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use: 'cpu', 'cuda:0', 'mps' etc.")
    parser.add_argument('--temperature', '-t', type=str, default='1.0', help='Temperature for MCTS.')

    # 플레이 관련 옵션
    parser.add_argument("--play", action="store_true", default=True,
                        help="If set, run player vs AI after optional training.")
    parser.add_argument("--player-black", action="store_true", default=False,
                        help="If set, human player is black (default is white).")
    parser.add_argument("--ai-vs-ai", action="store_true", default=False,
                        help="If set, run AI vs AI game.")

    parser.add_argument('--random-vs-ai', action="store_true", default=False,
                        help="If set, run random vs AI game.")
    
    # 학습 관련 옵션
    parser.add_argument("--train", action="store_true", default=False,
                        help="If set, train the model.")
    parser.add_argument("--num-iterations", type=int, default=10,
                        help="Number of training iterations (default=10)")
    parser.add_argument("--games-per-iteration", type=int, default=2,
                        help="Number of self-play games per iteration (default=2)")
    parser.add_argument("--num-simulations", type=int, default=50,
                        help="Number of MCTS simulations per move (default=50)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for training (default=16)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs per iteration (default=100)")
    parser.add_argument("--capacity", type=int, default=2000,
                        help="Replay buffer capacity (default=2000)")
    parser.add_argument("--train-device", '-d', type=str, default="cpu")
    parser.add_argument("--learning-rate", '-lr', type=float, default=0.001,
                        help="Learning rate for training (default=0.001)")
    parser.add_argument("--save-path", "-s", type=str, default=None,
                        help="Path to save the trained model.")
    parser.add_argument("--exploration", default=1., type=float,
                        help="Exploration constant for MCTS (default=1.)")
    parser.add_argument("--network-trust", default=0.25, type=float,
                        help="Network trust for MCTS (default=0.25)")

    # 모델 경로
    parser.add_argument("--pretrained-model-path", "-p", type=str, default=None,
                        help="Path to a pretrained model (.pt).")

    args = parser.parse_args()
    args.temperature = float(args.temperature)
    if args.pretrained_model_path is not None:
        if not args.pretrained_model_path.startswith('models/'):
            args.pretrained_model_path = f'models/{args.pretrained_model_path}'
        if not args.pretrained_model_path.endswith(f'{args.board_size}x{args.board_size}.pt'):
            args.pretrained_model_path += f'_{args.board_size}x{args.board_size}.pt'
        
    if args.save_path is not None:
        if not args.save_path.startswith('models/'):
            args.save_path = f'models/{args.save_path}'
        if not args.save_path.endswith(f'{args.board_size}x{args.board_size}.pt'):
            args.save_path += f'_{args.board_size}x{args.board_size}.pt'

    os.environ.setdefault('BOARD_SIZE', str(args.board_size))
    # --------------------------------------------------------
    # 1) 학습(train)
    # --------------------------------------------------------
    # train 옵션을 넣었거나, 모델이 없으면 학습을 진행합니다.
    if args.train or f'cho_pha_go_{args.board_size}x{args.board_size}.pt' not in os.listdir('models'):
        import cho_pha_go_train
        # cho_pha_go_train.train에 필요한 인자를 넘겨줍니다.
        cho_pha_go_train.train(
            board_size=args.board_size,
            num_iterations=args.num_iterations,
            games_per_iteration=args.games_per_iteration,
            num_simulations=args.num_simulations,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            capacity=args.capacity,
            device=args.device,
            pretrained_model_path=args.pretrained_model_path,
            save_model_path=args.save_path,
            temperature=args.temperature,
            exploration=args.exploration,
            network_trust=args.network_trust
        )

    # --------------------------------------------------------
    # 2) 플레이(player vs AI)
    # --------------------------------------------------------
    if args.play:
        import interactive_go
        # 플레이 세팅
        # interactive_go 모듈에서 board_size 등을 받을 수 있게 구현했다면 넘겨줍니다
        interactive = interactive_go.InteractiveGo(
            board_size=args.board_size,
        )
        if args.ai_vs_ai: 
            interactive.run_ai_vs_ai(
                model_path=args.pretrained_model_path,
                device=args.device,
                num_simulations=args.num_simulations,
                temperature=args.temperature
            )
        elif args.random_vs_ai:
            interactive.run_random_vs_ai(
                model_path=args.pretrained_model_path,
                device=args.device,
                num_simulations=args.num_simulations,
                temperature=args.temperature
            )
        else:    
            interactive.run_player_vs_ai(
                player_black=args.player_black,
                model_path=args.pretrained_model_path,
                device=args.device,
                num_simulations=args.num_simulations,
                temperature=args.temperature
            )

    # 만약 train이나 play 옵션 둘 다 없으면, 도움말 출력
    if not args.train and not args.play:
        parser.print_help()


if __name__ == "__main__":
    main()


"""
python main.py --train --board-size 5 --num-iterations 30 --games-per-iteration 3 --num-simulations 100 --batch-size 16 --epochs 100 --capacity 100 --device mps
python main.py --train --board-size 5 --num-iterations 50 --games-per-iteration 3 --num-simulations 100 --batch-size 16 --epochs 100 --capacity 1000 --device mps -s super_cho_pha_go
python main.py --train --board-size 5 --num-iterations 200 --games-per-iteration 4  --num-simulations 400 --batch-size 16 --epochs 200 --capacity 1000 --device cpu --pretrained-model-path cho_pha_go
"""
