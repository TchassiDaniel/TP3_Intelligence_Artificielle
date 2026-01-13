# -*- coding: utf-8 -*-
"""
Main entry point for the TP3 project.

This script allows running the different exercises from the command line.

@author: Tchassi Daniel
@matricule: 21P073
"""
import argparse
import sys

def main():
    """
    Parses command-line arguments and runs the selected exercise.
    """
    parser = argparse.ArgumentParser(
        description="Runner for TP3 exercises on CNNs.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--exercise',
        '-e',
        type=int,
        choices=[1, 2, 4],
        required=True,
        help="""
        The exercise number to run:
        1: Basic CNN classification on CIFAR-10.
        2: ResNet classification on CIFAR-10.
        4: Neural Style Transfer.
        """
    )
    # Arguments for style transfer (Exercise 4)
    parser.add_argument('--content', type=str, help='Path to the content image for style transfer.')
    parser.add_argument('--style', type=str, help='Path to the style image for style transfer.')


    args = parser.parse_args()

    if args.exercise == 1:
        print("="*60)
        print("EXECUTING EXERCISE 1: BASIC CNN")
        print("="*60)
        from exercises.exercise_1_cnn import run_exercise_1
        run_exercise_1()

    elif args.exercise == 2:
        print("="*60)
        print("EXECUTING EXERCISE 2: RESNET")
        print("="*60)
        from exercises.exercise_2_resnet import run_exercise_2
        run_exercise_2()

    elif args.exercise == 4:
        print("="*60)
        print("EXECUTING EXERCISE 4: NEURAL STYLE TRANSFER")
        print("="*60)
        from exercises.exercise_4_style_transfer import run_exercise_4
        if not args.content or not args.style:
            print("Error: Please provide paths for --content and --style images for exercise 4.", file=sys.stderr)
            sys.exit(1)
        run_exercise_4(content_path=args.content, style_path=args.style)


if __name__ == '__main__':
    main()
