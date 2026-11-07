from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.common.resolution.prequantized_resolution import PrequantizedResolution
from mflux.models.common.training.runner import TrainingRunner
from mflux.models.common.training.state.training_spec import TrainingSpec
from mflux.utils.exceptions import StopTrainingException


def main():
    # Training keeps the upstream weights in this release (#938)
    with PrequantizedResolution.disabled():
        _main()


def _main():
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Finetune a LoRA adapter")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_training_arguments()
    args = parser.parse_args()

    config_path = args.config
    resume_path = args.resume
    if config_path is not None and not config_path.exists():
        parser.error(f"Config file not found: {config_path}")
    if resume_path is not None and not resume_path.exists():
        parser.error(f"Checkpoint not found: {resume_path}")

    if args.dry_run:
        TrainingSpec.resolve(
            config_path=str(config_path) if config_path is not None else None,
            resume_path=str(resume_path) if resume_path is not None else None,
            create_output_dir=False,
        )
        print("✅ Training config validated.")
        return

    try:
        TrainingRunner.train(
            config_path=str(config_path) if config_path is not None else None,
            resume_path=str(resume_path) if resume_path is not None else None,
        )
    except StopTrainingException as stop_exc:
        print(stop_exc)


if __name__ == "__main__":
    main()
