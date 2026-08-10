import argparse
from pathlib import Path

from mflux.cli.defaults import defaults as ui_defaults
from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.flux.variants.in_context.utils.in_context_loras import LORA_NAME_MAP


class CompletionGenerator:
    def __init__(self):
        self.commands = [
            "mflux-generate",
            "mflux-generate-flux2",
            "mflux-generate-flux2-edit",
            "mflux-generate-controlnet",
            "mflux-generate-kontext",
            "mflux-generate-in-context",
            "mflux-generate-in-context-edit",
            "mflux-generate-in-context-catvton",
            "mflux-generate-fill",
            "mflux-generate-depth",
            "mflux-generate-redux",
            "mflux-generate-qwen",
            "mflux-generate-qwen-edit",
            "mflux-generate-fibo",
            "mflux-generate-fibo-edit",
            "mflux-generate-z-image",
            "mflux-generate-z-image-turbo",
            "mflux-generate-krea2",
            "mflux-refine-fibo",
            "mflux-inspire-fibo",
            "mflux-concept",
            "mflux-concept-from-image",
            "mflux-save",
            "mflux-save-depth",
            "mflux-train",
            "mflux-upscale-controlnet",
            "mflux-upscale-seedvr2",
            "mflux-lora-library",
            "mflux-info",
            "mflux-completions",
        ]

    def create_parser_for_command(self, command: str) -> CommandLineParser:
        parser = CommandLineParser(prog=command, add_help=False)

        if command == "mflux-generate":
            parser.add_general_arguments()
            parser.add_model_arguments(require_model_arg=False)
            parser.add_lora_arguments()
            parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
            parser.add_image_to_image_arguments()
            parser.add_image_outpaint_arguments()
            parser.add_output_arguments()

        elif command == "mflux-generate-flux2":
            parser.add_general_arguments()
            parser.add_model_arguments(require_model_arg=False)
            parser.add_lora_arguments()
            parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
            parser.add_output_arguments()

        elif command == "mflux-generate-flux2-edit":
            parser.add_general_arguments()
            parser.add_model_arguments(require_model_arg=False)
            parser.add_lora_arguments()
            parser.add_argument(
                "--image-paths",
                type=Path,
                nargs="+",
                required=True,
                help="Local paths to one or more init images. For single image editing, provide one path. For multiple image editing, provide multiple paths.",
            )
            parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
            parser.add_output_arguments()

        elif command == "mflux-generate-controlnet":
            parser.add_general_arguments()
            parser.add_model_arguments(require_model_arg=True)
            parser.add_lora_arguments()
            parser.add_controlnet_arguments()
            parser.add_image_generator_arguments()
            parser.add_output_arguments()

        elif command == "mflux-generate-kontext":
            parser.add_general_arguments()
            parser.add_model_arguments(require_model_arg=False)
            parser.add_lora_arguments()
            parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
            parser.add_image_to_image_arguments(required=True)
            parser.add_output_arguments()

        elif command == "mflux-generate-in-context":
            parser.add_general_arguments()
            parser.add_model_arguments(require_model_arg=False)
            parser.add_image_generator_arguments()
            parser.add_image_to_image_arguments(required=True)
            parser.add_in_context_arguments()
            parser.add_output_arguments()

        elif command == "mflux-generate-in-context-edit":
            parser.add_general_arguments()
            parser.add_model_arguments(require_model_arg=False)
            parser.add_lora_arguments()
            parser.add_image_generator_arguments(supports_metadata_config=False, require_prompt=False)
            parser.add_in_context_edit_arguments()
            parser.add_in_context_arguments()
            parser.add_output_arguments()

        elif command == "mflux-generate-in-context-catvton":
            parser.add_general_arguments()
            parser.add_model_arguments(require_model_arg=False)
            parser.add_lora_arguments()
            parser.add_image_generator_arguments(supports_metadata_config=False, require_prompt=False)
            parser.add_catvton_arguments()
            parser.add_in_context_arguments()
            parser.add_output_arguments()

        elif command == "mflux-generate-fill":
            parser.add_general_arguments()
            parser.add_model_arguments(require_model_arg=False)
            parser.add_lora_arguments()
            parser.add_image_generator_arguments()
            parser.add_fill_arguments()
            parser.add_output_arguments()

        elif command == "mflux-generate-depth":
            parser.add_general_arguments()
            parser.add_model_arguments(require_model_arg=False)
            parser.add_lora_arguments()
            parser.add_image_generator_arguments()
            parser.add_depth_arguments()
            parser.add_output_arguments()

        elif command == "mflux-generate-redux":
            parser.add_general_arguments()
            parser.add_model_arguments(require_model_arg=False)
            parser.add_lora_arguments()
            parser.add_image_generator_arguments()
            parser.add_redux_arguments()
            parser.add_output_arguments()

        elif command == "mflux-generate-qwen":
            parser.add_general_arguments()
            parser.add_model_arguments(require_model_arg=False)
            parser.add_lora_arguments()
            parser.add_image_generator_arguments(supports_metadata_config=True)
            parser.add_image_to_image_arguments()
            parser.add_output_arguments()

        elif command == "mflux-generate-qwen-edit":
            parser.add_general_arguments()
            parser.add_model_arguments(require_model_arg=False)
            parser.add_lora_arguments()
            parser.add_image_generator_arguments(supports_metadata_config=True)
            parser.add_argument("--image-paths", type=Path, nargs="+", required=True, help="Local paths to init images")
            parser.add_output_arguments()

        elif command == "mflux-generate-fibo":
            parser.add_general_arguments()
            parser.add_model_arguments(require_model_arg=False)
            parser.add_lora_arguments()
            parser.add_image_generator_arguments(supports_metadata_config=True)
            parser.add_image_to_image_arguments()
            parser.add_output_arguments()

        elif command == "mflux-generate-fibo-edit":
            parser.add_general_arguments()
            parser.add_model_arguments(require_model_arg=False)
            parser.set_defaults(model="fibo-edit")
            parser.add_lora_arguments()
            parser.add_image_generator_arguments(
                supports_metadata_config=True,
                require_prompt=False,
                supports_dimension_scale_factor=True,
            )
            parser.add_argument(
                "--image-path", type=Path, required=False, help="Local path to source image for editing."
            )
            parser.add_argument(
                "--mask-path", type=Path, default=None, help="Optional mask image path for localized edits."
            )
            parser.add_output_arguments()

        elif command == "mflux-generate-z-image-turbo":
            parser.add_general_arguments()
            parser.add_model_arguments(require_model_arg=False)
            parser.add_lora_arguments()
            parser.add_image_generator_arguments(supports_metadata_config=True)
            parser.add_image_to_image_arguments()
            parser.add_output_arguments()

        elif command == "mflux-generate-krea2":
            parser.add_general_arguments()
            parser.add_model_arguments(require_model_arg=False)
            parser.add_lora_arguments()
            parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
            parser.add_image_to_image_arguments(required=False)
            parser.add_output_arguments()

        elif command == "mflux-refine-fibo":
            parser.add_argument("--prompt-file", type=Path, required=True, help="Path to JSON prompt file to refine")
            parser.add_argument("--instructions", type=str, required=True, help="Text instructions for refinement")
            parser.add_argument("--output", type=Path, help="Output path for refined JSON prompt")
            parser.add_argument("--path", type=str, help="Local path for VLM model")
            parser.add_argument("--top-p", type=float, help="Top-p sampling for VLM")
            parser.add_argument("--temperature", type=float, help="Temperature for VLM")
            parser.add_argument("--max-tokens", type=int, help="Max tokens for VLM generation")
            parser.add_argument("--seed", type=int, help="Seed for VLM generation")

        elif command == "mflux-inspire-fibo":
            parser.add_argument("--image-path", type=Path, required=True, help="Path to image file to inspire from")
            parser.add_argument("--prompt", type=str, help="Optional text prompt to blend with the image")
            parser.add_argument("--output", type=Path, help="Output path for generated JSON prompt")
            parser.add_argument("--path", type=str, help="Local path for VLM model")
            parser.add_argument("--top-p", type=float, help="Top-p sampling for VLM")
            parser.add_argument("--temperature", type=float, help="Temperature for VLM")
            parser.add_argument("--max-tokens", type=int, help="Max tokens for VLM generation")
            parser.add_argument("--seed", type=int, help="Seed for VLM generation")

        elif command == "mflux-concept":
            parser.add_general_arguments()
            parser.add_model_arguments(require_model_arg=True)
            parser.add_concept_attention_arguments()
            parser.add_image_generator_arguments()
            parser.add_output_arguments()

        elif command == "mflux-concept-from-image":
            parser.add_general_arguments()
            parser.add_model_arguments(require_model_arg=True)
            parser.add_concept_from_image_arguments()
            parser.add_image_generator_arguments()
            parser.add_output_arguments()

        elif command == "mflux-save":
            parser.add_model_arguments(path_type="save", require_model_arg=True)

        elif command == "mflux-save-depth":
            parser.add_save_depth_arguments()
            parser.add_output_arguments()

        elif command == "mflux-train":
            parser.add_model_arguments(require_model_arg=False)
            parser.add_training_arguments()

        elif command == "mflux-upscale-controlnet":
            parser.add_general_arguments()
            parser.add_model_arguments(require_model_arg=False)
            parser.add_lora_arguments()
            parser.add_lora_arguments()
            parser.add_image_generator_arguments(supports_metadata_config=False, supports_dimension_scale_factor=True)
            parser.add_controlnet_arguments(require_image=True)
            parser.add_output_arguments()

        elif command == "mflux-upscale-seedvr2":
            parser.add_general_arguments()
            parser.add_seedvr2_upscale_arguments()
            parser.add_output_arguments()

        elif command == "mflux-lora-library":
            # Special case: has subcommands
            subparsers = parser.add_subparsers(dest="command")
            list_parser = subparsers.add_parser("list")
            list_parser.add_argument(
                "--paths", nargs="+", type=Path, help="Override LORA_LIBRARY_PATH with these directories"
            )

        elif command == "mflux-info":
            parser.add_info_arguments()

        elif command == "mflux-completions":
            parser.add_argument("--print", action="store_true", help="Print completion script to stdout")
            parser.add_argument("--dir", type=Path, help="Custom installation directory")
            parser.add_argument("--update", action="store_true", help="Update existing installation")
            parser.add_argument("--check", action="store_true", help="Check installation status")

        return parser

    def escape_description(self, desc: str) -> str:
        if not desc:
            return ""
        # Escape brackets, quotes, and other special characters
        desc = desc.replace("\\", "\\\\")  # Escape backslashes first
        desc = desc.replace("[", "\\[").replace("]", "\\]")
        desc = desc.replace("'", "")  # Remove single quotes entirely
        desc = desc.replace('"', '\\"')
        desc = desc.replace("`", "\\`")
        desc = desc.replace("$", "\\$")
        return desc

    def format_argument_spec(self, action: argparse.Action) -> list[str]:
        specs = []

        # Handle subparsers (simplified: treat subcommand as positional and add all sub-options)
        if isinstance(action, argparse._SubParsersAction):
            choices = " ".join(action.choices.keys())
            dest = action.dest or "subcommand"
            specs.append(f"'1:{dest}:({choices})'")
            for subparser in action.choices.values():
                for subaction in subparser._actions:
                    if isinstance(subaction, argparse._HelpAction):
                        continue
                    specs.extend(self.format_argument_spec(subaction))
            return specs

        # Handle options with both short and long forms
        if action.option_strings:
            opts = action.option_strings
            desc = self.escape_description(action.help or "")

            # For store_true/store_false actions, no value spec needed
            is_flag = isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction))

            if len(opts) == 2:
                # Both short and long form - create exclusion group
                opt_spec = f"'({opts[0]} {opts[1]})'{{{opts[0]},{opts[1]}}}"
            else:
                # Only one form
                opt_spec = f"'{opts[0]}'"

            opt_spec += f"'[{desc}]"

            # Add value spec for non-flag arguments
            if not is_flag and (action.type or action.choices or action.nargs):
                opt_spec += self.get_value_spec(action)

            opt_spec += "'"
            specs.append(opt_spec)

        # Handle positional arguments
        else:
            desc = self.escape_description(action.help or action.dest)
            value_spec = self.get_value_spec(action)
            # If get_value_spec already provides a name, use it
            if value_spec.startswith(":"):
                specs.append(f"':{desc}{value_spec}'")
            else:
                specs.append(f"':{desc}:{value_spec}'")

        return specs

    def get_value_spec(self, action: argparse.Action) -> str:
        # Check for special cases first
        if action.dest == "model":
            return ":model:_mflux_models"
        elif action.dest == "quantize":
            return ":quantization:_mflux_quantize"
        elif action.dest == "lora_style":
            return ":style:_mflux_lora_styles"

        if action.choices:
            # Fixed choices
            choices = " ".join(str(c) for c in action.choices)
            return f":value:({choices})"

        elif action.type == Path or (action.dest and ("path" in action.dest or "file" in action.dest)):
            # File completion
            return ":file:_files"

        elif action.type is int:
            # Integer value
            return ":number:"

        elif action.type is float:
            # Float value
            return ":float:"

        elif callable(action.type):
            # Callable type (like lambda) - usually returns int or float
            # Check the dest name for hints
            if "percentage" in (action.dest or "") or "percent" in (action.dest or ""):
                return ":percentage:(1 5 10 15 20 25 30 40 50 60 70 80 90 95 99)"
            return ":value:"

        elif action.nargs in ["*", "+"]:
            # Multiple values
            if action.type == Path or (action.dest and ("path" in action.dest)):
                return ":files:_files"
            return ":values:"

        elif action.type is str or action.type is None:
            # String value
            return f":{action.dest or 'value'}:"

        return ""

    def generate_command_function(self, command: str, parser: CommandLineParser) -> str:
        func_name = command.replace("-", "_")
        lines = [f"_{func_name}() {{"]
        lines.append("    local -a args")
        lines.append("    args=(")

        # Extract all actions from the parser
        for action in parser._actions:
            if isinstance(action, argparse._HelpAction):
                continue
            arg_specs = self.format_argument_spec(action)
            lines.extend(f"        {spec}" for spec in arg_specs)

        lines.append("    )")
        lines.append("    _arguments -s -S $args")
        lines.append("}")
        lines.append("")

        return "\n".join(lines)

    def generate_header(self) -> str:
        commands = " ".join(self.commands)
        return f"""#compdef {commands}
# ZSH completion for mflux commands
# Generated by mflux-completions

"""

    def generate_helper_functions(self) -> str:
        helpers = []

        # Model completion helper
        model_choices = " ".join(f"'{m}[{m} model]'" for m in ui_defaults.MODEL_CHOICES)
        helpers.append(f"""_mflux_models() {{
    _values 'model' \\
        {model_choices} \\
        '*:Hugging Face repo:(org/model)'
}}
""")

        # LoRA style completion helper
        if LORA_NAME_MAP:
            lora_choices = " ".join(f"'{k}[{v}]'" for k, v in LORA_NAME_MAP.items())
            helpers.append(f"""_mflux_lora_styles() {{
    _values 'lora style' \\
        {lora_choices}
}}
""")

        # Quantization choices
        quant_choices = " ".join(f"'{q}[{q}-bit quantization]'" for q in ui_defaults.QUANTIZE_CHOICES)
        helpers.append(f"""_mflux_quantize() {{
    _values 'quantization' \\
        {quant_choices}
}}
""")

        return "\n".join(helpers)

    def generate_main_function(self) -> str:
        lines = ["# Main completion dispatcher", "_mflux() {", "    local cmd=$words[1]", "    case $cmd in"]

        for command in self.commands:
            func_name = command.replace("-", "_")
            lines.append(f"        {command})")
            lines.append(f"            _{func_name}")
            lines.append("            ;;")

        lines.extend(["        *)", "            _default", "            ;;", "    esac", "}", "", "_mflux"])

        return "\n".join(lines)

    def generate(self) -> str:
        script = [self.generate_header()]
        script.append(self.generate_helper_functions())

        # Generate completion functions for each command
        for command in self.commands:
            parser = self.create_parser_for_command(command)
            script.append(self.generate_command_function(command, parser))

        script.append(self.generate_main_function())

        return "\n".join(script)


if __name__ == "__main__":
    # Test the generator by printing out the completion function to stdout
    generator = CompletionGenerator()
    print(generator.generate())
