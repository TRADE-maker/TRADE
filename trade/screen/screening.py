import shutil

from trade.constants import TARGET_SET_PATH, ORIGINAL_FILE, \
    PHYCHEM_LAYER_PATH, STRUCTURE_LAYER_PATH, RANKING_LAYER_PATH
from trade.screen import filter
from tap import tapify

from pathlib import Path

width = shutil.get_terminal_size().columns


def str_to_bool(
        string: str = None
) -> bool:
    """Convert a string representation of truth to boolean.

    :param string: Input string.
    :return: Boolean value (True or False).
    """
    if string == 'True':
        return True
    else:
        return False


def screening(
        set_path: Path = TARGET_SET_PATH,

        verbose: str = False,
        replicate: str = False
) -> None:
    """Perform drug-like compound screening using multiple filtering layers.

    :param set_path: Path to the training dataset.
    :param verbose: If True, print detailed progress information.
    :param replicate: If True, replicate results for consistency.
    """
    verbose = str_to_bool(verbose)
    replicate = str_to_bool(replicate)
    drug_like_filter = filter.Filter(set_path=set_path, verbose=verbose, replicate=replicate)
    # Define filtering steps with descriptive messages
    filtering_steps = [
        ("Empirical Layer (*-[SH], *-[NH2], *=[S] restriction)",
         lambda: drug_like_filter.add_Emp_layer(ORIGINAL_FILE, '*-[SH]', '*-[NH2]', '*=[S]')),
        (f"PhyChem_layer with model in {Path(*PHYCHEM_LAYER_PATH.parts[-3:])}", drug_like_filter.add_phychem_layer),
        (f"Structure_layer with model in {Path(*STRUCTURE_LAYER_PATH.parts[-3:])}", drug_like_filter.add_structure_layer),
        (f"Ranking_layer with model in {Path(*RANKING_LAYER_PATH.parts[-3:])}", drug_like_filter.add_ranking_layer),
        (f"Clustering_Layer", drug_like_filter.add_Clustering_layer),
    ]

    # Execute filtering steps with verbose output
    if verbose:
        print("─" * width)
    for step_name, step_function in filtering_steps:
        if verbose:
            print(f"Adding {step_name}...")
        step_function()
    if verbose:
        print()

    drug_like_filter.run()


def generate_command_line() -> None:
    """Run generate function from command line."""
    tapify(screening)

