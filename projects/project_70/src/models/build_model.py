import os


def run_shell(shell):
    """
    Calling this function will call the samtools related shell scripts stored under ./sh_scripts
    """
    os.system(f"{shell}")
    return

def run_R(file_name):
    """
    This method will run R file
    """
    os.system(f"Rscript {file_name}")