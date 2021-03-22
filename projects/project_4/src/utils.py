import os
import nbformat
from nbconvert import HTMLExporter


def convert_notebook(name, **kwargs):
    cfg = {}
    for key, value in kwargs.items():
        cfg[key] = value

    curdir = os.path.abspath(os.getcwd())
    indir = os.path.join(curdir, 'notebooks')
    outdir, _ = os.path.split(cfg['outdir'])
    os.makedirs(outdir, exist_ok=True)

    config = {
        "ExecutePreprocessor": {"enabled": True},
        "TemplateExporter": {"exclude_output_prompt": True, "exclude_input": True, "exclude_input_prompt": True},
    }

    report_in_path = os.path.join(indir, '{}.ipynb'.format(name))

    nb = nbformat.read(open(report_in_path), as_version=4)
    html_exporter = HTMLExporter(config=config)

    # change dir to notebook dir, to execute notebook
    os.chdir(indir)
    body, resources = (
        html_exporter
        .from_notebook_node(nb)
    )

    # change back to original directory
    os.chdir(curdir)

    report_out_path = os.path.join(outdir, '{}.html'.format(name))

    with open(report_out_path, 'w') as fh:
        fh.write(body)