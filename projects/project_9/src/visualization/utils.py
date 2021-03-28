from nbconvert import HTMLExporter
import nbformat
import os


def convert_jupyter_to_html(report_in_path, report_destination):

    
    config = {
        "ExecutePrepocessor": {"enabled": True},
        "TemplateExporter": {"exclude_output_prompt": True, "exclude_input_prompt":True, "exclude_input": True},
    }


    nb = nbformat.read(open(report_in_path), as_version=4)
    html_exporter = HTMLExporter(config=config)



    body, resources = (
        html_exporter
        .from_notebook_node(nb)
    )


    with open(report_destination, 'w') as fh:
        fh.write(body)
