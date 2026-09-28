"""Build and execute the analytics notebook from analysis.py's percent cells."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parent
cells = []
for section in (root / 'analysis.py').read_text(encoding='utf-8').split('# %%')[1:]:
    header, _, body = section.partition('\n')
    if '[markdown]' in header:
        content = '\n'.join(line[2:] if line.startswith('# ') else '' if line == '#' else line
                            for line in body.strip().splitlines())
        cells.append(nbformat.v4.new_markdown_cell(content))
    else:
        cells.append(nbformat.v4.new_code_cell(body.strip()))
notebook = nbformat.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
})
NotebookClient(notebook, timeout=180, kernel_name='python3',
               resources={'metadata': {'path': str(root)}}).execute()
# Reference committed figures rather than duplicating large raster payloads in JSON.
# Tables and metrics retain their executed outputs.
figures = iter(sorted((root / 'analysis' / 'figures').glob('*.png')))
compact_cells = []
for cell in notebook.cells:
    compact_cells.append(cell)
    if cell.cell_type != 'code':
        continue
    outputs = []
    for output in cell.outputs:
        if 'image/png' in output.get('data', {}):
            figure = next(figures)
            compact_cells.append(nbformat.v4.new_markdown_cell(
                f'![{figure.stem.replace("_", " ")}]({figure.relative_to(root).as_posix()})'))
        else:
            outputs.append(output)
    cell.outputs = outputs
notebook.cells = compact_cells
nbformat.validate(notebook)
nbformat.write(notebook, root / 'PremierLeagueAnalysis.ipynb')
print('Executed and saved PremierLeagueAnalysis.ipynb')
