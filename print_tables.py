import pandas as pd
import numpy as np
import re

data = 'out/performance_regr_short.csv'
df = pd.read_csv(data)

datasets = ['polution', 'diabetes', 'us_crime']
p1, p2 = 2.0, np.inf
delta1, delta2 = 0.01, 0.05

for dset in datasets:
    dff = df[df['dset'] == dset].copy()
    dff.loc[dff['method'] == 'akr', 'method'] = 'Adv Kern'
    dff.loc[dff['method'] == 'kr_cv', 'method'] = 'Ridge CV'

    keep = ['method', 'p', 'radius', 'r2_score']
    dff = dff.iloc[:, 1:][keep]

    dff = dff[dff['p'].isin([p1, p2])]
    dff = dff[dff['radius'].isin([delta1, delta2])]

    # reshape so columns are (p, radius) and the cell value is R^2
    pivot = (
        dff.reset_index()
        .pivot(
            index='method',  # rows
            columns=['p', 'radius'],  # two-level header (p on top, delta underneath)
            values='r2_score',
        )  # cell entries
        .sort_index(axis=1, level=[0, 1])
    )

    # set 'Method' as index
    pivot.columns.names = [r'', r'Method']
    pivot = pivot.rename_axis(None, axis=0)

    # input p and delta values into column names
    pivot.columns = pivot.columns.set_levels(
        [
            pivot.columns.levels[0].map(
                lambda x: fr'$p = \infty$' if x == np.inf else fr'$p = {x:g}$'
            ),
            pivot.columns.levels[1].map(lambda x: fr'$\delta = {x}$'),
        ],
        level=[0, 1],
    )

    # set column format
    col_fmt = 'l|' + '|'.join(
        ['c' * len(pivot.columns.levels[1]) for _ in pivot.columns.levels[0]]
    )

    # create latex table
    latex = pivot.to_latex(
        float_format='%.3f',
        multicolumn=True,
        multicolumn_format='c|',  # centre the grouped headers
        escape=False,
        column_format=col_fmt,
    )

    # knock off the *last* | that sits inside a \multicolumn
    lines = latex.splitlines()
    for i, ln in enumerate(lines):
        if r'\multicolumn' in ln:
            # turn the last "{c|}" on the line into "{c}"
            lines[i] = re.sub(r'(\{c\|})(?!.*\{c\|})', r'{c}', ln)
            break  # only need to patch the first header row

    # drop \toprule and \bottomrule
    lines = [
        ln
        for ln in lines
        if not ln.startswith(r'\toprule') and not ln.startswith(r'\bottomrule')
    ]

    latex = '\n'.join(lines)
    print(f'\nDataset = {dset}:\n{latex}')
