import pandas as pd
import numpy as np
import re

data = 'out/performance_regr_short.csv'
df = pd.read_csv(data)

datasets = ['polution', 'diabetes', 'us_crime', 'wine', 'abalone']
p1, p2, p3 = 0.0, 2.0, np.inf
delta1, delta2, delta3 = 0.0, 0.05, 0.5

# Define the desired order of methods
method_order = [
    'Adv. Kern (ours)',
    'Ridge CV',
    r'Adv. Input ($\ell_2$)',
    r'Adv. Input ($\ell_{\inf}$)',
]

for dset in datasets:
    dff = df[df['dset'] == dset].copy()
    dff.loc[dff['method'] == 'akr', 'method'] = 'Adv. Kern (ours)'
    dff.loc[dff['method'] == 'kr_cv', 'method'] = 'Ridge CV'
    dff.loc[dff['method'] == 'adv-inp-2', 'method'] = r'Adv. Input ($\ell_2$)'
    dff.loc[dff['method'] == 'adv-inp-inf', 'method'] = r'Adv. Input ($\ell_{\inf}$)'

    keep = ['method', 'p', 'radius', 'r2_score']
    dff = dff.iloc[:, 1:][keep]

    dff = dff[dff['p'].isin([p1, p2, p3])]
    dff = dff[dff['radius'].isin([delta1, delta2, delta3])]

    dff = dff[dff['method'] != 'amkl']  # filter out amkl

    # Convert 'method' to categorical to enforce order
    dff['method'] = pd.Categorical(dff['method'], categories=method_order, ordered=True)
    dff = dff.dropna(subset=['method'])  # Drop rows with methods not in method_order

    # Duplicate rows where p == 0, setting p to 2 and np.inf for the duplicates
    p_zero_rows = dff[dff['p'] == 0].copy()
    p_two_rows = p_zero_rows.copy()
    p_two_rows['p'] = 2.0
    p_inf_rows = p_zero_rows.copy()
    p_inf_rows['p'] = np.inf

    dff = pd.concat([dff, p_two_rows, p_inf_rows], ignore_index=True)
    dff = dff[dff['p'] != 0]  # remove rows where p == 0

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
