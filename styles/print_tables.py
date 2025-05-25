import pandas as pd
import numpy as np
import re


if __name__ == '__main__':
    data = 'out/performance_regr_short.csv'
    df = pd.read_csv(data)

    datasets = ['polution', 'diabetes', 'us_crime', 'wine', 'abalone']
    ps = [0.0, 2.0, np.inf]

    # Example: map each p to its allowed deltas
    delta_map = {
        0.0: [0.0,],
        2: [0.05, 0.1],
        np.inf: [0.01, 0.05],  # different deltas for p = ∞
    }

    # Flatten to get the full list of (p, delta) combinations to keep
    valid_pairs = set((p, d) for p, ds in delta_map.items() for d in ds)



    for dset in datasets:
        dff = df[df['dset'] == dset].copy()
        dff = dff[dff['method'] != 'amkl']
        dff.loc[dff['method'] == 'adv-inp-inf', 'method'] = 'Adv Input $(\ell_\infty)$'
        dff.loc[dff['method'] == 'adv-inp-2', 'method'] = 'Adv Input $(\ell_2)$'
        dff.loc[dff['method'] == 'akr', 'method'] = 'Adv Kern'
        dff.loc[dff['method'] == 'kr_cv', 'method'] = 'Ridge Kernel'

        keep = ['method', 'p', 'radius', 'r2_score']
        dff = dff.iloc[:, 1:][keep]

        #dff = dff
        #dff = dff[dff['p'].isin(ps)]
        #dff = dff[dff['radius'].isin(deltas)]

        dff = dff[dff.apply(lambda row: (row['p'], row['radius']) in valid_pairs, axis=1)]


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

        txt_attack = r'{\rm attack}'
        pivot.columns = pivot.columns.set_levels(
            [
                pivot.columns.levels[0].map(
                    lambda x: fr'$p = \infty$' if x == np.inf else fr'$p = {x:g}$'
                ),
                pivot.columns.levels[1].map(lambda x: fr'$\delta = {x}$'),
            ],
            level=[0, 1],
        )


        # create latex table
        latex = pivot.to_latex(
            float_format='%.3f',
            escape=False,
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