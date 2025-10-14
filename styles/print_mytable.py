import pandas as pd
import numpy as np
import re


if __name__ == '__main__':
    data = 'out/performance_regr_short.csv'
    df = pd.read_csv(data)
    data2 = 'out/performance_rebutall2.csv'
    df2 = pd.read_csv(data2)

    df = pd.concat([df, df2], ignore_index=True)

    datasets = ['polution', 'diabetes']
    ps = [0.0, 2.0, np.inf]

    # Example: map each p to its allowed deltas
    delta_map = {
        0.0: [0.0,],
        2: [0.01, 0.1],
        np.inf: [0.01, 0.1],  # different deltas for p = ∞
    }

    # Flatten to get the full list of (p, delta) combinations to keep
    valid_pairs = set((p, d) for p, ds in delta_map.items() for d in ds)



    for dset in datasets:
        dff = df[df['dset'] == dset].copy()

        dff = dff[dff['method'] != 'amkl']

        # enforce desired order using original codes
        method_order = ['akr', 'akr-0.01', 'akr-0.1', 'adv-inp-2', 'adv-inp-inf', 'kr_cv']

        # filter and set categorical order
        dff = dff[dff['method'] != 'amkl']
        dff['method'] = pd.Categorical(dff['method'], categories=method_order, ordered=True)

        # rename after ordering
        rename_map = {
            'akr': 'Adv Kern $\{\|\dx\|_\RKHS\le   \tfrac{1}{\sqrt{n}}\}$',
            'kr_cv': 'Ridge Kernel',
            'akr-0.01': 'Adv Kern $\{\|\dx\|_\RKHS \le 0.01\}$',
            'akr-0.1': 'Adv Kern $\{\|\dx\|_\RKHS \le 0.1\}$',
            'adv-inp-2': 'Adv Input $\{\|\Delta \\x\|_2 \le 0.1\}$',
            'adv-inp-inf': 'Adv Input $\{\|\Delta \\x\|_\infty \le 0.1\}$',
        }
        dff['method'] = dff['method'].map(rename_map)

        #dff = dff
        #dff = dff[dff['p'].isin(ps)]
        #dff = dff[dff['radius'].isin(deltas)]

        dff['formatted'] = dff.apply(
            lambda row: f"{row['r2_score']:.2f} ({row['r2_scoreq1']:.2f}–{row['r2_scoreq3']:.2f})",
            axis=1
        )
        keep = ['method', 'p', 'radius', 'formatted']
        dff = dff.iloc[:, 1:][keep]

        dff = dff[dff.apply(lambda row: (row['p'], row['radius']) in valid_pairs, axis=1)]


        # reshape so columns are (p, radius) and the cell value is R^2
        pivot = (
            dff.reset_index()
            .pivot(
                index='method',  # rows
                columns=['p', 'radius'],  # two-level header (p on top, delta underneath)
                values='formatted',
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