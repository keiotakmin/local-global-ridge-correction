"""Recompute the paper's summary table from the condition-level records and check it against records/summary.csv.

  python reproduce.py

Main view: the 126 conditions of the seven datasets with the filled Weather file (table == 'filled'); the standard
Weather file (table == 'standard', dataset == 'weather') is reported separately. Retained state is the largest
value over the three backbones. mean_panels weights the 12 dataset-backbone panels equally (the four ETT series
form one dataset); median_126 is the median over the 126 conditions.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RECORDS = Path(__file__).resolve().parent / 'records'
ETT = ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2')


def main():
    rows = pd.read_csv(RECORDS / 'conditions_main.csv')
    rows['main'] = rows.table.eq('filled') | (rows.table.eq('standard') & rows.dataset.ne('weather'))
    rows['scope'] = np.where(rows.dataset.isin(ETT), 'ETT', rows.dataset)
    main = rows[rows.main]
    assert main.groupby('method').size().eq(126).all()
    panels = main.groupby(['method', 'scope', 'backbone']).mse_reduction_pct.mean()
    out = pd.DataFrame({
        'mean_126': main.groupby('method').mse_reduction_pct.mean(),
        'mean_panels': panels.groupby('method').mean(),
        'median_126': main.groupby('method').mse_reduction_pct.median(),
        **{s: main[main.scope.eq(s)].groupby('method').mse_reduction_pct.mean()
           for s in ('ETT', 'weather', 'electricity', 'traffic')},
        'weather_standard': rows[rows.table.eq('standard') & rows.dataset.eq('weather')]
        .groupby('method').mse_reduction_pct.mean(),
        'bytes_ETT': main[main.scope.eq('ETT')].groupby('method').state_bytes.max(),
        'bytes_traffic': main[main.scope.eq('traffic')].groupby('method').state_bytes.max(),
    })
    expected = pd.read_csv(RECORDS / 'summary.csv').set_index('method')
    out = out.loc[expected.index, expected.columns]
    np.testing.assert_allclose(out.to_numpy(float), expected.to_numpy(float), rtol=1e-8)
    wide = main.pivot_table(index=['table', 'dataset', 'backbone', 'seed', 'phase'], columns='method', values='mse')
    print(out.round(2).to_string())
    for m in ('local8', 'local4', 'global'):
        print(f'{m}: lower MSE than ELF in {int((wide[m] < wide.elf).sum())} of 126 conditions')
    print('summary.csv reproduced')


if __name__ == '__main__':
    main()
