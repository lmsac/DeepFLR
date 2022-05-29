import numpy as np
from scipy.interpolate import interp1d
import statsmodels.api as sm
import pandas as pd
def interpolate(x, y, new_x):
    x, index = np.unique(x, return_index=True)
    y = y[index]
    interp = interp1d(x, y, fill_value='extrapolate')
    return interp(new_x)


def lowess(x, y, **kwargs):
    r = sm.nonparametric.lowess(y, x, **kwargs)

    # https://github.com/statsmodels/statsmodels/issues/2449
    if any(np.isnan(r[:, 1])):
        data = pd.DataFrame.from_dict({'x': x, 'y': y}) \
            .groupby(x).mean()
        x = data['x']
        y = data['y']
        r = sm.nonparametric.lowess(y, x, **kwargs)

    return r[:, 0], r[:, 1]

def calculate_rt(reference_data, data,smooth_func,model_func,smooth_args=None):
        merged_data = pd.merge(
            reference_data.drop(columns=['index', 'run']), data,
            on=reference_data.columns.drop(['index', 'run', 'rt']).tolist(),
            suffixes=['_reference', '']
        )

        index = np.argsort(merged_data['rt_reference'])
        y = merged_data['rt_reference'][index].values
        x = merged_data['rt'][index].values

        if smooth_func is not None:
            x, y = smooth_func(x, y, **(smooth_args or {}))

        if any(map(lambda x: not x > 0, x)):
            raise ValueError(x)
        if any(map(lambda x: not x > 0, y)):
            raise ValueError(y)

        y_new = model_func(x, y, data['rt'].values)

        #        if any(map(lambda x: not x > 0, y_new)):
        #            raise ValueError(y_new)

        return y_new