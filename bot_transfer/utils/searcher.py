'''
Conducts Hyperparameter searches over different values.
'''

import numpy as np
from bot_transfer.utils.trainer import train
from bot_transfer.utils.loader import ModelParams

from collections import namedtuple, OrderedDict
import copy

SearchParam = namedtuple('SearchParam', ['name', 'low', 'high', 'n', 'scale'])

def cvt_arg(arg, typ):
    assert isinstance(arg, str)
    components = arg.split(' ')
    if len(components) == 1:
        return False, typ(components[0])
    elif len(components) == len(SearchParam._fields) - 1:
        return True, (typ(components[0]), typ(components[1]), int(components[2]), components[3])
    else:
        raise ValueError("Incorrect length search parameter. Use Format low high n scaling")

def grid_search(params):
    search_params = list()
    for param_name, param_value in params['alg_args'].items():
        if hasattr(param_value, '__len__') and len(param_value) == len(SearchParam._fields) - 1:
            search_params.append(SearchParam(param_name, *param_value))
    
    search_values = OrderedDict()
    for search_param in search_params:
        if search_param[4] == 'linear':
            search_values[search_param] = np.linspace(search_param[1], search_param[2], search_param[3])
        elif search_param[4] == 'geom':
            search_values[search_param] = np.geomspace(search_param[1], search_param[2], search_param[3])
        else:
            raise ValueError("Scaling for gridsearch not correctly specified")
    
    experiments = np.meshgrid(*[v for v in search_values.values()])
    experiments = zip(*[param_vals.flatten() for param_vals in experiments])


    for experiment in experiments:
        print(experiment)
        experiment_params = copy.deepcopy(params)
        name_str = ""
        for i in range(len(search_params)):
            param_name = search_params[i][0]
            name_str += '_' + param_name + '_' + str(round(experiment[i], 4)).replace('.', 'p')
            experiment_params['alg_args'][param_name] = experiment[i]
        experiment_params['name'] = name_str[1:]
        train(experiment_params)


