def config(act = 'train', model_name = 'reg', epochs = 100, data = None):
    ''' 
    act (action) options: train, generate
    model options: [('ic-gan', 'gan'), ('vq-vae', vae'), 'diff']
    '''
    return {'act': act,
            'model': model_name,
            'data' : data,
            'hp': config_hyperparam(model_name),
            'epochs': epochs
            }


def config_hyperparam(model):
    """
    Determine configuration based on the model. 
    type = 'o' means original, or "vanilla" model. 
    :return: config params  
    """
    hp_config = {'optim': {'lr': 1e-4}}

    if model == 'diffusion':
        if type == 'o':
            hp_config = {
                    'optim': {
                            'lr': 0.00001
                            },  
                    'model' : {
                                'use_batch_norm': True
                            }
                    }

    elif model == 'gan':
        if type == 'o':
            hp_config = {
                    'optim': {
                        'lr': 1e-4
                        },  
                    'model' : {
                                'use_batch_norm': True
                            }
                    }
    return hp_config