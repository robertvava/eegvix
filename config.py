def config(act = 'train', model = 'gan', epochs = 100, data = None):
    ''' 
    act (action) options: train, generate
    model options: [('ic-gan', 'gan'), ('vq-vae', vae'), 'diff']
    '''
    return {'act': act,
            'model': model,
            'data' : data,
            'hp': config_hyperparam(model),
            'epochs': epochs
            }


def config_hyperparam(model):
    """
    Determine configuration based on the model. 
    type = 'o' means original, or "vanilla" model. 
    :return: config params  
    """
    if model == 'diffusion':
        if type == 'o':
            config = {
                    'optim': {
                            'lr': 0.00001
                            },  
                    'model' : {
                                'use_batch_norm': True
                            }
                    }

    elif model == 'gan':
        if type == 'o':
            config = {
                    'optim': {
                        'lr': 1e-4
                        },  
                    'model' : {
                                'use_batch_norm': True
                            }
                    }
    return config