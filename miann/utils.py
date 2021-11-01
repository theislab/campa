import logging
import os

def init_logging(level=logging.INFO):
    import warnings
    #from numba.errors import NumbaPerformanceWarning
    logging.basicConfig(level=level) # need one of this?
    #logging.getLogger().setLevel(level) # need one of this?
    # ignore tensorflow warnings
    #logging.getLogger('tensorflow').setLevel(logging.ERROR)
    # ignore scanpy / anndata warnings
    #logging.getLogger('scanpy').setLevel(logging.WARNING)
    #logging.getLogger('anndata').setLevel(logging.ERROR)
    #logging.getLogger('get_version').setLevel(logging.WARNING)
    #logging.getLogger('numexpr.utils').setLevel(logging.WARNING)
    # ignore number warnings
    #warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
    #warnings.filterwarnings("ignore", category=FutureWarning)
    
def load_config(config_file):
    """load config file and return config object"""
    import importlib.machinery, importlib.util
    loader = importlib.machinery.SourceFileLoader(os.path.basename(config_file), config_file)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    config = importlib.util.module_from_spec(spec)
    loader.exec_module(config)
    return config