class Config:
    def __init__(self, config_dict):
        for k, v in config_dict.items():
            setattr(self, k, Config(v) if isinstance(v, dict) else v)