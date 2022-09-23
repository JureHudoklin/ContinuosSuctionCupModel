import yaml


class Config:
    def __init__(self, config_file):
        self.config_file = config_file

    def load(self):
        with open(self.config_file, 'r') as f:
            config = yaml.safe_load(f)
        print('Load configuration from `{}`'.format(self.config_file))
        print(config)
        return config

    def save(self, config, save_dir):
        print('Save configuration on `{}`'.format(save_dir))
        print(config)
        with open(save_dir, 'w') as f:
            yaml.safe_dump(config, f)