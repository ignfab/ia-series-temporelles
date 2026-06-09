import json
import os


class Logger:
    def __init__(self, save_path):
        self.save_path = save_path
        self.create_log_file()

    def create_log_file(self):
        self.log_file = os.path.join(self.save_path, "log.json")
        self.logs = {}

    def log(self, curr_dict):
        for key, value in curr_dict.items():
            if key not in self.logs:
                self.logs[key] = []
            if 'acc' in key:
                value = value * 100
            self.logs[key].append(value)
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=4)