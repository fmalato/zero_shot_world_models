import json


class ActionMapper:
    def __init__(self, game):
        self.game = game
        with open("mappings.json", "r") as f:
            self.mapper = json.load(f)[self.game]

    def __len__(self):
        return len(self.mapper["gym"].keys())

    # TODO: Freeway fails due to action mismatch (18 on farama --> 3 in reality)
    def map_gym_to_ale(self, a):
        for k, v in self.mapper["gym"].items():
            if v == a:
                return self.mapper["ale"][k]

        return self.mapper["ale"]["NOOP"]

    def map_ale_to_gym(self, a):
        for k, v in self.mapper["ale"].items():
            if v == a:
                return self.mapper["gym"][k]

        return self.mapper["gym"]["NOOP"]
