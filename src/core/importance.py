import os

class Importance:
    def __init__(self, degree, output_dir='results'):
        self.degree = degree

        self.importance = [0 for _ in range(degree)]

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.output_dir = output_dir

    def add_importance(self, i, importance):
        self.importance[i] = max(self.importance[i], importance)

    def get_importance(self):
        return self.importance