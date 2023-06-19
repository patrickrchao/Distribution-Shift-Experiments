class Bound:
    def __init__(self, name, perturbation_type, color, fn):
        self.name = name
        self.perturbation_type = perturbation_type
        self.color = color
        self.fn = fn
        
    def __str__(self):
        return self.name
    
    def __call__(self, epsilon):
        return self.fn(epsilon)
