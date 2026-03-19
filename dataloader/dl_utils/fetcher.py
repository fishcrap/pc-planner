class FastFetcher():
    def __init__(self, dataset):
        self.dataset = dataset
        
    def fetch(self, idx):
        data = self.dataset.__getitems__(idx)
        return data