
class Config:
    in_channel = 52
    unet_depth = 5
    unetpp_depth = 4

    segment_class = 8
    sample_rate = 60
    def __init__(self, dataset_name):
        if dataset_name == 'HTHI':
            self.in_channel = 60
            self.sample_rate = 160
            self.num_class = 7
        elif dataset_name == 'ARIL':
            self.in_channel = 52
            self.sample_rate = 60
            self.num_class = 6
        elif dataset_name == 'WiAR':
            self.in_channel = 90
            self.sample_rate = 100
            self.num_class = 7
