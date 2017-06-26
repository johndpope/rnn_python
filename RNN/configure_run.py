class ConfigRun():

    #In this class the run parameters will be set

    #default configuration
    def __init__(self):
        self.random_seed = None
        self.save_summaries_steps = 5
        self.save_checkpoint_secs = None
        self.save_checkpoint_steps = None
        self.keep_checkpoint_steps = None
        self.keep_checkpoint_max = 5
        self.keep_checkpoint_every_n_hours = 4
        self.gpu_memory_fraction =1.0
        self.gpu_allow_growth = False
        self.log_device_placement = False

    def set_save_summaries_steps(self, sum_steps):
        self.save_summaries_steps = sum_steps
        return self

    def set_random_seed(self, random_seed):
        self.random_seed = random_seed
        return self

    def set_save_checkpoint_secs(self, save_checkpoints_secs):
        self.save_checkpoint_secs = save_checkpoints_secs
        return self

    def set_save_checkpoint_steps(self, save_checkpoints_steps):
        self.save_checkpoint_steps = save_checkpoints_steps
        return self

    def set_keep_checkpoint_steps(self, keep_checkpoints_steps):
        self.keep_checkpoint_steps = keep_checkpoints_steps
        return self

    def set_keep_checkpoint_max(self, keep_checkpoint_max):
        self.keep_checkpoint_max = keep_checkpoint_max
        return self

    def set_keep_checkpoint_every_n_hours(self, keep_checkpoint_every_n_hours):
        self.keep_checkpoint_every_n_hours = keep_checkpoint_every_n_hours
        return self

    def set_gpu_memory_fraction(self, gpu_memory_fraction):
        self.gpu_memory_fraction = gpu_memory_fraction
        return

    def set_log_device_placement(self, log_device_placement):
        self.log_device_placement = log_device_placement
        return self

    def set_gpu_allow_growth(self, gpu_allow_growth):
        self.gpu_allow_growth = gpu_allow_growth
        return self
