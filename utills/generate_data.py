#Create dataloader class for each device.
from torch.utils.data import  Dataset

class generate_data(Dataset):
  def __init__(self,x,y):
      self.x ,self.y= x,y
  def __len__(self):
      return len(self.x)
  def __getitem__(self,idx):
      inp=self.x[idx]
      labels = self.y[idx]
      return inp,labels

class SlidingWindowDataset(Dataset):
    def __init__(self, features, labels, window_size, step_size):
        # Ensure features are converted to numpy arrays
        features = features.values if isinstance(features, pd.DataFrame) else features
        labels = labels.values if isinstance(labels, (pd.DataFrame, pd.Series)) else labels
        
        self.features = self._create_sliding_window(features, window_size, step_size)
        self.labels = self._create_label_windows(labels, window_size, step_size)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return len(self.features)

    @staticmethod
    def _create_sliding_window(data, window_size, step_size):
        shape = ((data.shape[0] - window_size) // step_size + 1, window_size, data.shape[1])
        strides = (data.strides[0] * step_size, data.strides[0], data.strides[1])
        return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides).astype(np.float32)

    @staticmethod
    def _create_label_windows(labels, window_size, step_size):
        if labels.ndim > 1:
            labels = np.argmax(labels, axis=1)        
        labels = np.asarray(labels)
        return np.lib.stride_tricks.as_strided(labels,
                                               shape=((labels.shape[0] - window_size) // step_size + 1, window_size),
                                               strides=(labels.strides[0] * step_size, labels.strides[0]))[:, -1].astype(np.uint8)
