from mozilla_common_voice import MozillaCommonVoiceDataset
from urban_sound_8K import UrbanSound8K
from dataset import Dataset
import warnings
k=15
warnings.filterwarnings(action='ignore')

mozilla_basepath = '/content/drive/My Drive/daitan/implementation/largescale/clean'
urbansound_basepath = '/content/drive/My Drive/daitan/implementation/largescale/noise'

mcv = MozillaCommonVoiceDataset(k,val_dataset_size=0)
clean_train_filenames, clean_val_filenames = mcv.get_train_val_filenames()

us8K = UrbanSound8K(k,val_dataset_size=0)
noise_train_filenames, noise_val_filenames = us8K.get_train_val_filenames()

windowLength = 256
config = {'windowLength': windowLength,
          'overlap': round(0.25 * windowLength),
          'fs': 16000,
          'audio_max_duration': 0.8}

val_dataset = Dataset(clean_val_filenames, noise_val_filenames,k, **config)
val_dataset.create_tf_record(prefix='val', subset_size=100)
#subset_size is the number of records
train_dataset = Dataset(clean_train_filenames, noise_train_filenames,k, **config)
train_dataset.create_tf_record(prefix='train', subset_size=100)

# Create Test Set
#clean_test_filenames = mcv.get_test_filenames()

#noise_test_filenames = us8K.get_test_filenames()

#test_dataset = Dataset(clean_test_filenames, noise_test_filenames, **config)
#test_dataset.create_tf_record(prefix='test', subset_size=1000, parallel=False)

