import pandas as pd
import numpy as np
import os

np.random.seed(999)
noisepath='/content/drive/My Drive/daitan/largescale/noise'
class UrbanSound8K:
  def __init__(self,k,val_dataset_size):
    #self.basepath = basepath
    self.val_dataset_size = val_dataset_size
    #self.class_ids = class_ids
    self.k=k
    print(self.k)

  def _get_urban_sound_8K_filenames(self,pathname,dataframe_name):
    noise_files=[]
    noise_metadata = pd.read_csv(os.path.join(pathname, dataframe_name), sep='\t')
    noise_files = noise_metadata[noise_metadata.columns[0]].values
    #np.random.shuffle(noise_files)
    print("Total number of training examples:", len(noise_files))
    return noise_files

  def get_train_val_filenames(self):
    urbansound_train_filenames_1 = self._get_urban_sound_8K_filenames(pathname=noisepath,dataframe_name='noise.csv')
    urbansound_train_filenames_2=urbansound_train_filenames_1[2000*self.k:2000*(self.k+1)]
    urbansound_train_filenames = [os.path.join(noisepath,filename) for filename in urbansound_train_filenames_2]
    # separate noise files for train/validation
    #urbansound_train_filenames=[file1 for file1 in urbansound_train_filenames_1 if os.path.isfile(file1)]
    urbansound_val = urbansound_train_filenames[-self.val_dataset_size:]
    urbansound_train = urbansound_train_filenames[:-self.val_dataset_size]
    print("Noise training:", len(urbansound_train))
    print("Noise validation:", len(urbansound_val))
    return urbansound_train, urbansound_val

  def get_test_filenames(self):
    urbansound_metadata = self._get_urban_sound_8K_filenames(pathname=noisepath)
    #change to something else
    return urbansound_metadata
'''
# fold 10 is used for testing only
urbansound_train = urbansound_metadata[urbansound_metadata.fold!= 10]

urbansound_test_filenames = self._get_filenames_by_class_id(urbansound_train)
np.random.shuffle(urbansound_test_filenames)

print("# of Noise testing files:", len(urbansound_test_filenames))
return urbansound_test_filenames
'''