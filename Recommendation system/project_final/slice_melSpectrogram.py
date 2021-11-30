import os
import re
from PIL import Image

"""
Slice the spectrogram into multiple 128x128 images which will be the input to the
Convolutional Neural Network.
"""
def slice_spectrogram(mode='Train'):
    if mode=="Train":
        image_folder = "Train_Spectogram_Images"
        if not os.path.exists('Train_Sliced_Images'):
            os.makedirs('Train_Sliced_Images')
        folder_name='Train_Sliced_Images'
        pattern='Train_Spectogram_Images/.*_(.+?).jpg'
        
    elif mode=='Test':
        image_folder ="Test_Spectogram_Images"
        if not os.path.exists('Test_Sliced_Images'):
            os.makedirs('Test_Sliced_Images')
        folder_name='Test_Sliced_Images'
        pattern='Test_Spectogram_Images/(.+?).jpg'
        
    filenames = [image_folder + '/' + f for f in os.listdir(image_folder)
                   if f.endswith(".jpg")]
    counter = 0
    for f in filenames:
        genre_variable = re.search(pattern, f).group(1)
        img = Image.open(f)
        subsample_size = 128
        width, height = img.size
        number_of_samples = width // subsample_size
        for i in range(number_of_samples):
            start = i*subsample_size
            img_temporary = img.crop((start, 0., start + subsample_size, subsample_size))
            img_temporary.save(folder_name+ "/"+str(counter)+"_"+genre_variable+".jpg")
            counter = counter + 1
if __name__ == '__main__':
    slice_spectrogram(mode='Train')