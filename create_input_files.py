from utils import create_input_files

if __name__ == '__main__':
    create_input_files(dataset='flickr8k',
                       karpathy_json_path=r'C:\Users\thera\Downloads\a-PyTorch-Tutorial-to-Image-Captioning-master\a-PyTorch-Tutorial-to-Image-Captioning-master\dataset_flickr8k.json',
                       image_folder=r'C:\Users\thera\Downloads\a-PyTorch-Tutorial-to-Image-Captioning-master\a-PyTorch-Tutorial-to-Image-Captioning-master\Data\Images',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder=r'C:\Users\thera\Downloads\a-PyTorch-Tutorial-to-Image-Captioning-master\a-PyTorch-Tutorial-to-Image-Captioning-master',
                       max_len=50)