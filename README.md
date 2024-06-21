# TF_WaveNet
An unofficial tensorflow/keras implementation of [WaveNet: A Generative Model For Raw Audio (Oord et al, 2016)](https://arxiv.org/abs/1609.03499)

[Some personal notes regarding my implementation of the model (Written in Korean)](https://github.com/95dykim/ToyProject_WaveNetImplementation/blob/main/personal_notes.md)


# Requirements
tensorflow>=2.9.0

tensorflow-datasets>=4.8.0

librosa>=0.9

numpy>=1.22.0

pandas>=1.4.0

pydub

# Dataset
Beside of [groove dataset](https://magenta.tensorflow.org/datasets/groove), I have used the following albums as a training dataset:
- The Arrogant Sons of Bitches - Three Cheers for Disappointment
- Bomb the Music Industry! - Album Minus Band
- Bomb the Music Industry! - Goodbye Cool World!
- Bomb the Music Industry! - Get Warmer
- Bomb the Music Industry! - Scrambles
- Bomb the Music Industry! - Vacation
- Jeff Rosenstock - I Look Like Shit
- Jeff Rosenstock - We Cool?
- Jeff Rosenstock - WORRY.
- Jeff Rosenstock - POST-
- Jeff Rosenstock - NO DREAM
- Jeff Rosenstock - SKA DREAM

You can download all of them for free from a website of Jeff Rosenstock's record label, [Quote Unquote Records](http://www.quoteunquoterecords.com/).

In order to use load_dataset_rosenstock function, mp3 file paths should follow this format:
./dataset/rosenstock/(AlbumName)/(AudiofileName).mp3

If you're planning to use the same dataset as I did or simply liked the music, then consider supporting him by buying the albums or donating to the label!

# Results

https://user-images.githubusercontent.com/115688680/218380283-0825a8e1-5ec8-4833-bb7b-c5bf73402e9f.mp4

https://user-images.githubusercontent.com/115688680/218380290-81c3d0e6-a46e-4d74-a25d-ad41b90f8e3a.mp4

These are generated audio samples by Unconditional WaveNet trained in 10000 epochs with groove dataset.

# Notes

I have not asked for the musician's permission on using his works to train this DL model. In order to respect his ownership of his music, I kept this repository private most of the time.

However, since I decided to include this particulary page as a part of my resume, **this repository is made public only temporarily.**
