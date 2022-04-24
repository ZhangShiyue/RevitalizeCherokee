#### Data
chr_voice contains the cherokee audio data plus its phonetic transcriptions
open-sourced by https://github.com/CherokeeLanguage/cherokee-audio-data

In the paper, we also used private data shared by Michael Conrad.

#### Script
run.py is the script of training an ASR model


#### Models
pretrained models are hosted on Hugging Face

| finetuned on | Hugging Face hub name|
| ------------- | ----------- |
| public audio to phonetic text  | shiyue/wav2vec2-large-xlsr-53-chr-phonetic |
| private+public audio to phonetic text   | shiyue/wav2vec2-large-xlsr-53-chr-phonetic-with-private-data |
| private audio to syllabic text  | shiyue/wav2vec2-large-xlsr-53-chr-syllabary |


#### Acknowledgment
We thank Michael Conrad for compiling the data and sharing it with us.
