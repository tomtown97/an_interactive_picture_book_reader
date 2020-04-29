# 语音识别预测
import argparse

import automatic_speech_recognition as asr


# file = 'data/cv-valid-test/sample-000000.mp3'  # sample rate 16 kHz, and 16 bit depth
# file = 'sample-en.wav'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--voice_path', type=str, default='Data/test/sample-en.wav',
                        help='Voice Path')

    args = parser.parse_args()

    print("Sound File:", args.voice_path)

    # load sound sample
    sample = asr.utils.read_audio(args.voice_path)

    # load model
    pipeline = asr.load('deepspeech2', lang='en')

    # print model architecture
    # pipeline.model.summary()     # TensorFlow model

    # get predicted result
    sentences = pipeline.predict([sample])

    # print predicted result
    print(sentences)

    return sentences


# file = 'data/cv-valid-test/sample-000000.mp3'  # sample rate 16 kHz, and 16 bit depth

if __name__ == '__main__':
    main()
