import subprocess
from audiotsm import phasevocoder
from audiotsm.io.wav import WavReader, WavWriter
from scipy.io import wavfile
import numpy as np
import math
from shutil import rmtree
import os
import argparse
from pytube import YouTube, StreamQuery
from collections import deque
from multiprocessing import Process
from random import randint

# VIDEO INITIALISATION INFO
parser = argparse.ArgumentParser(
    description='Modifies a video file to play at different speeds when there is sound vs. silence.')
parser.add_argument('--input_file', type=str, help='Video file to modify')
parser.add_argument('--input_dir', type=str, help='Directory with videos to modify')
parser.add_argument('--url_file', type=str, help='Path to file with youtube urls')
parser.add_argument('--url', type=str, help='Youtube url of video')
parser.add_argument('--output_file', type=str, default="",
                    help="the output file")
parser.add_argument('--output_dir', type=str, default="output_videos",
                    help="Directory for output videos, default is \"..\output_videos\"")

# CONSTANTS
parser.add_argument('--parallel_all', type=int, default=0,
                    help='Download and process all videos at one time(1), or one by one(0)? Default is 0.'
                         'Use it wisely - if you insert some big videos, then parallel processing can'
                         ' kill your computer')
parser.add_argument('--resolution', type=int, default=480, help='Default resolution of youtube video to download')
parser.add_argument('--silent_threshold', type=float, default=0.03,
                    help="Volume value that frames' audio needs to surpass to be consider \"sounded\". "
                         "It ranges from 0 (silence) to 1 (max volume)")
parser.add_argument('--sounded_speed', type=float, default=1.00,
                    help="Speed that sounded (spoken) frames should be played at, usually 1")
parser.add_argument('--silent_speed', type=float, default=5.00,
                    help="Speed that silent frames should be played at")
parser.add_argument('--frame_margin', type=float, default=1,
                    help="Some silent frames adjacent to sounded frames are included to provide context. "
                         "This variable shows how many frames on either the side of speech should be included")
parser.add_argument('--sample_rate', type=float, default=44100, help="Sample rate of the input and output videos")
parser.add_argument('--frame_quality', type=int, default=3,
                    help="Quality of frames to be extracted from input video. "
                         "1 is highest, 31 is lowest, 3 is the default.")
args = parser.parse_args()

SAMPLE_RATE = args.sample_rate
SILENT_THRESHOLD = args.silent_threshold
FRAME_SPREADAGE = args.frame_margin
NEW_SPEED = [args.silent_speed, args.sounded_speed]
FRAME_QUALITY = args.frame_quality
OUTPUT_DIR = args.output_dir
RESOLUTION = args.resolution
PARALLEL_ALL = args.parallel_all


class Video:
    def __init__(self, output_file, url=None, file_path=None):
        self.temp_folder = "TEMP" + str(randint(1, 10 ** 5))
        if file_path:
            self.filename = file_path
            self.fps = self.get_fps()
        elif url:
            streams: StreamQuery = YouTube(url).streams
            highest_stream = streams.get_highest_resolution()
            highest_res = int(highest_stream.resolution[:-1])
            stream = None
            if highest_res > RESOLUTION:
                for cur_stream in streams:
                    if int(cur_stream.resolution[:-1]) == RESOLUTION:
                        stream = cur_stream
                        break
                if not stream:
                    stream = highest_res
            else:
                stream = highest_stream
            self.fps = stream.fps
            name = stream.download()
            self.filename = name.replace(' ', '_')
            os.rename(name, self.filename)
        else:
            raise ValueError('cannot initialize video')

        self.output_filename = output_file if output_file else self.get_output_filename()

    def get_output_filename(self):
        basename = os.path.basename(self.filename)
        dot_idx = basename.rfind(".")
        return os.path.join(OUTPUT_DIR, basename[:dot_idx] + "_ALTERED" + basename[dot_idx:])

    def get_fps(self):
        command = "ffprobe -v 0 -of csv=p=0 -select_streams v:0 -show_entries stream=r_frame_rate " + self.filename
        fps_str = str(subprocess.check_output(command, shell=True))
        return int(fps_str[2:4])

    def save_audio(self):
        command = f"ffmpeg -i {self.filename} -ab 160k -ac 2 -ar {SAMPLE_RATE} -vn {self.temp_folder}/audio.wav"
        subprocess.call(command, shell=True)

    def save_video(self):
        command = f"ffmpeg -i {self.filename} -qscale:v {FRAME_QUALITY} {self.temp_folder}/frame%06d.jpg -hide_banner"
        subprocess.call(command, shell=True)

    def final_concatenation(self):
        command = f"ffmpeg -framerate {self.fps} -i " + self.temp_folder + "/newFrame%06d.jpg -i " \
                  + self.temp_folder + "/audioNew.wav -strict -2 " + self.output_filename
        subprocess.call(command, shell=True)

    def copy_frame(self, input_frame: int, output_frame: int):
        src = self.temp_folder + "/frame{:06d}".format(input_frame + 1) + ".jpg"
        dst = self.temp_folder + "/newFrame{:06d}".format(output_frame + 1) + ".jpg"
        if not os.path.isfile(src):
            return False
        os.rename(src, dst)
        if output_frame == 1 or output_frame % 1000 == 999:
            print(str(output_frame + 1) + " time-altered frames saved.")
        return True

    def process_and_concatenate(self):
        audio_fade_envelope_size = 400  # smooth out transition's audio by quickly fading in/out

        self.save_audio()

        sample_rate, audio_data = wavfile.read(self.temp_folder + "/audio.wav")
        audio_sample_count = audio_data.shape[0]
        max_audio_volume = get_max_volume(audio_data)

        samples_per_frame = sample_rate / self.fps

        audio_frame_count = int(math.ceil(audio_sample_count / samples_per_frame))

        has_loud_audio = np.zeros(audio_frame_count)

        for i in range(audio_frame_count):
            start = int(i * samples_per_frame)
            end = min(int((i + 1) * samples_per_frame), audio_sample_count)
            audio_chunks = audio_data[start:end]
            max_chunks_volume = float(get_max_volume(audio_chunks)) / max_audio_volume
            if max_chunks_volume >= SILENT_THRESHOLD:
                has_loud_audio[i] = 1

        chunks = [[0, 0, 0]]
        should_include_frame = np.zeros(audio_frame_count)

        last_idx = 0
        for i in range(audio_frame_count):
            start = int(max(0, i - FRAME_SPREADAGE))
            end = int(min(audio_frame_count, i + 1 + FRAME_SPREADAGE))
            should_include_frame[i] = np.max(has_loud_audio[start:end])
            if i >= 1 and should_include_frame[i] != should_include_frame[i - 1]:  # Did we flip?
                chunks.append([chunks[-1][1], i, should_include_frame[i - 1]])
            last_idx = i

        chunks.append([chunks[-1][1], audio_frame_count, should_include_frame[last_idx - 1]])
        chunks = chunks[1:]

        output_audio_data = np.zeros((0, audio_data.shape[1]))
        output_pointer = 0

        last_existing_frame = None

        command = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {self.filename}"
        duration = subprocess.check_output(command, shell=True)

        frames_num = int(float(duration) * self.fps)
        signed_frames = [False for _ in range(frames_num)]
        output_frames = []

        for chunk in chunks:
            audio_chunk = audio_data[int(chunk[0] * samples_per_frame):int(chunk[1] * samples_per_frame)]

            s_file = self.temp_folder + "/tempStart.wav"
            e_file = self.temp_folder + "/tempEnd.wav"
            wavfile.write(s_file, SAMPLE_RATE, audio_chunk)
            with WavReader(s_file) as reader:
                with WavWriter(e_file, reader.channels, reader.samplerate) as writer:
                    tsm = phasevocoder(reader.channels, speed=NEW_SPEED[int(chunk[2])])
                    tsm.run(reader, writer)
            _, altered_audio_data = wavfile.read(e_file)
            leng = altered_audio_data.shape[0]
            end_pointer = output_pointer + leng
            output_audio_data = np.concatenate((output_audio_data, altered_audio_data / max_audio_volume))

            if leng < audio_fade_envelope_size:
                output_audio_data[output_pointer:end_pointer] = 0
            else:
                pre_mask = np.arange(audio_fade_envelope_size) / audio_fade_envelope_size
                mask = np.repeat(pre_mask[:, np.newaxis], 2, axis=1)
                output_audio_data[output_pointer:output_pointer + audio_fade_envelope_size] *= mask
                output_audio_data[end_pointer - audio_fade_envelope_size:end_pointer] *= 1 - mask

            start_output_frame = int(math.ceil(output_pointer / samples_per_frame))
            end_output_frame = int(math.ceil(end_pointer / samples_per_frame))

            for outputFrame in range(start_output_frame, end_output_frame):
                input_frame = int(chunk[0] + NEW_SPEED[int(chunk[2])] * (outputFrame - start_output_frame))
                if input_frame < frames_num - 2:
                    signed_frames[input_frame + 1] = True
                    last_existing_frame = input_frame
                else:
                    signed_frames[last_existing_frame] = True
                output_frames.append(outputFrame)

            output_pointer = end_pointer

        j = 0
        for i, frame_sign in enumerate(signed_frames):
            if frame_sign:
                self.copy_frame(i, j)
                j += 1
        wavfile.write(self.temp_folder + "/audioNew.wav", SAMPLE_RATE, output_audio_data)

        self.final_concatenation()
        delete_path(self.temp_folder)


def valid_format(filename):
    formats = ['.mp4', '.mov', '.avi', '.wmv']
    dot_idx = filename.rfind('.')
    return filename[dot_idx:] in formats


def get_max_volume(s):
    min_volume = float(np.min(s))
    max_volume = float(np.max(s))
    return max(max_volume, -min_volume)


def create_path(s):
    try:
        os.mkdir(s)
    except OSError:
        assert False, "Creation of the directory %s failed: TEMP folder may already exist"


def delete_path(s):
    try:
        rmtree(s, ignore_errors=False)
    except Exception as e:
        print("Deletion of the directory %s failed" % s)
        print(e)


if __name__ == '__main__':
    print('Started')

    q = deque()
    if args.url:
        q.append(Video(url=args.url, output_file=args.output_file))
    elif args.url_file:
        file_path = args.url_file
        abspath = os.path.abspath(file_path)
        assert os.path.isfile(abspath), f"invalid urls file path: {abspath}"
        with open(file_path, 'r') as f:
            for url in f.read().split('\n'):
                q.append(Video(url=url, output_file=args.output_file))
    elif args.input_file:
        abspath = os.path.abspath(args.input_file)
        assert os.path.isfile(abspath), f"invalid input file path: {abspath}"
        q.append(Video(file_path=args.input_file, output_file=args.output_file))
    elif args.input_dir:
        abspath = os.path.abspath(args.input_dir)
        assert os.path.isdir(abspath), f"invalid directory: {abspath}"
        for filename in os.listdir(args.input_dir):
            full_filename = os.path.join(args.input_dir, filename)
            if not os.path.isfile(full_filename):
                print(f'file {full_filename} does not exist')
                continue
            if valid_format(full_filename):
                q.append(Video(file_path=full_filename, output_file=args.output_file))
            else:
                print(f'Invalid file format: {full_filename}')
    else:
        raise ValueError("no input file")

    i = 0
    while len(q) != 0:
        video = q.popleft()
        try:
            print(f'Downloading {i} video')
            if os.path.exists(video.temp_folder):
                delete_path(video.temp_folder)
            create_path(video.temp_folder)

            if not os.path.exists(OUTPUT_DIR):
                create_path(OUTPUT_DIR)
            frames_saving_process = Process(target=video.save_video)
            audio_processing_process = Process(target=video.process_and_concatenate)
            frames_saving_process.start()
            audio_processing_process.start()
            if not PARALLEL_ALL:
                frames_saving_process.join()
                audio_processing_process.join()

            i += 1
        except Exception as ex:
            print(f'Exception at {video.filename}:', ex)
