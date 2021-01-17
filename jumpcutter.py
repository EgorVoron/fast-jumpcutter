import subprocess
from audiotsm import phasevocoder
from audiotsm.io.wav import WavReader, WavWriter
from scipy.io import wavfile
import numpy as np
import re
import math
from shutil import rmtree
import os
import argparse
from pytube import YouTube
from collections import deque


def download_file(url):
    stream = YouTube(url).streams.get_highest_resolution()
    fps = stream.fps
    name = stream.download()
    new_name = name.replace(' ', '_')
    os.rename(name, new_name)
    return new_name, fps


def valid_format(filename):
    formats = ['.mp4', '.mov', '.avi', '.wmv']
    dot_idx = filename.rfind('.')
    return filename[dot_idx:] in formats


def get_max_volume(s):
    min_volume = float(np.min(s))
    max_volume = float(np.max(s))
    return max(max_volume, -min_volume)


def copy_frame(input_frame, output_frame, tmp_folder):
    src = tmp_folder + "/frame{:06d}".format(input_frame + 1) + ".jpg"
    dst = tmp_folder + "/newFrame{:06d}".format(output_frame + 1) + ".jpg"
    if not os.path.isfile(src):
        return False
    os.rename(src, dst)
    if output_frame == 1 or output_frame % 1000 == 999:
        print(str(output_frame + 1) + " time-altered frames saved.")
    return True


def output_filename(filename, output_dir):
    basename = os.path.basename(filename)
    dot_idx = basename.rfind(".")
    return os.path.join(output_dir, basename[:dot_idx] + "_ALTERED" + basename[dot_idx:])


def create_path(s):
    try:
        os.mkdir(s)
    except OSError:
        assert False, "Creation of the directory %s failed: TEMP folder may already exist"


def delete_path(s):
    try:
        rmtree(s, ignore_errors=False)
    except OSError() as e:
        print("Deletion of the directory %s failed" % s)
        print(e)


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
parser.add_argument('--frame_rate', type=float, default=25,
                    help="Fps of input and output videos")
parser.add_argument('--frame_quality', type=int, default=3,
                    help="Quality of frames to be extracted from input video. "
                         "1 is highest, 31 is lowest, 3 is the default.")
args = parser.parse_args()

frame_rate = args.frame_rate
SAMPLE_RATE = args.sample_rate
SILENT_THRESHOLD = args.silent_threshold
FRAME_SPREADAGE = args.frame_margin
NEW_SPEED = [args.silent_speed, args.sounded_speed]

q = deque()
if args.url:
    path, fps = download_file(args.url)
    q.append(path)
    frame_rate = fps
elif args.url_file:
    file_path = args.url_file
    assert os.path.isfile(file_path), f"invalid file path: {file_path}"
    with open(file_path, 'r') as f:
        for url in f.read().split('\n'):
            path, fps = download_file(url)
            q.append(path)
            frame_rate = fps
elif args.input_file:
    q.append(args.input_file)
elif args.input_dir:
    for filename in os.listdir(args.input_dir):
        full_filename = os.path.join(args.input_dir, filename)
        if valid_format(full_filename):
            q.append(full_filename)
        else:
            print(f'Invalid file format: {full_filename}')
else:
    raise ValueError("no input file")
URL = args.url
FRAME_QUALITY = args.frame_quality


def run(input_file, frame_rate=frame_rate):
    output_file = args.output_file if len(args.output_file) >= 1 else output_filename(input_file, args.output_dir)

    temp_folder = "TEMP"
    audio_fade_envelope_size = 400  # smooth out transition's audio by quickly fading in/out

    if os.path.exists(temp_folder):
        delete_path(temp_folder)
    create_path(temp_folder)

    if not os.path.exists(args.output_dir):
        create_path(args.output_dir)

    command = "ffmpeg -i " + input_file + " -qscale:v " + str(
        FRAME_QUALITY) + " " + temp_folder + "/frame%06d.jpg -hide_banner"
    subprocess.call(command, shell=True)

    command = "ffmpeg -i " + input_file + " -ab 160k -ac 2 -ar " + str(
        SAMPLE_RATE) + " -vn " + temp_folder + "/audio.wav"

    subprocess.call(command, shell=True)

    command = "ffmpeg -i " + temp_folder + "/input.mp4 2>&1"
    with open(temp_folder + "/params.txt", "w") as f:
        subprocess.call(command, shell=True, stdout=f)

    sample_rate, audio_data = wavfile.read(temp_folder + "/audio.wav")
    audio_sample_count = audio_data.shape[0]
    max_audio_volume = get_max_volume(audio_data)

    with open(temp_folder + "/params.txt", 'r+') as f:
        pre_params = f.read()
    params = pre_params.split('\n')
    for line in params:
        m = re.search('Stream #.*Video.* ([0-9]*) fps', line)
        if m is not None:
            frame_rate = float(m.group(1))

    samples_per_frame = sample_rate / frame_rate

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
    for i in range(audio_frame_count):
        start = int(max(0, i - FRAME_SPREADAGE))
        end = int(min(audio_frame_count, i + 1 + FRAME_SPREADAGE))
        should_include_frame[i] = np.max(has_loud_audio[start:end])
        if i >= 1 and should_include_frame[i] != should_include_frame[i - 1]:  # Did we flip?
            chunks.append([chunks[-1][1], i, should_include_frame[i - 1]])

    chunks.append([chunks[-1][1], audio_frame_count, should_include_frame[i - 1]])
    chunks = chunks[1:]

    output_audio_data = np.zeros((0, audio_data.shape[1]))
    output_pointer = 0

    last_existing_frame = None
    for chunk in chunks:
        audio_chunk = audio_data[int(chunk[0] * samples_per_frame):int(chunk[1] * samples_per_frame)]

        s_file = temp_folder + "/tempStart.wav"
        e_file = temp_folder + "/tempEnd.wav"
        wavfile.write(s_file, SAMPLE_RATE, audio_chunk)
        with WavReader(s_file) as reader:
            with WavWriter(e_file, reader.channels, reader.samplerate) as writer:
                tsm = phasevocoder(reader.channels, speed=NEW_SPEED[int(chunk[2])])
                tsm.run(reader, writer)
        _, altered_audio_data = wavfile.read(e_file)
        leng = altered_audio_data.shape[0]
        end_pointer = output_pointer + leng
        output_audio_data = np.concatenate((output_audio_data, altered_audio_data / max_audio_volume))

        # smooth out transition's audio by quickly fading in/out
        if leng < audio_fade_envelope_size:
            output_audio_data[output_pointer:end_pointer] = 0  # audio is less than 0.01 sec, let's just remove it.
        else:
            pre_mask = np.arange(audio_fade_envelope_size) / audio_fade_envelope_size
            mask = np.repeat(pre_mask[:, np.newaxis], 2, axis=1)  # make the fade-envelope mask stereo
            output_audio_data[output_pointer:output_pointer + audio_fade_envelope_size] *= mask
            output_audio_data[end_pointer - audio_fade_envelope_size:end_pointer] *= 1 - mask

        start_output_frame = int(math.ceil(output_pointer / samples_per_frame))
        end_output_frame = int(math.ceil(end_pointer / samples_per_frame))
        for outputFrame in range(start_output_frame, end_output_frame):
            input_frame = int(chunk[0] + NEW_SPEED[int(chunk[2])] * (outputFrame - start_output_frame))
            did_it_work = copy_frame(input_frame, outputFrame, temp_folder)
            if did_it_work:
                last_existing_frame = input_frame
            else:
                copy_frame(last_existing_frame, outputFrame, temp_folder)

        output_pointer = end_pointer

    wavfile.write(temp_folder + "/audioNew.wav", SAMPLE_RATE, output_audio_data)

    command = "ffmpeg -framerate " + str(
        frame_rate) + " -i " + temp_folder + "/newFrame%06d.jpg -i " + temp_folder + "/audioNew.wav -strict -2 " + output_file
    subprocess.call(command, shell=True)

    delete_path(temp_folder)


while len(q) != 0:
    file = q.popleft()
    try:
        run(file)
    except Exception as ex:
        print(f'Exception at {file}:', ex)
