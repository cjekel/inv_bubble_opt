#!/bin/bash

ffmpeg -framerate 1 -i figs/Test_1_j=%02d.png -c:v libx264 test_1.mp4
ffmpeg -framerate 1 -i figs/Test_2_j=%02d.png -c:v libx264 test_2.mp4
ffmpeg -framerate 1 -i figs/Test_3_j=%02d.png -c:v libx264 test_3.mp4
ffmpeg -framerate 1 -i figs/Test_4_j=%02d.png -c:v libx264 test_4.mp4