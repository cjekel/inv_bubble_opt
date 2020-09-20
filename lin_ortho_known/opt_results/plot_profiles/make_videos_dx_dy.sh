#!/bin/bash

ffmpeg -framerate 1 -i figs_dx_dy/Test_1_j=%02d.png -c:v libx264 test_dxdy_1.mp4
ffmpeg -framerate 1 -i figs_dx_dy/Test_2_j=%02d.png -c:v libx264 test_dxdy_2.mp4
ffmpeg -framerate 1 -i figs_dx_dy/Test_3_j=%02d.png -c:v libx264 test_dxdy_3.mp4
ffmpeg -framerate 1 -i figs_dx_dy/Test_4_j=%02d.png -c:v libx264 test_dxdy_4.mp4