#/usr/bin/bash
#rem re run all test 04 runs...
python3 -u ../ind_04.py > iso04noweight.txt
python3 -u ../ind_04w.py > iso04weight.txt
python3 -u ../optimization_blue_cv01.py > cv01noweight.txt
python3 -u ../optimization_blue_cv01w.py > cv01weight.txt
python3 -u ../optimization_blue_cv02.py > cv02noweight.txt
python3 -u ../optimization_blue_cv02w.py > cv02weight.txt
python3 -u ../optimization_blue_cv03.py > cv03noweight.txt
python3 -u ../optimization_blue_cv03w.py > cv03weight.txt