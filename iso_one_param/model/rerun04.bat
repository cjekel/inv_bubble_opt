rem re run all test 04 runs...
python -u ..\ind04.py > iso04noweight.txt
python -u ..\ind04w.py > iso04weight.txt
python -u ..\optimization_blue_cv01.py > cv01noweight.txt
python -u ..\optimization_blue_cv01w.py > cv01weight.txt
python -u ..\optimization_blue_cv02.py > cv02noweight.txt
python -u ..\optimization_blue_cv02w.py > cv02weight.txt
python -u ..\optimization_blue_cv03.py > cv03noweight.txt
python -u ..\optimization_blue_cv03w.py > cv03weight.txt