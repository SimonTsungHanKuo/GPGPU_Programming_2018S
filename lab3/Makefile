target:
	nvcc -g -std=c++11 -Wno-deprecated-gpu-targets main.cu lab3.cu pgm.cpp -o a.out
clean:
	rm -fr lab3.exe
go:
	rm -f output.ppm
	sudo ./a.out img_background.ppm img_target.ppm img_mask.pgm 130 600 output.ppm
