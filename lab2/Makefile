target:
	nvcc -g -std=c++11 -Wno-deprecated-gpu-targets main.cu counting.cu -o lab2.exe
clean:
	rm -fr lab2.exe
go:
	sudo ./lab2.exe
convert:
	avconv -i ./result.y4m /tmp/lab1_video.mp4
all:
	rm -fr result.y4m /tmp/lab1_video.mp4
	nvcc -g -std=c++11 -Wno-deprecated-gpu-targets main.cu lab1.cu -o lab1.exe
	sudo ./lab1.exe
	avconv -i ./result.y4m /tmp/lab1_video.mp4
	


