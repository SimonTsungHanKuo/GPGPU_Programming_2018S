target:
	nvcc -g -std=c++11 -Wno-deprecated-gpu-targets main.cu lab1.cu -o lab1.exe
clean:
	rm -fr lab1.exe	/tmp/lab1_video.mp4
go:
	sudo ./lab1.exe
convert:
	avconv -i ./result.y4m /tmp/lab1_video.mp4
all:
	rm -fr result.y4m /tmp/lab1_video.mp4
	nvcc -g -std=c++11 -Wno-deprecated-gpu-targets main.cu lab1.cu -o lab1.exe
	sudo ./lab1.exe
	avconv -i ./result.y4m /tmp/lab1_video.mp4
	


