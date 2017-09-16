objectness:
	g++ -I ~/include objectness.cpp -L ~/lib -lopencv_saliency -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -o objectness
