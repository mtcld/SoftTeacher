pre:
	python -m pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
	mkdir -p thirdparty
	git clone https://github.com/thangnx183/CBNetV2.git thirdparty/mmdetection
	cd thirdparty/mmdetection && python -m pip install -e .
install:
	make pre
	python -m pip install -e .
clean:
	rm -rf thirdparty
	rm -r ssod.egg-info
