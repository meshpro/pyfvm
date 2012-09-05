
default:
	@echo "\"make upload\"?"

upload:
	python setup.py sdist upload --sign
