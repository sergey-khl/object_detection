# object detection of medical tools with the resnet CNN pretrained model
## init (only needs to be done once) :frog:
1. pip install streamlit
2. pip install pyngrok
3. pip install tensorflow --upgrade
4. pip install Cython pandas tf-slim lvis
5. (In the root directory) git clone https://github.com/tensorflow/models
6. cd models/research
6. copy object_detection\packages\tf2\setup.py setup.py
7. Add the protoc bin folder to path: setx path "%path%;c:\directoryPath"
8. protoc object_detection\protos\*.proto --python_out=.
9. python setup.py install

## Running the website locally (make sure you are in the root directory) :chicken:
1. ngrok authtoken 1ubZJpRfO6t6ivwtEosnP0IHwMm_5q9MoQmQh9UBkqknqFVRo
2. start /min streamlit run master_app.py &