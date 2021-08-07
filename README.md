# object detection of medical tools with the resnet CNN pretrained model
## init (only needs to be done once)
1. pip install streamlit
2. pip install pyngrok
3. pip install tensorflow --upgrade
4. pip install Cython pandas tf-slim lvis
5. (In the models/research directory) python setup.py

## Running the website locally (make sure you are in the root directory)
1. ngrok authtoken 1ubZJpRfO6t6ivwtEosnP0IHwMm_5q9MoQmQh9UBkqknqFVRo
2. start /min streamlit run master_app.py &