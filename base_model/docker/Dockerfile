FROM python
RUN pip install torch torchvision
RUN pip install pyzmq

RUN git clone https://github.com/Yasaman-A/split-learning.git
WORKDIR split-learning
RUN git switch base_model

#CMD ["python", "server_splitnn.py", "5550"]
#CMD ["python", "client_splitnn.py", "localhost", "5550"]
#ENTRYPOINT ["python", "client_splitnn.py", "localhost", "5550"]

