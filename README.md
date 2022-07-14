# Base Implementation

This is the server and single client implmentation of the spilt learning.


To run the server: 
* Have the server and convert files in the same location.
* run the server using: `python server_splitnn.py $PORT$ $DEVICE$`


To run the clinet:
* Have the client and convert files in the same location.
* run the client using: `python client_splitnn.py $IP$ $PORT$ $DEVICE$`

The `$DEVICE$` parameter can be set to 'cpu' to specifically utilize cpu. If any other value is passed, the program will check for cuda, if it is available, it will use it, otherwise, it will use cpu.
