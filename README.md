# split-learning

## Round Robin Multiclient Implementation
This is the server and multiple client implmentation of the spilt learning.

To run the server:

Have the server and convert files in the same location.
run the server using: `python server_rr_2cl.py $Total_no_of_clients$ $Starting_Port_No$ $DEVICE(cpu or gpu)$`

To run the clinet:

Have the client and convert files in the same location.
run the client using: `python client_rr_2cl.py $IP$ $PORT$ $DEVICE$ $CLIENT_NO$`.
The `$DEVICE$` parameter can be set to 'cpu' to specifically utilize cpu. If any other value is passed, the program will check for cuda, if it is available, it will use it, otherwise, it will use cpu.

To run using docker file: (will be updated ...)