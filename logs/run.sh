mkdir $1
collectl -sCj > $1/"cpu-"$1 &
collectl -sm > $1/"mem-"$1 &
collectl -sn > $1/"net-"$1 &
collectl -sd > $1/"disk-"$1 
