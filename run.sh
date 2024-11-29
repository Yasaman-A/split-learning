#! /usr/bin/bash

device=$(hostname)
time=$(date +%F_%H_%M_%S)
TERMINAL=$(tty)
#collect_data()
#inputs:
# $1 -> output directory.
#    -> Should be /$HOME/dir/.../dir/
# $2 -> current iteration
function collect_data() {    
    mkdir -p "$1$2" || { echo "Failed to create directory"; exit 1; }

    collectl -oT -sCj > "$1$2/cpu.txt"&
    collectl -oT -sM > "$1$2/mem.txt"&
    sar -n DEV --iface=ens4 1 > "$1$2/net.txt"&
}

# 1 = config
# 2 = num clients
# 3 = split ip
# 4 = split port
# 5 = fed ip
# 6 = fed port
# 7 = output_file -> data file used by data server
function change_setup_config() {
    clear
    yq -yi ".client_total = $2" "$1"
    yq -yi ".split_server.server_ip = \"tcp://$3\"" "$1"
    yq -yi ".split_server.server_start_port = $4" "$1"
    
    yq -yi ".fed_server.server_ip = \"tcp://$5\"" "$1"
    yq -yi ".fed_server.server_start_port = $6" "$1"

    yq -yi ".data_server.server_address = \"http://$3:8000\"" "$1" #keep data server on split-server
    yq -yi ".data_server.output_file = $7" "$1"

}


# 1 = config
# 2 = cut layer
# 3 = epoch
# 4 = round
# 5 = split_type
function update_hyperparameters() {

 #src/split-learning/splitfed_v1/config

    yq -yi ".cut_layer: = $2" "$1"
    yq -yi ".epoch: = $3" "$1"
    yq -yi ".round: = $4" "$1"
    yq -yi ".split_type: = $5" "$1"
}


#run_client
#inputs: $1 = output directory
#        $2 = current iteration
function run_client() {
    
    cd /$HOME/split-learning
    source venv/bin/activate
    
    #launch based on client name
    if [[ "$device" == "split-server" ]]; then
        echo "Starting the Split Server and Data Server"
        server_ip=$(ip address show dev ens4 | awk '/inet / {split($2, a, "/"); print a[1]}')
        python3 -m http.server -b $server_ip 8000 -d data&
        server_pid=$!
        trap "kill $server_pid" EXIT
        
        collect_data "$1" "$2"
        python3 -m src.split-learning --mode splitfed_v1 --server

        kill $server_pid

    elif [[ "$device" == "fed-server" ]]; then
        echo "Starting the fed server."
        collect_data "$1" "$2"
        python3 -m src.split-learning --mode splitfed_v1 --fed

    elif [[ "$device" =~ ^client-[0-9]+ ]]; then
        sleep 3 #give the split-server enough time to start up
        client_num="${device##*-}"
        client_num="${client_num#"0"}"
        client_num="${client_num:-0}"
        echo "Starting client-$client_num";
        collect_data "$1" "$2" 
        python3 -m src.split-learning --mode splitfed_v1 --client $client_num
    else
        echo "Invalid device configuration. Did you set up the instance correctly?"
        echo "Device name: $device"
    fi

    deactivate
}

#=========================================================================================





#===========================
# Automatic
#===========================


#run auto
# 1 = filepath to file containing run hyperparameters
function run_auto() {
    currRun=0

    output_dir="/$HOME/auto_experiments/$time/"


    #TODO: Add progress bar of some sort?
    while IFS= read -r line
    do
        run_params=($line)
        
        update_hyperparameters "${run_params[@]}"

        run_client "$output_dir" "$currRun" 
        
        ((currRun++))

    done < "$1"

}


#Automatic()
#no inputs
function automatic() {
    
    dialog --title "Pick automation file" --fselect "$HOME/" $HEIGHT $WIDTH
    result=$?
    
    if [ "$result" -eq 1 ]; then
        dialog --infobox "Cancelled File Selection" 10 30
        sleep 1
        main_menu
    elif [ "$result" -eq 0 ]; then
        dialog --infobox "No file selected" 10 30
        sleep 1
        main_menu
    else
        dialog --infobox "Running from file $result!" 10 30
        sleep 1

        clear

        run_auto "$result"
        fi

}



#===========================
# Manual mode
#===========================
# $1 = config

function manual_input() {

    echo "0" | dialog --no-clear --gauge "Getting Config Values" 15 50 0
    cut_layer=$(yq '.cut_layer' "$1")
    
    echo "25" | dialog --no-clear --gauge "Getting Config Values" 15 50 25
    epoch=$(yq '.epoch' "$1")
    
    echo "50" | dialog --no-clear --gauge "Getting Config Values" 15 50 25
    round=$(yq '.round' "$1")

    echo "75" | dialog --no-clear --gauge "Getting Config Values" 15 50 25
    split_type=$(yq '.split_type' "$1")

    OPTIONS=("Cut Layer:" 1 1 "$cut_layer" 1 20 30 0
             "Epochs:" 2 1 "$epoch" 2 20 30 0
             "Rounds:" 3 1 "$round" 3 20 30 0
             "Split Type:" 4 1 "$split_type" 4 20 30 0)

    CHOICE=$(dialog --clear \
            --backtitle "$BACKTITLE - Manual" \
            --title "" \
            --form "hyperparameters" \
            $HEIGHT $WIDTH $CHOICE_HEIGHT \
            "${OPTIONS[@]}" \
            2>&1 > $TERMINAL)
    
    if [ $? -eq 0 ]; then
        cut_layer=$(echo "$CHOICE" | sed -n '1p')
        epoch=$(echo "$CHOICE" | sed -n '2p')
        round=$(echo "$CHOICE" | sed -n '3p')
        split_type=$(echo "$CHOICE" | sed -n '4p')

        update_hyperparameters "$1" "$cut_layer" "$epoch" "$round" "$split_type"
        dialog --infobox "Successfully updated hyperparameters" 10 30
        sleep 2

        run_client "$output_dir" "0"

    else
        dialog --infobox "Aborted changes to hyperparameters" 10 30
        sleep 2
        manual "$1"
    fi

}




# $1 = config
function manual() {

    
    output_dir="$/HOME/manual_experiments/$time/"

    clear 

    OPTIONS=(1 "Run with current hyperparameters"
             2 "Run with new hyperparameters")
    
    CHOICE=$(dialog --clear \
                    --backtitle "$BACKTITLE - Manual" \
                    --title "" \
                    --menu "Run options" \
                    $HEIGHT $WIDTH $CHOICE_HEIGHT \
                    "${OPTIONS[@]}" \
                    2>&1 >$TERMINAL)


    case $CHOICE in 
        1) clear 
            run_client "$output_dir" "0"
            sleep 30
            ;;
        2) manual_input "$1"
            ;;
        *) clear
           main_menu
            ;;
    esac

}


#===========================
# Modify config
#===========================
function modify_config() {
    
    echo "0" | dialog --no-clear --gauge "Getting Config Values" 15 50 0
    num_clients=$(yq '.client_total' "$1")
    
    echo "17" | dialog --no-clear --gauge "Getting Config Values" 15 50 17
    split_ip=$(yq '.split_server.server_ip' "$1" | sed 's/^"tcp:\/\///;s/"$//')
    
    echo "33" | dialog --no-clear --gauge "Getting Config Values" 15 50 33
    split_port=$(yq '.split_server.server_start_port' "$1")

    echo "50" | dialog --no-clear --gauge "Getting Config Values" 15 50 50
    fed_ip=$(yq '.fed_server.server_ip' "$1" | sed 's/^"tcp:\/\///;s/"$//')

    echo "66" | dialog --no-clear --gauge "Getting Config Values" 15 50 66
    fed_port=$(yq '.fed_server.server_start_port' "$1")

    echo "83" | dialog --no-clear --gauge "Getting Config Values" 15 50 83
    output=$(yq '.data_server.output_file' "$1")
    
    OPTIONS=("Number of clients" 1 1 "$num_clients" 1 20 30 0
             "Split Server IP" 2 1 "$split_ip" 2 20 30 0
             "Split Server Port" 3 1 "$split_port" 3 20 30 0
             "Fed Server IP" 4 1 "$fed_ip" 4 20 30 0
             "Fed Server Port" 5 1 "$fed_port" 5 20 30 0
             "Output File" 6 1 "$output" 6 20 30 0)
    
    CHOICE=$(dialog --clear \
                    --backtitle "$BACKTITLE - Configuration" \
                    --title "" \
                    --form "Configuration" \
                    $HEIGHT $WIDTH 0 \
                    "${OPTIONS[@]}" \
                    2>&1 >$TERMINAL)

    if [ $? -eq 0 ]; then
        num_clients=$(echo "$CHOICE" | sed -n '1p')
        split_ip=$(echo "$CHOICE" | sed -n '2p')
        split_port=$(echo "$CHOICE" | sed -n '3p')
        fed_ip=$(echo "$CHOICE" | sed -n '4p')
        fed_port=$(echo "$CHOICE" | sed -n '5p')
        output=$(echo "$CHOICE" | sed -n '6p')
        
        change_setup_config "$1" "$num_clients" "$split_ip" "$split_port" "$fed_ip" "$fed_port" "$output"

        dialog --infobox "Config successfully updated." 10 30
        sleep 2
    else
        dialog --infobox "Aborted changes to config" 10 30
        sleep 2




    fi
    
    clear

    main_menu

    #case choice in 
}


HEIGHT=15
WIDTH=50
CHOICE_HEIGHT=4

function main_menu() {

    BACKTITLE="Split Learning Executor"
    CONFIG="/$HOME/split-learning/src/split-learning/splitfed_v1/config.yaml"

    OPTIONS=(1 "Automatic (file driven)"
            2 "Manual (user driven)"
            3 "Configure Setup Variables")

    CHOICE=$(dialog --clear \
                    --backtitle "$BACKTITLE" \
                    --title "Operating mode" \
                    --menu "Choose one of the following operating modes" \
                    $HEIGHT $WIDTH $CHOICE_HEIGHT \
                    "${OPTIONS[@]}" \
                    2>&1 >$TERMINAL)

    clear

    case $CHOICE in 
        1)
            automatic "$CONFIG"
            ;;
        2)
            manual "$CONFIG"
            ;;
        3)
            modify_config "$CONFIG"
            ;;
    esac


}


trap clear SIGINT

main_menu
