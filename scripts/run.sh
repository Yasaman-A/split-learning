#! /usr/bin/bash


#parse launch for client or server
case "$1" in
    --server)
        device="split-server"
        ;;
    --fed)
        device="fed-server"
        ;;
    --client)
        device="$2"
        ;;
    *)
        echo "Usage: $0 --server | --fed | --client <n>"
        exit 1
        ;;
esac

time=$(date +%F_%H_%M_%S)
TERMINAL=$(tty)


#global array to keep track of logging processes.
loggers=()

PYTHON=~/venv/bin/python3

#terminate_loggers
#kills all logging functions of that round. Pevents logging beyond scope of runs.
function terminate_loggers(){
    for pid in "${loggers[@]}"; do
        kill $pid
    done
    loggers=()
}



#collect_data()
#inputs:
# $1 -> output directory.
#    -> Should be /$HOME/dir/.../dir/
# $2 -> current iteration
function collect_data() {    
    mkdir -p "$1$2" || { echo "Failed to create directory"; exit 1; }

    collectl -oT -sCj > "$1$2/cpu.txt"&
    loggers+=($!)
    collectl -oT -sM > "$1$2/mem.txt"&
    loggers+=($!)
    sar -n DEV --iface=ens4 1 > "$1$2/net.txt"&
    loggers+=($!)
}

# 1 = config
# 2 = num clients
# 3 = split ip
# 4 = split port
# 5 = fed ip
# 6 = fed port
# 7 = data ip
# 8 = data port
# 9 = output_file -> data file used by data server
function change_setup_config() {
    clear
    yq -yi ".client_total = $2" "$1"
    yq -yi ".split_server.server_ip = \"tcp://$3\"" "$1"
    yq -yi ".split_server.server_start_port = $4" "$1"
    
    yq -yi ".fed_server.server_ip = \"tcp://$5\"" "$1"
    yq -yi ".fed_server.server_start_port = $6" "$1"

    yq -yi ".data_server.server_address = \"http://$7:$8\"" "$1" 
    yq -yi ".data_server.output_file = $9" "$1"

}


# 1 = config
# 2 = cut layer
# 3 = epoch
# 4 = round
# 5 = split_type
# 6 = patience
# 7 = test_cut_layer
# 8 = custom_cut_mode

function update_hyperparameters() {

    if [[ $8 == "false" ]]; then
        yq -yi ".cut_layer = $2" "$1"
    fi    
    yq -yi ".epoch = $3" "$1"
    yq -yi ".round = $4" "$1"
    yq -yi ".split_type = \"$5\"" "$1"
    yq -yi ".patience = $6" "$1"
    yq -yi ".test_cut_Layer = $7" "$1"

}


#run_client
#inputs: $1 = output directory
#        $2 = current iteration
#        $3 = custom split information
function run_client() {
    
    cd /$HOME/split-learning
    
    #launch based on client name
    if [[ "$device" == "split-server" ]]; then
        echo "Starting the Split Server"
        
        collect_data "$1" "$2"
        
        if [[ "$3" != *','* ]]; then
            $PYTHON -m src.split-learning --mode splitfed_v1 --server
        else
            $PYTHON -m src.split-learning --mode splitfed_v1_custom_cut --server --extra $3
        fi

        kill $server_pid

    elif [[ "$device" == "fed-server" ]]; then
        echo "starting the data server"
        
        $PYTHON -m http.server 8000&
        server_pid=$!
        trap "kill $server_pid" EXIT
        
        echo "Starting the fed server."
        collect_data "$1" "$2"
    
        if [[ "$3" != *','* ]]; then
            $PYTHON -m src.split-learning --mode splitfed_v1 --fed
        else
            $PYTHON -m src.split-learning --mode splitfed_v1_custom_cut --fed
        fi

    #check for client num
    elif [[ "$device" =~ ^[0-9]+$ ]]; then
        client_num=$device
        echo "Starting client-$client_num";
        collect_data "$1" "$2" 
        
        if [[ "$3" != *','* ]]; then
            $PYTHON -m src.split-learning --mode splitfed_v1 --client $client_num
        else
            IFS=',' read -r -a split_points <<< "$3"
            split_point=${split_points[$client_num - 1]}
            echo "Split point set to $split_point"
            $PYTHON -m src.split-learning --mode splitfed_v1_custom_cut --client $client_num --extra $split_point
        fi
    else
        echo "Invalid device configuration. Did you specify the right name on launch?"
        echo "Device name: $device"
    fi
    
    terminate_loggers

    deactivate

    mv logs/*.log "$1$2/"
}

#=========================================================================================





#===========================
# Automatic
#===========================


#run auto
# 1 = filepath to file containing run hyperparameters
# 2 = parameter file
# 3 = custom_cut_mode
function run_auto() {
    currRun=0

    output_dir="/$HOME/auto_experiments/$time/"


    while IFS= read -r line
    do
        run_params=($line)

        update_hyperparameters "$1" "${run_params[@]}" "$3"

        run_client "$output_dir" "$currRun" "${run_params[0]}"
        
        ((currRun++))

    done < "$2"

}


#Automatic()
# 1 = input config dir
# 2 = custom_cut_mode
function automatic() {
    
    file=$(dialog --title "Pick automation file" --fselect "$HOME/" $HEIGHT $WIDTH 2>&1 >$TERMINAL)
    result=$?

    if [ "$result" -eq 1 || "$result" -eq 255 ]; then
        dialog --infobox "Cancelled File Selection" 10 30
        sleep 1
        main_menu
    elif [ "$result" -eq 0 ]; then
        dialog --infobox "Running from file $file!" 10 30
        sleep 1

        clear

        run_auto "$1" "$file" "$2"
    else
        dialog --infobox "Unexpected error! Dialog returned some arbitrary value" 10 30
        clear
        main_menu
        fi

}



#===========================
# Manual mode
#===========================
# $1 = config
# $2 = custom_cut_mode

function manual_input() {

    if [[ $2 == false ]]; then
        echo "0" | dialog --no-clear --gauge "Getting Config Values" 15 50 0
        cut_layer=$(yq '.cut_layer' "$1")
    else
        cut_layer="X,X,X"
    fi
    
    echo "16" | dialog --no-clear --gauge "Getting Config Values" 15 50 25
    epoch=$(yq '.epoch' "$1")
    
    echo "33" | dialog --no-clear --gauge "Getting Config Values" 15 50 25
    round=$(yq '.round' "$1")

    echo "50" | dialog --no-clear --gauge "Getting Config Values" 15 50 25
    split_type=$(yq '.split_type' "$1" | sed 's/^"\|"$//g')

    echo "66" | dialog --no-clear --gauge "Getting Config Values" 15 50 25
    patience=$(yq '.patience' $1)

    echo "83" | dialog --no-clear --gauge "Getting Config Values" 15 50 25
    test_cut_Layer=$(yq 'test_cut_layer' $1)

    OPTIONS=("Cut Layer:" 1 1 "$cut_layer" 1 20 30 0
             "Epochs:" 2 1 "$epoch" 2 20 30 0
             "Rounds:" 3 1 "$round" 3 20 30 0
             "Split Type:" 4 1 "$split_type" 4 20 30 0
             "Patience:" 5 1 "$patience" 5 20 30 0
             "Test Cut layer:" 6 1 $test_cut_layer 6 20 30 0)

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
        patience=$(echo "$CHOICE" | sed -n '5p')
        test_cut_layer=$(echo "$CHOICE" | sed -n '6p')

        update_hyperparameters "$1" "$cut_layer" "$epoch" "$round" "$split_type" "$patience" "$test_cut_layer" $2
        dialog --infobox "Successfully updated hyperparameters" 10 30
        sleep 2

        clear

        run_client "$output_dir" "0" "$cut_layer"

    else
        dialog --infobox "Aborted changes to hyperparameters" 10 30
        sleep 2
        manual "$1" $2
    fi

}


# $1 = config
# $2 = custom_cut_mode
function manual() {

    
    output_dir="/$HOME/manual_experiments/$time/"

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
            if [[ $2 == "true" ]]; then

                cut_layer=$(dialog --title "Cut Layer?" \
                       --inputbox "Enter All cut layers separated by commas e.g. 3,5,6" \
                       $HEIGHT $WIDTH $CHOICE_HEIGHT \
                       2>&1 >$TERMINAL)

                option=$?

                if [[ $option > 0 ]]; then
                    manual "$1" $2
                    exit
                fi
            
            else
                cut_layer=""
            fi
            
            clear
            run_client "$output_dir" "0" "$cut_layer"
            ;;
        2) manual_input "$1" $2
            ;;
        *) clear
           main_menu "$1" $2
            ;;
    esac

}


#===========================
# Modify config
#===========================
#
# 1 = config path
# 2 = custom_cut_mode
#
function modify_config() {
    
    echo "0" | dialog --no-clear --gauge "Getting Config Values" 15 50 0
    num_clients=$(yq '.client_total' "$1")
    
    echo "14" | dialog --no-clear --gauge "Getting Config Values" 15 50 17
    split_ip=$(yq '.split_server.server_ip' "$1" | sed 's/^"tcp:\/\///;s/"$//')
    
    echo "29" | dialog --no-clear --gauge "Getting Config Values" 15 50 33
    split_port=$(yq '.split_server.server_start_port' "$1")

    echo "43" | dialog --no-clear --gauge "Getting Config Values" 15 50 50
    fed_ip=$(yq '.fed_server.server_ip' "$1" | sed 's/^"tcp:\/\///;s/"$//')

    echo "57" | dialog --no-clear --gauge "Getting Config Values" 15 50 66
    fed_port=$(yq '.fed_server.server_start_port' "$1")

    echo "71" | dialog --no-clear --gauge "Getting Config Values" 15 50 66
    data_address=$(yq '.data_server.server_address' "$1" | sed 's/^"//;s/"$//')
    data_ip=$(echo "$data_address" | sed 's|http://\([^:]*\):.*|\1|')
    data_port=$(echo "$data_address" | sed 's|.*:\([0-9]*\)|\1|')

    echo "86" | dialog --no-clear --gauge "Getting Config Values" 15 50 83
    output=$(yq '.data_server.output_file' "$1")

    
    OPTIONS=("Number of clients" 1 1 "$num_clients" 1 20 30 0
             "Split Server IP" 2 1 "$split_ip" 2 20 30 0
             "Split Server Port" 3 1 "$split_port" 3 20 30 0
             "Fed Server IP" 4 1 "$fed_ip" 4 20 30 0
             "Fed Server Port" 5 1 "$fed_port" 5 20 30 0
             "Data Server IP" 6 1 "$data_ip" 6 20 30 0
             "Data Server Port" 7 1 "$data_port" 7 20 30 0
             "Output File" 8 1 "$output" 8 20 30 0)
    
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
        data_ip=$(echo "$CHOICE" | sed -n '6p')
        data_port=$(echo "$CHOICE" | sed -n '7p')
        output=$(echo "$CHOICE" | sed -n '8p')
        
        change_setup_config "$1" "$num_clients" "$split_ip" "$split_port" "$fed_ip" "$fed_port" "$data_ip" "$data_port" "$output"

        dialog --infobox "Config successfully updated." 10 30
        sleep 2
    else
        dialog --infobox "Aborted changes to config" 10 30
        sleep 2
    fi
    
    clear

    main_menu "$1" $2

    #case choice in 
}


HEIGHT=15
WIDTH=50
CHOICE_HEIGHT=4



#main_menu
# 1 = config path
# 2 = custom_cut_mode
function main_menu() {

    BACKTITLE="Split Learning Executor"

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
            automatic "$1" $2
            ;;
        2)
            manual "$1" $2
            ;;
        3)
            modify_config "$1" $2
            ;;
        *)
            home
            ;;
    esac
}

function home() {
    
    OPTIONS=(1 "SplitFedV1"
            2 "SplitFedV1 with Custom Cut")

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
            CONFIG="/$HOME/split-learning/src/split-learning/splitfed_v1/config.yaml"
            main_menu "$CONFIG" false
            ;;
        2)
            CONFIG="/$HOME/split-learning/src/split-learning/splitfed_v1_custom_cut/config.yaml"
            main_menu "$CONFIG" true
            ;;
    esac
}



trap clear SIGINT

home
