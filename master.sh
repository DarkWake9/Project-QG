#!/bin/bash
cd /scratch/vibhavasu.phy.iith/Project-QG/

chmod +x xerr_run_sub.sh
chmod +x xerr_run_sup.sh

# Run the first bash script
./xerr_run_sub.sh

# Check the exit status of the first script
if [ $? -eq 0 ]; then
    echo "The first script succeeded. The second script will be executed." >> /scratch/vibhavasu.phy.iith/Project-QG/log.txt
    # The first script succeeded, so run the second bash script
    ./xerr_run_sup.sh

    # Check the exit status of the second script
    if [ $? -eq 0 ]; then
        echo "The second script succeeded." >> /scratch/vibhavasu.phy.iith/Project-QG/log.txt
    else
        echo "The second script failed." >> /scratch/vibhavasu.phy.iith/Project-QG/log.txt
    fi
else
    echo "The first script failed. The second script will not be executed." >> /scratch/vibhavasu.phy.iith/Project-QG/log.txt
fi
