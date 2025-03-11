#!/bin/bash
# Execute the Python script
python3 ./neurons/miner.py \
    --netuid 1 \
    --subtensor.network finney \
    --wallet.name <your_coldkey> \
    --wallet.hotkey <your_hotkey> \
    --axon.port <your_port>
