#!/bin/bash

# Execute the Python script
python3 ./neurons/validator.py \
    --netuid 1 \
    --subtensor.network finney \
    --wallet.name <test_coldkey> \
    --wallet.hotkey <test_hotkey> \
    --axon.port <your_port> \