import bittensor as bt

def build_config() -> bt.Config:
    config = bt.Config()
    config.netuid = 1           # Subnet ID (example)
    config.chain_endpoint = "ws://127.0.0.1:9944"
    # "wss://test.finney.opentensor.ai:443"
    config.subtensor.network = "local"
    config.subtensor.chain_endpoint = config.chain_endpoint
    # Additional hyperparams for the captioning logic
    config.captioning = bt.Config()
    config.captioning.max_audio_size_mb = 5
    config.captioning.language = "en"
    # You can add more specialized parameters for scoring
    return config
