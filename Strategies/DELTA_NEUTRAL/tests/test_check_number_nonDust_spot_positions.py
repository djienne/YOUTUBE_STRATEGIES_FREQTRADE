

def _get_required_env(env_name):
    import os
    """Get a required environment variable or raise an informative error."""
    value = os.getenv(env_name)
    if not value or value == "your_private_key_here" or value == "your_eth_address_here":
        raise ValueError(f"{env_name} environment variable not set or has default placeholder value")
    return value

def GET_NUMBER_SPOT_POSITION():
    from hyperliquid.info import Info
    from hyperliquid.utils import constants

    def count_spot_position(info, address):
        spot_user_state = info.spot_user_state(address)
        #print(spot_user_state)
        count = sum(
            1 for item in spot_user_state['balances']
            if item['coin'] != 'USDC' and float(item['entryNtl']) > 11
        )
        return count

    # Load credentials from environment variables
    address = _get_required_env("HYPERLIQUID_ADDRESS")

    # Fetch spot metadata
    info = Info(constants.MAINNET_API_URL, skip_ws=True)

    return count_spot_position(info, address)

if __name__=="__main__":
    print(GET_NUMBER_SPOT_POSITION())
