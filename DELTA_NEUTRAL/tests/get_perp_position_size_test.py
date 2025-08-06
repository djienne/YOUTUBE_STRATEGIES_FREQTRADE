
def get_perp_position_size(coin_name: str) -> float:
    from hyperliquid.exchange import Exchange
    from hyperliquid.info import Info
    from hyperliquid.utils import constants
    import eth_account
    from eth_account.signers.local import LocalAccount
    import os
    import math

    def _get_required_env(env_name):
        """Get a required environment variable or raise an informative error."""
        value = os.getenv(env_name)
        if not value or value == "your_private_key_here" or value == "your_eth_address_here":
            raise ValueError(f"{env_name} environment variable not set or has default placeholder value")
        return value
    private_key = _get_required_env("HYPERLIQUID_PRIVATE_KEY")
    address = _get_required_env("HYPERLIQUID_ADDRESS")

    account: LocalAccount = eth_account.Account.from_key(private_key)
    exchange = Exchange(account, constants.MAINNET_API_URL, account_address=address)
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    try:
        user_state = info.user_state(address)

        # Rebuild perp position information for every entry we get back
        for entry in user_state.get("assetPositions", []):
            if entry["type"] != "oneWay" or "position" not in entry:
                continue

            pos      = entry["position"]
            c_name   = pos["coin"]
            if c_name == coin_name:
                return float(pos["szi"])
        
    except Exception as exc:
        print(f"Failed to get perp positions: {exc}", exc_info=True)
        return None
        # fall through – will still try to read any cached data

    return 0.0
    

print(get_perp_position_size("PUMP"))
