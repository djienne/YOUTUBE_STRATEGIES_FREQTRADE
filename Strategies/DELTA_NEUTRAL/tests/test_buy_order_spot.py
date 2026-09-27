def is_order_error(response: dict) -> bool:
    """Return True if the Hyperliquid order response contains an error."""
    if response.get("status") != "ok":
        # Unexpected structure
        return True

    resp = response.get("response", {})
    data = resp.get("data", {})
    statuses = data.get("statuses", [])

    # Look for any status item containing an "error" key
    for status in statuses:
        if isinstance(status, dict) and "error" in status:
            return True
    return False

def _get_spot_price(coin_name):
    from hyperliquid.info import Info
    from hyperliquid.utils import constants
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    mid_price = info.all_mids()
    for key, value in mid_price.items():
        if key == coin_name:
            return float(value)
    return 0

def get_coin_info(coin):
    
    def count_decimal_digits(num):
        import math
        return math.log10(num)*-1
    
    match coin:
        case "BTC":
            size_tick = 0.00001
            price_tick = 1
            return {
                "HL_spot_pair": "UBTC/USDC",
                "size_tick": size_tick,
                "size_decimal_digits": count_decimal_digits(size_tick),
                "price_tick": price_tick,
                "price_decimal_digits": count_decimal_digits(price_tick)
            }
        case "ETH":
            size_tick = 0.0001
            price_tick = 0.1
            return {
                "HL_spot_pair": "UETH/USDC",
                "size_tick": size_tick,
                "size_decimal_digits": count_decimal_digits(size_tick),
                "price_tick": price_tick,
                "price_decimal_digits": count_decimal_digits(price_tick)
            }
        case "SOL":
            size_tick = 0.001
            price_tick = 0.01
            return {
                "HL_spot_pair": "USOL/USDC",
                "size_tick": size_tick,
                "size_decimal_digits": count_decimal_digits(size_tick),
                "price_tick": price_tick,
                "price_decimal_digits": count_decimal_digits(price_tick)
            }
        case "HYPE":
            size_tick = 0.01
            price_tick = 0.001
            return {
                "HL_spot_pair": "HYPE/USDC",
                "size_tick": size_tick,
                "size_decimal_digits": count_decimal_digits(size_tick),
                "price_tick": price_tick,
                "price_decimal_digits": count_decimal_digits(price_tick)
            }
        case "PUMP":
            size_tick = 1
            price_tick = 0.0000001
            return {
                "HL_spot_pair": "UPUMP/USDC",
                "size_tick": size_tick,
                "size_decimal_digits": count_decimal_digits(size_tick),
                "price_tick": price_tick,
                "price_decimal_digits": count_decimal_digits(price_tick)
            }
        case "FARTCOIN":
            size_tick = 0.1
            price_tick = 0.0001
            return {
                "HL_spot_pair": "UFART/USDC",
                "size_tick": size_tick,
                "size_decimal_digits": count_decimal_digits(size_tick),
                "price_tick": price_tick,
                "price_decimal_digits": count_decimal_digits(price_tick)
            }
        case "PURR":
            size_tick = 1
            price_tick = 0.00001
            return {
                "HL_spot_pair": "PURR/USDC",
                "size_tick": size_tick,
                "size_decimal_digits": count_decimal_digits(size_tick),
                "price_tick": price_tick,
                "price_decimal_digits": count_decimal_digits(price_tick)
            }
        case _:
            raise ValueError(f"{coin} is unsupported coin")

def HL_buy_spot_market(coin, spot_size):
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

    def floor_to_n_digits(value, n):
        factor = 10 ** n
        return math.floor(value * factor) / factor
    
    coin_info = get_coin_info(coin)
    HL_spot_pair = coin_info['HL_spot_pair']
    size_decimal_digits = coin_info['size_decimal_digits']
    price_decimal_digits = coin_info['price_decimal_digits']

    # Load credentials from environment variables
    private_key = _get_required_env("HYPERLIQUID_PRIVATE_KEY")
    address = _get_required_env("HYPERLIQUID_ADDRESS")

    account: LocalAccount = eth_account.Account.from_key(private_key)
    exchange = Exchange(account, constants.MAINNET_API_URL, account_address=address)
    info = Info(constants.MAINNET_API_URL, skip_ws=True)

    resp = exchange.update_leverage(3, 'BTC', is_cross=True)

    #print(resp)
    #spot_meta = info.spot_meta()
    #spot_coins = spot_meta["tokens"]

    #print(spot_coins)
    last_price = _get_spot_price(coin)
    #print(last_price)
    rounded_spot_buy_size = floor_to_n_digits(spot_size, size_decimal_digits)
    #print(price_decimal_digits)
    limit_buy_price = floor_to_n_digits(last_price*1.05, price_decimal_digits)
    #print(limit_buy_price)
    # True -> buy
    spot_order_result = exchange.order(HL_spot_pair, True, rounded_spot_buy_size, limit_buy_price, {"limit": {"tif": "Gtc"}})
    if is_order_error(spot_order_result):
        print(spot_order_result)
        raise ValueError('sell spot_order_result error')
    print(spot_order_result)

def HL_sell_spot_market(coin):
    from hyperliquid.exchange import Exchange
    from hyperliquid.info import Info
    from hyperliquid.utils import constants
    import eth_account
    from eth_account.signers.local import LocalAccount
    import os
    import math

    def count_decimal_digits(num):
        if num == 0:
            return 0
        s = f"{num:.20f}".rstrip('0')  # Limit precision and remove trailing zeros
        if '.' in s:
            return len(s.split('.')[1])
        return 0

    def _get_required_env(env_name):
        """Get a required environment variable or raise an informative error."""
        value = os.getenv(env_name)
        if not value or value == "your_private_key_here" or value == "your_eth_address_here":
            raise ValueError(f"{env_name} environment variable not set or has default placeholder value")
        return value
    
    def get_spot_position_size(info, address, wanted_coin_name):
        if wanted_coin_name=='PUMP':
            wanted_coin_name = 'UPUMP'
        if wanted_coin_name=='FARTCOIN':
            wanted_coin_name = 'UFART'
        spot_user_state = info.spot_user_state(address)
        #print(spot_user_state)
        for balance in spot_user_state.get("balances", []):
            if float(balance["total"]) > 0:
                coin_name = balance["coin"]
                #print(coin_name)
                if wanted_coin_name in coin_name:
                    return float(balance["total"])
        return 0.0

    def floor_to_n_digits(value, n):
        factor = 10 ** n
        return math.floor(value * factor) / factor
    
    coin_info = get_coin_info(coin)
    HL_spot_pair = coin_info['HL_spot_pair']
    size_decimal_digits = coin_info['size_decimal_digits']
    price_decimal_digits = coin_info['price_decimal_digits']

    # Load credentials from environment variables
    private_key = _get_required_env("HYPERLIQUID_PRIVATE_KEY")
    address = _get_required_env("HYPERLIQUID_ADDRESS")

    account: LocalAccount = eth_account.Account.from_key(private_key)
    exchange = Exchange(account, constants.MAINNET_API_URL, account_address=address)
    info = Info(constants.MAINNET_API_URL, skip_ws=True)

    spot_size = get_spot_position_size(info, address, coin)

    #print(spot_size)

    #spot_meta = info.spot_meta()
    #spot_coins = spot_meta["tokens"]

    #print(spot_coins)
    last_price = _get_spot_price(coin)
    #print(last_price)
    rounded_spot_sell_size = floor_to_n_digits(spot_size, size_decimal_digits)
    limit_sell_price = floor_to_n_digits(last_price*0.95, price_decimal_digits)
    # True -> buy
    spot_order_result = exchange.order(HL_spot_pair, False, rounded_spot_sell_size, limit_sell_price, {"limit": {"tif": "Gtc"}})
    if is_order_error(spot_order_result):
        print(spot_order_result)
        raise ValueError('sell spot_order_result error')
    print(spot_order_result)

if __name__=="__main__":
    import time
    coin = 'FARTCOIN'
    USDC_to_buy = 12.0
    coin_price = _get_spot_price(coin)
    print(coin_price)
    size_to_buy = USDC_to_buy/float(coin_price)
    print(size_to_buy)
    HL_buy_spot_market(coin, size_to_buy)
    time.sleep(5)
    HL_sell_spot_market(coin)