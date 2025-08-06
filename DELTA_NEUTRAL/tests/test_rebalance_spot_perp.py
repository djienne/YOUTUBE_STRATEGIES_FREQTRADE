
TEST = False # if True, will not execute the rebalancing on the account

def _get_spot_account_available_USDC(info, address):
    spot_user_state = info.spot_user_state(address)
    #print(spot_user_state)
    for balance in spot_user_state["balances"]:
        #print(balance)
        if balance["coin"] == "USDC":
            return float(balance["total"])
    return 0

def _get_perp_account_available_USDC(info, address):
    user_state = info.user_state(address)
    #print(user_state)
    perp_user_state = account_balance_available = float(user_state['crossMarginSummary'].get('accountValue', 0))
    total_account_balance = float(user_state['marginSummary'].get('accountValue', 0))
    return account_balance_available, total_account_balance

def _get_required_env(env_name):
    import os
    from dotenv import load_dotenv
    load_dotenv()
    """Get a required environment variable or raise an informative error."""
    value = os.getenv(env_name)
    if not value or value == "your_private_key_here" or value == "your_eth_address_here":
        raise ValueError(f"{env_name} environment variable not set or has default placeholder value")
    return value

def REBALANCE_PERP_SPOT():
    from hyperliquid.exchange import Exchange
    from hyperliquid.info import Info
    from hyperliquid.utils import constants
    import eth_account
    from eth_account.signers.local import LocalAccount
    import math

    # Load credentials from environment variables
    private_key = _get_required_env("HYPERLIQUID_PRIVATE_KEY_ACCOUNT")
    address = _get_required_env("HYPERLIQUID_ADDRESS")

    # Initialize exchange
    account: LocalAccount = eth_account.Account.from_key(private_key)
    exchange = Exchange(account, constants.MAINNET_API_URL, account_address=address)

    # Fetch spot metadata
    info = Info(constants.MAINNET_API_URL, skip_ws=True)

    spot_usdc_available = _get_spot_account_available_USDC(info, address)
    perp_usdc_available, _ = _get_perp_account_available_USDC(info, address)

    amt_avilable_total = spot_usdc_available + perp_usdc_available
    target_each = amt_avilable_total / 2.0

    # deviation in percent
    def pct_diff(current, target):
        return abs(current - target) / target * 100

    spot_pct = pct_diff(spot_usdc_available, target_each)
    perp_pct = pct_diff(perp_usdc_available, target_each)

    print(f"Spot: {spot_usdc_available:.4f}, Perp: {perp_usdc_available:.4f}, target {target_each:.4f}")
    print(f"Deviations – spot: {spot_pct:.2f}%, perp: {perp_pct:.2f}%")

    amt_avilable_total = spot_usdc_available+perp_usdc_available

    # If deviation below threshold: skip
    if spot_pct < 5.0 and perp_pct < 5.0:
        print("Balances within 5% threshold — no rebalance needed")
        return False
    else:
        # decide which direction to transfer
        if spot_usdc_available > target_each:
            amt = spot_usdc_available - target_each
            amt = math.floor(amt*100.0)/100.0
            print(f"Considering transferring {amt:.2f} USDC from spot to perp")
            if amt_avilable_total < 22.0:
                print(f"But is very little USDC available ({round(amt_avilable_total,2)}), there may be a position open. No rebalacing.")
                return False
            if not TEST:
                transfer_result = exchange.usd_class_transfer(amt, True)
                print(f"Transfer result: {transfer_result}")
            else:
                print("Not doing it because it is a test.")
            return True
        else:
            amt = perp_usdc_available - target_each
            amt = math.floor(amt*100.0)/100.0
            print(f"Considering transferring {amt:.2f} USDC from perp to spot")
            if amt_avilable_total < 22.0:
                print(f"But is very little USDC available ({round(amt_avilable_total,2)}), there may be a position open. No rebalacing.")
                return False
            if not TEST:
                transfer_result = exchange.usd_class_transfer(amt, False)
                print(f"Transfer result: {transfer_result}")
            else:
                print("Not doing it because it is a test.")
            return True

if __name__=="__main__":
    REBALANCE_PERP_SPOT()
