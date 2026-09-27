"""Credential-free tests of production code; all sockets are blocked."""
import copy
import json
import socket
import sys
import tempfile
import types
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'user_data/strategies'))
from delta_hedge import SpotHedge, check_signer, floor_size, number
import DELTA_NEUTRAL as dn
from freqtrade.enums import RunMode
from freqtrade.persistence import LocalTrade, Order
from freqtrade.resolvers import StrategyResolver


class Cloid:
    def __init__(self,value): self.value=value
    @classmethod
    def from_str(cls,value): return cls(value)
    def to_raw(self): return self.value


class Venue:
    """Small exchange fixture with independently controlled execution and visibility."""
    def __init__(self):
        self.spot=0.0
        self.cash=1000.0
        self.hold=0.0
        self.perp=1.0
        self.withdrawable=1000.0
        self.leverage={'type':'isolated','value':1}
        self.orders=[]
        self.calls=[]
        self.transfers=[]
        self.fee=0.001
        self.fraction=1.0
        self.visible=True
        self.behavior='fill'
        self.abstraction='disabled'
    def post(self,*args): return self.abstraction
    def spot_meta(self):
        return {'tokens':[{'index':0,'name':'USDC','szDecimals':8},{'index':1,'name':'UBTC','szDecimals':3}],
                'universe':[{'index':7,'name':'@7','tokens':[1,0]}]}
    def user_fees(self,address): return {'userSpotCrossRate':str(self.fee)}
    def spot_user_state(self,address):
        return {'balances':[{'coin':'USDC','token':0,'total':str(self.cash),'hold':str(self.hold),'entryNtl':'0'},
                            {'coin':'UBTC','token':1,'total':str(self.spot),'hold':'0','entryNtl':'999999'}]}
    def user_state(self,address):
        positions=[] if self.perp==0 else [{'position':{'coin':'BTC','szi':str(-self.perp),'leverage':self.leverage}}]
        return {'assetPositions':positions,'withdrawable':str(self.withdrawable),
                'marginSummary':{'accountValue':'10000'},'crossMarginSummary':{'accountValue':'9000'}}
    def all_mids(self): return {'@7':'100','BTC':'99999'}
    def open_orders(self,address): return self.orders
    def market_open(self,market,buy,size,price,slippage,cloid=None):
        self.calls.append((market,buy,size,price,cloid.to_raw()))
        if self.behavior=='timeout': raise TimeoutError('lost response')
        if self.behavior=='reject': return {'status':'ok','response':{'data':{'statuses':[{'error':'rejected'}]}}}
        if self.behavior=='resting': return {'status':'ok','response':{'data':{'statuses':[{'resting':{'oid':1}}]}}}
        filled=floor_size(size*self.fraction,3)
        if self.visible:
            self.spot += filled*(1-self.fee) if buy else -filled
            self.cash += -filled*price if buy else filled*price*(1-self.fee)
        return {'status':'ok','response':{'type':'order','data':{'statuses':[{'filled':{'oid':len(self.calls),'totalSz':str(filled),'avgPx':str(price)}}]}}}
    def usd_class_transfer(self,amount,to_perp):
        self.transfers.append((amount,to_perp))
        if self.behavior=='timeout': raise TimeoutError('lost transfer response')
        if self.behavior=='reject': return {'status':'err','response':'rejected'}
        if self.visible:
            self.cash += -amount if to_perp else amount
            self.withdrawable += amount if to_perp else -amount
        return {'status':'ok','response':{'type':'default'}}


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'state.json'
        self.venue=Venue()
        self.config={'dry_run':False,'exchange':{'walletAddress':'TEST_ACCOUNT','pair_whitelist':['BTC/USDC:USDC']}}
        for method in ('connect','connect_ex','sendto'):
            blocker=patch.object(socket.socket,method,side_effect=AssertionError('Network forbidden in offline checks'))
            blocker.start(); self.addCleanup(blocker.stop)
        sdk=types.ModuleType('hyperliquid.utils.types'); sdk.Cloid=Cloid
        modules=patch.dict(sys.modules,{'hyperliquid.utils.types':sdk})
        modules.start(); self.addCleanup(modules.stop)
    def hedge(self):
        return SpotHedge(self.config,self.path,info=self.venue,exchange=self.venue,transfer_exchange=self.venue)
    def strategy(self,mode=RunMode.LIVE):
        config={'runmode':mode,'max_open_trades':2,'dry_run':mode!=RunMode.LIVE}
        strategy=dn.DELTA_NEUTRAL(config)
        strategy._initialized=True
        strategy._trading_ready=True
        strategy._hedge=self.hedge() if mode==RunMode.LIVE else None
        strategy._expected_positions={}
        strategy._rebalance_needed=False
        strategy.CURRENT_POSITION_PAIRS=[]
        strategy.BEST_PAIRS=['BTC/USDC:USDC']
        return strategy

    def test_finite_values_and_downward_size_rounding(self):
        self.assertEqual(floor_size(1.2349,3),1.234)
        for value in (-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError): number(value)

    def test_signing_agent_must_belong_to_the_actual_master_account(self):
        info=Mock()
        info.post.side_effect=[{'role':'user'},{'role':'agent','data':{'user':'TARGET'}}]
        check_signer(info,'TARGET','AGENT')
        info.post.side_effect=[{'role':'user'},{'role':'agent','data':{'user':'DIFFERENT'}}]
        with self.assertRaises(ValueError): check_signer(info,'TARGET','AGENT')
        info.post.side_effect=[{'role':'subAccount'}]
        with self.assertRaises(ValueError): check_signer(info,'TARGET','TARGET')

    def test_partial_fills_reconcile_remaining_inventory(self):
        hedge=self.hedge()
        self.venue.fraction=0.5
        self.assertFalse(hedge.reconcile())
        first=self.venue.spot
        self.venue.fraction=1
        self.assertFalse(hedge.reconcile())
        self.assertLess(self.venue.calls[1][2],self.venue.calls[0][2])
        self.assertGreater(self.venue.spot,first)
        self.assertTrue(hedge.reconcile())
        self.assertEqual(len(self.venue.calls),2)
        self.assertLess(abs(self.venue.spot-self.venue.perp),0.002)
        self.assertEqual(self.venue.calls[0][0],'@7')

    def test_acknowledged_fill_waits_for_balance_visibility(self):
        hedge=self.hedge(); self.venue.visible=False
        self.assertFalse(hedge.reconcile())
        for _ in range(3): self.assertFalse(hedge.reconcile())
        self.assertEqual(len(self.venue.calls),1)
        restarted=self.hedge()
        self.assertFalse(restarted.reconcile())
        self.assertEqual(len(self.venue.calls),1)

    def test_timeout_survives_restart_and_never_retries(self):
        hedge=self.hedge(); self.venue.behavior='timeout'
        with self.assertRaises(TimeoutError): hedge.reconcile()
        self.assertIn('cloid',json.loads(self.path.read_text())['pending'])
        self.venue.behavior='fill'
        with self.assertRaises(RuntimeError): self.hedge().reconcile()
        self.assertEqual(len(self.venue.calls),1)

    def test_journal_write_failure_sends_nothing_and_latches_closed(self):
        hedge=self.hedge()
        with patch('delta_hedge.os.replace',side_effect=OSError('disk unavailable')):
            with self.assertRaises(OSError): hedge.reconcile()
        self.assertEqual(self.venue.calls,[])
        with self.assertRaises(RuntimeError): hedge.reconcile()
        self.assertFalse(hedge.can_fund('BTC',1))
        with patch('delta_hedge.os.replace',side_effect=OSError('disk unavailable')):
            with self.assertRaises(OSError): self.hedge()

    def test_resting_and_rejected_status_are_not_fills(self):
        for behavior in ('reject','resting'):
            with self.subTest(behavior=behavior):
                if self.path.exists(): self.path.unlink()
                self.venue.behavior=behavior
                hedge=self.hedge()
                with self.assertRaises(RuntimeError): hedge.reconcile()
                self.assertFalse(hedge.ready)
                self.assertEqual(self.venue.spot,0)

    def test_partial_exit_sells_only_excess_and_floors(self):
        self.venue.spot=1.0004; self.venue.perp=0.4
        hedge=self.hedge()
        self.assertFalse(hedge.reconcile())
        self.assertFalse(self.venue.calls[0][1])
        self.assertAlmostEqual(self.venue.calls[0][2],0.6)
        self.assertTrue(hedge.reconcile())
        self.assertAlmostEqual(self.venue.spot,0.4004)

    def test_stale_perp_snapshot_and_inflight_orders_block_actions(self):
        hedge=self.hedge()
        self.assertFalse(hedge.reconcile({'BTC':0.4}))
        self.venue.orders=[{'coin':'BTC','oid':1}]
        self.assertFalse(hedge.reconcile())
        self.assertEqual(self.venue.calls,[])

    def test_entry_preflight_accounts_for_precision_and_existing_dust(self):
        self.venue.perp=0
        hedge=self.hedge(); self.assertTrue(hedge.reconcile())
        self.assertFalse(hedge.can_fund('BTC',0.1005))  # rounds below venue minimum
        self.assertTrue(hedge.can_fund('BTC',1))
        self.venue.spot=0.09  # $9 orphan dust, not an executed perp position
        self.assertTrue(hedge.reconcile())
        self.assertFalse(hedge.can_fund('BTC',0.15))  # $6 hedge gap is not tradeable
        self.assertEqual(self.venue.calls,[])

    def test_material_subminimum_gap_blocks_new_entries(self):
        self.venue.spot=0.95
        hedge=self.hedge()
        with self.assertRaises(RuntimeError): hedge.reconcile()
        self.assertFalse(hedge.ready)
        self.assertFalse(hedge.can_fund('BTC',1))
        self.assertEqual(self.venue.calls,[])

    def test_unpriced_holdings_and_corrupt_state_fail_closed(self):
        self.venue.all_mids=Mock(return_value={'@7':'nan'})
        with self.assertRaises(ValueError): self.hedge().reconcile()
        with self.assertRaises(ValueError): self.hedge().equity()
        self.path.write_text('{invalid')
        with self.assertRaises(ValueError): self.hedge()
        self.assertEqual(self.path.read_text(),'{invalid')

    def test_wrong_position_side_or_leverage_is_rejected(self):
        for perp,leverage in [(-1,1),(1,2)]:
            self.venue.perp=perp; self.venue.leverage['value']=leverage
            with self.assertRaises(ValueError): self.hedge().reconcile()
        self.assertEqual(self.venue.calls,[])

    def test_balance_is_matched_by_token_not_substring_or_entry_value(self):
        self.venue.spot_user_state=Mock(return_value={'balances':[{'coin':'UBTC_EXTRA','token':9,'total':'1','hold':'0','entryNtl':'0'}]})
        with self.assertRaises(ValueError): self.hedge().reconcile()
        self.assertEqual(self.venue.calls,[])

    def test_spot_equity_uses_spot_mid_not_perpetual_price(self):
        self.venue.spot=2
        self.assertEqual(self.hedge().equity(),11200)

    def test_fee_changes_are_refreshed_before_planning(self):
        self.venue.perp=0
        hedge=self.hedge()
        hedge._fee_updated-=3601
        self.venue.fee=0.0007
        self.assertTrue(hedge.reconcile())
        self.assertEqual(hedge.fee,0.0007)

    def test_replacement_uses_valid_low_funding_not_missing_data(self):
        strategy=self.strategy(RunMode.DRY_RUN)
        strategy.FUNDINGS={'BTC/USDC:USDC':5,'ETH/USDC:USDC':30}
        strategy.QUOTE_VOLUMES={p:3_000_000 for p in strategy.FUNDINGS}
        strategy.BEST_PAIRS=strategy.select_BEST_PAIRS(strategy.QUOTE_VOLUMES,strategy.FUNDINGS)
        self.assertEqual(strategy.BEST_PAIRS,['ETH/USDC:USDC'])
        self.assertTrue(strategy.PAIR_SHOULD_BE_REPLACED('BTC/USDC:USDC'))
        del strategy.FUNDINGS['BTC/USDC:USDC']
        self.assertFalse(strategy.PAIR_SHOULD_BE_REPLACED('BTC/USDC:USDC'))

    def test_transfer_uses_free_cash_and_withdrawable_not_equity(self):
        self.venue.perp=0; self.venue.cash=250; self.venue.hold=50; self.venue.withdrawable=0
        hedge=self.hedge(); self.assertTrue(hedge.reconcile())
        self.assertTrue(hedge.rebalance())
        self.assertEqual(self.venue.transfers,[(100.0,True)])
        self.assertFalse(hedge.rebalance())
        self.assertTrue(hedge.reconcile())

    def test_empty_accounts_do_not_divide_by_zero_or_transfer(self):
        self.venue.perp=0; self.venue.cash=0; self.venue.withdrawable=0
        hedge=self.hedge(); self.assertTrue(hedge.reconcile())
        self.assertFalse(hedge.rebalance())

    def test_transfer_timeout_persists_and_rejection_is_not_success(self):
        self.venue.perp=0; self.venue.cash=300; self.venue.withdrawable=100
        hedge=self.hedge(); self.assertTrue(hedge.reconcile())
        self.venue.behavior='reject'
        with self.assertRaises(RuntimeError): hedge.rebalance()
        self.assertIsNone(hedge.state['pending'])
        self.venue.behavior='timeout'; self.assertTrue(hedge.reconcile())
        with self.assertRaises(TimeoutError): hedge.rebalance()
        with self.assertRaises(RuntimeError): self.hedge().reconcile()
        self.assertEqual(len(self.venue.transfers),2)

    def test_unsupported_account_mode_and_dry_run_are_rejected(self):
        for mode in ('unifiedAccount','portfolioMargin','default'):
            self.venue.abstraction=mode
            with self.assertRaises(ValueError): self.hedge()
        self.config['dry_run']=True
        with self.assertRaises(ValueError): self.hedge()

    def test_account_mode_change_cannot_reuse_old_balance_assumptions(self):
        self.venue.perp=0; self.venue.cash=300; self.venue.withdrawable=100
        hedge=self.hedge(); self.assertTrue(hedge.reconcile())
        self.venue.abstraction='unifiedAccount'
        with self.assertRaises(ValueError): hedge.rebalance()
        with self.assertRaises(ValueError): hedge.reconcile()
        self.assertEqual(self.venue.transfers,[])
        self.assertEqual(self.venue.calls,[])

    def test_confirmation_never_trades_spot_and_exits_are_never_vetoed(self):
        strategy=self.strategy()
        strategy._hedge.can_fund=Mock(return_value=True)
        now=datetime.now(timezone.utc)
        with patch.object(dn.Trade,'get_open_trade_count',return_value=0):
            self.assertTrue(strategy.confirm_trade_entry('BTC/USDC:USDC','market',1,100,'gtc',now,None,'short'))
            self.assertFalse(strategy.confirm_trade_entry('BTC/USDC:USDC','market',1,100,'gtc',now,None,'short'))
        self.assertTrue(strategy.confirm_trade_exit('BTC/USDC:USDC',None,'market',1,100,'gtc','stop_loss',now))
        self.assertEqual(self.venue.calls,[])

    def test_actual_freqtrade_trade_fill_callback_uses_remaining_amount(self):
        strategy=self.strategy()
        strategy._hedge.reconcile=Mock(return_value=False)
        trade=LocalTrade(pair='BTC/USDC:USDC',amount=0.4,is_open=True,is_short=True)
        order=Order(filled=0.6)
        strategy.order_filled(trade.pair,trade,order,datetime.now(timezone.utc))
        strategy._hedge.reconcile.assert_called_once_with({'BTC':0.4})
        self.assertFalse(strategy._trading_ready)
        trade.is_open=False
        strategy.order_filled(trade.pair,trade,order,datetime.now(timezone.utc))
        self.assertEqual(strategy._expected_positions,{'BTC':0.0})

    def test_freqtrade_loads_the_published_strategy_without_account_access(self):
        project=Path(__file__).resolve().parents[1]
        config=json.loads((project/'user_data/config.json').read_text())
        config.update(strategy='DELTA_NEUTRAL',strategy_path=str(project/'user_data/strategies'),
                      user_data_dir=project/'user_data',runmode=RunMode.DRY_RUN,dry_run=True)
        strategy=StrategyResolver.load_strategy(config)
        self.assertEqual(strategy.__class__.__name__,'DELTA_NEUTRAL')
        self.assertEqual(strategy._max_positions(),2)
        self.assertFalse(strategy._initialized)

    def test_full_callback_flow_with_partial_entry_then_close(self):
        now=datetime.now(timezone.utc)
        self.venue.perp=0
        hedge=self.hedge()
        config=copy.deepcopy(self.config)
        config.update(runmode=RunMode.LIVE,max_open_trades=2,trading_mode='futures',margin_mode='isolated',stake_currency='USDC')
        config['exchange']['name']='hyperliquid'
        strategy=dn.DELTA_NEUTRAL(config)
        self.venue.funding_history=Mock(side_effect=[[
            {'coin':'BTC','time':int((now.replace(minute=0,second=0,microsecond=0)-timedelta(hours=i)).timestamp()*1000),
             'fundingRate':'0.00002'} for i in (2,1,0)],[]])
        strategy.dp=Mock()
        strategy.dp.ticker.return_value={'info':{'funding':'0.00002'},'quoteVolume':3_000_000}
        with patch.object(dn,'SpotHedge',return_value=hedge), patch.object(dn,'__file__',str(Path(self.temp.name)/'strategy.py')), \
             patch.object(dn,'log_equity'), patch.object(dn.Trade,'get_trades_proxy',return_value=[]) as trades, \
             patch.object(dn.Trade,'get_open_trade_count',return_value=0):
            strategy.bot_start()
            strategy.bot_loop_start(now)
            frame=dn.pd.DataFrame({'date':[now],'close':[100.]})
            signals=strategy.populate_entry_trend(strategy.populate_indicators(frame,{'pair':'BTC/USDC:USDC'}),{})
            self.assertEqual(signals.enter_short.iloc[-1],1)
            self.assertTrue(strategy.confirm_trade_entry('BTC/USDC:USDC','market',1,100,'gtc',now,None,'short'))
            self.assertEqual(self.venue.calls,[])
            # Only 0.6 of the proposed 1.0 perp entry actually fills.
            self.venue.perp=0.6; self.venue.withdrawable=940
            trade=LocalTrade(pair='BTC/USDC:USDC',amount=0.6,is_short=True,is_open=True,leverage=1)
            order=Order(filled=0.6)
            trades.return_value=[trade]
            strategy.order_filled(trade.pair,trade,order,now)
            self.assertLess(self.venue.calls[0][2],1)
            strategy.bot_loop_start(now)
            self.assertTrue(strategy._trading_ready)
            self.assertLess(abs(self.venue.spot-self.venue.perp),0.001)
            # Exit confirmation is pure; hedge sale follows the confirmed close.
            self.assertTrue(strategy.confirm_trade_exit(trade.pair,trade,'market',0.6,100,'gtc','stop_loss',now))
            self.assertEqual(len(self.venue.calls),1)
            self.venue.perp=0; self.venue.withdrawable=1000; trade.is_open=False
            trades.return_value=[]
            strategy.order_filled(trade.pair,trade,order,now)
            strategy.bot_loop_start(now)
            self.assertTrue(strategy._trading_ready)
            self.assertGreaterEqual(self.venue.spot,0)
            self.assertLess(self.venue.spot*100,0.10)
            self.assertEqual(len(self.venue.calls),2)

    def test_dry_run_fill_and_loop_do_not_query_private_account(self):
        strategy=self.strategy(RunMode.DRY_RUN)
        strategy._pairs=[]; strategy._info=Mock(); strategy._funding_updated=datetime.now(timezone.utc)
        strategy._equity_logged=None; strategy.nb_loop=0
        strategy.order_filled('BTC/USDC:USDC',Mock(),Mock(),datetime.now(timezone.utc))
        with patch.object(dn.Trade,'get_trades_proxy',return_value=[]):
            strategy.bot_loop_start(datetime.now(timezone.utc))
        self.assertEqual(strategy._info.mock_calls,[])

    def test_stake_handles_arbitrary_slots_minimum_and_failures(self):
        strategy=self.strategy(RunMode.DRY_RUN); strategy.config['max_open_trades']=4
        with patch.object(dn.Trade,'get_open_trade_count',return_value=1):
            self.assertEqual(strategy.custom_stake_amount('BTC/USDC:USDC',None,100,10,10,300,1,None,'short'),99.5)
            self.assertEqual(strategy.custom_stake_amount('BTC/USDC:USDC',None,100,10,100,300,1,None,'short'),0)
            self.assertEqual(strategy.custom_stake_amount('BTC/USDC:USDC',None,100,10,None,-1,1,None,'short'),0)

    def test_future_forecasts_missing_hours_and_nonfinite_funding(self):
        now=datetime(2026,1,2,12,30,tzinfo=timezone.utc); path=Path(self.temp.name)/'funding.json'
        db={dn._iso_ms(now.replace(minute=0)-timedelta(hours=i)):{'BTC/USDC:USDC':3.0} for i in range(3)}
        db[dn._iso_ms(now+timedelta(hours=1))]={'BTC/USDC:USDC':-999.0}
        dn.save_db(path,db)
        self.assertEqual(dn.avg_funding_last_hours('BTC/USDC:USDC',nb_hours=3,file_path=path,now_utc=now),3)
        db.pop(dn._iso_ms(now.replace(minute=0)-timedelta(hours=1)))
        dn.save_db(path,db)
        with self.assertRaises(ValueError): dn.avg_funding_last_hours('BTC/USDC:USDC',nb_hours=3,file_path=path,now_utc=now)
        with self.assertRaises(ValueError): dn.record_hourly_funding_by_pair('BTC/USDC:USDC',float('nan'),file_path=path)

    def test_funding_pagination_and_no_corruption_reset(self):
        now=datetime(2026,1,2,tzinfo=timezone.utc); path=Path(self.temp.name)/'funding.json'
        t=int((now-timedelta(hours=1)).timestamp()*1000)
        info=Mock(); info.funding_history.side_effect=[[
            {'coin':'BTC','time':t,'fundingRate':'0.00001'}],
            [{'coin':'BTC','time':t+3600000,'fundingRate':'0.00002'}]]
        dn.get_funding_history(path,'BTC',1,info,now)
        self.assertEqual(info.funding_history.call_args_list[1].kwargs['startTime'],t+1)
        self.assertEqual(len(json.loads(path.read_text())),2)
        path.write_text('{invalid')
        with self.assertRaises(ValueError): dn.record_hourly_funding_by_pair('BTC/USDC:USDC',1,file_path=path)
        self.assertEqual(path.read_text(),'{invalid')

    def test_settled_rates_replace_predictions_without_promoting_other_forecasts(self):
        path=Path(self.temp.name)/'funding.json'
        when=datetime(2026,1,2,12,tzinfo=timezone.utc)
        key=dn._iso_ms(when)
        dn.save_db(path,{key:{'harvested_datetime':key,'BTC/USDC:USDC':999,'ETH/USDC:USDC':999}})
        dn.update_funding_db(path,[{'time':int(when.timestamp()*1000),'fundingRate':'0.00001'}],'BTC/USDC:USDC')
        values=json.loads(path.read_text())[key]
        self.assertEqual(set(values),{'BTC/USDC:USDC'})
        self.assertAlmostEqual(values['BTC/USDC:USDC'],8.76)


if __name__=='__main__':
    unittest.main()
