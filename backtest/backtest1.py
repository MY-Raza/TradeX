import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class Backtest():
	def __init__(self, obj_dataset, df_predictions, starting_balance=1000, take_profit=100, stop_loss=100, buy_after_minutes=0, transaction_fee=0.05, leverage=1.0, slippage=0.0):

		df_predictions['datetime'] = pd.to_datetime(df_predictions['datetime'])
		self.index_pred_datetime = df_predictions.columns.get_loc("datetime")
		self.index_pred_direction = df_predictions.columns.get_loc("predicted_direction")
		self.np_model_predctions = df_predictions.to_numpy()
		# print(df_predictions)
		# print(obj_dataset.np_1m)
		# print(len(df_predictions))

		self.pnl_percent_all = 0
		self.starting_balance = self.current_balance = starting_balance
		self.breaking_balance = self.current_balance * 0.5
		
		self.buy_price = 0
		self.sell_price = 0
		self.leverage = leverage
		self.slippage = slippage 
		
		self.take_profit_percent = take_profit / 100
		self.stop_loss_percent = stop_loss / 100
		self.buy_after_minutes = int(buy_after_minutes) #in minutes
		self.transaction_fee_percent = transaction_fee * self.leverage
		self.in_position = False
		self.array_to_save = []						
		self.header_names = [
						'datetime',
						'predicted_direction',
						'action',
						'buy_price',
						'sell_price',
						'balance',
						'pnl'
					]		
		

		#
		self.obj_dataset = obj_dataset
		# print(self.obj_dataset.np_1m)
			
		self.index_1m_open = obj_dataset.index_open
		self.index_1m_high = obj_dataset.index_high
		self.index_1m_low = obj_dataset.index_low
		self.index_1m_datetime = obj_dataset.index_datetime

	def buy(self, np_temp):
		self.buy_price = np_temp[self.buy_after_minutes][self.index_1m_open]
		self.sell_price = 0
		pnl = self.transaction_fee_percent * -1
		pnl -= self.slippage
		self.current_balance = self.current_balance + (self.current_balance * (pnl/100))
		self.in_position = True
		
		self.record_trade(np_temp[self.buy_after_minutes][self.index_1m_datetime], 'buy', pnl)

	def pnl_direction_change(self, sell_datetime):
		if self.in_position:
			pnl = 0
			if self.previous_pred_direction > 0:
				pnl = ((self.sell_price - self.buy_price)/self.buy_price) * 100
				pnl *= self.leverage
				pnl = pnl - (self.transaction_fee_percent) 
			else:
				pnl = ((self.buy_price - self.sell_price)/self.buy_price) * 100
				pnl *= self.leverage
				pnl = pnl - (self.transaction_fee_percent) 

			pnl -= self.slippage
			self.current_balance += self.current_balance * (pnl/100)
			self.in_position = False

			self.record_trade(sell_datetime, 'sell - direction change', pnl)
		
	def find_tp_sl_index(self, take_profit_amount, stop_loss_amount, np_temp_high, np_temp_low):
		if self.current_pred_direction > 0:
			list_minute_high_indices = np.where(np_temp_high >= take_profit_amount)[0]
			list_minute_low_indices = np.where(np_temp_low <= stop_loss_amount)[0]
		else:
			list_minute_high_indices = np.where(np_temp_high >= stop_loss_amount)[0]
			list_minute_low_indices = np.where(np_temp_low <= take_profit_amount )[0]

		if len(list_minute_high_indices) == 0 and len(list_minute_low_indices) == 0:
			return False, -1
		elif len(list_minute_high_indices) > 0 and len(list_minute_low_indices) == 0:
			df_index = list_minute_high_indices[0]
			self.sell_price = np_temp_high[df_index]
			return True, df_index
		
		elif len(list_minute_high_indices) == 0 and len(list_minute_low_indices) > 0:
			df_index = list_minute_low_indices[0]
			self.sell_price = np_temp_low[df_index]
			return True, df_index
		else:
			if list_minute_high_indices[0] < list_minute_low_indices[0]:
				df_index = list_minute_high_indices[0]
				self.sell_price = np_temp_high[df_index]	
				return True, df_index
			else:
				df_index = list_minute_low_indices[0]
				self.sell_price = np_temp_low[df_index]
				return True, df_index

	def check_tp_sl(self, np_temp, np_temp_high, np_temp_low):
		if self.in_position:
			tp_sl_condition = False
			if self.current_pred_direction > 0: #long
				take_profit_amount = self.buy_price + (self.buy_price * self.take_profit_percent) 
				stop_loss_amount = self.buy_price - (self.buy_price * self.stop_loss_percent)
				tp_sl_condition, df_temp_index = self.find_tp_sl_index(take_profit_amount, stop_loss_amount, np_temp_high, np_temp_low)
			else:
				take_profit_amount = self.buy_price - (self.buy_price * self.take_profit_percent)
				stop_loss_amount = self.buy_price + (self.buy_price * self.stop_loss_percent)
				tp_sl_condition, df_temp_index = self.find_tp_sl_index(take_profit_amount, stop_loss_amount, np_temp_high, np_temp_low)

			if tp_sl_condition:				
				pnl = 0
				if self.previous_pred_direction > 0:
					pnl = ((self.sell_price - self.buy_price)/self.buy_price) * 100
					pnl *= self.leverage
					pnl = pnl - (self.transaction_fee_percent) 
				else:
					pnl = ((self.buy_price - self.sell_price)/self.buy_price) * 100
					pnl *= self.leverage
					pnl = pnl - (self.transaction_fee_percent) 

				pnl -= self.slippage
				self.current_balance += self.current_balance * (pnl/100)
				self.in_position = False

				str_tp_sl = ''
				if pnl > 0:
					str_tp_sl = ' - take_profit'
				else:
					str_tp_sl = ' - stop_loss'


				self.record_trade(np_temp[df_temp_index][self.index_1m_datetime], 'sell' + str_tp_sl, pnl)

	def record_trade(self, datetime, action, pnl):
		# print(datetime)
		self.array_to_save.append( 
								[ datetime, 
									'long' if self.current_pred_direction > 0 else 'short',
									action,  
									self.buy_price,
									self.sell_price,
									self.current_balance,
									pnl
								]
							)
		
	def get_interval_min_data(self, index):
		# get minutes data for the current prediction time using numpy
		start_time = np.datetime64(self.np_model_predctions[index][self.index_pred_datetime] )
		end_time   = np.datetime64(self.np_model_predctions[index+1][self.index_pred_datetime] )

		# np_1m_indices = np.where( (self.np_1m_datetime >= start_time) & (self.np_1m_datetime < end_time) )[0]
		np_1m_indices = np.where( (self.obj_dataset.np_1m[:, self.index_1m_datetime] >= start_time) & (self.obj_dataset.np_1m[:, self.index_1m_datetime] < end_time) )[0]
		# print(len(np_1m_indices))
		np_temp = self.obj_dataset.np_1m[np_1m_indices]

		# get minutes high and low data for the current prediction time using numpy
		np_temp_high = self.obj_dataset.np_1m[np_1m_indices, self.index_1m_high]
		np_temp_low = self.obj_dataset.np_1m[np_1m_indices, self.index_1m_low]
		return np_temp, np_temp_high, np_temp_low

	def run(self):					
		# first predicted direction
		self.previous_pred_direction = self.current_pred_direction = self.np_model_predctions[0][self.index_pred_direction] 
		break_on_huge_loss = False
		
		for i in range (0, len(self.np_model_predctions)-1):
			self.current_pred_direction = self.np_model_predctions[i][self.index_pred_direction] 

			if self.current_pred_direction == 0:
				if self.previous_pred_direction == 0:
					self.previous_pred_direction = self.current_pred_direction
					continue
				self.current_pred_direction = self.previous_pred_direction

			## get current interval's minute level data
			np_temp, np_temp_high, np_temp_low = self.get_interval_min_data(i)
			if not len(np_temp)>10:
				continue

			## if in position, and the new direction is same as the previous direction
			if self.in_position:
				if self.previous_pred_direction == self.current_pred_direction:
					self.previous_pred_direction = self.current_pred_direction

					self.record_trade(np_temp[self.buy_after_minutes][self.index_1m_datetime], 'same direction', 0)

			### if not in position then buy
			if not self.in_position: 
				self.buy(np_temp)
				self.previous_pred_direction = self.current_pred_direction
				
			### sell -> change in direction
			if self.current_pred_direction != self.previous_pred_direction:
				if len(np_temp)>=10:
					self.sell_price = np_temp[self.buy_after_minutes][self.index_1m_open]
					sell_datetime = np_temp[self.buy_after_minutes][self.index_1m_datetime]
				else:
					continue
					# self.sell_price = np_temp[0][self.index_1m_open]
					# sell_datetime = np_temp[0][self.index_1m_datetime]

				self.pnl_direction_change(sell_datetime)
				self.previous_pred_direction = self.current_pred_direction

				### buy again after direction change
				if not self.in_position: #buy
					self.buy(np_temp)
					self.previous_pred_direction = self.current_pred_direction

			### check if during the time horizon it hits take profit or stop loss
			self.check_tp_sl(np_temp, np_temp_high, np_temp_low) 

			self.previous_pred_direction = self.current_pred_direction

			if self.current_balance < self.breaking_balance:
				break_on_huge_loss = True
				break
		
		### backtest dataframe
		df_ledger = pd.DataFrame(self.array_to_save, columns = self.header_names)
		df_ledger["pnl_sum"] = df_ledger["pnl"].cumsum()
		df_ledger[['balance', 'pnl', 'pnl_sum']] = df_ledger[['balance', 'pnl', 'pnl_sum']].round(2)



		### pnl percent
		if len(df_ledger)>1:
			pnl_percent = np.round(df_ledger["pnl_sum"].iloc[-1], 2)

			if break_on_huge_loss:
				# print(df_ledger)
				return df_ledger, -1000, pnl_percent
			else:
				# print(df_ledger)
				return df_ledger, round(self.current_balance, 2), round(pnl_percent, 2)
		else:
			return df_ledger, 0, 0

	


