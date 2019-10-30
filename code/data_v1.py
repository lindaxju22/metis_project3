#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 21:24:13 2019

@author: lindaxju
"""

#%%
import pandas as pd
import numpy as np

import pickle
from datetime import datetime
#%%
def count_nan_col(cols_list,df):
    df_temp = pd.DataFrame({'col_name': [], 'total_nan': []})
    for col_name in cols_list:
        df_temp = df_temp.append({'col_name': col_name, 'total_nan': df[col_name].isna().sum()}, ignore_index=True)
    return df_temp
#%%
df_orig = pd.read_csv('data/prosperLoanData.csv')
#%%
df_orig.shape
#%%
#df_orig.info()
#%%
list(df_orig.columns)
#%%
df_orig['LoanStatus'].unique()
#%%
pd.Series(df_orig['LoanStatus']).value_counts().plot('bar')
#%%
# only include relevant loan statuses (target)
LoanStatus_col = ['Completed', 'Defaulted', 'Chargedoff', 'Past Due (61-90 days)',
                  'Past Due (91-120 days)','Past Due (>120 days)']
df_filter1 = df_orig[df_orig.LoanStatus.isin(LoanStatus_col)]
df_filter1.reset_index(drop=True,inplace=True)
print(df_filter1.shape)
print(df_filter1.LoanStatus.unique())
#%%
#df_filter1.info()
#%%
df_filter1['ListingCreationDate'] = df_filter1['ListingCreationDate'].astype('datetime64[ns]')
#%%
# only include listings after July 2009
df_filter2 = df_filter1[(df_filter1['ListingCreationDate'] > '2009-08-01')]
df_filter2.reset_index(drop=True,inplace=True)
print(df_filter2.shape)
print(df_filter2['ListingCreationDate'].max())
print(df_filter2['ListingCreationDate'].min())
#%%
#df_filter2.info()
df_filter2.shape
#%%
# remove one row whose income was not verified
df_filter3 = df_filter2.copy()
df_filter3 = df_filter3[(df_filter3['IncomeVerifiable'] == True)]
df_filter3.reset_index(drop=True,inplace=True)
print(df_filter3.shape)
#%%
df_filter3_colnameclean = df_filter3.rename(columns={'ProsperRating (numeric)':'ProsperRating_num',
                                                     'ListingCategory (numeric)':'ListingCategory_num',
                                                     'TradesNeverDelinquent (percentage)':'TradesNeverDelinquent_perc'})
#%%
relevant_cols_list = ['Term','LoanStatus','BorrowerAPR','BorrowerRate','ProsperRating_num',
               'ProsperScore','ListingCategory_num','IsBorrowerHomeowner','CreditScoreRangeLower',
               'CreditScoreRangeUpper','CurrentCreditLines','OpenCreditLines',
               'TotalCreditLinespast7years','OpenRevolvingAccounts',
               'OpenRevolvingMonthlyPayment','InquiriesLast6Months','TotalInquiries',
               'CurrentDelinquencies','AmountDelinquent','DelinquenciesLast7Years',
               'PublicRecordsLast10Years','PublicRecordsLast12Months',
               'RevolvingCreditBalance','BankcardUtilization','AvailableBankcardCredit',
               'TotalTrades','TradesNeverDelinquent_perc','TradesOpenedLast6Months',
               'DebtToIncomeRatio','StatedMonthlyIncome','Recommendations']
#%%
count_nan_col(relevant_cols_list,df_filter3_colnameclean)
#%%
# only include listings with DebtToIncomeRatio
df_filter4 = df_filter3_colnameclean.dropna(subset=['DebtToIncomeRatio'])
df_filter4.reset_index(drop=True,inplace=True)
df_filter4
#%%
#df_filter4.info()
#%%
df_filter4['LoanStatus'].value_counts()
#%%
df_filter4['LoanStatus_target'] = (df_filter4['LoanStatus'] == 'Completed')
df_filter4.LoanStatus_target = df_filter4.LoanStatus_target.astype(int)
#df_filter4.info()
df_filter4.shape
#%%
count_nan_col(relevant_cols_list,df_filter4)
#%%
timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
filename = 'df_all_'+timestamp
#df_all.to_csv(r'data/'+filename+'.csv')
with open('data/'+filename+'.pickle', 'wb') as to_write:
    pickle.dump(df_filter4, to_write)
#%%
#%%
#%%
#%%
#%%
