from datetime import datetime
from opensearchpy import OpenSearch
import numpy as np
import pandas as pd
from pytz import utc
from statsmodels.tsa.api import ExponentialSmoothing
from sqlalchemy import create_engine
from dateutil.tz import *
import os
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

# df = pd.read_csv('monthly-car-sales.csv')

# df2 = pd.DataFrame({'Month': ['1969-1-1','1969-2-1','1969-3-1','1969-4-1','1969-5-1','1969-6-1']})
# df = df.append(df2, ignore_index = True)

# df.Month = pd.to_datetime(df.Month, format='%Y-%m-%d')

# df.index = df.Month
# df = df.drop(columns=['Month'])

# train_end = datetime(1965,12,1)
# test_end  = datetime(1969,6,1)
# train_data = df[:train_end]
# test_data  = df[train_end + timedelta(days=1):test_end]

# holt_winter = ExponentialSmoothing(np.asarray(train_data['Sales']), seasonal_periods=36, seasonal='add')
# hw_fit      = holt_winter.fit()
# hw_forecast = hw_fit.forecast(len(test_data))

# hw_residuals = test_data['Sales'] - hw_forecast

# df1 = test_data
# df1['Sales']=hw_forecast

# # engine = create_engine("mysql+pymysql://{user}:{pw}@{ip}:{port}/{db}".format(user='nksglx',pw='G4xK8qLa',db='netkaquartz',ip='172.21.21.56',port='3307'))
# # df1.to_sql('monthlycarsale', con = engine, if_exists = 'replace', chunksize = 20, index=False)
# print(df1)



NAIOPS_OPENSEARCH_HOST        = os.getenv("NAIOPS_OPENSEARCH_HOST"      ,'172.21.21.55')
NAIOPS_OPENSEARCH_PORT        = os.getenv("NAIOPS_OPENSEARCH_PORT"      ,'9200')
NAIOPS_OPENSEARCH_USERNAME    = os.getenv("NAIOPS_OPENSEARCH_USERNAME"  ,'admin')
NAIOPS_OPENSEARCH_PASSWORD    = os.getenv("NAIOPS_OPENSEARCH_PASSWORD"  ,'admin')
NAIOPS_OPENSEARCH_INDEX_NAME  = os.getenv("NAIOPS_OPENSEARCH_INDEX_NAME","nnm*")
NAIOPS_PREDICT_NODE           = os.getenv("NAIOPS_PREDICT_NODE"         ,"node")
NAIOPS_PREDICT_NODE_NAME      = os.getenv("NAIOPS_PREDICT_NODE_NAME"    ,"nks-rtr-04")
NAIOPS_PREDICT_IFDESCR        = os.getenv("NAIOPS_PREDICT_IFDESCR"      ,"ifdescr")
NAIOPS_PREDICT_IFDESCR_NAME   = os.getenv("NAIOPS_PREDICT_IFDESCR_NAME" ,"FastEthernet0/0")
NAIOPS_PREDICT_METRIC         = os.getenv("NAIOPS_PREDICT_METRIC"       ,"locifoutbitssec")
NAIOPS_PREDICT_TIMESTAMP      = os.getenv("NAIOPS_PREDICT_TIMESTAMP"    ,"@timestamp")
NAIOPS_PREDICT_HISTORY        = os.getenv("NAIOPS_PREDICT_HISTORY"      ,"72") 
NAIOPS_PREDICT_WINDOW         = os.getenv("NAIOPS_PREDICT_WINDOW"       ,"8") 

NAIOPS_DB_URI                 = os.getenv("NAIOPS_DB_URI"               ,"172.21.21.56")
NAIOPS_DB_PORT                = os.getenv("NAIOPS_DB_PORT"              ,"3307")
NAIOPS_DB_USERNAME            = os.getenv("NAIOPS_DB_USERNAME"          ,"nksglx")
NAIOPS_DB_PASSWORD            = os.getenv("NAIOPS_DB_PASSWORD"          ,"G4xK8qLa")
NAIOPS_DB_DATABASE_NAME       = os.getenv("NAIOPS_DB_DATABASE_NAME"     ,"policyndpp")
NAIOPS_DB_TABLE_NAME          = os.getenv("NAIOPS_DB_TABLE_NAME"        ,"predict_analysis")

auth = (NAIOPS_OPENSEARCH_USERNAME, NAIOPS_OPENSEARCH_PASSWORD)

client = OpenSearch(
    hosts               = [{'host': NAIOPS_OPENSEARCH_HOST, 'port': NAIOPS_OPENSEARCH_PORT}],
    http_compress       = True, # enables gzip compression for request bodies
    http_auth           = auth,
    use_ssl             = True,
    verify_certs        = False,
    ssl_assert_hostname = False,
    ssl_show_warn       = False
)

index_name = NAIOPS_OPENSEARCH_INDEX_NAME
query1 = {
    "size":10000,
   "_source":
    {
        "includes":
        [
            NAIOPS_PREDICT_TIMESTAMP, NAIOPS_PREDICT_METRIC, NAIOPS_PREDICT_IFDESCR, NAIOPS_PREDICT_NODE 
        ]
    },
     "query":
        {
    "bool": {
      "must": [],
      "filter": [
        {
          "exists": {
            "field": NAIOPS_PREDICT_NODE
          }
        },
        {
          "match_phrase": {
            "node": NAIOPS_PREDICT_NODE_NAME
          }
        },
        {
          "match_phrase": {
            "ifdescr": NAIOPS_PREDICT_IFDESCR_NAME
          }
        },
        
        {
          "range": {
            "@timestamp": {
              "gte": "now-" + NAIOPS_PREDICT_HISTORY + "h"
            }
          }
        }
      ],
      "should": [],
      "must_not": []
    }
        }
}

response1 = client.search( body = query1, index = index_name )

df01 = pd.DataFrame(response1)

value = []
for i in range(len(df01['hits']['hits'])):
    value.append(df01['hits']['hits'][i]['_source'])

df = pd.DataFrame.from_dict(value)

df['pertime'] = pd.to_datetime(df[NAIOPS_PREDICT_TIMESTAMP], format='%Y-%m-%d')
df=df.sort_values([NAIOPS_PREDICT_TIMESTAMP])

df.index = df.pertime
hour = df.drop(columns=[NAIOPS_PREDICT_TIMESTAMP])

hour.index = df[NAIOPS_PREDICT_TIMESTAMP]

hour.index = pd.to_datetime(hour.index, utc=True)
hour = hour.resample('H').sum()

time_max = hour.index.max()
rng      = pd.date_range(time_max, periods=int(7), freq='H')
hours     = hour.reindex(hour.index.union(rng))

newhour = hour.head(int((len(hours)/100)*70))

train_data = hours[:int((len(hours)/100)*70)]
test_data  = hours[int((len(hours)/100)*70) :int(len(hours))]

holt_winter = ExponentialSmoothing(np.asarray(train_data[NAIOPS_PREDICT_METRIC]), seasonal_periods=4, seasonal='add', trend='add')
hw_fit      = holt_winter.fit()
hw_forecast = hw_fit.forecast(len(test_data))

hw_residuals = test_data[NAIOPS_PREDICT_METRIC] - hw_forecast

df1 = test_data

df1[NAIOPS_PREDICT_METRIC]=hw_forecast

hours=pd.concat([newhour,df1])

hours[NAIOPS_PREDICT_NODE]=NAIOPS_PREDICT_NODE_NAME

hours[NAIOPS_PREDICT_IFDESCR]=NAIOPS_PREDICT_IFDESCR_NAME

hours.reset_index(level=0, inplace=True)

hours.rename(columns = {'index':'timestamp'}, inplace = True)

engine = create_engine("mysql+pymysql://{user}:{pw}@{ip}:{port}/{db}".format(user=NAIOPS_DB_USERNAME,pw=NAIOPS_DB_PASSWORD,db=NAIOPS_DB_DATABASE_NAME,ip=NAIOPS_DB_URI,port=NAIOPS_DB_PORT))
hours.to_sql(NAIOPS_DB_TABLE_NAME, con = engine, if_exists = 'replace', chunksize = 1000, index=False)
print(hours)
