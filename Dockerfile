FROM python:3.10

ADD main1.py .

RUN pip install --upgrade pip sqlalchemy pandas pymysql PySide2 DateTime statsmodels matplotlib opensearch-py numpy mysql-connector-python

CMD ["python", "./main1.py"]