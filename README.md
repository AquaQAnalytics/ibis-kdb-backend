# ibis-kdb-backend

## Installation

![Ibis](https://ibis-project.org/docs/3.2.0/) is a python library so you must have python installed on your device to use it. Installation instructions can be found ![here](https://www.python.org/downloads/).

To get started with Ibis you need to install the Ibis framework with:
```
pip install ibis-framework
```

Next, clone this repo into an easily accessable directory. Then cd into your clone and run the following commands:

```
pip install -r requirements.txt
pip install -e .
```
For Ibis to work properly with kdb you will need to ensure you are running the corrcet version of pandas and numpy. To check this run:
```
pip list
```
Pandas should be version 1.3.5 and numpy should be version 1.23.4

If you need to install a different version of them run the following commands. This may take some time.

```
python -m pip install pandas==1.3.5
python -m pip install numpy==1.23.4 
```
Finally in order to use the kdb backend you will need to edit the entry_points.txt file. This will be located wherever your python dependancies are; for me the file path was:
```
/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/ibis_framework-3.2.0.dist-info/entry_points.txt
```
You will need to add the following line to the end of the file:
```
kdb=ibis.backends.kdb
```

## Getting started 
To use ibis you must first start a python session.
Once in the session you will need to import ibis:
```
>>> import ibis
```
Next open a connection to your kdb session. By default this will be looking for a process running on your localhost at port 8000.
```
>>> q=ibis.kdb.connect()
```
This should print the following confirming it is connected:
```
:localhost:8000
IPC version: 3. Is connected: True
```
Now that your connection is establish you can pass queries into it.

## Example query
Right now the only function that has been programmed to work with kdb to apply aggregations is the table() function.
This function takes the following arguments:
```
table(self, table: str, select="", by="", where="")
```
So you could pass in the following items:
```
q.table(table="trade",select="avg price",by="sym")
```
You should get something that looks like this:

|sym     |price |
|--------|------|
|b'APPL' |109.34|
|b'CAT'  |252.32|
|b'GOOG' |209.11|
|b'NYSE' |57.59 |
