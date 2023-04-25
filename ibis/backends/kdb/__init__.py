from __future__ import annotations
from io import StringIO

import importlib
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Mapping

import pandas as pd
from datetime import timedelta

import ibis.common.exceptions as com
import ibis.config
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir

from ibis.backends.kdb.compiler import KDBCompiler

from ibis.backends.base import BaseBackend
from qpython import qconnection

from ibis.backends.pandas.client import (
    PandasDatabase,
    PandasTable,
    ibis_schema_to_pandas,
)

if TYPE_CHECKING:
    import pyarrow as pa


class BaseKDBBackend(BaseBackend):
    """Base class for backends based on pandas."""

    name = "pandas"
    backend_table_type = pd.DataFrame

    table_class = ops.DatabaseTable
    table_expr_class = ir.Table
    compiler = KDBCompiler
    dictionary={}

    class Options(ibis.config.Config):
        enable_trace: bool = False
     
    def do_connect(
        self,
        host: str = "localhost",
        port: int = 8000,
    ) -> None:
        """Construct a client.

        Parameters
        ----------
        host
            hostname or IP
        port
            interger port number

        Examples
        --------
        >>> import ibis
        >>> ibis.kdb.connect(host="localhost",port=8000)
        <ibis.kdb.Backend at 0x...>
        """
        
        #Connect the q session
        q = qconnection.QConnection(host = host, port = port)
        qpandas = qconnection.QConnection(host=host, port=port, pandas=True)

        #Open handle
        q.open()
        qpandas.open()

        print(q)
        print('IPC version: %s. Is connected: %s' % (q.protocol_version, q.is_connected()))

        self.q = q
        self.qpandas = qpandas

    def close(self):
        self.q.close()
        self.qpandas.close()

    @property
    def version(self) -> str:
        return pd.__version__

    @property
    def current_database(self):
        raise NotImplementedError('KDB+ backend does not support databases')
    def list_databases(self, like=None):
        raise NotImplementedError('KDB+ backend does not support databases')
    def database(self, name=None):
        raise NotImplementedError('KDB+ backend does not support databases')

    #######

    def table(self, table: str, database: str | None = None) -> ir.Table:
        """ Creates IBIS table expression from table name """
        # If table exists in KDB+ process make ibis table client side
        if self.qpandas("`" + table + " in key`."):
            schema = self._get_schema(table)
            node = self.table_class(table, schema, self) 
            return self.table_expr_class(node)
        else:
            raise FileNotFoundError(
                "Table: " + table + " doesn't exist in the KDB+ process."
                )
    
    def table_from_schema(self, name: str, schema):
        """ Make an ibis table from an ibis schema"""
        if name in locals() or name in globals():
            raise FileExistsError(
                "Table: " + name + " already exists in the python process."
                )
        else: 
            node = self.table_class(name, schema, self) 
            return self.table_expr_class(node)

    def compile(
        self,
        expr: ir.Expr,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: str = 'default',
        **kwargs: Any,
    ):
        """ Compiles the IBIS table expression into a QSQL statement """
        if not isinstance(expr, ir.Expr):
            raise TypeError(
                "`expr` has type {!r}, expected ibis.expr.types.Expr".format(
                    type(expr).__name__
                )
            )

        kwargs.pop('timecontext', None)
        query_ast = self.compiler.to_ast_ensure_limit(expr, limit, params=params)
        sql = query_ast.compile()
        return sql

    def execute(
        self,
        expr: ir.Expr,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: str = 'default',
        **kwargs: Any,
    ):
        """ Calls the compile function and executes the QSQL server side. """
        sql = self.compile(expr)
        tab=self.qpandas(sql)

        if type(tab.index[0])==bytes:               # Converts the index to a string
            tab=self._convert_idx_from_byte(tab)

        for typs in tab.dtypes:
            if typs=="object":
                tab=self._convert_from_byte(tab)
                return tab
                
        return tab
    
    def list_tables(self):
        """ Lists tables in root namespace on KDB+ process """
        return self.qpandas("tables[]").str.decode('utf-8')
        
    def head(self, table: str):
        """ Returns first 5 rows from table. """
        if self.qpandas("`" + table + " in key`."):
            return self.qpandas("5#" + table)
        else:
            raise FileNotFoundError(
                "Table: " + table + " doesn't exist in the KDB+ process."
                )
        
    def create_table(
        self, 
        table_name:str, 
        expr: ir.Table | None = None, 
        schema: None = None
    ) -> ir.Table:
        """Creates a table.
        Using either table expression schema or ibis schema as a template, creates a table
        on KDB process and makes a local version to use.

        Parameters
        ----------
        name
            Table name to create
        expr
            Table expression to use as the data source
        schema
            An ibis schema, have to specify, schema=sch, in function call.
        """

        if expr is None and schema is None:
            raise ValueError('You must pass either an table expression or a schema')
        
        if expr is not None and schema is not None:
            if not expr.schema().equals(ibis.schema(schema)):
                raise TypeError(
                    'Expression schema is not equal to passed schema. '
                    'Try passing the expression without the schema'
                )
            
        if self.qpandas("`" + table_name + " in key`."):
            raise FileExistsError(
                "Table: " + table_name + " already exists in the KDB+ process."
                )

        if expr is not None:
            # get schema from table expression
            schema = expr.schema()      

        tnames = schema.names                                         
        ttypes = list(map(self._create_tab_mapping,schema.types))

        # make "create table" string in kdb   
        buf = StringIO()

        buf.write(f'{table_name}:([]')

        for i in range(len(tnames)):
            buf.write(f'{tnames[i]}:`{ttypes[i]}$()')

            if i<len(tnames)-1:          # each column
                buf.write(f';')
            else:
                buf.write(f')')          # end of statement

        self.value=buf.getvalue()        # for debugging
        self.qpandas(buf.getvalue())

        # create ibis table client side
        node = self.table_class(table_name, schema, self) 
        return self.table_expr_class(node)

    def insert(
        self,
        table_name=str,
        obj=None,
        database=None,
        overwrite=False,
        partition=None,
        values=None,
        validate=True,
    ):
        """Insert data into an existing table.
        Completely overwrite contents
        >>> q.insert("table", table_expr, overwrite=True)
        """
        if not self.qpandas("`" + table_name + " in key`."):
            raise FileNotFoundError(
                "Table: " + table_name + " doesn't exist in the KDB+ process."
                )
        
        if overwrite:
            schema = self._get_schema(table_name)
            self.drop_table(table_name)
            self.create_table(table_name,schema)

        # If we've been passed a `memtable`, pull out the underlying dataframe
        if isinstance(obj, ir.Table) and isinstance(in_mem_table := obj.op(), ops.InMemoryTable):
            obj = in_mem_table.data.to_frame()

        if isinstance(obj, pd.DataFrame):
            obj.to_sql(
                table_name,
                self.con,
                index=False,
                if_exists='replace' if overwrite else 'append',
                schema=self._current_schema,
            )

        elif isinstance(obj, ir.Table):
            to_table_expr = self.table(table_name)
            to_table_schema = to_table_expr.schema()

            if overwrite:
                self.drop_table(table_name, database=database)
                self.create_table(
                    table_name,
                    schema=to_table_schema,
                    database=database,
                )

            to_table = self._get_sqla_table(table_name, schema=database)

            from_table_expr = obj

            with self.begin() as bind:
                if from_table_expr is not None:
                    bind.execute(
                        to_table.insert().from_select(
                            list(from_table_expr.columns),
                            from_table_expr.compile(),
                        )
                    )

        elif isinstance(obj, (list, dict)):
            to_table = self.table(table_name)                      # gets ibis table from server table
            
            if type(obj)==dict:
                vals=list(obj.values())
            else:
                vals=obj

            # start string write for insert
            buf = StringIO()
            buf.write(f'`{table_name} insert (')

            buf = self._write_val_for_insert(buf,vals,0)           # separate function for recursion purposes

            string = buf.getvalue()[:-1] + ')'                     # ending ) only works in non-nested list, this just overrides ; or ) to end query
            self.value=string                                      # for debugging query
            self.qpandas(string)
            
        else:
            raise ValueError(
                "No operation is being performed. Either the obj parameter "
                "is not a pandas DataFrame or is not a ibis Table."
                f"The given obj is of type {type(obj).__name__} ."
            )

    def drop_table(
        self,
        table_name: str,
        database: str | None = None,
        force: bool = True,
    ) -> None:
        """Drop a table.

        Parameters
        ----------
        table_name
            Table to drop
        database
            Database to drop table from
        force
            Check for existence before dropping
        """
        if force:
            if self.qpandas("`" + table_name + " in key`."):
                self.qpandas("delete " + table_name + " from `.")
                return (table_name + " deleted from KDB+ process.")
            else:
                raise FileNotFoundError(
                    "Table: " + table_name + " doesn't exist in the KDB+ process."
                    )
        else:
            self.qpandas("delete " + table_name + " from `.")
            return (table_name + " deleted from KDB+ process.")
    
    # internal functions
    def _convert_from_byte(self,table):
        """ Converts Object columns in a pandas dataframe to string """
        str_tab = table.select_dtypes([object])
        str_tab = str_tab.stack().str.decode('utf-8').unstack()
        for col in str_tab:
            table[col] = str_tab[col]
        return table

    def _convert_idx_from_byte(self,table):
        """ Converts bytes index column in a pandas dataframe to string """
        str_tab = table.index
        str_tab = str_tab.str.decode('utf-8')
        table.index = str_tab
        return table

    def _convert_timedelta_to_str(self,delta:timedelta):
        buf = StringIO()

        # int division floors answer
        s=delta.seconds
        hrs=s//(60*60)
        mins=(s//60) - (hrs*60)
        secs= s - (mins*60) - (hrs*60*60)
        msecs= delta.microseconds//1000

        if hrs<10:
            buf.write(f'0')
        buf.write(f"{hrs}:")
        if mins<10:
            buf.write(f'0')
        buf.write(f"{mins}:")
        if secs<10:
            buf.write(f'0')
        buf.write(f"{secs}.")
        if msecs<10:
            buf.write(f"00")
        elif msecs<100:
            buf.write(f"0")
        buf.write(f"{msecs}")

        return buf.getvalue()
    
    def _mapping(self, val):
        """ Mapping function for KDB+ datatypes to python/IBIS """
        int_vals = ["i","h","j"]
        float_vals = ["f","e"]
        #date_time = ["p","m","d","z"]
        time = ["n","u","v","t"]
        if val in int_vals:
            return "int"    # int64 in ibis
        elif val in float_vals:
            return "float"  # float 64 in ibis
        elif val in time:
            return "time"
        else:
            return "string"

    def _get_schema(self, name: str):
        """ Retrieve meta of table from query to KDB+ process. Converts to IBIS schema"""
        tab=self.qpandas("meta " + name)
        tab=self._convert_idx_from_byte(tab)
        idx,typ=[],[]

        for i in range(len(tab)):
            idx.append(tab.index[i])
            typ.append(tab['t'][i])

        typ = list(map(self._mapping,typ))
        schema = ibis.schema(dict(zip(idx, typ)))
        return schema
    
    def _create_tab_mapping(self, val):
        """ Mapping function for python/IBIS datatypes to KDB+ """
        int_vals = ["int64"]
        float_vals = ["float64"]
        time_vals = ["time"]
        if str(val) in int_vals:
            return "int"    # int in KDB+
        elif str(val) in float_vals:
            return "float"  # float in KDB+
        elif str(val) in time_vals:
            return "time"   # time in KDB+
        else:
            return ""       # defaults to symbol

    def _write_val_for_insert(self,buf,vals,level):
        """ Function to write insert string for insert function """
        for i in range(len(vals)):

            if type(vals[i])==list:                     # if list of lists, have to go down a level
                self._write_val_for_insert(buf,vals[i],1)
            else:

                if type(vals[i])==str:                  # assume symbol if its a string
                    buf.write(f'`')
                
                if type(vals[i])==timedelta:            # for time values
                    test = StringIO()
                    if vals[i].seconds//(60*60) <10:    # if hrs<10 have to preppend a 0 bc the way it handles this type
                        test.write(f'0')
                    test.write(f'{vals[i]}')
                    write_vals = test.getvalue()[:-3]   # remove trailing 000 for <milleseconds, no rounding
                else:
                    write_vals = vals[i]

                buf.write(f'{write_vals}')

                if level==1:                            # if in list
                    if i==len(vals)-1:
                        buf.write(f';')                 # at end of current list
                    elif type(vals[i])!=str:
                        buf.write(f' ')                 # if its not a symbol, separate
                else:
                    if i<len(vals)-1:
                        buf.write(f';')                 # end of current value
                    else:
                        buf.write(f')')                 # end of query
        return buf
           
    #####

    @classmethod
    def _supports_conversion(cls, obj: Any) -> bool:
        return True

    @staticmethod
    def _convert_schema(schema: sch.Schema):
        return ibis_schema_to_pandas(schema)

    @staticmethod
    def _from_pandas(df: pd.DataFrame) -> pd.DataFrame:
        return df

    @classmethod
    def _convert_object(cls, obj: Any) -> Any:
        return cls.backend_table_type(obj)

    @classmethod
    @lru_cache
    def _get_operations(cls):
        backend = f"ibis.backends.{cls.name}"

        execution = importlib.import_module(f"{backend}.execution")
        execute_node = execution.execute_node

        # import UDF to pick up AnalyticVectorizedUDF and others
        importlib.import_module(f"{backend}.udf")

        dispatch = importlib.import_module(f"{backend}.dispatch")
        pre_execute = dispatch.pre_execute

        return frozenset(
            op
            for op, *_ in execute_node.funcs.keys() | pre_execute.funcs.keys()
            if issubclass(op, ops.Value)
        )

    @classmethod
    def has_operation(cls, operation: type[ops.Value]) -> bool:
        # Pandas doesn't support geospatial ops, but the dispatcher implements
        # a common base class that makes it appear that it does. Explicitly
        # exclude these operations.
        if issubclass(operation, (ops.GeoSpatialUnOp, ops.GeoSpatialBinOp)):
            return False
        op_classes = cls._get_operations()
        return operation in op_classes or any(
            issubclass(operation, op_impl) for op_impl in op_classes
        )


class Backend(BaseKDBBackend):
    name = 'pandas'
    #database_class = PandasDatabase
    #table_class = PandasTable

    def to_pyarrow(
        self,
        expr: ir.Expr,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
    ) -> pa.Table:
        pa = self._import_pyarrow()
        output = self.execute(expr, params=params, limit=limit)

        if isinstance(output, pd.DataFrame):
            return pa.Table.from_pandas(output)
        elif isinstance(output, pd.Series):
            return pa.Array.from_pandas(output)
        else:
            return pa.scalar(output)
