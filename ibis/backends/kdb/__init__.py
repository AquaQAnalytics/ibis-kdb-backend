from __future__ import annotations

import importlib
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Mapping

import pandas as pd

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
        raise NotImplementedError('pandas backend does not support databases')

    def list_databases(self, like=None):
        raise NotImplementedError('pandas backend does not support databases')
 
    def database(self, name=None):
        return self.database_class(name, self)

    #######
    
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

    def _mapping(self, val):
        """ Mapping function for KDB+ datatypes to python/IBIS """
        int_vals = ["i","h","j"]
        float_vals = ["f","e"]
        date_time = ["p","m","d","z"]
        time_delta = ["n","u","v","t"]
        if val in int_vals:
            return "int"    # int64 in ibis
        elif val in float_vals:
            return "Float"  # float 64 in ibis
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

    def table(self, table: str, database: str | None = None) -> ir.Table:
        """ Creates IBIS table expression from table name """
        if self.qpandas("`" + table + " in key`."):
            schema = self._get_schema(table)
            node = self.table_class(table, schema, self) 
            return self.table_expr_class(node)
        else:
            raise FileNotFoundError(
                "Table: " + table + " doesn't exist in the KDB+ process."
                )

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

        if type(tab.index[0])==bytes: # Converts the index to a string
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

    #####
    

    def create_table(self, table_name, obj=None, schema=None):
        """Create a table."""
        if obj is None and schema is None:
            raise com.IbisError('Must pass expr or schema')

        if obj is not None:
            if not self._supports_conversion(obj):
                raise com.BackendConversionError(
                    f"Unable to convert {obj.__class__} object "
                    f"to backend type: {self.__class__.backend_table_type}"
                )
            df = self._convert_object(obj)
        else:
            pandas_schema = self._convert_schema(schema)
            dtypes = dict(pandas_schema)
            df = self._from_pandas(pd.DataFrame(columns=dtypes.keys()).astype(dtypes))

        self.dictionary[table_name] = df

        if schema is not None:
            self.schemas[table_name] = schema

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
