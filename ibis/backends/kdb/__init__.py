from __future__ import annotations

import importlib
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Mapping, MutableMapping

import pandas as pd

import ibis.common.exceptions as com
import ibis.config
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir

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

    class Options(ibis.config.Config):
        enable_trace: bool = False
     
    def do_connect(
        self,
        #host: str = "localhost",
        #port: int = 8000,
        host: str = "81.150.99.19",
        port: int = 8001,
    ) -> None:
        """Construct a client from a dictionary of pandas DataFrames.

        Parameters
        ----------
        dictionary
            Mutable mapping of string table names to pandas DataFrames.

        Examples
        --------
        >>> import ibis
        >>> ibis.pandas.connect({"t": pd.DataFrame({"a": [1, 2, 3]})})
        <ibis.backends.pandas.Backend at 0x...>
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

    def from_dataframe(
        self,
        df: pd.DataFrame,
        name: str = 'df',
        client: BaseKDBBackend | None = None,
    ) -> ir.Table:
        """Construct an ibis table from a pandas DataFrame.

        Parameters
        ----------
        df
            A pandas DataFrame
        name
            The name of the pandas DataFrame
        client
            Client dictionary will be mutated with the name of the DataFrame,
            if not provided a new client is created

        Returns
        -------
        Table
            A table expression
        """
        if client is None:
            return self.connect({name: df}).table(name)
        client.dictionary[name] = df
        return client.table(name)

    @property
    def version(self) -> str:
        return pd.__version__

    @property
    def current_database(self):
        raise NotImplementedError('pandas backend does not support databases')

    def list_databases(self, like=None):
        raise NotImplementedError('pandas backend does not support databases')

    def list_tables(self, like=None, database=None):
        return self._filter_with_like(list(self.dictionary.keys()), like)

    def database(self, name=None):
        return self.database_class(name, self)

    """
    #This will push processing down to kdb but I don't like the way it is implemented
    def table(self, table: str, select="", by="", where="", columns="", aggregation=""):
        # select=[]
        # columns=columns.split(",")
        # for column in columns:x
        #     select.append(aggregation + " " + column)
        # select=",".join(select)
        if by!="" and where!="":
            return self.qpandas("select " + select + " by " + by + " from " + table + " where " + where)
        elif by!="":
            return self.qpandas("select " + select + " by " + by + " from " + table)
        elif where!="":
            return self.qpandas("select " + select + " from " + table + " where " + where)
        # elif columns!="":
        #     return self.qpandas("select " + select + " from " + table)
        elif select!="":
            return self.qpandas("select " + select + " from " + table)
        
        return self.qpandas(table)
    
    def q_table(self, table: str, select="", by="", where="", columns="", aggregation=""):
        # select=[]
        # columns=columns.split(",")
        # for column in columns:x
        #     select.append(aggregation + " " + column)
        # select=",".join(select)
        if by!="" and where!="":
            return self.q("select " + select + " by " + by + " from " + table + " where " + where)
        elif by!="":
            return self.q("select " + select + " by " + by + " from " + table)
        elif where!="":
            return self.q("select " + select + " from " + table + " where " + where)
        # elif columns!="":
        #     return self.qpandas("select " + select + " from " + table)
        elif select!="":
            return self.q("select " + select + " from " + table)
        
        return self.q(table)
    """

    #####
    # example query
    # query = q.qrybuild("trade").select("avg price").by("sym").where("amount>150")
    # qry_str = q.compile(query)
    # q.execute(qry_str)
    
    def execute(self, query): 
        return self.qpandas(query)

    def compile(self, expr, *args, **kwargs):
        expr=expr.query
        return " ".join(expr)
    
    class qrybuild():
        def __init__(self, name):
            self.name=name

        def select(self,input=""):
            self.bycnt=0                                            # counters so that if you run the by/where again it'll add to query correctly
            self.wherecnt=0
            self.query=["select",input,"from",self.name]
            return self
        def by(self,input):
            if self.bycnt==0:
                self.query.insert(2,"by " + input)
                self.bycnt=1
            else:
                self.query.insert(3,"," + input)
            return self
        def where(self,input):
            if self.wherecnt==0:
                self.query.append("where " + input)
                self.wherecnt=1
            else:
                self.query.append("," + input)
            return self
        def return_query(self):                                      # a debugging function, replaced by compile
            return " ".join(self.query)

    #####
        
    def head(self, table: str):
        return self.qpandas("5#" + table)

    def get_schema(self, table_name, database=None):
        schemas = self.schemas
        try:
            schema = schemas[table_name]
        except KeyError:
            schemas[table_name] = schema = sch.infer(self.dictionary[table_name])
        return schema

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
    database_class = PandasDatabase
    table_class = PandasTable

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
    """
    def execute(self, query, params=None, limit='default', **kwargs):
        from ibis.backends.pandas.core import execute_and_reset

        if limit != 'default' and limit is not None:
            raise ValueError(
                'limit parameter to execute is not yet implemented in the '
                'pandas backend'
            )

        if not isinstance(query, ir.Expr):
            raise TypeError(
                "`query` has type {!r}, expected ibis.expr.types.Expr".format(
                    type(query).__name__
                )
            )

        node = query.op()

        if params is None:
            params = {}
        else:
            params = {k.op() if hasattr(k, 'op') else k: v for k, v in params.items()}

        return execute_and_reset(node, params=params, **kwargs)
        """
