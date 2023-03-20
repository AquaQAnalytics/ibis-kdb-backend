from ibis.backends.kdb.query_builder import KDBSelect
from ibis.backends.base.sql.compiler.query_builder import Union
from ibis.backends.base.sql.registry import quote_identifier
from ibis.backends.kdb.registry import operation_registry
import ibis.expr.operations as ops

from ibis.backends.base.sql.compiler import (
    Compiler,
    ExprTranslator,
    SelectBuilder,
    TableSetFormatter,
)



class KDBExprTranslator(ExprTranslator):
    """Translates ibis expressions into a compilation target."""
    _registry = operation_registry
    _require_order_by = (
        ops.DenseRank,
        ops.MinRank,
        ops.FirstValue,
        ops.LastValue,
        ops.PercentRank,
        ops.CumeDist,
        ops.NTile,
    )
    def name(self, translated, name, force=True):
        return f'{quote_identifier(name, force=force)}:{translated}'
class KDBTableSetFormatter(TableSetFormatter):
    _join_names = {
        ops.InnerJoin: 'ALL INNER JOIN',
        ops.LeftJoin: 'ALL LEFT OUTER JOIN',
        ops.RightJoin: 'ALL RIGHT OUTER JOIN',
        ops.OuterJoin: 'ALL FULL OUTER JOIN',
        ops.CrossJoin: 'CROSS JOIN',
        ops.LeftSemiJoin: 'LEFT SEMI JOIN',
        ops.LeftAntiJoin: 'LEFT ANTI JOIN',
        ops.AnyInnerJoin: 'ANY INNER JOIN',
        ops.AnyLeftJoin: 'ANY LEFT OUTER JOIN',
    }

    def _format_in_memory_table(self, op):
        # We register in memory tables as external tables because clickhouse
        # doesn't implement a generic VALUES statement
        return op.name
class KDBSelectBuilder(SelectBuilder):
    def _convert_group_by(self, exprs):
        return exprs
class KDBUnion(Union):
    @classmethod
    def keyword(cls, distinct):
        return 'UNION DISTINCT' if distinct else 'UNION ALL'



class KDBCompiler(Compiler):
    cheap_in_memory_tables = True
    translator_class = KDBExprTranslator
    table_set_formatter_class = KDBTableSetFormatter
    select_builder_class = KDBSelectBuilder
    select_class = KDBSelect
    union_class = KDBUnion




