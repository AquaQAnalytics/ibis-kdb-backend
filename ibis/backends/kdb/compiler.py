from ibis.backends.kdb.query_builder import KDBSelect
from ibis.backends.base.sql.compiler.query_builder import Union
from ibis.backends.base.sql.registry import quote_identifier
from ibis.backends.kdb.registry import operation_registry
import ibis.expr.operations as ops
from ibis.backends.base.sql.compiler.base import SetOp
import ibis.util as util
import ibis.common.exceptions as com

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
    def _format_table(self, op):
        # TODO: This could probably go in a class and be significantly nicer
        ctx = self.context

        ref_op = op
        if isinstance(op, ops.SelfReference):
            ref_op = op.table

        if isinstance(ref_op, ops.InMemoryTable):
            result = self._format_in_memory_table(ref_op)
            is_subquery = True
        elif isinstance(ref_op, ops.PhysicalTable):
            name = ref_op.name
            # TODO(kszucs): add a mandatory `name` field to the base
            # PhyisicalTable instead of the child classes, this should prevent
            # this error scenario
            if name is None:
                raise com.RelationError(f'Table did not have a name: {op!r}')
            result = self._quote_identifier(name)
            is_subquery = False
        else:
            # A subquery
            if ctx.is_extracted(ref_op):
                # Was put elsewhere, e.g. WITH block, we just need to grab its
                # alias
                alias = ctx.get_ref(op)

                # HACK: self-references have to be treated more carefully here
                if isinstance(op, ops.SelfReference):
                    return f'{ctx.get_ref(ref_op)} {alias}'
                else:
                    return alias

            subquery = ctx.get_compiled_expr(op)
            result = f'({util.indent(subquery, self.indent)})'
            is_subquery = True

        if is_subquery or ctx.need_aliases(op):
            result += f' {ctx.get_ref(op)}'

        return result

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

class KDBIntersection(SetOp):
    _keyword = "INTERSECT"

class KDBDifference(SetOp):
    _keyword = "EXCEPT"

class KDBCompiler(Compiler):
    cheap_in_memory_tables = True
    translator_class = KDBExprTranslator
    table_set_formatter_class = KDBTableSetFormatter
    select_builder_class = KDBSelectBuilder
    select_class = KDBSelect
    union_class = KDBUnion
    intersect_class = KDBIntersection
    difference_class = KDBDifference




