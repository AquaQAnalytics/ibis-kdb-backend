from ibis.backends.kdb.query_builder import KDBSelect
from ibis.backends.base.sql.compiler import (
    Compiler,
)
from ibis.backends.clickhouse.compiler import (
    ClickhouseExprTranslator,
    ClickhouseTableSetFormatter,
    ClickhouseSelectBuilder,
    ClickhouseSelect,
    ClickhouseUnion,
)

from ibis.backends.base.sql.registry import quote_identifier

class KDBExprTranslator(ClickhouseExprTranslator):
    """Translates ibis expressions into a compilation target."""

    def name(self, translated, name, force=True):
        return f'{quote_identifier(name, force=force)}:{translated}'


class KDBCompiler(Compiler):
    cheap_in_memory_tables = False
    translator_class = KDBExprTranslator
    table_set_formatter_class = ClickhouseTableSetFormatter
    select_builder_class = ClickhouseSelectBuilder
    select_class = KDBSelect
    union_class = ClickhouseUnion




