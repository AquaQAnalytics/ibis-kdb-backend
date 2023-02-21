from io import StringIO
import ibis.expr.operations as ops
import ibis.util as util
from ibis.backends.base.sql.compiler import Select


class KDBSelect(Select):

    """A SELECT statement which, after execution, might yield back to the user
    a table, array/list, or scalar value, depending on the expression that
    generated it."""

    def _all_exprs(self): # need to change
        return tuple(
            *self.select_set,
            self.table_set,
            *self.where,
            *self.group_by,
            *self.having,
            *self.order_by,
            *self.subqueries,
        )

    def compile(self):
        """This method isn't yet idempotent; calling multiple times may yield
        unexpected results."""
        # Can't tell if this is a hack or not. Revisit later
        self.context.set_query(self)

        # If any subqueries, translate them and add to beginning of query as
        # part of the WITH section
        with_frag = self.format_subqueries()

        # SELECT
        select_frag = self.format_select_set()

        # FROM, JOIN, UNION
        from_frag = self.format_table_set()

        # WHERE
        where_frag = self.format_where()

        # GROUP BY and HAVING
        groupby_frag = self.format_group_by()

        # ORDER BY
        order_frag = self.format_order_by()

        # LIMIT
        limit_frag = self.format_limit()

        # Glue together the query fragments and return
        query = ' '.join(
            filter(
                None,
                [
                    with_frag,
                    select_frag,
                    groupby_frag,
                    from_frag,
                    where_frag,
                    order_frag,
                    limit_frag,
                ],
            )
        )
        query=query.replace("`", "")

        return query

    def format_subqueries(self):
        if not self.subqueries:
            return

        context = self.context

        buf = []

        for expr in self.subqueries:
            formatted = util.indent(context.get_compiled_expr(expr), 2)
            alias = context.get_ref(expr)
            buf.append(f'({formatted}):{alias}')

        return 'WITH {}'.format(','.join(buf))

    def format_select_set(self):
        # TODO:
        context = self.context
        formatted = []
        for node in self.select_set:
            if isinstance(node, ops.Value):
                expr_str = self._translate(node, named=True)
            elif isinstance(node, ops.TableNode):
                # A * selection, possibly prefixed
                if context.need_aliases(node):
                    alias = context.get_ref(node)
                    expr_str = f'{alias}.*' if alias else '*'
                else:
                    expr_str = ''
            else:
                raise TypeError(node)
            formatted.append(expr_str)

        # select puts groupby col name in select cols for some reason this is to stop that
        if len(self.group_by) > 0:
            columns = [f'{op.name}' for op in self.group_by]

        buf = StringIO()
        line_length = 0
        max_length = 70
        tokens = 0
        cnt=0
        for i, val in enumerate(formatted):
            # always line-break for multi-line expressions
            #dummy.append(type(columns[0]))
            if len(self.group_by) > 0 and cnt == 0:
                # don't write to buffer
                cnt=1
            elif val.count('\n'):
                if i:
                    buf.write(',')
                buf.write('')
                indented = util.indent(val, self.indent)
                buf.write(indented)

                # set length of last line
                line_length = len(indented.split('\n')[-1])
                tokens = 1
            elif tokens > 0 and line_length and len(val) + line_length > max_length:
                # There is an expr, and adding this new one will make the line
                # too long
                buf.write(',       ') if i else buf.write('')
                buf.write(val)
                line_length = len(val) + 7
                tokens = 1
            else:
                if i and len(self.group_by) == 0:
                    buf.write(',')
                if i>1 and len(self.group_by) > 0:
                    buf.write(',')
                buf.write(' ')
                buf.write(val)
                tokens += 1
                line_length += len(val) + 2

        if self.distinct:
            select_key = 'SELECT DISTINCT'
        else:
            select_key = 'select'

        return f'{select_key}{buf.getvalue()}'

    def format_table_set(self):
        if self.table_set is None:
            return None

        fragment = 'from '

        helper = self.table_set_formatter_class(self, self.table_set)
        fragment += helper.get_result()

        return fragment

    def format_where(self):
        if not self.where:
            return None

        buf = StringIO()
        buf.write('where ')
        fmt_preds = []
        npreds = len(self.where)
        for pred in self.where:
            new_pred = self._translate(pred, permit_subquery=True)
            if npreds > 1:
                new_pred = f'({new_pred})'
            fmt_preds.append(new_pred)

        conj = ' ,{}'.format(' ' * 6)
        buf.write(conj.join(fmt_preds))
        return buf.getvalue()
 
    def format_group_by(self):
        if not len(self.group_by):
            # There is no aggregation, nothing to see here
            return None

        lines = []
        if len(self.group_by) > 0:
            columns = [f'{op.name}' for op in self.group_by]
            clause = 'by {}'.format(', '.join(columns))
            lines.append(clause)

        if len(self.having) > 0:
            trans_exprs = []
            for expr in self.having:
                translated = self._translate(expr)
                trans_exprs.append(translated)
            lines.append('HAVING {}'.format(' AND '.join(trans_exprs)))

        return ' '.join(lines)