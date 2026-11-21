"""Find which columns of an unpacked SQL fixture row can reach a test assertion."""

import ast


class FixtureProjection:
    """Follow direct uses of a fixture list through ordinary Python loop targets.

    Unsupported consumers, including reflection, can observe every column.
    The returned columns describe input dependencies on the pristine test code.
    """

    def __init__(self, source, owner, width):
        tree = ast.parse(source)
        self.owner = next(item for item in tree.body if isinstance(item, ast.ClassDef)
                          and item.name == owner)
        self.methods = {item.name: item for item in self.owner.body if isinstance(item, ast.FunctionDef)}
        self.parent = {child: node for node in ast.walk(self.owner) for child in ast.iter_child_nodes(node)}
        self.all = frozenset(range(width))

    def unpacked(self, target, fn):
        if not isinstance(target, (ast.Tuple, ast.List)) or len(target.elts) != len(self.all):
            return self.all
        if any(not isinstance(node, ast.Name) for node in target.elts):
            return self.all
        if any(isinstance(node, ast.Name) and node.id in {
            "eval", "exec", "getattr", "locals", "globals", "vars", "_getframe", "currentframe"
        } for node in ast.walk(fn)):
            return self.all
        loaded = {node.id for node in ast.walk(fn) if isinstance(node, ast.Name)
                  and isinstance(node.ctx, ast.Load)}
        return {i for i, name in enumerate(target.elts) if name.id in loaded}

    def row(self, node, fn):
        parent = self.parent.get(node)
        if isinstance(parent, ast.IfExp) and node in (parent.body, parent.orelse):
            return self.row(parent, fn)
        if isinstance(parent, ast.Assign) and parent.value is node and len(parent.targets) == 1:
            return self.unpacked(parent.targets[0], fn)
        return self.all

    def sequence(self, node, fn, seen):
        parent = self.parent.get(node)
        if isinstance(parent, ast.Subscript) and parent.value is node and not isinstance(parent.slice, ast.Slice):
            return self.row(parent, fn)
        if isinstance(parent, (ast.For, ast.comprehension)) and parent.iter is node:
            return self.unpacked(parent.target, fn)
        if not isinstance(parent, ast.Call):
            return self.all
        if isinstance(parent.func, ast.Name) and parent.func.id == "enumerate" and parent.args[:1] == [node]:
            loop = self.parent.get(parent)
            if isinstance(loop, (ast.For, ast.comprehension)) and loop.iter is parent:
                target = loop.target
                if isinstance(target, (ast.Tuple, ast.List)) and len(target.elts) == 2:
                    return self.unpacked(target.elts[1], fn)
        if not (isinstance(parent.func, ast.Attribute) and isinstance(parent.func.value, ast.Name)
                and parent.func.value.id == "self" and node in parent.args
                and not any(isinstance(arg, ast.Starred) for arg in parent.args)):
            return self.all
        name = parent.func.attr
        if name not in self.methods or name in seen:
            return self.all
        other = self.methods[name]
        args = [*other.args.posonlyargs, *other.args.args]
        index = parent.args.index(node) + 1
        if index >= len(args):
            return self.all
        name_at_call = args[index].arg
        if any(isinstance(n, ast.Name) and n.id == name_at_call and not isinstance(n.ctx, ast.Load)
               for n in ast.walk(other)):
            return self.all
        used = set()
        for n in ast.walk(other):
            if isinstance(n, ast.Name) and n.id == name_at_call:
                used.update(self.sequence(n, other, seen | {name}))
        return used

    def columns(self, method, line, field):
        fn = self.methods.get(method)
        if fn is None:
            return self.all
        matches = [n for n in ast.walk(fn) if isinstance(n, ast.Attribute) and n.attr == field
                   and isinstance(n.value, ast.Name) and n.value.id == "self"
                   and n.lineno <= line <= n.end_lineno]
        if not matches:
            return self.all
        return set().union(*(self.sequence(n, fn, {method}) for n in matches))
