import itertools
from ontolearn.ea_utils import OperatorVocabulary, escape
from ontolearn.knowledge_base import KnowledgeBase
from owlapy.model import BooleanOWLDatatype, DoubleOWLDatatype, IntegerOWLDatatype, OWLClass, OWLClassExpression, \
    OWLDataProperty, OWLNamedIndividual, OWLObjectProperty, OWLPropertyExpression, Python_Types
import random
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, List, Tuple, Union
from deap.gp import Primitive, PrimitiveSetTyped, Terminal
from deap import creator


class AbstractEAInitialization(metaclass=ABCMeta):
    """Abstract base class for initialization methods for evolutionary algorithms.

    """
    __slots__ = ()

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_population(self, container: Callable,
                       pset: PrimitiveSetTyped,
                       population_size: int = 0) -> List['creator.Individual']:
        pass

    @abstractmethod
    def get_expression(self, pset: PrimitiveSetTyped) -> List[Union[Primitive, Terminal]]:
        pass


class EARandomInitialization(AbstractEAInitialization):
    """Rnndom initialization methods for evolutionary algorithms.

    """
    __slots__ = 'min_height', 'max_height', 'method'

    min_height: int
    max_height: int
    method: str

    def __init__(self, min_height: int = 3, max_height: int = 6, method: str = "rhh"):
        """
        Args:
            min_height: minimum height of trees
            max_height: maximum height of trees
            method: initialization method possible values: rhh, grow, full
        """
        self.min_height = min_height
        self.max_height = max_height
        self.method = method

    def get_population(self, container: Callable,
                       pset: PrimitiveSetTyped,
                       population_size: int = 0) -> List['creator.Individual']:
        return [container(self.get_expression(pset)) for _ in range(population_size)]

    def get_expression(self, pset: PrimitiveSetTyped, type_: type = None) -> List[Union[Primitive, Terminal]]:
        if type_ is None:
            type_ = pset.ret

        use_grow = (self.method == 'grow' or (self.method == 'rhh' and random.random() < 0.5))

        expr: List[Union[Primitive, Terminal]] = []
        height = random.randint(self.min_height, self.max_height)
        self._build_tree(expr, pset, height, 0, type_, use_grow)
        return expr

    def _build_tree(self, tree,
                    pset: PrimitiveSetTyped,
                    height: int,
                    current_height: int,
                    type_: type,
                    use_grow: bool):

        if current_height == height or len(pset.primitives[type_]) == 0:
            tree.append(random.choice(pset.terminals[type_]))
        else:
            operators = []
            if use_grow and current_height >= self.min_height:
                operators = pset.primitives[type_] + pset.terminals[type_]
            else:
                operators = pset.primitives[type_]

            operator = random.choice(operators)
            tree.append(operator)

            if isinstance(operator, Primitive):
                for arg_type in operator.args:
                    self._build_tree(tree, pset, height, current_height+1, arg_type, use_grow)


# BIG TODO: Fix type hints 
class EARandomWalkInitialization(AbstractEAInitialization):
    """Random walk initialization for description logic learning.

    """
    __slots__ = 'max_r', 'jump_pr', 'type_counts', 'use_data_properties', 'dp_to_prim_type', 'dp_splits'

    connection_pr: float = 0.5

    max_r: int
    jump_pr: float

    type_counts: Dict[OWLClass, int]
    use_data_properties: bool
    dp_to_prim_type: Dict[OWLDataProperty, Any]
    dp_splits: Dict[OWLDataProperty, List[Python_Types]]

    def __init__(self, max_r: int = 2, jump_pr: float = 0.5):
        """
        Args:
            max_r: number of paths
            jump_pr: probability to explore paths of length 2
        """
        self.max_r = max_r
        self.jump_pr = jump_pr

        self.type_counts = dict()
        self.use_data_properties = False
        self.dp_to_prim_type = dict()
        self.dp_splits = dict()

    def get_population(self, container: Callable,
                       pset: PrimitiveSetTyped,
                       population_size: int = 0,
                       pos: List[OWLNamedIndividual] = None,
                       use_data_properties: bool = False,
                       dp_to_prim_type: Dict[OWLDataProperty, Any] = None,
                       dp_splits: Dict[OWLDataProperty, List[Python_Types]] = None,
                       kb: KnowledgeBase = None) -> List['creator.Individual']:
        assert pos is not None
        assert kb is not None
        assert dp_to_prim_type is not None
        assert dp_splits is not None

        self.use_data_properties = use_data_properties
        self.dp_to_prim_type = dp_to_prim_type
        self.dp_splits = dp_splits
        self.type_counts = self._compute_type_counts(pos, kb)

        count = 0
        population = []
        for ind in itertools.cycle(pos):
            population.append(container(self.get_expression(pset, ind, kb)))
            count += 1
            if count == population_size:
                break

        return population

    def get_expression(self, pset: PrimitiveSetTyped,
                       ind: OWLNamedIndividual = None,
                       kb: KnowledgeBase = None) -> List[Union[Primitive, Terminal]]:
        assert ind is not None
        assert kb is not None
        type_ = self._select_type(ind, kb)
        paths = self._select_paths(self._select_properties(ind, kb), ind, kb)

        expr: List[Union[Primitive, Terminal]] = []
        if len(paths) > 0:
            self._add_intersection_or_union(expr, pset)
        self._add_object_terminal(expr, pset, type_)

        for idx, path in enumerate(paths):
            if idx != len(paths) - 1:
                self._add_intersection_or_union(expr, pset)

            if isinstance(path[0], OWLObjectProperty):
                self._build_object_property(expr, ind, path, pset, kb)
            elif isinstance(path[0], OWLDataProperty):
                ranges = set(kb.reasoner().data_property_ranges(path[0]))
                if BooleanOWLDatatype in ranges:
                    self._build_bool_property(expr, path, pset)
                elif IntegerOWLDatatype in ranges or DoubleOWLDatatype in ranges:
                    self._build_float_property(expr, path, pset)

        return expr

    def _compute_type_counts(self, pos: List[OWLNamedIndividual], kb: KnowledgeBase) -> Dict[OWLClass, int]:
        types = itertools.chain.from_iterable({kb.reasoner().types(ind, direct=True) for ind in pos})
        type_counts = dict.fromkeys(types, 0)

        for ind in pos:
            common_types = type_counts.keys() & set(kb.reasoner().types(ind))
            for t in common_types:
                type_counts[t] += 1

        return type_counts

    def _select_type(self, ind: OWLNamedIndividual, kb: KnowledgeBase) -> OWLClass:
        types_ind = list(self.type_counts.keys() & set(kb.reasoner().types(ind)))
        weights = [self.type_counts[t] for t in types_ind]
        return random.choices(types_ind, weights=weights)[0]

    def _select_properties(self, ind: OWLNamedIndividual, kb: KnowledgeBase) -> List[OWLPropertyExpression]:
        properties: List[OWLPropertyExpression] = list(kb.reasoner().object_properties(ind))
        for p in kb.reasoner().data_properties(ind):
            if p in self.dp_to_prim_type:
                properties.append(p)
        return properties

    def _select_paths(self, properties: List[OWLPropertyExpression], ind: OWLNamedIndividual,
                      kb: KnowledgeBase) -> List[Tuple[OWLPropertyExpression, Union[OWLNamedIndividual, Python_Types]]]:
        ind_neighbours: Dict[OWLPropertyExpression, List[Union[OWLNamedIndividual, Python_Types]]] = dict()
        ind_neighbours = {p: self._get_property_values(ind, p, kb) for p in properties}

        paths = []
        if len(properties) < self.max_r:
            paths = [(p, random.choice(ind_neighbours[p])) for p in properties]
        else:
            temp_props = random.sample(properties, k=self.max_r)
            paths = [(p, random.choice(ind_neighbours[p])) for p in temp_props]

        # If not enough paths selected, also taking duplicate properties to different objects
        temp_paths = []
        if len(paths) < self.max_r:
            temp_paths = [(p, o) for p in properties for o in ind_neighbours[p] if (p, o) not in paths]

            remaining_paths = self.max_r - len(paths)
            if len(temp_paths) > remaining_paths:
                paths += random.sample(temp_paths, k=remaining_paths)
            else:
                paths += temp_paths

        return paths

    def _build_object_property(self, expr: List[Union[Primitive, Terminal]], ind: OWLNamedIndividual,
                               path: Tuple[OWLPropertyExpression, OWLNamedIndividual],
                               pset: PrimitiveSetTyped, kb: KnowledgeBase):
        assert isinstance(path[0], OWLObjectProperty)
        self._add_primitive(expr, pset, path[0], OperatorVocabulary.EXISTENTIAL)

        second_ind = path[1]
        assert isinstance(second_ind, OWLNamedIndividual)

        properties = self._select_properties(second_ind, kb)

        # Select next path while prohibiting a loop back to the first individual
        next_path = None
        while next_path is None and len(properties) > 1:
            temp_prop = random.choice(properties)
            objs = self._get_property_values(second_ind, temp_prop, kb)

            if isinstance(temp_prop, OWLObjectProperty):
                try:
                    objs.remove(ind)
                except ValueError:
                    pass

            if len(objs) > 0:
                next_path = (temp_prop, random.choice(objs))

            properties.remove(temp_prop)

        if next_path is not None and random.random() < self.jump_pr:
            if isinstance(next_path[0], OWLObjectProperty):
                self._add_primitive(expr, pset, next_path[0], OperatorVocabulary.EXISTENTIAL)
                assert isinstance(next_path[1], OWLNamedIndividual)
                type_ = random.choice(list(kb.reasoner().types(next_path[1], direct=True)))
                self._add_object_terminal(expr, pset, type_)
            elif isinstance(next_path[0], OWLDataProperty):
                ranges = set(kb.reasoner().data_property_ranges(next_path[0]))
                if BooleanOWLDatatype in ranges:
                    self._build_bool_property(expr, next_path, pset)
                elif IntegerOWLDatatype in ranges or DoubleOWLDatatype in ranges:
                    self._build_float_property(expr, next_path, pset)

        else:
            type_ = random.choice(list(kb.reasoner().types(second_ind, direct=True)))
            self._add_object_terminal(expr, pset, type_)

    def _build_bool_property(self, expr: List[Union[Primitive, Terminal]],
                             path: Tuple[OWLPropertyExpression, bool],
                             pset: PrimitiveSetTyped):
        assert isinstance(path[0], OWLDataProperty)
        self._add_primitive(expr, pset, path[0], OperatorVocabulary.DATA_HAS_VALUE)
        self._add_data_terminal(expr, pset, path)

    def _build_float_property(self, expr: List[Union[Primitive, Terminal]],
                              path: Tuple[OWLPropertyExpression, float],
                              pset: PrimitiveSetTyped):
        assert isinstance(path[0], OWLDataProperty)
        splits = self.dp_splits[path[0]]
        nearest_value = min(splits, key=lambda k: abs(k-path[1])) if len(splits) > 0 else 0
        vocab = OperatorVocabulary.DATA_MIN_INCLUSIVE \
            if nearest_value <= path[1] else OperatorVocabulary.DATA_MAX_INCLUSIVE

        self._add_primitive(expr, pset, path[0], vocab)
        self._add_data_terminal(expr, pset, (path[0], nearest_value))

    def _get_property_values(self, ind: OWLNamedIndividual, property_: OWLPropertyExpression, kb: KnowledgeBase) \
            -> List[Union[OWLNamedIndividual, Python_Types]]:
        if isinstance(property_, OWLObjectProperty):
            return list(kb.reasoner().object_property_values(ind, property_))
        elif isinstance(property_, OWLDataProperty):
            return [lit.to_python() for lit in kb.reasoner().data_property_values(ind, property_)]
        else:
            raise ValueError

    def _add_intersection_or_union(self, expr: List[Union[Primitive, Terminal]], pset: PrimitiveSetTyped):
        if random.random() <= EARandomWalkInitialization.connection_pr:
            expr.append(pset.primitives[OWLClassExpression][2])
        else:
            expr.append(pset.primitives[OWLClassExpression][1])

    def _add_object_terminal(self, expr: List[Union[Primitive, Terminal]], pset: PrimitiveSetTyped, type_: OWLClass):
        for t in pset.terminals[OWLClass]:
            if t.name == escape(type_.get_iri().get_remainder()):
                expr.append(t)
                return

    def _add_data_terminal(self, expr: List[Union[Primitive, Terminal]], pset: PrimitiveSetTyped,
                           path: Tuple[OWLDataProperty, Python_Types]):
        for t in pset.terminals[self.dp_to_prim_type[path[0]]]:
            if t.name == str(path[1]):
                expr.append(t)
                return

    def _add_primitive(self, expr: List[Union[Primitive, Terminal]], pset: PrimitiveSetTyped,
                       path: Union[OWLObjectProperty, OWLDataProperty], vocab: OperatorVocabulary):
        for p in pset.primitives[OWLClassExpression]:
            if p.name == vocab + escape(path.get_iri().get_remainder()):
                expr.append(p)
                return
