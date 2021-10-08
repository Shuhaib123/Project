from enum import Enum
from typing import Any, Callable, Final, Tuple
from owlapy.model import OWLObjectPropertyExpression, OWLObjectSomeValuesFrom, OWLObjectUnionOf, \
    OWLClassExpression, OWLDataHasValue, OWLDataPropertyExpression, OWLDataSomeValuesFrom, OWLLiteral, \
    OWLObjectAllValuesFrom, OWLObjectIntersectionOf

from ontolearn.knowledge_base import KnowledgeBase
import re

from owlapy.model.providers import OWLDatatypeMaxInclusiveRestriction, OWLDatatypeMinInclusiveRestriction


class PrimitiveFactory:

    __slots__ = 'knowledge_base'

    def __init__(self, knowledge_base: KnowledgeBase):
        self.knowledge_base = knowledge_base

    def create_union(self) -> Callable[[OWLClassExpression, OWLClassExpression], OWLObjectUnionOf]:

        def union(A: OWLClassExpression, B: OWLClassExpression) -> OWLObjectUnionOf:
            return self.knowledge_base.union([A, B])

        return union

    def create_intersection(self) -> Callable[[OWLClassExpression, OWLClassExpression], OWLObjectIntersectionOf]:

        def intersection(A: OWLClassExpression, B: OWLClassExpression) -> OWLObjectIntersectionOf:
            return self.knowledge_base.intersection([A, B])

        return intersection

    def create_existential_universal(self, property_: OWLObjectPropertyExpression) \
            -> Tuple[Callable[[OWLClassExpression], OWLObjectSomeValuesFrom],
                     Callable[[OWLClassExpression], OWLObjectAllValuesFrom]]:

        def existential_restriction(filler: OWLClassExpression) -> OWLObjectSomeValuesFrom:
            return self.knowledge_base.existential_restriction(filler, property_)

        def universal_restriction(filler: OWLClassExpression) -> OWLObjectAllValuesFrom:
            return self.knowledge_base.universal_restriction(filler, property_)

        return existential_restriction, universal_restriction

    def create_data_some_values(self, property_: OWLDataPropertyExpression) \
            -> Tuple[Callable[[Any], OWLDataSomeValuesFrom], Callable[[Any], OWLDataSomeValuesFrom]]:

        def data_some_min_inclusive(value: Any) -> OWLDataSomeValuesFrom:
            filler = OWLDatatypeMinInclusiveRestriction(value)
            return self.knowledge_base.data_existential_restriction(filler, property_)

        def data_some_max_inclusive(value: Any) -> OWLDataSomeValuesFrom:
            filler = OWLDatatypeMaxInclusiveRestriction(value)
            return self.knowledge_base.data_existential_restriction(filler, property_)

        return data_some_min_inclusive, data_some_max_inclusive

    def create_data_has_value(self, property_: OWLDataPropertyExpression) -> Callable[[Any], OWLDataHasValue]:

        def data_has_value(value: Any) -> OWLDataHasValue:
            return self.knowledge_base.data_has_value_restriction(OWLLiteral(value), property_)

        return data_has_value


class OperatorVocabulary(str, Enum):
    UNION: Final = "union"  #:
    INTERSECTION: Final = "intersection"  #:
    NEGATION: Final = "negation"  #:
    EXISTENTIAL: Final = "exists"  #:
    UNIVERSAL: Final = "forall"  #:
    DATA_MIN_INCLUSIVE: Final = "dataMinInc"  #:
    DATA_MAX_INCLUSIVE: Final = "dataMaxInc"  #:
    DATA_HAS_VALUE: Final = "dataHasValue"  #:


def escape(name: str) -> str:
    name = re.sub(r'\W+', '', name)
    return name
