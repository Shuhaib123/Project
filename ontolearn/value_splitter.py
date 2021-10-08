from abc import ABCMeta, abstractmethod
from typing import ClassVar, Dict, List, Set

from ontolearn.knowledge_base import KnowledgeBase
from owlapy.model import DoubleOWLDatatype, IntegerOWLDatatype, OWLDataProperty, OWLDatatype, OWLNamedIndividual
from owlapy.model.providers import Provider_Split_Types

import math


class AbstractValueSplitter(metaclass=ABCMeta):
    """Abstract base class for split calculation for data properties.

    """
    __slots__ = 'max_nr_splits'

    supported_datatypes: ClassVar[Set[OWLDatatype]]

    max_nr_splits: int

    @abstractmethod
    def __init__(self, max_nr_splits: int):
        self.max_nr_splits = max_nr_splits

    @abstractmethod
    def compute_splits_properties(self, kb: KnowledgeBase, properties: List[OWLDataProperty] = None) \
            -> Dict[OWLDataProperty, List[Provider_Split_Types]]:
        pass

    def _get_all_properties(self, kb: KnowledgeBase) -> List[OWLDataProperty]:
        properties = []
        for p in kb.ontology().data_properties_in_signature():
            ranges = set(kb.reasoner().data_property_ranges(p))
            if self.supported_datatypes & ranges:
                properties.append(p)
        return properties


class BinningValueSplitter(AbstractValueSplitter):
    """Calculate a number of bins as splits.

    """
    __slots__ = ()

    supported_datatypes: ClassVar[Set[OWLDatatype]] = {IntegerOWLDatatype, DoubleOWLDatatype}

    def __init__(self, max_nr_splits: int = 10):
        super().__init__(max_nr_splits)

    def compute_splits_properties(self, kb: KnowledgeBase, properties: List[OWLDataProperty] = None) \
            -> Dict[OWLDataProperty, List[Provider_Split_Types]]:
        if properties is None:
            properties = self._get_all_properties(kb)
        dp_splits = dict()
        for p in properties:
            values = self._get_values_property(kb, p)
            dp_splits[p] = self._compute_splits_property(values)
        return dp_splits

    def _compute_splits_property(self, values: List[Provider_Split_Types]) -> List[Provider_Split_Types]:
        values = sorted(values)
        nr_splits = min(self.max_nr_splits, len(values))

        splits = set()
        if len(values) > 0:
            splits.add(values[0])
        if len(values) > 1:
            splits.add(values[len(values)-1])

        for i in range(1, nr_splits):
            index = max(math.floor(i * len(values) / nr_splits),
                        math.floor(i * len(values) / (nr_splits - 1) - 1))
            splits.add(self._combine_values(values[index], values[min(index + 1, len(values)-1)]))

        return sorted(list(splits))

    def _combine_values(self, a: Provider_Split_Types, b: Provider_Split_Types) -> Provider_Split_Types:
        assert type(a) == type(b)

        if isinstance(a, int):
            return (a + b) // 2
        elif isinstance(a, float):
            return round((a + b) / 2, 3)
        else:
            raise ValueError

    def _get_values_property(self, kb: KnowledgeBase, property_: OWLDataProperty) -> List[Provider_Split_Types]:
        values = set()
        for i in kb.ontology().individuals_in_signature():
            values.update({lit.to_python() for lit in kb.reasoner().data_property_values(i, property_)})
        return list(values)


class EntropyValueSplitter(AbstractValueSplitter):
    """Calculate the splits depending on the entropy of the resulting sets.

    """
    __slots__ = ()

    def __init__(self, max_nr_splits: int = 5):
        super().__init__(max_nr_splits)

    def compute_splits(self, kb: KnowledgeBase,
                       properties: List[OWLDataProperty] = None,
                       pos: Set[OWLNamedIndividual] = None,
                       neg: Set[OWLNamedIndividual] = None) -> Dict[OWLDataProperty, List[Provider_Split_Types]]:
        assert pos is not None
        assert neg is not None

        if properties is None:
            properties = self._get_all_properties(kb)
