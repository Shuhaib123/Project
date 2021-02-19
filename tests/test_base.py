""" Test the base module"""

from ontolearn import KnowledgeBase
import os

PATH_FAMILY = 'KGs/Family/family-benchmark_rich_background.owl'
PATH_FATHER = 'KGs/father.owl'


def test_knowledge_base():
    kb = KnowledgeBase(PATH_FAMILY)
    assert kb.name == 'family-benchmark_rich_background'

    assert kb.property_hierarchy
    assert kb.property_hierarchy.all_properties
    assert len(kb.property_hierarchy.all_properties) >= \
           len(kb.property_hierarchy.data_properties)
    assert len(kb.property_hierarchy.all_properties) >= \
           len(kb.property_hierarchy.object_properties)


def test_multiple_knowledge_bases():
    KnowledgeBase(PATH_FAMILY)

    # There should not be an exception here
    # (that refers to the family ontology)
    KnowledgeBase(PATH_FATHER)


def test_knowledge_base_save():
    kb = KnowledgeBase(PATH_FAMILY)
    kb.save('test_kb_save', rdf_format='nt')
    assert os.stat('test_kb_save.nt').st_size > 0
    os.remove('test_kb_save.nt')
