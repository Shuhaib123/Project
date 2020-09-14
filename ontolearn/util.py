import datetime
import logging
import os
import pickle
import time
import copy
from owlready2 import get_ontology
import types
from typing import Iterable
import matplotlib.pyplot as plt
from rdflib import Graph, Literal, RDF, URIRef
from rdflib.namespace import OWL, RDFS
from collections import deque
from sklearn.manifold import TSNE
import random

# DEFAULT_FMT = '[{elapsed:0.8f}s] {name}({args}) -> {result}'
DEFAULT_FMT = 'Func:{name} took {elapsed:0.8f}s'

flag_for_performance = False


def parametrized_performance_debugger(fmt=DEFAULT_FMT):
    def decorate(func):
        def clocked(*_args):
            t0 = time.time()
            _result = func(*_args)
            elapsed = time.time() - t0
            name = func.__name__
            args = ', '.join(repr(arg) for arg in _args)
            result = repr(_result)
            if flag_for_performance:
                print(fmt.format(**locals()))
            return _result

        return clocked

    return decorate


def performance_debugger(func_name):
    def function_name_decoratir(func):
        def debug(*args, **kwargs):
            long_string = ''
            starT = time.time()
            # print('######', func_name, ' func ', end=' ')
            r = func(*args, **kwargs)
            print(func_name, ' took ', round(time.time() - starT, 4), ' seconds')
            #           long_string += str(func_name) + ' took:' + str(time.time() - starT) + ' seconds'

            return r

        return debug

    return function_name_decoratir


def decompose(number, upperlimit, bisher, combosTmp):
    """
    TODO: Explain why we need it. We have simply hammered the java code into python here
    TODO: After fully understanding, we could optimize the computation if necessary
    TODO: By simply vectorizing the computations.
    :param number:
    :param upperlimit:
    :param bisher:
    :param combosTmp:
    :return:
    """
    i = min(number, upperlimit)
    while i >= 1:
        newbisher = list()

        if i == 0:
            newbisher = bisher
            newbisher.append(i)
        elif number - i != 1:
            newbisher = copy.copy(bisher)
            newbisher.append(i)

        if number - i > 1:
            decompose(number - i - 1, i, newbisher, combosTmp)
        elif number - i == 0:
            combosTmp.append(newbisher)

        i -= 1


def getCombos(length: int, max_length: int):
    """
    	/**
	 * Methods for computing combinations with the additional restriction
	 * that <code>maxValue</code> is the highest natural number, which can
	 * occur.
	 * @see #getCombos(int)
	 * @param length Length of construct.
	 * @param maxValue Maximum value which can occur in sum.
	 * @return A two dimensional list constructed in {@link #getCombos(int)}.
	 */

    :param i:
    :param max_length:
    :return:
    """
    combosTmp = []
    decompose(length, max_length, [], combosTmp)
    return combosTmp


def incCrossProduct(baseset, newset, exp_gen):
    retset = set()

    if len(baseset) == 0:
        for c in newset:
            retset.add(c)
        return retset
    for i in baseset:
        for j in newset:
            retset.add(exp_gen.union(i, j))
    return retset


def create_experiment_folder(folder_name='Logs'):
    directory = os.getcwd() + '/' + folder_name + '/'
    folder_name = str(datetime.datetime.now())
    path_of_folder = directory + folder_name
    os.makedirs(path_of_folder)
    return path_of_folder, path_of_folder[:path_of_folder.rfind('/')]


def serializer(*, object_: object, path: str, serialized_name: str):
    with open(path + '/' + serialized_name + ".p", "wb") as f:
        pickle.dump(object_, f)
    f.close()


def deserializer(*, path: str, serialized_name: str):
    with open(path + "/" + serialized_name + ".p", "rb") as f:
        obj_ = pickle.load(f)
    f.close()
    return obj_


def get_full_iri(x):
    return x.namespace.base_iri + x.name


def create_logger(*, name, p):
    logger = logging.getLogger(name)

    logger.setLevel(logging.INFO)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(p + '/info.log')
    fh.setLevel(logging.INFO)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


def retrieve_concept_hierarchy(c) -> Iterable:
    """
    c: a node object

    hierarchy: an iterable containing a path of concepts from c to T.
    """
    hierarchy = deque()
    hierarchy.appendleft(c.parent_node)
    while hierarchy[-1].parent_node is not None:
        hierarchy.append(hierarchy[-1].parent_node)
    hierarchy.appendleft(c)
    return hierarchy


def serialize_concepts(concepts: Iterable, serialize_name: str, metric: str, attribute: str, rdf_format: str):
    """
    Given a set of node objects containing concepts.
    1) For each concept obtain its hierarchy based on subsumption.
    2) Serialize each object with its hierarchy including quality.
    """
    g = Graph()
    for c in concepts:
        hierarchy = retrieve_concept_hierarchy(c)  # retrieve concept subsuming c.
        p_quality = URIRef('https://dice-research.org/' + attribute + ':' + metric)

        integer_mapping = {p: str(th) for th, p in enumerate(hierarchy)}
        for th, i in enumerate(hierarchy):

            subject = URIRef('https://dice-research.org/' + integer_mapping[i])
            g.add((subject, RDF.type, RDFS.Class))  # https://dice-research.org/1 type Class.
            g.add((subject, RDFS.label,
                   Literal(i.concept.str)))  # https://dice-research.org/1 type (Mother ⊔ (Son ⊔ ¬Daughter)).
            try:  # if a concept has not been tested.
                val = round(getattr(i, attribute), 3)
            except TypeError:  # if attribute is None.
                val = 0.0
            g.add((subject, p_quality, Literal(val)))

            if i.parent_node:
                g.add((subject, RDFS.subClassOf,
                       URIRef('https://dice-research.org/' + integer_mapping[i.parent_node])))

        g.serialize(destination=serialize_name, format=rdf_format)


def apply_TSNE_on_df(df)-> None:
    low_emb = TSNE(n_components=2).fit_transform(df)
    plt.scatter(low_emb[:, 0], low_emb[:, 1])
    plt.title('Instance Representatons via TSNE')
    plt.show()


def balanced_sets(a: set, b: set) -> (set, set):
    if len(a) > len(b):
        sampled_a = random.sample(a, len(b))
        return sampled_a, b
    elif len(b) > len(a):
        sampled_b = random.sample(b, len(a))
        return a, sampled_b
    else:
        assert len(a) == len(b)
        return a, b
