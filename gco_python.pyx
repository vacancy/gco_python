# cython: experimental_cpp_class_def=True
import numpy as np
cimport numpy as np
from cpython cimport bool

np.import_array()

cdef extern from "GCoptimization.h":
    cdef cppclass GCoptimizationGridGraph:
        cppclass SmoothCostFunctor:
            int compute(int s1, int s2, int l1, int l2)

        GCoptimizationGridGraph(int width, int height, int n_labels)
        void setDataCost(int *)
        void setSmoothCost(int *)
        void expansion(int n_iterations)
        void swap(int n_iterations)
        void setSmoothCostVH(int* pairwise, int* V, int* H)
        void setSmoothCostFunctor(SmoothCostFunctor* f)
        void setLabelOrder(bool RANDOM_LABEL_ORDER)
        int whatLabel(int node)
        void setVerbosity(int level)

    cdef cppclass GCoptimizationGeneralGraph:
        GCoptimizationGeneralGraph(int n_vertices, int n_labels)
        void setDataCost(int *)
        void setSmoothCost(int *)
        void setNeighbors(int, int)
        void setNeighbors(int, int, int)
        void expansion(int n_iterations)
        void swap(int n_iterations)
        int whatLabel(int node)

cdef cppclass InpaintFunctor(GCoptimizationGridGraph.SmoothCostFunctor):
    int w
    int h
    int n_labels
    int* image
    int* offsets
    int* known

    # Since I always get confused
    # x+y*width = site label
    # col + row * width
    # channel + (x + y*width) * 3
    # offsets [:,0] row/y offset
    # offsets [:,1] col/x offset
    __init__(int w_, int h_, int n_labels_, int* image_, int* offsets_, int* known_):
        this.w = w_
        this.h = h_
        this.n_labels = n_labels_
        this.image = image_
        this.offsets = offsets_
        this.known = known_

    int imageIndexFromSubs(int x, int y, int c):
        return c + (x + y *this.w) * 3

    int is_known(int x, int y):
        return this.known[x + y * this.w] == 1

    int is_valid(int x, int y):
        return x >=0 and x < this.w and y >= 0 and y < this.h and is_known(x,y)

    int compute_seam(int s, int l1, int l2):
        cdef int x = s % this.w
        cdef int y = (s - x) / this.w

        cdef int x1 = x + this.offsets[1 + 2 * l1]
        cdef int y1 = y + this.offsets[0 + 2 * l1]

        cdef int x2 = x + this.offsets[1 + 2 * l2]
        cdef int y2 = y + this.offsets[0 + 2 * l2]

        # for destination pixels that are not known, bail with 0 energy
        # since single site infinity handles it
        if(not is_valid(x1,y1)):
            return 0
        if(not is_valid(x2,y2)):
            return 0

        cdef int c
        cdef int res
        cdef int t1
        cdef int t2
        res = 0
        for c in range(3):
            t1 = this.image[imageIndexFromSubs(x1,y1,c)]
            t2 = this.image[imageIndexFromSubs(x2,y2,c)]
            tmp = t1 - t2
            res += tmp * tmp
        return res

    int compute(int s1, int s2, int l1, int l2):
        # ||I(s1 + l1) - I(s1 + l2)||^2 + ||I(s2 + l1) - I(s2 + l2)||^2
        if(l1 == l2): return 0
        cdef int e1 = compute_seam(s1,l1,l2)
        cdef int e2 = compute_seam(s2,l1,l2)
        return e1 + e2

def cut_inpaint(np.ndarray[np.int32_t, ndim=3, mode='c'] unary_cost,
        np.ndarray[np.int32_t, ndim=2, mode='c'] offsets,
        np.ndarray[np.int32_t, ndim=3, mode='c'] image,
        np.ndarray[np.int32_t, ndim=2, mode='c'] known,
        n_iter=5,
        algorithm='swap',
        randomizeOrder = False,
        verbosity = 0):
    """
    Apply multi-label graphcuts to grid graph using smoothing inpaint functor for
    pairwise costs

    Parameters
    ----------
    unary_cost: ndarray, int32, shape=(height, width, n_labels)
        Unary potentials
    offets: ndarray, int32, shape=(n_labels, 2)
        Offset for each label
    image: ndarray, int32, shape = (height, width, 3)
        RGB image for calculating pairwise costs
    known: ndarray, int32, shape = (height, width)
        Whether a pixel is in known or unknown region (1 = known, 0 unknown)
    n_iter: int, (default=5)
        Number of iterations
    algorithm: string, `expansion` or `swap`, default=expansion
        Whether to perform alpha-expansion or alpha-beta-swaps.
    randomizeOrder: boolean, default = False
        Whether to randomize min-cut order of swaps/expansions
    verbosity: int, (0 = none, 1 = medium, 2 = max)
        Control debug output from min-cut algorithm
    """

    if unary_cost.shape[2] != offsets.shape[0]:
        raise ValueError("unary_cost and offsets have incompatible shapes.\n"
            "unary_cost must be height x width x n_labels, offsets must be n_labels x 2.\n"
            "Got: unary_cost: (%d, %d, %d), pairwise_cost: (%d, %d)"
            %(unary_cost.shape[0], unary_cost.shape[1], unary_cost.shape[2],
                offsets.shape[0], offsets.shape[1]))
    if image.shape[0] != unary_cost.shape[0] and image.shape[1] != unary_cost.shape[1]:
        raise ValueError("unaray_cost shape must much image shape")
    if image.shape[2] != 3:
        raise ValueError("Image must be RGB")
    if image.shape[0] != known.shape[0] and image.shape[1] != known.shape[1]:
        raise ValueError("known shape must match image shape")

    # everything is ROW major at this point x = col, y = row

    cdef int h = unary_cost.shape[0]
    cdef int w = unary_cost.shape[1]
    cdef int n_labels = offsets.shape[0]

    cdef GCoptimizationGridGraph* gc = new GCoptimizationGridGraph(w, h, n_labels)
    gc.setDataCost(<int*>unary_cost.data)
    gc.setSmoothCostFunctor(<InpaintFunctor*>new InpaintFunctor(w, h, n_labels, <int*>image.data, <int*>offsets.data, <int*>known.data))
    if(randomizeOrder):
        print "Randomizing label order"
        gc.setLabelOrder(True)
    print "Verbosity {0}".format(verbosity)
    gc.setVerbosity(verbosity)
    if algorithm == 'swap':
        gc.swap(n_iter)
    elif algorithm == 'expansion':
        gc.expansion(n_iter)
    else:
        raise ValueError("algorithm should be either `swap` or `expansion`. Got: %s" % algorithm)

    cdef np.npy_intp result_shape[2]
    result_shape[0] = h
    result_shape[1] = w
    cdef np.ndarray[np.int32_t, ndim=2] result = np.PyArray_SimpleNew(2, result_shape, np.NPY_INT32)
    cdef int * result_ptr = <int*>result.data
    for i in xrange(w * h):
        result_ptr[i] = gc.whatLabel(i)
    return result

def cut_simple(np.ndarray[np.int32_t, ndim=3, mode='c'] unary_cost,
        np.ndarray[np.int32_t, ndim=2, mode='c'] pairwise_cost, n_iter=5,
        algorithm='expansion'):
    """
    Apply multi-label graphcuts to grid graph.

    Parameters
    ----------
    unary_cost: ndarray, int32, shape=(width, height, n_labels)
        Unary potentials
    pairwise_cost: ndarray, int32, shape=(n_labels, n_labels)
        Pairwise potentials for label compatibility
    n_iter: int, (default=5)
        Number of iterations
    algorithm: string, `expansion` or `swap`, default=expansion
        Whether to perform alpha-expansion or alpha-beta-swaps.
    """

    if unary_cost.shape[2] != pairwise_cost.shape[0]:
        raise ValueError("unary_cost and pairwise_cost have incompatible shapes.\n"
            "unary_cost must be height x width x n_labels, pairwise_cost must be n_labels x n_labels.\n"
            "Got: unary_cost: (%d, %d, %d), pairwise_cost: (%d, %d)"
            %(unary_cost.shape[0], unary_cost.shape[1], unary_cost.shape[2],
                pairwise_cost.shape[0], pairwise_cost.shape[1]))
    if pairwise_cost.shape[1] != pairwise_cost.shape[0]:
        raise ValueError("pairwise_cost must be a square matrix.")
    cdef int h = unary_cost.shape[1]
    cdef int w = unary_cost.shape[0]
    cdef int n_labels = pairwise_cost.shape[0]
    if (pairwise_cost != pairwise_cost.T).any():
        raise ValueError("pairwise_cost must be symmetric.")

    cdef GCoptimizationGridGraph* gc = new GCoptimizationGridGraph(h, w, n_labels)
    gc.setDataCost(<int*>unary_cost.data)
    gc.setSmoothCost(<int*>pairwise_cost.data)
    if algorithm == 'swap':
        gc.swap(n_iter)
    elif algorithm == 'expansion':
        gc.expansion(n_iter)
    else:
        raise ValueError("algorithm should be either `swap` or `expansion`. Got: %s" % algorithm)

    cdef np.npy_intp result_shape[2]
    result_shape[0] = w
    result_shape[1] = h
    cdef np.ndarray[np.int32_t, ndim=2] result = np.PyArray_SimpleNew(2, result_shape, np.NPY_INT32)
    cdef int * result_ptr = <int*>result.data
    for i in xrange(w * h):
        result_ptr[i] = gc.whatLabel(i)
    return result

def cut_simple_vh(np.ndarray[np.int32_t, ndim=3, mode='c'] unary_cost,
        np.ndarray[np.int32_t, ndim=2, mode='c'] pairwise_cost,
        np.ndarray[np.int32_t, ndim=2, mode='c'] costV,
        np.ndarray[np.int32_t, ndim=2, mode='c'] costH,
        n_iter=5,
        algorithm='expansion'):
    """
    Apply multi-label graphcuts to grid graph.

    Parameters
    ----------
    unary_cost: ndarray, int32, shape=(width, height, n_labels)
        Unary potentials
    pairwise_cost: ndarray, int32, shape=(n_labels, n_labels)
        Pairwise potentials for label compatibility
    costV: ndarray, int32, shape=(width, height)
        Vertical edge weights
    costH: ndarray, int32, shape=(width, height)
        Horizontal edge weights
    n_iter: int, (default=5)
        Number of iterations
    algorithm: string, `expansion` or `swap`, default=expansion
        Whether to perform alpha-expansion or alpha-beta-swaps.
    """

    if unary_cost.shape[2] != pairwise_cost.shape[0]:
        raise ValueError("unary_cost and pairwise_cost have incompatible shapes.\n"
            "unary_cost must be height x width x n_labels, pairwise_cost must be n_labels x n_labels.\n"
            "Got: unary_cost: (%d, %d, %d), pairwise_cost: (%d, %d)"
            %(unary_cost.shape[0], unary_cost.shape[1], unary_cost.shape[2],
                pairwise_cost.shape[0], pairwise_cost.shape[1]))
    if pairwise_cost.shape[1] != pairwise_cost.shape[0]:
        raise ValueError("pairwise_cost must be a square matrix.")
    cdef int h = unary_cost.shape[1]
    cdef int w = unary_cost.shape[0]
    cdef int n_labels = pairwise_cost.shape[0]
    if (pairwise_cost != pairwise_cost.T).any():
        raise ValueError("pairwise_cost must be symmetric.")
    if costV.shape[0] != w or costH.shape[0] != w or costV.shape[1] != h or costH.shape[1] != h:
        raise ValueError("incorrect costV or costH dimensions.")

    cdef GCoptimizationGridGraph* gc = new GCoptimizationGridGraph(h, w, n_labels)
    gc.setDataCost(<int*>unary_cost.data)
    gc.setSmoothCostVH(<int*>pairwise_cost.data, <int*>costV.data, <int*>costH.data)
    if algorithm == 'swap':
        gc.swap(n_iter)
    elif algorithm == 'expansion':
        gc.expansion(n_iter)
    else:
        raise ValueError("algorithm should be either `swap` or `expansion`. Got: %s" % algorithm)

    cdef np.npy_intp result_shape[2]
    result_shape[0] = w
    result_shape[1] = h
    cdef np.ndarray[np.int32_t, ndim=2] result = np.PyArray_SimpleNew(2, result_shape, np.NPY_INT32)
    cdef int * result_ptr = <int*>result.data
    for i in xrange(w * h):
        result_ptr[i] = gc.whatLabel(i)
    return result


def cut_from_graph(np.ndarray[np.int32_t, ndim=2, mode='c'] edges,
        np.ndarray[np.int32_t, ndim=2, mode='c'] unary_cost,
        np.ndarray[np.int32_t, ndim=2, mode='c'] pairwise_cost, n_iter=5,
        algorithm='expansion'):
    """
    Apply multi-label graphcuts to arbitrary graph given by `edges`.

    Parameters
    ----------
    edges: ndarray, int32, shape(n_edges, 2 or 3)
        Rows correspond to edges in graph, given as vertex indices.
        if edges is n_edges x 3 then third parameter is used as edge weight
    unary_cost: ndarray, int32, shape=(n_vertices, n_labels)
        Unary potentials
    pairwise_cost: ndarray, int32, shape=(n_labels, n_labels)
        Pairwise potentials for label compatibility
    n_iter: int, (default=5)
        Number of iterations
    algorithm: string, `expansion` or `swap`, default=expansion
        Whether to perform alpha-expansion or alpha-beta-swaps.
    """
    if (pairwise_cost != pairwise_cost.T).any():
        raise ValueError("pairwise_cost must be symmetric.")

    if unary_cost.shape[1] != pairwise_cost.shape[0]:
        raise ValueError("unary_cost and pairwise_cost have incompatible shapes.\n"
            "unary_cost must be height x width x n_labels, pairwise_cost must be n_labels x n_labels.\n"
            "Got: unary_cost: (%d, %d), pairwise_cost: (%d, %d)"
            %(unary_cost.shape[0], unary_cost.shape[1],
                pairwise_cost.shape[0], pairwise_cost.shape[1]))
    if pairwise_cost.shape[1] != pairwise_cost.shape[0]:
        raise ValueError("pairwise_cost must be a square matrix.")
    cdef int n_vertices = unary_cost.shape[0]
    cdef int n_labels = pairwise_cost.shape[0]

    cdef GCoptimizationGeneralGraph* gc = new GCoptimizationGeneralGraph(n_vertices, n_labels)
    for e in edges:
        if e.shape[0] == 3:
            gc.setNeighbors(e[0], e[1], e[2])
        else:
            gc.setNeighbors(e[0], e[1])
    gc.setDataCost(<int*>unary_cost.data)
    gc.setSmoothCost(<int*>pairwise_cost.data)
    if algorithm == 'swap':
        gc.swap(n_iter)
    elif algorithm == 'expansion':
        gc.expansion(n_iter)
    else:
        raise ValueError("algorithm should be either `swap` or `expansion`. Got: %s" % algorithm)

    cdef np.npy_intp result_shape[1]
    result_shape[0] = n_vertices
    cdef np.ndarray[np.int32_t, ndim=1] result = np.PyArray_SimpleNew(1, result_shape, np.NPY_INT32)
    cdef int * result_ptr = <int*>result.data
    for i in xrange(n_vertices):
        result_ptr[i] = gc.whatLabel(i)
    return result
