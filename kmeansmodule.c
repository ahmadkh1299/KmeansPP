
#define  PY_SSIZE_T_CLEANS
#include <Python.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

/* Interface */
static PyObject* PseudoMain(PyObject *Py_Points, PyObject* Py_initialCent, int PointCount, int dimensions, int k, int max_iter,double epsilon);
int CentroidIndex(double* Point, double** Coordinations, int k,int dimensions);
double* AVG(double* Cluster, int ClusterSize, int ClusterDim);
double distance(double* Point1, double* Point2, int dimensions);
static PyObject* kmeans_capi(PyObject *self, PyObject *args);
static double **PyToC_Arr(PyObject* Py_list, int num_row, int num_col);

/* Wrapping the function (part 2) */
static PyObject* kmeans_capi(PyObject *self, PyObject *args){

    PyObject *data, *centroids;
    int rows, dim, K, MAX_ITER;
    double epsilon;
    if(!PyArg_ParseTuple(args, "OOiiiid", &data, &centroids, &rows, &dim, &K, &MAX_ITER ,&epsilon)){
        return NULL;
    }
    return Py_BuildValue("O", PseudoMain(data, centroids, rows, dim, K, MAX_ITER, epsilon));
}

/* tPyMethodDef (part 3) */
static PyMethodDef capiMethods[] = {

        {"fit",
                (PyCFunction) kmeans_capi,
                METH_VARARGS,
                PyDoc_STR("calculates the centroids using kmeans algorithm")},
        {NULL, NULL, 0, NULL}
};

/* PyModuleDef (part 4) */
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "mykmeanssp",           //name of module
        NULL,                   //module documentation
        -1,                     //size of per-interpreter
        capiMethods
};

/* PyMODINIT_FUNC (part 5) */
PyMODINIT_FUNC PyInit_mykmeanssp(void) {
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    return m;
}

static double **PyToC_Arr(PyObject* Py_list, int num_row, int num_col) {

    int i,j;
    Py_ssize_t Py_i,Py_j;
    double **result;
    PyObject* row; PyObject* num;
    result= malloc(num_row* sizeof(double*));
    if (result == NULL) {
        printf("An Error Has Occurred");
        exit(1);
    }
    for(i=0;i<num_row;i++){
        Py_i = (Py_ssize_t)i;
        result[Py_i] = malloc(num_col * sizeof(double));
        if (result[Py_i] == NULL) {
            printf("An Error Has Occurred");
            exit(1);
        }
        row = PyList_GetItem(Py_list, Py_i);
        if (!PyList_Check(row)){
            continue;
        }
        for(j=0;j<num_col;j++){
            Py_j = (Py_ssize_t)j;
            num= PyList_GetItem(row,Py_j);
            if (!PyFloat_Check(num)) {
                continue;}
            result[Py_i][Py_j]= PyFloat_AsDouble(num);
        }
    }
    return result;
}

double distance(double* Point1, double* Point2, int dimensions) {
    double sum = 0, diff = 0;
    int i;
    for (i = 0; i < dimensions; i++) {
        diff = Point1[i] - Point2[i];
        sum = sum + pow(diff, 2);
    }
    sum = sqrt(sum);
    return sum;
}

int CentroidIndex(double* Point, double** Coordinations, int k,int dimensions) {
    int Index = 0;
    double dist = distance(Coordinations[0], Point, dimensions);
    double min = dist;
    int i;
    for (i = 1; i < k; ++i) {
        dist = distance(Coordinations[i], Point, dimensions);
        if (dist < min) {
            Index = i;
            min = dist;
        }
    }
    return Index;
}

double* AVG(double* Cluster,int ClusterSize,int ClusterDim) {
    double *result;
    int i;
    result = (double *) malloc(ClusterDim * sizeof(double));
    if (result == NULL) {
        printf("An Error Has Occurred");
        exit(1);
    }
    for (i = 0; i < ClusterDim; i++) {
        result[i] = Cluster[i] / ClusterSize;
    }
    return result;
}

/* the main algorithm */
static PyObject* PseudoMain(PyObject *Py_Points, PyObject* Py_initialCent, int PointCount, int dimensions, int k, int max_iter,double epsilon) {

    int i, j,Index=0;
    int flag = 1;
    double eps = epsilon;
    int iterations=0;
    double **centroids, **Coordinations, **ClustersSum,**UpdatedCent;
    int *ClusterSize;
    PyObject *Py_centroids, *Curr_Centroid, *num;

    Py_centroids = PyList_New(k);

    Coordinations = PyToC_Arr(Py_Points,PointCount,dimensions);
    UpdatedCent = PyToC_Arr(Py_initialCent,k,dimensions);


    ClusterSize = (int*) malloc(k * sizeof(int));
    if (ClusterSize == NULL) {
        printf("An Error Has Occurred");
        exit(1);
    }
    ClustersSum = (double **) malloc(k * sizeof(double *));
    if (ClustersSum == NULL) {
        printf("An Error Has Occurred");
        exit(1);
    }
    for (i = 0; i < k; i++) {
        ClustersSum[i] = (double *) malloc(dimensions * sizeof(double));
        if (ClustersSum[i] == NULL) {
            printf("An Error Has Occurred");
            exit(1);
        }
    }
    centroids = (double **) malloc(k * sizeof(double*));
    if (centroids == NULL) {
        printf("An Error Has Occurred");
        exit(1);
    }
    for (i = 0; i < k; i++) {
        centroids[i] = (double *) malloc(dimensions * sizeof(double));
        if (centroids[i] == NULL) {
            printf("An Error Has Occurred");
            exit(1);
        }
    }

    while (iterations < max_iter && flag == 1) {
        iterations++;
        for (i = 0; i < k; i++) {
            ClusterSize[i] = 0;
            for (j = 0; j < dimensions; j++)
                ClustersSum[i][j] = 0;
        }
        for(i=0;i<PointCount;i++){
            Index= CentroidIndex(Coordinations[i],UpdatedCent,k,dimensions);
            ClusterSize[Index]++;
            for(j=0;j<dimensions;j++){
                ClustersSum[Index][j]+=Coordinations[i][j];
            }
        }
        for(i=0;i<k;i++){
            for(j=0;j<dimensions;j++)
                centroids[i][j]=UpdatedCent[i][j];
        }
        for(i=0;i<k;i++){
            free(UpdatedCent[i]);
            UpdatedCent[i]= AVG(ClustersSum[i],ClusterSize[i],dimensions);
        }
        for(i=0;i<k;i++){
            if(distance(UpdatedCent[i],centroids[i],dimensions)>eps){
                flag=1;
                break;
            }
            flag=0;
        }

    }

    if(!Py_centroids)
        return NULL;
    for (i=0; i<k;i++){
        Curr_Centroid = PyList_New(dimensions);
        if (!Curr_Centroid){
            return NULL;
        }
        for(j=0;j<dimensions;j++){
            num = PyFloat_FromDouble(centroids[i][j]);
            if (!num){
                Py_DECREF(Curr_Centroid);
                return NULL;
            }
            PyList_SET_ITEM(Curr_Centroid,j,num);
        }
        PyList_SET_ITEM(Py_centroids,i,Curr_Centroid);
    }

    /*free memory*/
    for (i = 0; i < PointCount; i++) {
        free(Coordinations[i]);
    }
    free(Coordinations);

    for (i = 0; i < k; i++) {
        free(centroids[i]);
        free(ClustersSum[i]);
    }
    for (i=0;i<k;i++)
    {
        free(UpdatedCent[i]);
    }
    free(UpdatedCent);
    free(ClustersSum);
    free(centroids);
    free(ClusterSize);

    return Py_centroids;
}