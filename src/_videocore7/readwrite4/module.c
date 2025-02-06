
/*
 * Copyright (c) 2025- Idein Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#if defined(__arm__) && defined(__aarch64__)
#error "__arm__ and __aarch64__ are both defined"
#elif !defined(__arm__) && !defined(__aarch64__)
#error "__arm__ and __aarch64__ are both not defined"
#endif

#include <stdint.h>

static uint32_t read4(const void * const addr)
{
    uint32_t value = 0xDEADBEEF;

    asm volatile (
#if defined(__arm__)
            "ldr %[value], [%[addr]]\n\t"
#elif defined(__aarch64__)
            "ldr %w[value], [%[addr]]\n\t"
#endif
            : [value] "=r" (value)
            : [addr] "r" (addr)
            : "memory"
    );

    return value;
}

static void write4(const void * const addr, const uint32_t value)
{
    asm volatile (
#if defined(__arm__)
            "str %[value], [%[addr]]\n\t"
#elif defined(__aarch64__)
            "str %w[value], [%[addr]]\n\t"
#endif
            :
            : [value] "r" (value),
              [addr] "r" (addr)
            : "memory"
    );
}

static void* py2c_void_p(PyObject* py_c_void_p) {
    PyObject* value = PyObject_GetAttrString(py_c_void_p, "value");
    if (!value) {
        PyErr_SetString(PyExc_TypeError, "Argument is not a c_void_p (missing .value attribute).");
        return NULL;
    }
    void* c_void_p = PyLong_AsVoidPtr(value);
    Py_DECREF(value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return c_void_p;
}

static uint32_t py2c_uint32(PyObject* py_c_uint32) {
    PyObject* value = PyObject_GetAttrString(py_c_uint32, "value");
    if (!value) {
        PyErr_SetString(PyExc_TypeError, "Argument is not a c_uint32 (missing .value attribute).");
        return 0;
    }
    uint32_t c_uint32 = PyLong_AsUnsignedLong(value);
    Py_DECREF(value);
    if (PyErr_Occurred()) {
        return 0;
    }
    return c_uint32;
}

static PyObject* wrap_read4(PyObject *self, PyObject *args) {
    PyObject* obj_addr = NULL; // c_void_p object
    if (!PyArg_ParseTuple(args, "O", &obj_addr)) {
        return NULL;
    }
    void* addr = py2c_void_p(obj_addr);
    return PyLong_FromLong(read4(addr));
}

static PyObject* wrap_write4(PyObject *self, PyObject *args) {
    PyObject* obj_addr = NULL; // c_void_p object
    PyObject* obj_value = NULL; // c_uint32 object
    if (!PyArg_ParseTuple(args, "OO", &obj_addr, &obj_value)) {
        return NULL;
    }
    void* addr = py2c_void_p(obj_addr);
    uint32_t value = py2c_uint32(obj_value);
    write4(addr, value);
    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {
        "read4",
        wrap_read4,
        METH_VARARGS,
        "Read a 4-byte integer from the given register address."
    },
    {
        "write4",
        wrap_write4,
        METH_VARARGS,
        "Write a 4-byte integer value to the given register address."
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_videocore7.readwrite4",
    "Module that provides the V3D register read/rwrite func",
    -1,
    methods,
};

PyMODINIT_FUNC PyInit_readwrite4(void) {
    return PyModule_Create(&module);
}
