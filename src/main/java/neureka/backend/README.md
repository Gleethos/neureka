
# Backend #

This package consists of two important sub-packages:

- The `api` package responsible for defining an extensible / modular API for operations

- The `standard` package, which is the standard implementation for the backend API.

## Why is there a backend? ##

This library deals with nd-arrays / tensors as a core data structures
which can be used by a variety of operations.
It would be bad practise to hardcode these operation algorithms
tightly alongside or even into the nd-array / tensor data structure...
This leads to a clean divide of the library into a backend which
consists of operations, and a "frontend", which is simply the rest of the library. 
<br>

> For more information on how to implement custom operations, <br>
read through the
[backend API architecture documentation](api/README.md)!
