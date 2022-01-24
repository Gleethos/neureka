
# Neueka #

This is the root package containing the tensor class and the following sub-packages:

| Package                                | Description                                                                                                    |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| [devices](devices/README.md)           | a sub-package which enables cross platform acceleration (`OpenCLDevice`) and tensor persistence (`FileDevice`) |
| [calculus](calculus/README.md)         | a sub-package containing collections of functions and the ability to create custom ones                        |
| [optimization](optimization/README.md) | a sub-package for weight-gradient optimization                                                                 |
| [autograd](autograd/README.md)         | the guts of Neurekas autograd system                                                                           |
| [backend](backend/README.md)           | the backend containing both a consistent API and a standard implementation                                     |
| [dtype](dtype/README.md)               | type handling classes including the `DataType` multiton which is being used by tensors to represent their data |
| [ndim](ndim/README.md)                 | tensor specific internals for handling indexing and making multidimensional tensors possible.                  |
| [utility](ndim/README.md)              | common utilities for tensors and friends...                                                                    |


## Architecture ##

The previously listed packages form a layered architecture which governs the
structure of the library as a whole and should be kept in mind when extending it.
<br>
These conceptual layers are the following:

- 1. The data layer
  - `Tsr` alongside packages `ndim`, `dtype` and `framing`
  

- 2. The routing layer: 
  - `calculus`
  - `autograd`


- 3. The execution layer: 
  - `backend`
    - `api` containing 3 layers: `Operation` > `Algorithm` > `ImplementationFor`
    - `standard` : A default (but extendable) implementation of the above API.
  - `devices`
