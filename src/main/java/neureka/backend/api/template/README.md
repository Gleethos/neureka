# Templates #

This package holds reusable template implementations
of main interfaces exposed by the backend API, namely:

| layer | interfaces                            | purpose                                                                                                                          |
|-------|---------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| 0     | `Operation`                           | A high level representation of an operation, hosting one or more algorithms.                                                     |
| 1     | `Algorithm`, `DeviceAlgorithm`        | A simple `Algorithm` operating on an `ExecutionCall` or a `DeviceAlgorithm` hosting and calling device specific implementations. |
| 2     | `ImplementationFor<D extends Device>` | `Device` specific implementations hosted and called by `DeviceAlgorithm`s.                                                       |

- Visit the `neureka.backend.api.template.operations` package <br>to find reusable code to build `Operation`s. <br>
- Visit the `neureka.backend.api.template.algorithms` package <br>to find reusable code to build `Algorithm`s and `DeviceAlgorithm`s. <br>
- Visit the `neureka.backend.api.template.implementations` package <br>to find reusable code to build `ImplementationFor<D extends Device>` for `DeviceAlgorithm`s. <br>

---
