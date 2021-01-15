
# :fire: Motivation #

This library has been heavily inspired by [PyTorch](https://github.com/pytorch/pytorch).
A powerful deep learning framework that combines
[dynamic computation](https://medium.com/@omaraymanomar/dynamic-vs-static-computation-graph-2579d1934ecf), performance and debugging freedom!

Popular deep learning frameworks like PyTorch and Tensorflow are heavy weight code bases
which often do not carry with them the benefits of *'write once run everywhere'*.
This is especially true for dedicated <b>Hardware</b>! 

[On the state of Deep Learning outside of CUDAs walled garden.](https://towardsdatascience.com/on-the-state-of-deep-learning-outside-of-cudas-walled-garden-d88c8bbb4342)

This is due to the fact that the backends of these frameworks have been written in nvidia's cuda and C++. 
Which means that even developers willing to compile for all platforms
would still be [locked out of AMD, Intel, and ARM](https://discuss.pytorch.org/t/support-for-amd-rocm-gpu/90404/3) systems when it comes to performance.

For that reason Neureka is written in Java and OpenCl.
Although performance will certainly be impacted 
by this choice, modularity, extensibility, uncomplicated cross platform deployment, and ease of 
use are the benefits.
Additionally, the use of OpenCl even allows for FPGA utilization.

In general, the JVM ecosystem currently plays an underwhelming role in the Deep-Learning community despite
the fact that it is among the most dominant platforms.

[What Java needs for true Machine / Deep Learning support.](https://medium.com/@hsheil/what-java-needs-for-true-machine-deep-learning-support-1571ffdbb594)

Neureka has been built for the JVM not for Java.
