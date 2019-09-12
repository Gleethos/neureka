# NEUREKA

[![Build Status](https://travis-ci.com/Gleethos/neureka.svg?branch=master)](https://travis-ci.org/gleethos/neureka)


Neureka is a platform independent deep-learning library written in Java. 

  - Java
  - Aparapi (OpenCl)
  - JavaFX

# Features!

  - Dynamic computation graph.
  - Auto differentiation (forwards/backwards).
  - Unlimited tensor convolution.

You can also:
  - Create a computation graph via a gui interface in JavaFX

> Note:
> Many features are not yet complete/tested 
>

### Tech

This library is heavily inspired by PyTorch.
A powerful deep learning framework that combines
dynamic computation, performance and debugging freedom!

However it lacks one thing! :
Platform independence. 

This is due to the fact that PyTorxh's backend is written
in c++ and cuda.
Which means that it is locked out of many systems by default.

This is the reason why Neureka is written in Java and OpenCl (Aparapi).
Although performance will probably be impacted by this choice,
uncomplicated deployment and ease of use are the benefit.


#### Building for source
Execute the following:
```sh
$ gradlew build
```

### Dependencies

Neureka uses Aparapi and Javafx.


### Development

Want to contribute? Great!

There is currently a lack of sufficient documentation on this repository.
If you have questions simply contact me or read through the test cases 
of this project to understand what neureka is supposed to be!


### Todos

 - Write MORE Tests
 - Much MORE.
 - Allot!

License
----

**It's Free to use!**
