# NEUREKA - [![Build Status](https://travis-ci.com/Gleethos/neureka.svg?branch=master)](https://travis-ci.org/gleethos/neureka) - [![Code Coverage](https://img.shields.io/codecov/c/github/gleethos/neureka)](https://codecov.io/github/gleethos/neureka) - [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  

[![forthebadge](https://forthebadge.com/images/badges/made-with-java.svg)](https://forthebadge.com) 
[![forthebadge](https://forthebadge.com/images/badges/built-with-swag.svg)](https://forthebadge.com) 
[![forthebadge](https://forthebadge.com/images/badges/for-you.svg)](https://forthebadge.com) 
[![forthebadge](https://forthebadge.com/images/badges/certified-elijah-wood.svg)](https://forthebadge.com)
[![forthebadge](https://forthebadge.com/images/badges/check-it-out.svg)](https://forthebadge.com)
---

Neureka is a platform independent deep-learning library written in Java. 

  - Java
  - Aparapi (OpenCl)
  - JavaFX
  
[![Beerpay](https://beerpay.io/Gleethos/neureka/badge.svg?style=beer-square)](https://beerpay.io/Gleethos/neureka)  [![Beerpay](https://beerpay.io/Gleethos/neureka/make-wish.svg?style=flat-square)](https://beerpay.io/Gleethos/neureka?focus=wish) 

---  

# Features!

  - Dynamic computation graph.
  - Auto differentiation (forwards/backwards).
  - Unlimited tensor convolution.

Take a look:
```
    Tsr x = new Tsr(3).setRqsGradient(true);
    Tsr b = new Tsr(-4);
    Tsr w = new Tsr(2);
        
    Tsr y = new Tsr(new Tsr[]{x, b, w}, "((i0+i1)*i2)^2");
    
    /**
     *   f(x) = ((x-4)*2)^2; :=>  f(3) = 4
     *   f(x)' = 8*x - 32 ;  :=>  f(3)' = -8
     *   
     *   y.toString(): "[1]:(4.0); ->d[1]:(-8.0), "    
     * */
```
Matrix multiplication:
```
    x = new Tsr(
                new int[]{2, 3, 1},
                new double[]{
                        3,   2,
                       -1,  -2,
                        2,   4
                }
    );
    y = new Tsr(
            new int[]{1, 3, 2},
            new double[]{
                    4, -1,  3,
                    2,  3, -1
            });
    Tsr z = new Tsr(new Tsr[]{x, y}, "i[0] x i[1]");
    
    /**
     *   z.toString(): "[2x1x2]:(19.0, 22.0, 1.0, -6.0), "    
     * */
```
Convolution:
```
        x = new Tsr(
                new int[]{3, 3},
                new double[]{
                        1, 2, 5,
                        -1, 4, -2,
                        -2, 3, 4,
                }
        );
        y = new Tsr(
                new int[]{2, 2},
                new double[]{
                        -1, 3,
                        2, 3,
                });
        z = new Tsr(new Tsr[]{x, y}, "I0xi1");
        z.toString(): "[2x2]:(15.0, 15.0, 18.0, 8.0), "
        z.backward(new Tsr(new int[]{2, 2}, 1));
        /**
         *   y.toString(): "[2x2]:(-1.0, 3.0, 2.0, 3.0):g:(6.0, 9.0, 4.0, 9.0)"    
         * */
```

GPU execution:
```
        Device gpu = new Device("nvidia FP64");
        x = new Tsr(
                new int[]{3, 3},
                new double[]{
                        1, 2, 5,
                        -1, 4, -2,
                        -2, 3, 4,
                }
        );
        y = new Tsr(
                new int[]{2, 2},
                new double[]{
                        -1, 3,
                        2, 3,
                });
        gpu.add(x).add(y);        
        z = new Tsr(new Tsr[]{x, y}, "I0xi1");// <= executed on gpu!
        z.toString(): "[2x2]:(15.0, 15.0, 18.0, 8.0), "
        z.backward(new Tsr(new int[]{2, 2}, 1));
        /**
         *   y.toString(): "[2x2]:(-1.0, 3.0, 2.0, 3.0):g:(6.0, 9.0, 4.0, 9.0)"    
         * */
```


You can also:
  - Create a computation graph via a gui interface in JavaFX

> Note:
> Many features are not yet complete/tested 
>

### Tech

This library is heavily inspired by [PyTorch](https://github.com/pytorch/pytorch).
A powerful deep learning framework that combines
[dynamic computation](https://medium.com/@omaraymanomar/dynamic-vs-static-computation-graph-2579d1934ecf), performance and debugging freedom!

PyTorch however does not carry with it the benefit of *'write once run everywhere'*! 

This is due to the fact that internally PyTorch is written
in C++. Additionally, it's GPU acceleration is written in nvidia's cuda. 
Which means that even developers willing to compile for all platforms
would still be locked out of AMD Systems when it comes to performance.

For that reason Neureka is written in Java and OpenCl (Aparapi).
Although performance will certainly be impacted by this choice,
uncomplicated deployment and ease of use are the benefits.
Additionally, the use of OpenCl theoretically should allow for
FPGA utilization. This however has not been tested.

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
of this project to understand what Neureka is supposed to be!


### Todos

 - Write MORE Tests
 - Much MORE.
 - Allot!

License
----

**It's Free!** ... 

[![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.png?v=103)](https://github.com/ellerbrock/open-source-badges/)

---

## Support on Beerpay
Hey dude! Help me out for a couple of :beers:!

[![Beerpay](https://beerpay.io/Gleethos/neureka/badge.svg?style=beer-square)](https://beerpay.io/Gleethos/neureka)  [![Beerpay](https://beerpay.io/Gleethos/neureka/make-wish.svg?style=flat-square)](https://beerpay.io/Gleethos/neureka?focus=wish)