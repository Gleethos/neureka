

# [NEUREKA](https://gleethos.github.io/neureka/index.html) - [![Codacy Badge](https://api.codacy.com/project/badge/Grade/6bfd22ba9b8c410285b19e3d37f4fbc6)](https://www.codacy.com/manual/Gleethos/neureka?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Gleethos/neureka&amp;utm_campaign=Badge_Grade) [![Build Status](https://travis-ci.org/Gleethos/neureka.svg?branch=master)](https://travis-ci.org/Gleethos/neureka) [![Code Coverage](https://img.shields.io/codecov/c/github/gleethos/neureka)](https://codecov.io/github/gleethos/neureka) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![GitHub version](https://badge.fury.io/gh/Gleethos%2Fneureka.svg)](https://github.com/Gleethos/neureka)  #

[![forthebadge](https://forthebadge.com/images/badges/made-with-java.svg)](https://forthebadge.com) 
[![forthebadge](https://forthebadge.com/images/badges/built-with-swag.svg)](https://forthebadge.com) 
[![forthebadge](https://forthebadge.com/images/badges/for-you.svg)](https://forthebadge.com) 
[![forthebadge](https://forthebadge.com/images/badges/certified-elijah-wood.svg)](https://forthebadge.com)
[![forthebadge](https://forthebadge.com/images/badges/check-it-out.svg)](https://forthebadge.com)

[![HitCount](http://hits.dwyl.com/Gleethos/neureka.svg)](http://hits.dwyl.com/Gleethos/neureka)
---

Neureka is a platform independent deep-learning library written in Java. 

  - Java, Kotlin, Groovy, Scala, Jython, JRuby...
 
  - OpenCL accelerated.

  - nd-arrays / tensors.

  - flexible Tensor indexing and slicing.
  
Visit [Neurekas homepage](https://gleethos.github.io/neureka/index.html) for more information!
  
Try out the latest release: [neureka.jar](https://github.com/Gleethos/neureka/raw/master/production/lib/neureka-0.2.4-pre.jar)
  
[![Beerpay](https://beerpay.io/Gleethos/neureka/badge.svg?style=beer-square)](https://beerpay.io/Gleethos/neureka)  
[![Beerpay](https://beerpay.io/Gleethos/neureka/make-wish.svg?style=flat-square)](https://beerpay.io/Gleethos/neureka?focus=wish)

---  

## Features ##

  - dynamic computation graph

  - auto differentiation (forwards/backwards)

  - nd-convolution
  
  - nd-broadcasting

  - slicing
  
  - seeding
  
  - labeling

Take a quick look:
```
    Tsr x = new Tsr(3).setRqsGradient(true);
    Tsr b = new Tsr(-4);
    Tsr w = new Tsr(2);
        
    Tsr y = new Tsr(new Tsr[]{x, b, w}, "((i0+i1)*i2)^2");
    
    /*
     *   f(x) = ((x-4)*2)^2; :=>  f(3) = 4
     *   f(x)' = 8*x - 32 ;  :=>  f(3)' = -8
     *   
     *   y.toString(): "(1):[4.0]; ->d(1):[-8.0]"    
     */
```
Matrix multiplication:
```
    x = new Tsr(
                new int[]{2, 3, 1},
                new double[]{
                        3,   2, -1,
                        -2,  2,  4
                }
    );
    y = new Tsr(
            new int[]{1, 3, 2},
            new double[]{
                    4, -1,  
                    3,  2,  
                    3, -1
            });
    Tsr z = new Tsr(new Tsr[]{x, y}, "i[0] x i[1]");
    
    /*
     *   z.toString(): "(2x1x2):[15.0, 2.0, 10.0, 2.0]"    
     */
```
Convolution:
```
        x = new Tsr(
                new int[]{3, 3},
                new double[]{
                         1, 2, 5,
                        -1, 4,-2,
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

        // z.toString(): "(2x2):[15.0, 15.0, 18.0, 8.0)]"

        z.backward(new Tsr(new int[]{2, 2}, 1));
        /*
         *   y.toString(): "(2x2):[-1.0, 3.0, 2.0, 3.0]:g:[6.0, 9.0, 4.0, 9.0]"    
         */
```

GPU executorFor:
```
        Device gpu = Device.find("nvidia");
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
        z = new Tsr(new Tsr[]{x, y}, "I0xi1"); // <= executed on gpu!

        // z.toString(): "(2x2):[15.0, 15.0, 18.0, 8.0], "

        z.backward(new Tsr(new int[]{2, 2}, 1));
        /*
         *   y.toString(): "(2x2):[-1.0, 3.0, 2.0, 3.0]:g:[6.0, 9.0, 4.0, 9.0]"    
         */
```

---

## Tech ##

This library is being heavily inspired by [PyTorch](https://github.com/pytorch/pytorch).
A powerful deep learning framework that combines
[dynamic computation](https://medium.com/@omaraymanomar/dynamic-vs-static-computation-graph-2579d1934ecf), performance and debugging freedom!

PyTorch however does not carry with it the benefit of *'write once run everywhere'* on <b>every Hardware</b>! 

[On the state of Deep Learning outside of CUDA’s walled garden.](https://towardsdatascience.com/on-the-state-of-deep-learning-outside-of-cudas-walled-garden-d88c8bbb4342)

This is due to the fact that internally PyTorch is written
in C++, and it's GPU unit.acceleration is written in nvidia's cuda. 
Which means that even developers willing to compile for all platforms
would still be locked out of AMD and Intel Systems when it comes to performance.

For that reason Neureka is written in Java and OpenCl.
Although performance on nvidia devices will certainly be impacted 
by this choice, uncomplicated cross platform deployment and ease of use are the benefits.
Additionally, the use of OpenCl theoretically should allow for
FPGA utilization. This however has not been tested.

In general, the JVM ecosystem currently plays an underwhelming role in the Deep-Learning community despite
the fact that it is among the most dominant platforms.

[What Java needs for true Machine / Deep Learning support.](https://medium.com/@hsheil/what-java-needs-for-true-machine-deep-learning-support-1571ffdbb594)

---

## Building from source ##

Execute the following:
```sh
$ gradlew build
```

Tests:
```sh
$ gradlew check
```

Jar file:
```sh
$ gradlew jar
```

Min-jar file:
```sh
$ gradlew proguard
```

---

## Dependencies ##

- OpenCL - 2.^ (JOCL binding)

- Groovy - 3.^ (optional)

---

## Documentation ###

- [By example](https://gleethos.github.io/neureka/showcase.html) 

- [Java-Docs](https://gleethos.github.io/neureka/jdocs/index.html)

---

## Testing & Specification ###

- Spock! 

---

## Development - [![Commit activity 1 year](https://img.shields.io/github/commit-activity/y/Gleethos/neureka.svg?style=flat)]() - [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Gleethos/neureka/graphs/commit-activity) - [![GitHub commits](https://img.shields.io/github/commits-since/Gleethos/neureka/v0.0.0.svg)](https://GitHub.com/Gleethos/neurka/commit/) ##

Want to contribute? Great!

Although present, the documentation on this project still needs to mature.
So if you have questions simply contact me or read through the test suite 
of this project to understand what Neureka is supposed to be!

Feedback is being greatly appreciated!

---

## Todos - [![Issues](https://img.shields.io/github/issues-raw/Gleethos/neureka.svg?maxAge=25000)](https://github.com/Gleethos/neureka/issues)  ##

  - Make a wish! :)

---

## License ##

**It's Free!** ... 

[![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.png?v=103)](https://github.com/ellerbrock/open-source-badges/)

---

## Support on Beerpay ##
Help me out for a couple of :beers:!

[![Beerpay](https://beerpay.io/Gleethos/neureka/badge.svg?style=beer-square)](https://beerpay.io/Gleethos/neureka)  [![Beerpay](https://beerpay.io/Gleethos/neureka/make-wish.svg?style=flat-square)](https://beerpay.io/Gleethos/neureka?focus=wish)

---

[![Tweet](https://img.shields.io/twitter/url/https/github.com/Gleethos/neureka.svg?style=social)](https://twitter.com/intent/tweet?text=Check%20out%20Neureka!%20https://github.com/Gleethos/neureka)