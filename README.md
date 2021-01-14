
<center>
 
# :cyclone: [NEUREKA](https://gleethos.github.io/neureka/index.html) - [![HitCount](http://hits.dwyl.com/Gleethos/neureka.svg)](http://hits.dwyl.com/Gleethos/neureka) #
 
 ## **A platform independent tensor library written in Java.** ##
 
</center>

---
 
  > - OpenCL accelerated nd-arrays / tensors for Java, Kotlin, Groovy, Scala, Jython, JRuby...
 
  
- Visit [Neurekas homepage](https://gleethos.github.io/neureka/index.html) for more information!
  
- Try out the latest release: [neureka.jar](https://github.com/Gleethos/neureka/raw/master/production/lib/neureka-0.5.0.jar)
  
[![Beerpay](https://beerpay.io/Gleethos/neureka/badge.svg?style=beer-square)](https://beerpay.io/Gleethos/neureka)  
[![Beerpay](https://beerpay.io/Gleethos/neureka/make-wish.svg?style=flat-square)](https://beerpay.io/Gleethos/neureka?focus=wish)

| Current Build | Code Coverage | Version  | Codacy  | Licence |
| ------------- |:-------------:|:--------:|:-------:| -------:| 
| [![Build Status](https://circleci.com/gh/Gleethos/neureka.svg?branch=master&style=shield)](https://app.circleci.com/pipelines/github/Gleethos/neureka) [![Build Status](https://travis-ci.org/Gleethos/neureka.svg?branch=master)](https://travis-ci.org/Gleethos/neureka) | [![Code Coverage](https://img.shields.io/codecov/c/github/gleethos/neureka)](https://codecov.io/github/gleethos/neureka) | [![GitHub version](https://badge.fury.io/gh/Gleethos%2Fneureka.svg)](https://github.com/Gleethos/neureka) | [![Codacy Badge](https://api.codacy.com/project/badge/Grade/6bfd22ba9b8c410285b19e3d37f4fbc6)](https://www.codacy.com/manual/Gleethos/neureka?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Gleethos/neureka&amp;utm_campaign=Badge_Grade) | [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) |



---  

## :hammer_and_wrench: Features ##

  - dynamic computation graph

  - auto differentiation (forwards/backwards)

  - nd-convolution
  
  - nd-broadcasting

  - flexible tensor indexing and slicing
  
  - seeding
  
  - labeling

  - loading and saving as jpg/idx files

### Take a quick look: ###

- [Neureka with Java](docs/markdown/java_example.md) :coffee:

- [Neureka with Groovy](docs/markdown/groovy_example.md) :star:

---

## :robot: Tech ##

### Dynamic Autograd : Recording the Computation-Graph ### 

Neureka trains your neural network using a computation graph recorder.

This is contrary to the approaches found in other frameworks such as TensorFlow, Theano, Caffe, and CNTK 
which require the definition of a computation graph ahead of time. 
This means a developer has to build a neural network structure which 
cannot change during runtime.   

Neureka, uses the recorded computation graph in order to apply a technique called reverse-mode auto-differentiation, 
which allows your network structure to change during runtime arbitrarily with zero lag or overhead.<br>
This powerful feature has been inspired by PyTorch:
 

- [Motivation](docs/markdown/motivation.md) :fire:
 
### Main-Package Overview ###
 
| Package | Description |
| ---- | --- |
| [**neureka**](src/main/java/neureka/README.md) | the root package containing the tensor class and the following sub-packages |
| [**neureka.devices**](src/main/java/neureka/devices/README.md) | a sub-package which enables cross platform acceleration (`OpenCLDevice`) and tensor persistence (`FileDevice`) |
| [**neureka.calculus**](src/main/java/neureka/calculus/README.md) | a sub-package containing collections of functions and the ability to create custom ones |
| [**neureka.optimization**](src/main/java/neureka/optimization/README.md) | a sub-package for weight-gradient optimization |
| [**neureka.autograd**](src/main/java/neureka/autograd/README.md) | the guts of Neurekas autograd system |
| [**neureka.backend**](src/main/java/neureka/backend/README.md) | the backend containing both a consistent API and a standard implementation  |


 
 
---
## Getting started with Apache Maven: ##

```
<dependency>
  <groupId>com.github.gleethos</groupId>
  <artifactId>neureka</artifactId>
  <version>0.5.0</version>
</dependency>
```
---

## Getting started with Gradle: ##
Groovy DSL:
```
implementation 'com.github.gleethos:neureka:0.5.0'
```
Kotlin DSL:
```
implementation("com.github.gleethos:neureka:0.5.0")
```
...or alternatively: 

1. Add [jitpack](https://jitpack.io/) in your root build.gradle at the end of repositories:
```
allprojects {
	repositories {
		//...
		maven { url 'https://jitpack.io' }
	}
}
```
2. Add Neureka as dependency:
```
dependencies {
	implementation 'com.github.Gleethos:neureka:v0.5.0'
}
```

---

## Getting started with Groovy Grape: ##

```
@GrabResolver(name = 'jitpack.io', root = 'https://jitpack.io')
@Grab('com.github.Gleethos:neureka:v0.5.0')

import neureka.*
```

---

## :rocket: Building from source ##

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

## :mount_fuji: Dependencies ##

- OpenCL - 2.^ (JOCL binding)

- Groovy - 2.^ 

- SLF4J

---

## :book: Documentation ###

- [By example](https://gleethos.github.io/neureka/showcase.html) 

- [Java-Docs](https://gleethos.github.io/neureka/jdocs/index.html)

- [Living Documentation](https://gleethos.github.io/neureka/spock/reports/index.html)!

- [GitHub Wiki](https://github.com/Gleethos/neureka/wiki)

---

## :microscope: Testing & Specification :scroll: ###

- BDD & living documentation with Spock! 

- Yes ! Readable html test reports. [Check it out!](https://gleethos.github.io/neureka/spock/reports/index.html)!

---

## :seedling: Development - [![Commit activity 1 year](https://img.shields.io/github/commit-activity/y/Gleethos/neureka.svg?style=flat)]() - [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Gleethos/neureka/graphs/commit-activity) - [![GitHub commits](https://img.shields.io/github/commits-since/Gleethos/neureka/v0.0.0.svg)](https://GitHub.com/Gleethos/neurka/commit/) ##

Want to contribute? Great!

Although present, the documentation on this project still needs to mature.
So if you have questions simply contact me or read through the test suite 
of this project to understand what Neureka is supposed to be!

Feedback is greatly appreciated!

---

## :memo: Todos - [![Issues](https://img.shields.io/github/issues-raw/Gleethos/neureka.svg?maxAge=25000)](https://github.com/Gleethos/neureka/issues)  ##

  - Make a wish! :)

---

## :balance_scale: License ##

- MIT -> **It's Free!** ... 
    
[![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.png?v=103)](https://github.com/ellerbrock/open-source-badges/)

---

## :v: Support on Beerpay ##
Help me out for a couple of :beers:!

[![Beerpay](https://beerpay.io/Gleethos/neureka/badge.svg?style=beer-square)](https://beerpay.io/Gleethos/neureka)  [![Beerpay](https://beerpay.io/Gleethos/neureka/make-wish.svg?style=flat-square)](https://beerpay.io/Gleethos/neureka?focus=wish)

---

[![Tweet](https://img.shields.io/twitter/url/https/github.com/Gleethos/neureka.svg?style=social)](https://twitter.com/intent/tweet?text=Check%20out%20Neureka!%20https://github.com/Gleethos/neureka)
