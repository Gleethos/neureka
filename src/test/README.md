# Note #

## 1. ##
If you are looking at the test suite 
in order to find out how to use Neureka,
you might be interested in 
[the living documentation browser](https://gleethos.github.io/neureka/showcase.html).
There you can query the entire test suite using a simple search bar.
 
## 2. ##

As a devoper you might be interested in knowing that
Neureka relies on [Spock](https://github.com/spockframework/spock) as its primary testing framework.<br>
Therefore, the vast majority of tests are located in the [groovy directory](groovy).

If you are not familiar with Spock don't worry,
[here is a simple introduction for writing and reading 
Spock specifications](groovy/Example_Spec.groovy).

---

**TL;DR:**<br>
<br>
Spock test methods are referred to as "**features**" which are bundled into a single test class called "**specification**". <br>

<br>
Test method names can actually be strings! <br>
They describe the covered feature in the form of short sentences. 
A feature consists of multiple 
sections / code blocks which adhere to BDD and will be enforced by Spock.
These sections can receive documentation in the form of strings, which will
then be used to generate the contents of the living documentation browser.

A Spock specification can have the following fixture methods:

  - The `setupSpec()` method is invoked before the first feature method is invoked.
  - The `setup()` method is invoked before every feature method.
  - The `cleanup()` method is invoked after every feature method.
  - The `cleanupSpec()` method is invoked after all feature methods have been invoked.

