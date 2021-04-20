# Note #

Neureka uses the Spock testing framework. <br>
Here is some basic information if you are not familiar with it:<br>
<br>
Spock test methods are referred to as "features" which are bundled into a single test class called "specification". <br>
<br>
A Spock specification can have the following fixture methods:

  - The setupSpec() method is invoked before the first feature method is invoked.
  - The setup() method is invoked before every feature method.
  - The cleanup() method is invoked after every feature method.
  - The cleanupSpec() method is invoked after all feature methods have been invoked.

<br>
Test methods are actually String! <br>
They describe the given feature which this test
method ought to cover. 
Besides that there are also multiple 
sections of such a feature method which can also have descriptive labels
