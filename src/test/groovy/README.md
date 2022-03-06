# The Neureka Testsuite #

---

Neureka relies on [Spock](https://github.com/spockframework/spock) as its primary testing framework.
If you are not familiar with Spock don't worry,
[here is a simple introduction for writing and reading 
Spock specifications](Example_Spec.groovy).

## Structure ##

Neureka's testsuite is divided into the following 3 packages : <br>

- `st` : System-Tests : *high level system stress tests/scenarios*
- `it` : Integration-Tests : *cross component tests/scenarios*
- `ut` : Unit-Tests : *single feature tests/scenarios*

Package divisions within those test packages mirror those found in the <br>
main code base, namely : <br>
 ``devices``, ``autograd``, ``calculus``, ``framing``, ``ndim``, ``optimization``, ...<br>
Any of these package names might be present in the three test packages, <br>
however they don't have to be either if a related specification has not yet been implemented <br>
or simply does not make sense for a given reason. <br>

---

## Tests are living / executable documentation ##

This testsuite ought to be viewed as documentation first and foremost!<br>
If you want to add or change specifications and tests then please <br>
keep this in mind and write them as if they are being read as such. <br>

Specifications will automatically be turned into nice html reports located at "docs/spock/reports" <br>
which can even be accessed via [github pages](https://gleethos.github.io/neureka/spock/reports/index.html). <br>


