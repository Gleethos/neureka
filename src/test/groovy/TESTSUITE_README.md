
# The Neureka Testsuite #

---

## Structure

Neureka's testsuite is divided into the following 3 packages : <br>

- `st` : System-Tests : *high level system stress tests/scenarios*

- `it` : Integration-Tests : *cross component tests/scenarios*

- `ut` : Unit-Tests : *single feature tests/scenarios*

Package divisions within those test packages mirror those found in the <br>
main code base, namely : <br>
 ``acceleration``, ``autograd``, ``calculus``, ``framing``, ``ndim``, ``optimization``, ...<br>
Any of these package names might be present in the three test packages, <br>
however they don't have to be either if a related specification has not yet been implemented <br>
or simply does not make sense for a given reason. <br>

---

## Tests are living / executable documentation. ##

This testsuite ought to be viewed as documentation first and foremost!<br>
If specifications and tests shall be added or changed then one should <br>
keep this in mind and write them as if they are being read as such. <br>

Unfortunately Neureka has not enjoyed this development philosophy from the beginning on.
Therefore, many test cases are merely quick imports of former JUnit tests from <br>
before the migration to the Spock-Framework. <br>

Nevertheless, many Specifications can be read and understood very well. <br>

They are also represented as automatically generated reports located at "docs/spock/reports" <br>
which can even be accessed via github pages at "https://gleethos.github.io/neureka/spock/reports/index.html" <br>


