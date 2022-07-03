
# Unit Tests #

Tests within this package should only test <br>
code located in a single module where dependencies <br>
to other modules within this project are either mock objects or
extremely simple classes ( Value / Record / Utility classes ) <br>
and therefore almost certainly free of side effects.

Ideally the tested code only works with a single classes.

## What is a Unit? ##

The term **module** mentioned earlier
is commonly what's considered as a testable unit.
However, this project does not contain
highly independent modules in the traditional
sense, which is why the non-unit tests
otherwise found in this project are simply
more compute intensive tests.