
# Unit Tests #

Tests within this package should only test <br>
code located in a single class where dependencies <br>
to other classes within this project are either mock objects or
extremely simple classes ( Value / Record / Utility classes ) <br>
and therefore almost certainly free of side effects.

Ideally the tested code is completely independent of other classes.