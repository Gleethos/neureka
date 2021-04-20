# Implementations #

This package is disappointingly empty because the concrete  <br>
implementations of this layer are (if no plugins or extensions are present)  <br> 
only located in the acceleration package.  <br>

**Why?** <br>

Well, this layer simply hosts a basic implementation of the <br>
interface `ImplementationFor<TargetDevice extends Device>`  <br>
which is as one might guess **device specific**, meaning  <br>
the context of the implementation is almost always much more <br>
dependent on the device backends than on anything  <br>
contained in the `backend` package ! <br>

So if you want to see examples of this, simply visit any  <br>
sub package of the `backend.standard` package. <br>
It contains its own `implementations` package hosting <br>
implementations of the `ImplementationFor<TargetDevice extends Device>` class. <br>

