
# Neureka & Groovy without IDE #

This tutorial will show you how to use Neureka in Groovy <br>
after just a few simple linux commands, without an IDE!

---

## System Update ##

Before we start off, make sure that your system is up to date!

**Ubuntu:**
```
sudo apt update
sudo apt -y upgrade
sudo reboot
```

**Fedora:**
```
sudo dnf check-update
sudo dnf upgrade
sudo reboot
```

---

## Installing Groovy ##

First we get `sdkman`, a SDK installation manager, which we install through an installation script.

```
curl -s get.sdkman.io | bash
```

After that we need to set the path for `sdkman`:
```
source "/home/$USER/.sdkman/bin/sdkman-init.sh"
```

Now we need to install a Java SDK which contains the JVM used by the Groovy runtime.

```
sdk install java
```

Finally we install the Groovy SDK (and runtime)!

```
sdk install groovy
```
Let's see if it was successful:

```
groovy -version
```
This should print the latest Groovy version.

---

## Running Neureka in Groovy ##

Now in order to execute Groovy code, simply create a file (like for example: `SomeScript.groovy`)
and send it to the Groovy runtime like so: 
```
groovy SomeScript.groovy
```

One thing still missing here is Neureka! <br>
This, however, can easily be fixed by using Groovy's dynamic package manager: **Grape**<br>
Put the following code at the beginning of your script to load Neureka (or any other dependency you desire).

```groovy

@GrabResolver(name = 'jitpack.io', root = 'https://jitpack.io')
@Grab('com.github.Gleethos:neureka:vx.y.z')

import neureka.*
 
// Now we can start using Neureka:
println Tensor.of(1f, 2f, 0.34f, -9.3f) 

```


