

# Code Conventions #

The code of this project deviates slightly from official or common coding conventions
as one would recognize them in typical Java code.
These deviations boil down to just 3 simple rules which were chosen purposefully and are 
in no way to be mistaken with arbitrariness, or a lack of thought.
To the contrary, the code of this project borrows carefully chosen conventions which are standard
practice in other languages, namely : C#, C++ and Python.

---

## The rules are as follows : ##

---

***1. The names of things may be as long and as descriptive, as their purpose is non-trivial and unknown.***

> **Example :**
>```
>   Tsr<Double> tensor = new Tsr(...);
>   int number = 345335;
>   char letter = 'h';
>```
>   This rule must not be misunderstood : <br>
>   Descriptive names for things in this code base are almost always the preferred choice
>   over short abbreviations or arbitrary letters... 
>   However ! <br>
>   Sometimes the name of something that is generally well-known, widely-present and therefore 
>   easy to understand can be shortened to improve readability by not bloating
>   the code... <br>
>   The reason why the class ``Tsr`` is not called ``Tensor`` is the same reason
>   as to why ``int`` is not called ``integer``. <br>
>   **This tensor class is as essential to Neureka as the primitive integer type
>   is to the Java language.** <br>
>   Both types are extremely common and central to the environment that is used,
>   so much so in fact that giving them descriptive rather than merely recognizable
>   names is simply nonsensical.
>   

***2. Private or protected variable and method names always start with an underscore.***

> **Example :** 
>```
>    private String _stringVariable;
> 
>    protected int _intVariable;
> 
>    public double doubleVariable;
> 
>    private void _someInnerPrivateRoutine() {...}
> 
>    protected void _someInnerProtectedRoutine() {...}
> 
>    public void someOuterAccess() {...}
> ```
>   This rule, as one would typically find it in C++, C# or Python code,
>   improves the readability of code in this repository in one 
>   very considerable way : <br>
>   One can **immediately distinguish local- from field-variables**! 
>   Meaning the code improves in terms of : <br>
>   1. Clearer scope.
>   2. Clearer access levels.
>
>   **Why not just use ``this.fieldVar;`` instead?! :** <br>
>   Because of rule 1 : ``_`` is shorter than ``this`` and the meaning of both is trivial
>   and common enough to not fear confusion. 
>

***3. Round brackets have a white space padding for improved readability.***

>```
>    a += c + d;
>    a = ( a + b ) / ( c * d ); // Instead of : a = (a + b) / (c * d);
>    
>    while ( d++ == s++ ) { // Instead of : while (d++ == s++) { 
>        n++;
>    }
>    printSize( "size is " + foo + "\n" ); // Instead of : printSize("size is "+foo+"\n");
>```
>   This rule guarantees readability in two ways : <br>
>   1. It forces longer statements to be broken down into multiple lines simply due to their length.
>   2. The spaces around involved variables simply makes it easier to identify said variables.
>


---



