
# Dtype #

Classes within this package are responsible for
representing various data-types
for tensor instances. <br>
Although the JVM already supports
many primitive types, there is
a lack of support for unsigned integer types. <br>

Therefore, this package contains a type agnostic
abstraction over numeric data types. <br>
This leads to the following
interface : `NumericType` <br>
implementations for said interface would 
for example be :                <br>
- `I8` : byte                   <br>
- `UI8` : unsigned byte         <br>
- `I16` : short                 <br>
- `UI16` : unsigned short       <br>
 ...                            <br>
 
 Classes of this type are wrapped inside instances of
 the `DataType` class which contains
 a multiton implementation for managing unique instances
 for the wrapped `Class` instances if `NumericType` instances or 
 otherwise custom types...  <br>
 
 



