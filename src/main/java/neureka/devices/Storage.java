/*
MIT License

Copyright (c) 2019 Gleethos

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

       _____ _
      / ____| |
     | (___ | |_ ___  _ __ __ _  __ _  ___
      \___ \| __/ _ \| '__/ _` |/ _` |/ _ \
      ____) | || (_) | | | (_| | (_| |  __/
     |_____/ \__\___/|_|  \__,_|\__, |\___|
                                 __/ |
                                |___/

         ...a thing that stores things...

*/


package neureka.devices;

import neureka.Tsr;


/**
 *  This is an abstract interface which simply describes "a thing that stores tensors".
 *  Therefore the expected method signatures defining this abstract entity boil down
 *  to a "store" and a "restore" method.
 *  Classes like "OpenCLDevice" or "FileDevice" implement this interface indirectly (via the Device interface)
 *  because they are in essence also just entities that store tensors!
 *  Besides the "Device" interface this interface is also extended by the FileHead interface
 *  which is an internal component of the FileDevice architecture...
 *
 * @param <ValType>
 */
public interface Storage<ValType>
{
    /**
     *  Implementations of this method ought to store the value
     *  of the given tensor in whatever formant suites the underlying
     *  implementation and or final type.
     *  Classes like "OpenCLDevice" or "FileDevice" for example are tensor storages.
     *
     * @param tensor The tensor whose data ought to be stored.
     * @return A reference this object to allow for method chaining. (factory pattern)
     */
    Storage store( Tsr<ValType> tensor );

    /**
     * @param tensor The tensor whose data ought to be restored (loaded to RAM).
     * @return A reference this object to allow for method chaining. (factory pattern)
     */
    Storage restore( Tsr<ValType> tensor );


    int size();

    boolean isEmpty();

    boolean contains( Tsr<ValType> o );


}
