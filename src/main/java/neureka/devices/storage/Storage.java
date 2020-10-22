package neureka.devices.storage;

import neureka.Tsr;

/**
 *  This is an abstract interface which simply describes "a thing that stores tensors".
 *  Therefore the expected method signatures defining this abstract entity biol down
 *  to a "store" and a "restore" method.
 *  Classes like "OpenCLDevice" or "FileDevice" implement this interface indirectly (via the Device interface)
 *  because they are in essence also just entities that store tensors!
 *  Besides the "Device" interface this interface is also extended by the FileHead interface
 *  which is an internal component of the FileDevice architecture...
 *
 * @param <ValueType>
 */
public interface Storage<ValueType>
{
    /**
     *  Implementations of this method ought to store the value
     *  of the given tensor in whatever formant suites the underlying
     *  implementation and or final type.
     *  Classes like "OpenCLDevice" or "FileDevice" are also tensor storages.
     *
     * @param tensor The tensor whose data ought to be stored.
     * @return A reference this object to allow for method chaining. (factory pattern)
     */
    Storage store( Tsr<ValueType> tensor );

    /**
     * @param tensor The tensor whose data ought to be restored (loaded to RAM).
     * @return A reference this object to allow for method chaining. (factory pattern)
     */
    Storage restore( Tsr<ValueType> tensor );

}
