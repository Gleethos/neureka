package neureka.devices.storage;


import neureka.Tsr;
import neureka.dtype.DataType;

import java.io.IOException;

public interface FileHead<FinalType, ValueType> extends Storage<ValueType>
{
    /**
     *  An implementation of this method ought
     *  to create a new tensor instance containing the data which
     *  is stored in the file whose access this FileHead manages.
     *
     * @return A new tensor filled with the data from the targeted file.
     * @throws IOException If loading goes wrong an exception is being thrown.
     */
    Tsr<ValueType> load() throws IOException;

    /**
     *
     *  An implementation of this method ought to "free" up the memory used to store a tensor.
     *  Therefore the method is expected to delete the underlying file
     *  whose access this very FileHead implementation manages.
     *  The method also returns an instance of the final implementation of this class,
     *  meaning it adheres to the factory pattern.
     *
     * @return A reference of this very object in order to enable method chaining.
     * @throws IOException
     */
    FinalType free() throws IOException;

    /**
     *  This method return the size of the value which is stored
     *  in the tensor of the file which is managed by this FileHead.
     *  The size however does not represent the byte size of the data.
     *  This means that the returned size is dependent on the data type
     *  of the underlying data of the file...
     *
     * @return The size of the value of the underlying tensor body.
     */
    int getValueSize();

    /**
     *  This method returns the byte size of the data which is stored
     *  in the tensor of the file which is managed by this FileHead.
     *  The underlying datatype of the data within the file does not matter.
     *
     * @return The byte size of the data of the underlying tensor body.
     */
    int getDataSize();

    /**
     *  This method returns the number of bytes which are used to
     *  store the tensor in the file whose access is being managed by an implementation
     *  of th FileHead interface.
     *  Meta data stored inside the file will also be included in this returned size.
     *
     * @return The byte size of all the bytes used to represent the tensor in the file.
     */
    int getTotalSize();

    /**
     *
     *
     * @return The full path as well as name of the file which stores a tensor.
     */
    String getLocation();

    /**
     * @return The name of the file which stores a tensor.
     */
    String getFileName();

    /**
     * @return The data type of the tensor stored in the file which is managed by a FileHead.
     */
    DataType getDataType();

    /**
     * @return The shape of the tensor stored in the file which is managed by a FileHead.
     */
    int[] getShape();

    String extension();
}
