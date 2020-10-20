package neureka.devices.storage;


import neureka.Tsr;

import java.io.IOException;

public interface FileHead<FinalType, ValueType> extends Storage<ValueType>
{

    Tsr<ValueType> load() throws IOException;

    FinalType free() throws IOException;

}
